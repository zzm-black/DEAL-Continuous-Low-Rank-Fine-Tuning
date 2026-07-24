import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add

from uie_collator import DataCollatorForUIE
from uie_dataset_lora import gen_cache_path
from compute_metrics import compute_metrics, compute_grouped_metrics
from model.llama import LlamaForCausalLM_with_lossmask

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, init="ones"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        if init == "ones":
            nn.init.ones_(self.fc1.weight)
            nn.init.ones_(self.fc1.bias)
            nn.init.ones_(self.fc2.weight)
            nn.init.ones_(self.fc2.bias)
        elif init == "kaiming":
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc1.bias)
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc2.bias)
        else:
            raise ValueError(f"Unknown init method: {init}")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class LinearWaveletFilter(nn.Module):
    """Heat-kernel wavelet filtering followed by the DEAL update MLP."""

    def __init__(
        self,
        original_weight: torch.Tensor,
        original_bias: torch.Tensor | None = None,
        *,
        wavelet: str = "heat",
        input_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 8,
        num_kernels: int = 4,
        num_layers: int = 1,
        mlp_init: str = "ones",
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if wavelet.lower() not in {"heat", "heat_kernel"}:
            raise ValueError(
                "DEAL uses the heat kernel from Eq. (8)-(10); "
                "set --wavelet_kernel heat."
            )

        device = device or original_weight.device
        weight = original_weight.detach().clone().float().to(device)

        self.register_buffer("original_weight", weight)

        if original_bias is not None:
            self.register_buffer(
                "original_bias",
                original_bias.detach().clone().float().to(device),
            )
        else:
            self.original_bias = None

        out_features = self.original_weight.shape[0]
        input_dim = input_dim or out_features
        output_dim = output_dim or out_features
        self.num_layers = num_layers

        with torch.no_grad():
            singular_values = torch.linalg.svdvals(self.original_weight)

        self.spectral_rank = singular_values.numel()
        center_min = float(singular_values.min().item())
        center_max = float(singular_values.max().item())
        if abs(center_max - center_min) < 1e-6:
            center_max = center_min + 1e-3

        self.centers = nn.Parameter(
            torch.linspace(
                center_min,
                center_max,
                steps=num_kernels,
                device=device,
            )
        )
        sigma_sq = torch.linspace(
            1.0,
            float(num_kernels),
            steps=num_kernels,
            device=device,
        )
        sigma_sq = sigma_sq * ((center_max - center_min) / num_kernels)
        sigma_sq = sigma_sq.pow(2).clamp_min(1e-6)
        self.log_sigma_sq = nn.Parameter(
            sigma_sq.log()
        )
        self.diagonal_gains = nn.Parameter(
            torch.ones(num_kernels, self.spectral_rank, device=device)
        )

        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            init=mlp_init,
        ).to(device)

    def heat_kernel_features(self, singular_values: torch.Tensor) -> torch.Tensor:
        """Apply the spectral heat-kernel update from Eq. (9)-(10)."""

        features = singular_values

        for _ in range(self.num_layers):
            sigma_sq = self.log_sigma_sq.exp().clamp_min(1e-6)
            positive_diff_sq = (
                features.unsqueeze(0)
                - self.centers.view(-1, 1)
            ).pow(2)
            negative_diff_sq = (
                features.unsqueeze(0)
                + self.centers.view(-1, 1)
            ).pow(2)
            positive_kernel = torch.exp(
                -0.5 * positive_diff_sq / sigma_sq.view(-1, 1)
            )
            negative_kernel = torch.exp(
                -0.5 * negative_diff_sq / sigma_sq.view(-1, 1)
            )
            weighted = (
                positive_kernel
                * self.diagonal_gains
                * negative_kernel
                * features.unsqueeze(0)
            ).sum(dim=0)
            features = F.relu(weighted)

        return features

    def effective_weight(self) -> torch.Tensor:
        u, singular_values, vh = torch.linalg.svd(
            self.original_weight,
            full_matrices=False,
        )
        filtered_singular_values = self.heat_kernel_features(singular_values)
        filtered_weight = (u * filtered_singular_values.unsqueeze(0)) @ vh

        # 对权重矩阵应用MLP，而不是对LoRA输出激活应用MLP。
        # filtered_weight: [out_features, in_features]
        updated_weight = self.mlp(
            filtered_weight.transpose(0, 1)
        ).transpose(0, 1)

        return updated_weight

    def retention_parameters(self):
        """Parameters corresponding to theta_1 in Eq. 12."""
        return [self.centers, self.log_sigma_sq, self.diagonal_gains]

    def adaptation_parameters(self):
        """Parameters corresponding to theta_2 in Eq. 12."""
        return list(self.mlp.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        effective_weight = self.effective_weight().to(
            device=x.device,
            dtype=x.dtype,
        )

        bias = self.original_bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)

        return F.linear(x, effective_weight, bias)
        
def _replace_submodule(
    model: nn.Module,
    module_name: str,
    new_module: nn.Module,
) -> None:
    parent_name, _, child_name = module_name.rpartition(".")

    if parent_name:
        parent_module = model.get_submodule(parent_name)
    else:
        parent_module = model

    setattr(parent_module, child_name, new_module)


def materialize_wavelet_lora_factors(model: nn.Module) -> int:
    """Replace training-time wavelet filters with plain LoRA factor layers.

    DEAL uses the retention and update modules during training, but the paper's
    inference path directly replaces A/B with their updated low-rank matrices.
    Materializing here keeps saved adapters compatible with PEFT and avoids
    carrying the wavelet/MLP modules into inference.
    """

    wavelet_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LinearWaveletFilter)
    ]

    for name, module in wavelet_modules:
        effective_weight = module.effective_weight().detach()
        bias = module.original_bias

        replacement = nn.Linear(
            effective_weight.shape[1],
            effective_weight.shape[0],
            bias=bias is not None,
            device=effective_weight.device,
            dtype=effective_weight.dtype,
        )

        with torch.no_grad():
            replacement.weight.copy_(effective_weight)
            if bias is not None:
                replacement.bias.copy_(
                    bias.to(
                        device=effective_weight.device,
                        dtype=effective_weight.dtype,
                    )
                )

        _replace_submodule(model, name, replacement)

    return len(wavelet_modules)


def wrap_peft_lora_factors(
    model: nn.Module,
    strategy: str = "AB",
    wavelet: str = "heat",
    hidden_dim: int = 128,
    device: torch.device | None = None,
) -> int:
    """Replace selected PEFT LoRA factor modules by DEAL heat-kernel filters."""

    strategy = strategy.upper()

    if strategy not in {"A", "B", "AB"}:
        raise ValueError(
            "adapter_update_strategy must be one of: A, B, AB"
        )

    # 保存旧模块列表，避免替换时改变named_modules迭代结构。
    lora_modules = []

    for name, module in model.named_modules():
        if name.endswith("lora_A.default"):
            lora_modules.append(("A", name, module))
        elif name.endswith("lora_B.default"):
            lora_modules.append(("B", name, module))

    # 冻结旧模型，包括原始LoRA参数。
    for parameter in model.parameters():
        parameter.requires_grad = False

    num_replaced = 0

    for side, name, module in lora_modules:
        should_replace = (
            strategy == "AB"
            or strategy == side
        )

        if not should_replace:
            continue

        original_weight = module.weight
        original_bias = getattr(module, "bias", None)
        out_features = original_weight.shape[0]

        new_module = LinearWaveletFilter(
            original_weight=original_weight,
            original_bias=original_bias,
            wavelet=wavelet,
            input_dim=out_features,
            hidden_dim=max(hidden_dim, min(out_features, 512)),
            output_dim=out_features,
            mlp_init="kaiming",
            device=device or original_weight.device,
        )

        _replace_submodule(model, name, new_module)
        num_replaced += 1

    if num_replaced == 0:
        raise RuntimeError(
            "No LoRA factor was replaced. Check the PEFT module names "
            "and adapter_update_strategy."
        )

    return num_replaced
