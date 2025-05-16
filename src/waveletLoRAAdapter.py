import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional

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
from pytorch_wavelets import DWTForward, DWTInverse


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
    """Wavelet filtering of a weight matrix followed by an MLP knowledge update."""

    def __init__(
        self,
        original_weight: torch.Tensor,
        original_bias: torch.Tensor | None = None,
        *,
        wavelet: str = "haar",
        input_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 8,
        mlp_init: str = "ones",
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        device = device or original_weight.device
        self.register_buffer("original_weight", original_weight)
        if original_bias is not None:
            self.register_buffer("original_bias", original_bias)
        else:
            self.original_bias = None

        self.dwt = DWTForward(J=1, wave=wavelet, mode="zero").to(device)
        self.idwt = DWTInverse(wave=wavelet, mode="zero").to(device)

        weight_4d = original_weight.unsqueeze(0).unsqueeze(0).to(device)
        cA, _ = self.dwt(weight_4d)
        self.theta = nn.Parameter(torch.ones_like(cA))

        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            init=mlp_init,
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.theta.device)
        weight_4d = self.original_weight.unsqueeze(0).unsqueeze(0).to(self.theta.device)
        cA, cD = self.dwt(weight_4d)
        filtered_weight_4d = self.idwt((cA * self.theta, cD))
        filtered_weight = filtered_weight_4d.squeeze(0).squeeze(0)

        lin_out = F.linear(x, filtered_weight, self.original_bias)
        b, s, d = lin_out.shape
        mlp_in = lin_out.view(-1, d)
        mlp_out = self.mlp(mlp_in)
        return mlp_out.view(b, s, -1)
