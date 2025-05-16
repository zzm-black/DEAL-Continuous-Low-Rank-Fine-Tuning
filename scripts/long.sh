#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=2

datasets=("yelp" "amazon" "MNLI" "CB" "COPA" "QQP" "RTE" "IMDB" "SST-2" "dbpedia" "agnews" "yahoo" "MultiRC" "BoolQA" "WiC")
epochs=(1 1 2 1 1 1 1 1 1 1 1 1 1 1 1)

model_path="google/flan-t5-large"

for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  round=$((i + 1))
  output_dir="logs_and_outputs/long/outputs/${round}-${dataset}"

  python src/Llama3_run_wavelet.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path "$model_path" \
    --data_dir CL_Benchmark \
    --task_config_dir "configs/long_configs/${dataset}" \
    --instruction_file configs/instruction_config.json \
    --instruction_strategy single \
    --output_dir "$output_dir" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-05 \
    --num_train_epochs "${epochs[$i]}" \
    --run_name "long_round${round}" \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
    --add_task_name True \
    --add_dataset_name True \
    --overwrite_output_dir \
    --overwrite_cache \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 1500 \
    --lambda1 0.01 \
    --lambda2 0.001 \
    --theta_norm_p 10 \
    --mlp_norm_p 2

  model_path="${output_dir}/adapter"

  sleep 5
done
