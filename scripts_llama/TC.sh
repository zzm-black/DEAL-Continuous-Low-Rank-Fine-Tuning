#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=2

python src/Llama3_run_wavelet.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
   --data_dir CL_Benchmark \
   --task_config_dir configs/TC_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/TC/outputs/1-dbpedia \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 2 \
   --learning_rate 1e-05 \
   --num_train_epochs 1 \
   --run_name order1_round1 \
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


sleep 5

python src/Llama3_run_wavelet.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_llama/TC/outputs/1-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/TC_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/TC/outputs/2-yahoo \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 2 \
   --learning_rate 1e-05 \
   --num_train_epochs 1 \
   --run_name order1_round2 \
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

sleep 5

python src/Llama3_run_wavelet.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_llama/TC/outputs/2-yahoo/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/TC_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/TC/outputs/3-agnews \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 2 \
   --learning_rate 1e-05 \
   --num_train_epochs 1 \
   --run_name order1_round3 \
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


sleep 5
