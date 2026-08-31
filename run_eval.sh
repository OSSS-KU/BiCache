#!/bin/bash

model_args="model_path=GSAI-ML/LLaDA-8B-Instruct,\
gen_length=256,\
steps=128,\
show_speed=True,\
block_length=32,\
intra_request_cache_update_interval=16,\
threshold=0.97,\
save_path=./throughput.jsonl,\
num_profiling_data_per_ratio=500,\
max_length_profiling_data=200000,\
cache_budget=5000"

# BiCache
python -m accelerate.commands.launch --num-processes 1 eval.py --tasks arc_challenge_chat --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args ${model_args},caching_policy=bicache,task=arc --batch_size 1 >> accuracy.txt
python -m accelerate.commands.launch --num-processes 1 eval.py --tasks gpqa_main_generative_n_shot --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args ${model_args},caching_policy=bicache,task=gpqa --batch_size 1 >> accuracy.txt
python -m accelerate.commands.launch --num-processes 1 eval.py --tasks minerva_math500 --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args ${model_args},caching_policy=bicache,task=math --batch_size 1 >> accuracy.txt
python -m accelerate.commands.launch --num-processes 1 eval.py --tasks gsm8k --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args ${model_args},caching_policy=bicache,task=gsm8k --batch_size 1 >> accuracy.txt

# BiCache + Fast-dLLM
python -m accelerate.commands.launch --num-processes 1 eval.py --tasks arc_challenge_chat --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args ${model_args},caching_policy=bicache_fastdllm,task=arc --batch_size 1 >> accuracy.txt
python -m accelerate.commands.launch --num-processes 1 eval.py --tasks gpqa_main_generative_n_shot --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args ${model_args},caching_policy=bicache_fastdllm,task=gpqa --batch_size 1 >> accuracy.txt
python -m accelerate.commands.launch --num-processes 1 eval.py --tasks minerva_math500 --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args ${model_args},caching_policy=bicache_fastdllm,task=math --batch_size 1 >> accuracy.txt
python -m accelerate.commands.launch --num-processes 1 eval.py --tasks gsm8k --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args ${model_args},caching_policy=bicache_fastdllm,task=gsm8k --batch_size 1 >> accuracy.txt
