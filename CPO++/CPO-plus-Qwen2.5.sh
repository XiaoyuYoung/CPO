# 24GiB
# It is recommended to use padding_free. For more details, please refer to:
# https://github.com/modelscope/ms-swift/blob/main/examples/train/padding_free/dpo.sh
nproc_per_node=2

OMP_NUM_THREADS=8 \ 
CUDA_VISIBLE_DEVICES=0,1 \ 
NPROC_PER_NODE=$nproc_per_node \ 
MAX_PIXELS=1003520 \ 
RESIZED_HEIGHT=512 \ 
RESIZED_WIDTH=512 \ 
swift rlhf \
    --rlhf_type cpo \
    --model /home/xiaoyyan/Data/xiaoyu/models/Qwen2.5-VL-7B-Instruct \
    --train_type lora \
    --dataset /mimic-rft-counterfact-cot.json \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output-Qwen2.5-VL-7B-Instruct-MIMIC \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --rpo_alpha 0.1 \
    --dataset_num_proc 8

