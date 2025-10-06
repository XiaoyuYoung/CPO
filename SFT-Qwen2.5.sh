OMP_NUM_THREADS=12 \
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen2.5-VL-7B-Instruct \
    --dataset ./mimic-cot.json \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 15 \
    --per_device_eval_batch_size 1 \
    --freeze_vit false \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output-Qwen2.5-VL-7B-Instruct \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 12 \
    --dataset_num_proc 12