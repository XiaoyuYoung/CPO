nproc_per_node=2

OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
MAX_PIXELS=1003520 \
RESIZED_HEIGHT=512 \
RESIZED_WIDTH=512 \
swift rlhf \
    --rlhf_type dpo \
    --model /output-Qwen2.5-VL-7B-Instruct/v0/checkpoint-660-merged \
    --dataset ./mimic-cpo-cot.json \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --deepspeed zero2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output-Qwen2.5-VL-7B-Instruct-MIMIC-CPO \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8