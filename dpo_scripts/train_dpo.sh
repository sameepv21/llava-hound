input_model_name=${1:-"DAMO-NLP-SG/VideoLLaMA3-7B"}
output_model_name=${2:-"/home/cr8dl-user/sameep/experiments/videollama3_lp_combined"}
lr=${3:-"5e-7"}

cache_dir=/home/cr8dl-user/.cache
export cache_dir=$cache_dir

# export WANDB_MODE=disabled
export WANDB_PROJECT=video-llama3
export WANDB_NAME=video-llama3-ft

# gpu_ids=0
gpu_ids=3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

model_name_or_path=$input_model_name
output_dir=$output_model_name
mkdir -p $output_dir

# DATA
data_path=/home/cr8dl-user/sameep/datasets/timewarp/timewarp_combined_vl3_30k_frames.json

video_dir=/home/cr8dl-user/sameep/datasets/timewarp
image_dir="/"

# sudo chmod +x -R .
export PYTHONPATH="/home/cr8dl-user/arpit/miniforge3/envs/vl3_new/bin/python"
rand=$RANDOM
port=$((19000 + $rand % 1000))

# python -m dpo_scripts.run_dpo \
torchrun --nproc_per_node=$n_gpu --master_port=$port -m dpo_scripts.run_dpo \
    --deepspeed /home/cr8dl-user/sameep/Video-LLMs/finetune_all/video-llama3/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $model_name_or_path \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --version llama_3 \
    --data_path $data_path \
    --video_folder $video_dir \
    --image_folder $image_dir \
    --X "Image" "Video" --training_modal 'video' \
    "--image_tower" "DAMO-NLP-SG/VideoLLaMA3-7B" \
    "--video_tower" "DAMO-NLP-SG/VideoLLaMA3-7B" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_only_model True \
    --save_total_limit 1 \
    --learning_rate $lr --freeze_mm_mlp_adapter True \
    --weight_decay 0. --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 25 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --cache_dir $cache_dir \
    --report_to wandb 2>&1 | tee ./train.log