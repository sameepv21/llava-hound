input_model_name=${1:-"lmms-lab/LLaVA-NeXT-Video-7B"}
output_model_name=${2:-"/scratch/svani/experiments/llava-hound-experiments"}
lr=${3:-"5e-7"}

cache_dir=/scratch/svani/.cache
export cache_dir=$cache_dir

# export WANDB_MODE=disabled
export WANDB_PROJECT=llava-hound
export WANDB_NAME=dpo

# gpu_ids=0
gpu_ids=0
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

model_name_or_path=$input_model_name
output_dir=$output_model_name
mkdir -p $output_dir

# DATA
data_path=/scratch/svani/data/llava-hound/train/dpo/sft_dpo_17k.jsonl

video_dir=/scratch/svani/data/llava-hound/train/train_zip
image_dir="/"

# sudo chmod +x -R .
export PYTHONPATH="/home/svani/.conda/envs/llava-hound/bin/python"
rand=$RANDOM
port=$((19000 + $rand % 1000))

# python -m dpo_scripts.run_dpo \
torchrun --nproc_per_node=$n_gpu --master_port=$port -m dpo_scripts.run_dpo \
    --deepspeed /home/svani/Video-LLMs/LLaVA-Hound/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $model_name_or_path \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --version v1 \
    --data_path $data_path \
    --video_folder $video_dir \
    --image_folder $image_dir \
    --X "Image" "Video" --training_modal 'video' \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_tower LanguageBind/LanguageBind_Video_merge \
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
    --save_steps 24000 \
    --save_only_model True \
    --save_total_limit 1 \
    --learning_rate $lr --freeze_mm_mlp_adapter True \
    --weight_decay 0. --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 50 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --cache_dir $cache_dir \
    --report_to wandb 2>&1 | tee ./train.log