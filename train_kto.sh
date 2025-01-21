input_model_name=${1:-"ShareGPTVideo/LLaVA-Hound-SFT"}     # rhymes-ai/Aria-Base-8K. THIS SCRIPT NEEDS transformers==4.44
output_model_name=${2:-"/home/cr8dl-user/sameep/experiments/llava_hound_kto"}
lr=${3:-"5e-7"}

cache_dir=/home/cr8dl-user/.cache
export cache_dir=$cache_dir

# export WANDB_MODE=disabled
export WANDB_PROJECT=llava-hound
export WANDB_NAME=scaled_temporal_kto_v1
export WANDB_API_KEY=`echo $WAND_API_KEY`

# gpu_ids=0
gpu_ids=0,1,2,3
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

model_name_or_path=$input_model_name
output_dir=$output_model_name
mkdir -p $output_dir

# DATA
data_path=/home/cr8dl-user/sameep/datasets/llava-hound/temporal_kto_infused_good.json

video_dir=/home/cr8dl-user/sameep/datasets/llava-hound/
image_dir="/"

# sudo chmod +x -R .
export PYTHONPATH="/home/cr8dl-user/sameep/Video-LLMs/llava-hound/venv/bin/python"
rand=$RANDOM
port=$((19000 + $rand % 1000))

# python -m dpo_scripts.run_dpo # NOTE: total_batch_size must be ~64 to 128
torchrun --nproc_per_node=$n_gpu --master_port=$port -m dpo_scripts.run_kto \
    --deepspeed /home/cr8dl-user/sameep/Video-LLMs/llava-hound/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $model_name_or_path \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
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
    --loss_type apo_zero_unpaired \
    --logging_steps 25 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --cache_dir $cache_dir \
    --report_to wandb 2>&1 | tee ./train.log