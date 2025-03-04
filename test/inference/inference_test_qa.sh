cache_dir=$CACHE_DIR
export cache_dir=$cache_dir

data_path=$1
output_dir=$2
model_path=${3:-"Video-LLaVA-Finetune-frames-llava_instruction_623k-videochatgpt_99k"}
base_model_path=${4:-"None"}
load_peft=${5:-"None"}
cache_dir=${6:-"/scratch/svani/.cache"}
VIDEO_DATA_DIR=${7:-"/scratch/svani/data/llava-hound/test/video_data"}

mkdir -p $output_dir

echo data path: $data_path
echo save at $output_dir
echo model path: $model_path
echo base model path: $base_model_path
echo peft model path: $load_peft
echo cache dir: $cache_dir
echo video data dir: $VIDEO_DATA_DIR

# chunking and parallelism
gpu_list="5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

export PYTHONPATH=.

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 test/inference/inference_test_qa.py \
        --model_path ${model_path} --base_model_path ${base_model_path} \
        --cache_dir ${cache_dir} \
        --load_peft ${load_peft} \
        --data_path ${data_path} --video_dir $VIDEO_DATA_DIR \
        --output_dir ${output_dir} \
        --output_name ${CHUNKS}_${IDX}.jsonl \
        --chunks $CHUNKS \
        --chunk_idx $IDX &
done
wait

output_file=${output_dir}.jsonl
echo $output_file

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done