output_model_name=internvideo_sft_cm
model_path=OpenGVLab/InternVideo2_5_Chat_8B
model_base="None"
load_peft=/scratch/svani/experiments/internvideo

TEST_DATA_DIR=/scratch/svani/benchmarks/tempcompass
TEST_RESULT_DIR=/scratch/svani/evaluation

data_name=tempcompass
data_path=$TEST_DATA_DIR/cap_match.json
output_path=$TEST_RESULT_DIR/tempcompass/inference_test_official

cache_dir=/scratch/svani/.cache
VIDEO_DATA_DIR=/scratch/svani/benchmarks/tempcompass

bash test/inference/inference_test_qa.sh \
$data_path \
$output_path/${output_model_name} \
$model_path \
$model_base \
$load_peft \
$cache_dir \
$VIDEO_DATA_DIR

bash test/eval/eval_official_zeroshot_qa.sh $output_path/${output_model_name}.jsonl ${TEST_RESULT_DIR}/${data_name}/eval_test_official