output_model_name=llava-hound-sft
model_path=ShareGPTVideo/LLaVA-Hound-SFT
model_base="None"
load_peft="None"

data_name=video_mme
TEST_DATA_DIR=/scratch/svani/benchmarks/video-mme
TEST_RESULT_DIR=/scratch/svani/evaluation/${data_name}/${output_model_name}

data_path=$TEST_DATA_DIR/annotation.json
output_path=$TEST_RESULT_DIR/${data_name}/inference_test_official

cache_dir=/scratch/svani/.cache
VIDEO_DATA_DIR=$TEST_DATA_DIR

bash test/inference/inference_test_qa.sh \
$data_path \
$output_path/${output_model_name} \
$model_path \
$model_base \
$load_peft \
$cache_dir \
$VIDEO_DATA_DIR

bash test/eval/eval_official_zeroshot_qa.sh $output_path/${output_model_name}.jsonl ${TEST_RESULT_DIR}/${data_name}/eval_test_official
