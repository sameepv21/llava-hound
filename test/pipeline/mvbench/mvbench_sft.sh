output_model_name=llava-hound-sft
model_path=ShareGPTVideo/LLaVA-Hound-SFT
model_base="None"
load_peft="None"

data_name=mvbench
TEST_DATA_DIR=/home/cr8dl-user/sameep/datasets/llava-hound
TEST_RESULT_DIR=/home/cr8dl-user/sameep/evaluation/${data_name}/llava-hound-sft

data_path=$TEST_DATA_DIR/mv_bench.json
output_path=$TEST_RESULT_DIR/${data_name}/inference_test_official

cache_dir=/home/cr8dl-user/.cache
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