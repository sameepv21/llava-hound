output_model_name=llava-hound-temporal-6k
model_path=ShareGPTVideo/LLaVA-Hound-SFT
model_base="None"
load_peft=/home/cr8dl-user/sameep/experiments/scaled_temporal_infused_dpo_v1

data_name=nextqa
TEST_DATA_DIR=/home/cr8dl-user/sameep/datasets/llava-hound
TEST_RESULT_DIR=/home/cr8dl-user/sameep/evaluation/${data_name}/llava-hound-temporal-6k

data_path=$TEST_DATA_DIR/temporal_benchmark_sampled.json
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