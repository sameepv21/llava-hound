output_model_name=llava-hound-dpo-temporal-3k
model_path=ShareGPTVideo/LLaVA-Hound-SFT
model_base="None"
load_peft=/home/cr8dl-user/sameep/experiments/checkpoints_backup/llava-hound-tw-3k

TEST_DATA_DIR=/home/cr8dl-user/sameep/datasets/perceptiontest
TEST_RESULT_DIR=/home/cr8dl-user/sameep/evaluation/verify

data_name=perceptiontest
data_path=$TEST_DATA_DIR/llava_hound_valid.json
output_path=$TEST_RESULT_DIR/perceptiontest/inference_test_official

cache_dir=/home/cr8dl-user/.cache
VIDEO_DATA_DIR=/home/cr8dl-user/sameep/datasets/perceptiontest/frames

bash test/inference/inference_test_qa.sh \
$data_path \
$output_path/${output_model_name} \
$model_path \
$model_base \
$load_peft \
$cache_dir \
$VIDEO_DATA_DIR

bash test/eval/eval_official_zeroshot_qa.sh $output_path/${output_model_name}.jsonl ${TEST_RESULT_DIR}/${data_name}/eval_test_official