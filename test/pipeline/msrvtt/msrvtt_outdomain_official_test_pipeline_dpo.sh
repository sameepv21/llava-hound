output_model_name=llava-hound-dpo-temporal_scaled_6k
model_path=ShareGPTVideo/LLaVA-Hound-SFT
model_base="None"
load_peft=/home/cr8dl-user/sameep/experiments/scaled_temporal_infused_dpo_v1

data_name=msrvtt
TEST_DATA_DIR=/home/cr8dl-user/sameep/datasets/llava-hound
TEST_RESULT_DIR=/home/cr8dl-user/sameep/evaluation/${data_name}/llava_hound_dpo_temporal_v2_scaled_6k

data_path=$TEST_DATA_DIR/msrvtt.qa.jsonl
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