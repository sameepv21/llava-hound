output_model_name=llavahound_temporal_text
model_path=ShareGPTVideo/LLaVA-Hound-SFT
model_base="None"
load_peft=/home/cr8dl-user/sameep/experiments/llava-hound-tw

data_name=vinoground
TEST_DATA_DIR=/home/cr8dl-user/sameep/datasets/vinoground
TEST_RESULT_DIR=/home/cr8dl-user/sameep/evaluation/${data_name}/${output_model_name}

data_path=$TEST_DATA_DIR/vinoground_text_lh.json
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
