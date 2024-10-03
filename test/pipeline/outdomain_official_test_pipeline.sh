# # ---------------------------out domain official---------------------------------
# bash test/pipeline/outdomain_official_test_pipeline.sh \
# videollava_dpo \
# $SAVE_DIR/ShareGPT-VideoLLaVA/Video-LLaVA-DPO-Sample-ep3

output_model_name=LLaVA-Next-DPO-v1
model_path=lmms-lab/LLaVA-NeXT-Video-7B
peft_path=/scratch/svani/experiments/llava-hound-experiments/llava-next-video-v1 # None if evaluating baselines
model_base="None"

TEST_DATA_DIR=/scratch/svani/data/llava-hound/test/video_instruction/test
TEST_RESULT_DIR=/scratch/svani/evaluation/llava-hound/dpo

# data_names=(
#     msrvtt
#     msvd
#     tgif
# )
# for i in ${!data_names[@]}; do
    # data_name=${data_names[$i]}
    # data_path=$TEST_DATA_DIR/${data_name}.qa.official.jsonl
    # output_path=$TEST_RESULT_DIR/${data_name}/inference_test_official

    # bash test/inference/inference_test_qa.sh \
    # $data_path \
    # ${output_path}/${output_model_name} \
    # $model_path \
    # $model_base

    # bash test/eval/eval_official_zeroshot_qa.sh $output_path/${output_model_name}.jsonl \
    # ${TEST_RESULT_DIR}/${data_name}/eval_test_official_${GPT_MODEL_NAME} &
# done

data_name=msrvtt
data_path=$TEST_DATA_DIR/msrvtt.qa.official.jsonl
output_path=$TEST_RESULT_DIR/msrvtt/inference_test_official

bash test/inference/inference_test_qa.sh \
$data_path \
$output_path/${output_model_name} \
$model_path \
$model_base

bash test/eval/eval_official_zeroshot_qa.sh $output_path/${output_model_name}.jsonl ${TEST_RESULT_DIR}/msrvtt/eval_test_official