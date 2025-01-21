import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference, decode2frame

video_path = "examples/sample_msrvtt.mp4"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# options ["ShareGPTVideo/LLaVA-Hound-DPO", "ShareGPTVideo/LLaVA-Hound-SFT", "ShareGPTVideo/LLaVA-Hound-SFT-Image_only"]
model_path = "ShareGPTVideo/LLaVA-Hound-DPO" 
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = None, model_name=model_name)
inference_model = ModelInference(model=model, tokenizer=tokenizer, processor=processor, context_len=context_len)

# 
frame_dir, _ = os.path.splitext(video_path)
decode2frame(video_path, frame_dir, verbose=True)
question="What is the evident theme in the video?"
response = inference_model.generate(
    question=question,
    modal_path=frame_dir,
    temperature=0,
)
print(response)

# using decord 
response = inference_model.generate(
    question=question,
    modal_path=video_path,
    temperature=0,
    video_decode_backend="decord",
)
print(response)