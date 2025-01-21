# ignore this script; only valid for KTO finetuning of text-based models; For KTO finetuning of multimodal models, need to write LLaVA-like script
import os
import torch
from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType
import wandb

# Login to WandB
API_KEY = os.environ['WAND_API_KEY']
# wandb.login(key=API_KEY)


# Load the base model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)


# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # For text generation tasks
    inference_mode=False,             # Enable training mode
    target_modules=['q_proj', 'k_proj', 'v_proj', 'qkv'],
    r=8,                              # Rank of the low-rank updates
    lora_alpha=16,                    # Scaling factor
    lora_dropout=0.1,                 # Dropout rate
)
model = get_peft_model(model, lora_config)

# Tokenizer and Dataset
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
data_path = "/home/cr8dl-user/sameep/datasets/llava-hound/temporal_kto_infused_good.json"
train_dataset = load_dataset("json", data_files={"train": data_path}, split="train")

# Training arguments
training_args = KTOConfig(
    output_dir="Qwen2VL-Temporal-KTO-LoRA",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    warmup_steps=100,
    bf16=True,  # Use bfloat16 precision
    remove_unused_columns=False,  # Keep all columns for compatibility
)

# Initialize Trainer
trainer = KTOTrainer(
    model=model,
    args=training_args,
    processing_class=processor.tokenizer,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save LoRA Adapters
model.save_pretrained("Qwen2VL-Temporal-KTO-LoRA")
