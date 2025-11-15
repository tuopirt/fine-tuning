# train_lora_demo.py
"""
Tiny LoRA fine-tuning demo using distilgpt2 (CPU-friendly).
1) loads model & tokenizer
2) applies LoRA adapters via PEFT
3) tokenizes the small dataset
4) runs a short Trainer-based training loop
"""

from datasets import load_dataset                         # dataset utilities
from transformers import (                                # model + training utils
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# presets
MODEL_NAME = "distilgpt2"  # tiny GPT-style model (fast on CPU)
OUTPUT_DIR = "./lora-distilgpt2-demo"
BATCH_SIZE = 2
NUM_EPOCHS = 2
LR = 2e-4
MAX_LENGTH = 64

# initations
#dataset = load_dataset("json", data_files={"train": "tests/test1.jsonl"}, split="train")
dataset = load_dataset("json", data_files={"train": "data.jsonl"}, split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# tokenize
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer)) 

# Optional k-bit training for quantization later


# lora configs
lora_config = LoraConfig(
    r=8,              
    lora_alpha=16,    # scaling factor
    target_modules=["c_attn", "c_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# helper func for tokenization
def preprocess(example):
    text = example["prompt"] + " " + example["completion"] #given prompt and output
    tokens = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)


# training settings !!!
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=5,
    save_steps=100,
    fp16=False,  # CPU: keep float32
    report_to=[]  # disable huggingface logging integrations for local run
)

# train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)
trainer.train()

# output
model.save_pretrained(OUTPUT_DIR) 
print("Saved LoRA adapters to:", OUTPUT_DIR)
