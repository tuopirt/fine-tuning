# train_lora.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch
import evaluate

# ---- Config ----
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./lora-output"
BATCH_SIZE = 4   # per-GPU; lower if OOM
EPOCHS = 3
LR = 2e-4
MAX_LENGTH = 512

# ---- Load dataset ----
# Example: small alpaca dataset. Replace with your JSONL or dataset.
dataset = load_dataset("yahma/alpaca-cleaned")  # small instruct dataset
train = dataset["train"].select(range(500))     # quick subset for speed
val = dataset["train"].select(range(500, 600))

# ---- Tokenizer & Model ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# Ensure tokenizer has pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_8bit=False,  # set True if you use bitsandbytes + limited VRAM
    torch_dtype=torch.float16  # use fp16 if GPU supports it
)

# ---- Prepare LoRA config ----
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,            # low-rank dimension
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "dense"]  # typical targets; adapt per model
)

model = get_peft_model(model, peft_config)

# ---- Tokenize function ----
def preprocess(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    target = example.get("output", "")
    full = prompt + target
    tokens = tokenizer(full, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    # set labels so loss only computed on response tokens: naive approach — use full labels (okay for this quick run)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train = train.map(preprocess, remove_columns=train.column_names)
val = val.map(preprocess, remove_columns=val.column_names)
train.set_format(type="torch")
val.set_format(type="torch")

# ---- Trainer ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=val,
    tokenizer=tokenizer
)

# ---- Train ----
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("LoRA fine-tune complete. Model saved to", OUTPUT_DIR)
