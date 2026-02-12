# -*- coding: utf-8 -*-
"""
LoRA fine-tuning demo on a small Hugging Face model (CPU-only, Windows-compatible)

Environment (tested on Windows / Python 3.12):
    pip install pyarrow==15.0.2
    pip install datasets==2.18.0
    pip install transformers==4.38.2
    pip install peft==0.10.0
    pip install accelerate==0.27.2

This script demonstrates:
    - Loading a small pretrained language model (distilgpt2)
    - Injecting LoRA adapters using PEFT
    - Training only LoRA parameters on CPU
    - Saving and reloading the LoRA adapter
    - Performing text generation with the adapted model

Author: Ali Rıza SARAL
Date: 12 Feb 2026
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType


# ————————————————————————————
# 1) Load tokenizer and base model
# ————————————————————————————

model_name = "distilgpt2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 family models do not define a pad token by default.
# We set pad_token = eos_token to enable batching and padding.
tokenizer.pad_token = tokenizer.eos_token

# Load pretrained causal language model
model = AutoModelForCausalLM.from_pretrained(model_name)


# ————————————————————————————
# 2) Inject LoRA adapters (PEFT)
# ————————————————————————————

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # causal language modeling (GPT-style)
    inference_mode=False,          # training mode
    r=8,                           # low-rank dimension (controls adapter capacity)
    lora_alpha=16,                 # LoRA scaling factor
    lora_dropout=0.05              # regularization for LoRA layers
)

# Wrap the base model with LoRA adapters
model = get_peft_model(model, lora_config)

print("LoRA modules added → trainable params:")
model.print_trainable_parameters()


# ————————————————————————————
# 3) Prepare a minimal toy dataset
# ————————————————————————————

texts = [
    "The sky is blue and beautiful.",
    "Today is a sunny day and we are happy.",
    "The quick brown fox jumps over the lazy dog."
]

def tokenize_fn(ex):
    """
    Tokenize input text and create labels for causal language modeling.

    For GPT-style models, labels must be provided explicitly.
    The standard formulation uses labels = input_ids (next-token prediction).
    """
    tok = tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=32
    )

    # Labels are required for computing LM loss
    tok["labels"] = tok["input_ids"].copy()
    return tok

data = Dataset.from_dict({"text": texts})
data = data.map(tokenize_fn, batched=False)
data = data.remove_columns(["text"])
data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# ————————————————————————————
# 4) Training configuration (CPU-safe)
# ————————————————————————————

training_args = TrainingArguments(
    output_dir="./lora_distilgpt2_out",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    logging_steps=5,
    save_total_limit=1,
    fp16=False,                  # FP16 not supported on CPU
    evaluation_strategy="no",
    report_to=[]                 # disable WandB / TensorBoard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data
)


# ————————————————————————————
# 5) Train LoRA adapters (CPU)
# ————————————————————————————

trainer.train()


# ————————————————————————————
# 6) Save LoRA adapter (base model is not duplicated)
# ————————————————————————————

model.save_pretrained("./lora_distilgpt2_adapter")


# ————————————————————————————
# 7) Reload base model + LoRA adapter for inference
# ————————————————————————————

model = AutoModelForCausalLM.from_pretrained(model_name)
model = get_peft_model(model, lora_config)

adapter_name = "demo_lora"
model.load_adapter("./lora_distilgpt2_adapter", adapter_name=adapter_name)
model.set_adapter(adapter_name)

model.eval()


# ————————————————————————————
# 8) Generate sample text
# ————————————————————————————

prompt = "Today I feel"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id  # avoids generation warning
    )

print("\n=== Generated ===")
print(tokenizer.decode(out[0], skip_special_tokens=True))

"""
%runfile

LoRA modules added → trainable params:
trainable params: 147,456 || all params: 82,060,032 || trainable%: 0.17969283755580304

Map:   0%|          | 0/3 [00:00<?, ? examples/s]

=== Generated ===
Today I feel like I deserve to be a part of this game, so I wanted to add a little flair to the game. This is not just about fighting a
"""