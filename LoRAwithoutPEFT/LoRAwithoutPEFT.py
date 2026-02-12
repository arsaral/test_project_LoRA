# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:42:00 2026

@author: USER
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------
# LoRA Linear Layer
# ---------------------------
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base_layer

        # Freeze base weights
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


# ---------------------------
# Inject LoRA into DistilBERT
# ---------------------------
def inject_lora_into_distilbert(model, r=4, alpha=8):
    for layer in model.distilbert.transformer.layer:
        attn = layer.attention
        attn.q_lin = LoRALinear(attn.q_lin, r=r, alpha=alpha)
        attn.k_lin = LoRALinear(attn.k_lin, r=r, alpha=alpha)
        attn.v_lin = LoRALinear(attn.v_lin, r=r, alpha=alpha)


# ---------------------------
# Utilities
# ---------------------------
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_params(model):
    return sum(p.numel() for p in model.parameters())


def save_lora_weights(model, path="lora_weights.pt"):
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[name + ".lora_A"] = module.lora_A.detach().cpu()
            lora_state[name + ".lora_B"] = module.lora_B.detach().cpu()
    torch.save(lora_state, path)


def load_lora_weights(model, path="lora_weights.pt"):
    state = torch.load(path, map_location="cpu")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.copy_(state[name + ".lora_A"])
            module.lora_B.data.copy_(state[name + ".lora_B"])


def merge_lora_into_base(model):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta = (module.lora_B @ module.lora_A) * module.scaling
            module.base.weight.data += delta
    return model


# ---------------------------
# Main
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA
    inject_lora_into_distilbert(model, r=4, alpha=8)
    model.to(device)

    print("Trainable params:", count_trainable_params(model))
    print("All params:", count_all_params(model))

    # Dummy training data
    texts = [
        "LoRA is a parameter efficient fine tuning method",
        "Transformers are large neural networks",
        "Music perception relates to cognition",
        "Neural models learn representations"
    ]
    labels = torch.tensor([1, 0, 1, 0]).to(device)

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

    # Save LoRA
    save_lora_weights(model, "lora_weights.pt")
    print("LoRA weights saved.")

    # Reload LoRA (sanity check)
    load_lora_weights(model, "lora_weights.pt")
    print("LoRA weights reloaded.")

    # Optional: merge LoRA into base model
    merge_lora_into_base(model)
    print("LoRA merged into base weights (for deployment).")

    model.eval()
    with torch.no_grad():
        test_inputs = tokenizer(
            ["LoRA enables efficient fine tuning."],
            return_tensors="pt"
        ).to(device)
        logits = model(**test_inputs).logits
        print("Test logits:", logits)


if __name__ == "__main__":
    main()
"""   
%runfile 
Reloaded modules: torch.ops, torch.classes
Device: cpu
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Trainable params: 110592
All params: 67065602
Epoch 00 | Loss: 0.7113
Epoch 01 | Loss: 0.6735
Epoch 02 | Loss: 0.6788
Epoch 03 | Loss: 0.7015
Epoch 04 | Loss: 0.6902
Epoch 05 | Loss: 0.6680
Epoch 06 | Loss: 0.6450
Epoch 07 | Loss: 0.6279
Epoch 08 | Loss: 0.6571
Epoch 09 | Loss: 0.6264
LoRA weights saved.
LoRA weights reloaded.
LoRA merged into base weights (for deployment).
Test logits: tensor([[-0.1698,  0.1826]])
"""