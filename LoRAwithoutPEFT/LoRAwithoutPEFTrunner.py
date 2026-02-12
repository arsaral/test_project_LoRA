# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:59:39 2026

@author: Ali Rıza SARAL
"""

import math                       # Used for proper initialization scaling
import torch                      # Core PyTorch tensor library
import torch.nn as nn             # Neural network layers
import torch.optim as optim       # Optimizers (AdamW)
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Hugging Face model + tokenizer


# ---------------------------
# LoRA Linear Layer
# ---------------------------
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=4, alpha=8):
        super().__init__()

        # Store original (pretrained) linear layer
        self.base = base_layer

        # Freeze pretrained base weights (they must NOT be updated)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # LoRA rank (r) and scaling factor (alpha / r)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Low-rank matrices A and B (trainable parameters)
        # A: (r x in_features)
        # B: (out_features x r)
        self.lora_A = nn.Parameter(torch.zeros(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))

        # Initialize A with Kaiming init (good for linear layers)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Initialize B with zeros (LoRA paper recommendation: start with no effect)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Forward through frozen pretrained layer
        base_out = self.base(x)

        # Forward through LoRA low-rank update: (x @ A^T) @ B^T
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T

        # Combine base output with scaled LoRA update
        return base_out + self.scaling * lora_out


# ---------------------------
# Inject LoRA into DistilBERT Attention Layers
# ---------------------------
def inject_lora_into_distilbert(model, r=4, alpha=8):
    # Iterate over all transformer layers in DistilBERT
    for layer in model.distilbert.transformer.layer:
        attn = layer.attention  # Self-attention module

        # Replace query, key, value linear layers with LoRA-wrapped versions
        attn.q_lin = LoRALinear(attn.q_lin, r=r, alpha=alpha)
        attn.k_lin = LoRALinear(attn.k_lin, r=r, alpha=alpha)
        attn.v_lin = LoRALinear(attn.v_lin, r=r, alpha=alpha)


# ---------------------------
# Utilities
# ---------------------------
def count_trainable_params(model):
    # Count only parameters that require gradients (i.e., LoRA matrices)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_params(model):
    # Count all parameters of the model (frozen + trainable)
    return sum(p.numel() for p in model.parameters())


def save_lora_weights(model, path="lora_weights.pt"):
    # Dictionary for storing only LoRA parameters
    lora_state = {}

    # Iterate over all submodules in the model
    for name, module in model.named_modules():
        # Only capture LoRA layers
        if isinstance(module, LoRALinear):
            # Store A and B matrices by name
            lora_state[name + ".lora_A"] = module.lora_A.detach().cpu()
            lora_state[name + ".lora_B"] = module.lora_B.detach().cpu()

    # Save LoRA-only state dict
    torch.save(lora_state, path)


def load_lora_weights(model, path="lora_weights.pt"):
    # Load LoRA state dict from disk
    state = torch.load(path, map_location="cpu")

    # Copy saved LoRA parameters back into model
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.copy_(state[name + ".lora_A"])
            module.lora_B.data.copy_(state[name + ".lora_B"])


def merge_lora_into_base(model):
    # Permanently merge LoRA updates into frozen base weights (for deployment)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Compute delta_W = scaling * (B @ A)
            delta = (module.lora_B @ module.lora_A) * module.scaling

            # Add delta_W to original pretrained weight matrix
            module.base.weight.data += delta

    return model


# ---------------------------
# Main
# ---------------------------
def main():
    # Select device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Hugging Face model name (small and fast for testing)
    model_name = "distilbert-base-uncased"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load pretrained DistilBERT with classification head
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Freeze all pretrained parameters (we only want to train LoRA)
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA modules into attention layers
    inject_lora_into_distilbert(model, r=4, alpha=8)

    # Move model to device
    model.to(device)

    # Print parameter statistics
    print("Trainable params:", count_trainable_params(model))
    print("All params:", count_all_params(model))

    texts = [
    "This paper presents an efficient fine-tuning method.",
    "The method fails to converge in many cases.",
    "The experimental results are strong and convincing.",
    "The approach is weak and poorly evaluated."
]

    # Fake binary labels
    labels = torch.tensor([1, 0, 1, 0]).to(device)

    # Tokenize input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    # Optimizer only sees LoRA parameters (since others are frozen)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()                 # Reset gradients
        outputs = model(**inputs, labels=labels)  # Forward pass
        loss = outputs.loss                  # Compute loss
        loss.backward()                      # Backpropagate only through LoRA params
        optimizer.step()                     # Update LoRA weights

        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

    # Save LoRA parameters only
    save_lora_weights(model, "lora_weights.pt")
    print("LoRA weights saved.")

    # Reload LoRA parameters (sanity check)
    load_lora_weights(model, "lora_weights.pt")
    print("LoRA weights reloaded.")

    # Merge LoRA updates into base model weights (deployment step)
    merge_lora_into_base(model)
    print("LoRA merged into base weights (for deployment).")

    # Switch to evaluation mode
    model.eval()

    # ---------------------------
    # Driver code: real usage example (inference pipeline)
    # ---------------------------
    test_sentences = [
        "This method significantly improves fine-tuning efficiency.",
        "The results are disappointing and unclear."
    ]

    test_inputs = tokenizer(
        test_sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**test_inputs).logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    for s, p, pr in zip(test_sentences, preds.tolist(), probs.tolist()):
        print("\nInput:", s)
        print("Predicted class:", p)
        print("Class probabilities:", pr)


# Entry point
if __name__ == "__main__":
    main()

"""
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Trainable params: 110592
All params: 67065602
Epoch 00 | Loss: 0.7010
Epoch 01 | Loss: 0.6973
Epoch 02 | Loss: 0.6967
Epoch 03 | Loss: 0.6715
Epoch 04 | Loss: 0.6648
Epoch 05 | Loss: 0.7068
Epoch 06 | Loss: 0.6636
Epoch 07 | Loss: 0.6885
Epoch 08 | Loss: 0.6607
Epoch 09 | Loss: 0.6690
LoRA weights saved.
LoRA weights reloaded.
LoRA merged into base weights (for deployment).

Input: This method significantly improves fine-tuning efficiency.
Predicted class: 1
Class probabilities: [0.46789973974227905, 0.5321002006530762]

Input: The results are disappointing and unclear.
Predicted class: 1
Class probabilities: [0.480858713388443, 0.5191413164138794]


"""