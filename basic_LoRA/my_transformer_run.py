import torch
import torch.nn as nn
import torch.optim as optim
import random

from lora_custom_transformer import Transformer

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 20
SEQ_LEN = 6
BATCH_SIZE = 64

D_MODEL = 128   # MUST be divisible by HEADS
HEADS = 4
N_LAYERS = 2
D_FF = 256

# N_EPOCHS = 300
# LR = 1e-3

N_EPOCHS = 600
LR = 2e-3

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


# -----------------------------
# Toy dataset:
# target token = input token + 1 (mod vocab)
# -----------------------------
def generate_example(seq_len=SEQ_LEN):
    """
    Generates one (src, tgt) pair:
    - src: random tokens in [3, VOCAB_SIZE-2]
    - tgt: each token is src + 1 (mod VOCAB_SIZE)
    """
    src = [random.randint(3, VOCAB_SIZE - 2) for _ in range(seq_len)]
    tgt = [(x + 1) % VOCAB_SIZE for x in src]
    return src, tgt


def prepare_batch(batch_size=BATCH_SIZE):
    """
    Creates a batch of (src, tgt) pairs.
    Target sequence is wrapped with BOS and EOS tokens.
    """
    src_batch = []
    tgt_batch = []

    for _ in range(batch_size):
        src, tgt = generate_example()
        src_batch.append(src)
        tgt_batch.append([BOS_IDX] + tgt + [EOS_IDX])

    src_batch = torch.tensor(src_batch, dtype=torch.long, device=DEVICE)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.long, device=DEVICE)
    return src_batch, tgt_batch


# -----------------------------
# LoRA parameter freezing
# -----------------------------
def freeze_base_weights(model):
    """
    Freezes all base model parameters and leaves only LoRA parameters trainable.
    LoRA parameters are detected by:
        - name contains 'lora'
        - OR parameter name ends with '.A' or '.B'
    """
    trainable = 0
    total = 0

    for name, p in model.named_parameters():
        total += p.numel()
        print(name)

        # LoRA parameters: 'lora' in name OR endswith .A / .B
        if "lora" in name.lower() or name.endswith(".A") or name.endswith(".B"):
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False

    print(f"LoRA trainable params: {trainable:,} / {total:,}")


# -----------------------------
# Model
# -----------------------------
USE_LORA = False   # set True to enable LoRA

model = Transformer(
    src_vocab=VOCAB_SIZE,
    tgt_vocab=VOCAB_SIZE,
    d_model=D_MODEL,
    N=N_LAYERS,
    heads=HEADS,
    d_ff=D_FF,
    use_lora=USE_LORA
).to(DEVICE)


# -----------------------------
# Apply freezing (only for LoRA)
# -----------------------------
if USE_LORA:
    freeze_base_weights(model)


# -----------------------------
# Parameter sanity check
# -----------------------------
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable params: {trainable:,} / {total:,}")


# -----------------------------
# Optimizer / Loss
# -----------------------------
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# -----------------------------
# Training Loop
# -----------------------------
print("\nTraining...")

for epoch in range(N_EPOCHS):
    model.train()
    src, tgt = prepare_batch()

    optimizer.zero_grad()

    # Teacher forcing:
    #   input to decoder: tgt[:, :-1]
    #   target for loss:  tgt[:, 1:]
    out = model(src, tgt[:, :-1])
    loss = criterion(out.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))

    loss.backward()
    optimizer.step()

    if epoch % 25 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")


# -----------------------------
# Inference demo
# -----------------------------
model.eval()

with torch.no_grad():
    src, tgt = prepare_batch(1)
    out = model(src, tgt[:, :-1])
    pred = out.argmax(dim=-1)

    print("\nExample:")
    print("SRC:", src[0].tolist())
    print("TGT:", tgt[0, 1:-1].tolist())
    print("PRD:", pred[0].tolist())
  
# -----------------------------
# Experimental logs (kept as documentation)
# -----------------------------
"""
%runfile lora=true
Reloaded modules: lora_custom_transformer
D_MODEL: 128
HEADS: 4
D_MODEL % HEADS = 0
Trainable params: 409,876 / 707,092

Training...
Epoch    0 | Loss: 3.2245
Epoch   25 | Loss: 2.2544
Epoch   50 | Loss: 1.5822
Epoch   75 | Loss: 0.5902
Epoch  100 | Loss: 0.0830
Epoch  125 | Loss: 0.0114
Epoch  150 | Loss: 0.0074
Epoch  175 | Loss: 0.0039
Epoch  200 | Loss: 0.0030
Epoch  225 | Loss: 0.0018
Epoch  250 | Loss: 0.0015
Epoch  275 | Loss: 0.0013

Example:
SRC: [14, 9, 10, 10, 18, 14]
TGT: [15, 10, 11, 11, 19, 15]
PRD: [15, 10, 11, 11, 19, 15, 2]
"""
"""
%runfile lora=false
Reloaded modules: lora_custom_transformer
D_MODEL: 128
HEADS: 4
D_MODEL % HEADS = 0
Trainable params: 670,228 / 670,228

Training...
Epoch    0 | Loss: 3.1614
Epoch   25 | Loss: 0.9411
Epoch   50 | Loss: 0.0383
Epoch   75 | Loss: 0.0255
Epoch  100 | Loss: 0.0169
Epoch  125 | Loss: 0.0488
Epoch  150 | Loss: 0.0037
Epoch  175 | Loss: 0.0037
Epoch  200 | Loss: 0.0074
Epoch  225 | Loss: 0.0183
Epoch  250 | Loss: 0.0030
Epoch  275 | Loss: 0.0062

Example:
SRC: [14, 4, 8, 17, 10, 8]
TGT: [15, 5, 9, 18, 11, 9]
PRD: [15, 5, 9, 18, 11, 9, 2]

"""
"""  LoRA=true
LoRA trainable params: 36,864 / 707,092
Trainable params: 36,864 / 707,092

Training...
Epoch    0 | Loss: 3.2158
Epoch   25 | Loss: 2.8648
Epoch   50 | Loss: 2.7495
Epoch   75 | Loss: 2.3931
Epoch  100 | Loss: 2.1783
Epoch  125 | Loss: 2.1183
Epoch  150 | Loss: 2.0177
Epoch  175 | Loss: 1.8091
Epoch  200 | Loss: 1.6026
Epoch  225 | Loss: 1.3081
Epoch  250 | Loss: 1.0499
Epoch  275 | Loss: 0.5702

Example:
SRC: [8, 11, 14, 13, 17, 8]
TGT: [9, 12, 15, 14, 18, 9]
PRD: [9, 12, 15, 14, 18, 9, 2]
"""

""" LoRA=False
D_MODEL: 128
HEADS: 4
D_MODEL % HEADS = 0
Trainable params: 670,228 / 670,228

Training...
Epoch    0 | Loss: 3.0737
Epoch   25 | Loss: 1.0062
Epoch   50 | Loss: 0.0676
Epoch   75 | Loss: 0.0266
Epoch  100 | Loss: 0.0191
Epoch  125 | Loss: 0.1510
Epoch  150 | Loss: 0.0275
Epoch  175 | Loss: 0.0039
Epoch  200 | Loss: 0.0040
Epoch  225 | Loss: 0.0024
Epoch  250 | Loss: 0.0018
Epoch  275 | Loss: 0.0014

Example:
SRC: [12, 17, 12, 11, 10, 15]
TGT: [13, 18, 13, 12, 11, 16]
PRD: [13, 18, 13, 12, 11, 16, 2]

"""

"""
N_EPOCHS = 600
LR = 2e-3
lora_r = 16
lora_alpha = 16
Lora = True

LoRA trainable params: 36,864 / 707,092
Trainable params: 36,864 / 707,092

Training...
Epoch    0 | Loss: 3.1885
Epoch   25 | Loss: 2.7911
Epoch   50 | Loss: 2.4615
Epoch   75 | Loss: 2.0488
Epoch  100 | Loss: 1.8325
Epoch  125 | Loss: 1.7404
Epoch  150 | Loss: 1.5890
Epoch  175 | Loss: 1.1796
Epoch  200 | Loss: 0.8322
Epoch  225 | Loss: 0.4001
Epoch  250 | Loss: 0.2942
Epoch  275 | Loss: 0.1493
Epoch  300 | Loss: 0.1141
Epoch  325 | Loss: 0.0992
Epoch  350 | Loss: 0.0919
Epoch  375 | Loss: 0.0854
Epoch  400 | Loss: 0.0829
Epoch  425 | Loss: 0.0778
Epoch  450 | Loss: 0.0745
Epoch  475 | Loss: 0.0703
Epoch  500 | Loss: 0.0690
Epoch  525 | Loss: 0.0661
Epoch  550 | Loss: 0.0648
Epoch  575 | Loss: 0.0613

Example:
SRC: [3, 10, 18, 14, 13, 18]
TGT: [4, 11, 19, 15, 14, 19]
PRD: [4, 11, 15, 15, 14, 14, 2]

"""

"""
N_EPOCHS = 600
LR = 2e-3
lora_r = 16
lora_alpha = 16
Lora = False



"""