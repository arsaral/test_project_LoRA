# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

# -------------------------
# LoRA Linear Layer
# -------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):  # r=4
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Frozen base weight (pretrained-style)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02,
            requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # Trainable low-rank adapters
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x):
        base = x @ self.weight.T + self.bias
        lora = (x @ self.A.T) @ self.B.T * self.scaling
        return base + lora


# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with the model but is not trainable
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -------------------------
# Multi-Head Attention
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_lora=False, r=8, alpha=16):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        if use_lora:
            self.q_linear = LoRALinear(d_model, d_model, r=r, alpha=alpha)
            self.k_linear = LoRALinear(d_model, d_model, r=r, alpha=alpha)
            self.v_linear = LoRALinear(d_model, d_model, r=r, alpha=alpha)
        else:
            self.q_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)

        # Output projection (required for proper MHA)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, Tq, _ = q.size()
        B, Tk, _ = k.size()

        Q = self.q_linear(q)
        K = self.k_linear(k)
        V = self.v_linear(v)

        Q = Q.view(B, Tq, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V

        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.out_linear(out)


# -------------------------
# Feed Forward Network
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Encoder Layer
# -------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, use_lora=False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads, use_lora)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


# -------------------------
# Decoder Layer
# -------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, use_lora=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, use_lora)
        self.enc_attn = MultiHeadAttention(d_model, heads, use_lora)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.enc_attn(x, enc_out, enc_out, src_mask))
        x = self.norm3(x + self.ffn(x))
        return x


# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, use_lora=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads, d_ff, use_lora) for _ in range(N)
        ])

    def forward(self, x, mask=None):
        x = self.pe(self.embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return x


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, use_lora=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, heads, d_ff, use_lora) for _ in range(N)
        ])

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = self.pe(self.embed(x))
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x


# -------------------------
# Transformer
# -------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, N=2, heads=4, d_ff=256, use_lora=False):
        super().__init__()

        print("D_MODEL:", d_model)
        print("HEADS:", heads)
        print("D_MODEL % HEADS =", d_model % heads)

        self.encoder = Encoder(src_vocab, d_model, N, heads, d_ff, use_lora)
        self.decoder = Decoder(tgt_vocab, d_model, N, heads, d_ff, use_lora)
        self.out = nn.Linear(d_model, tgt_vocab)

        # Optional: freeze embeddings
        # for p in self.encoder.embed.parameters():
        #     p.requires_grad = False
        # for p in self.decoder.embed.parameters():
        #     p.requires_grad = False

    def make_pad_mask(self, seq, pad_idx=0):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_look_ahead_mask(self, size):
        return torch.tril(torch.ones(size, size)).bool().unsqueeze(0).unsqueeze(1)

    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src)
        tgt_mask = self.make_pad_mask(tgt) & self.make_look_ahead_mask(tgt.size(1)).to(tgt.device)

        enc = self.encoder(src, src_mask)
        dec = self.decoder(tgt, enc, tgt_mask, src_mask)
        return self.out(dec)