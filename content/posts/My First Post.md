---
title: "Attention Is All You Need Walkthrough"
date: 2026-02-13
description: "Step by step breakdown and PyTorch implementation"
type: "post"
tags: ["posts"]
showTableOfContents: true

---
## Introduction

The Transformer architecture introduced in *Attention Is All You Need* fundamentally changed deep learning for sequence modeling. Instead of recurrence (RNNs) or convolution (CNNs), it relies entirely on **self-attention mechanisms**.

This walkthrough breaks down:

- Why self-attention works
- The mathematics behind scaled dot-product attention
- Multi-head attention
- Positional encoding
- Encoder block structure
- A minimal PyTorch implementation

---

## 1. The Core Idea: Self-Attention

Traditional RNNs process tokens sequentially:

- Limited parallelization
- Difficulty capturing long-range dependencies
- Vanishing/exploding gradients

Self-attention computes relationships between all tokens simultaneously:

$$
Attention(Q, K, V) =
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:

- $Q$ = Query
- $K$ = Key
- $V$ = Value
- $d_k$ = dimension of key vectors


This enables:

- Full parallelization
- Global receptive field
- Stable gradient flow

---

## 2. Scaled Dot-Product Attention (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** 0.5

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, heads, seq_len, d_k)
        mask: optional attention mask
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)

        return output, attention
