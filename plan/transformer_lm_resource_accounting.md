# Transformer LM Resource Accounting

This note answers the **Transformer LM resource accounting** questions from `cs336_spring2025_assignment1_basics.pdf`, using the implementation in `cs336_basics/model.py`.

## Assumptions

- The model is this repo's `TransformerLM`, not the original GPT-2 implementation exactly.
- Token embeddings and the LM head are **not tied** in this implementation.
- `Linear.weight` has shape `(out_features, in_features)`, and `Linear.forward()` computes `x @ weight.T`.
- There are no bias parameters in this repo's `Linear` implementation.
- Batch size is treated as `1` for accounting.
- Context length is `T = 1024` unless stated otherwise.
- FP32 uses `4 bytes` per parameter.
- A matrix multiply `(m x k) @ (k x n)` costs `2*m*k*n` FLOPs.
- For FLOP accounting, only matrix multiplies are counted. Embedding lookup, RoPE, RMSNorm, softmax, masking, residual additions, and activation functions are ignored.

## Model Components in This Codebase

`TransformerLM` contains:

- `token_embeddings = Embedding(vocab_size, d_model)`
- `layers = ModuleList([...])` with `num_layers` Transformer blocks
- `ln_final = RMSNorm(d_model)`
- `lm_head = Linear(d_model, vocab_size)`

Each `TransformerBlock` contains:

- Multi-head self-attention:
  - `q_proj: Linear(d_model, d_model)`
  - `k_proj: Linear(d_model, d_model)`
  - `v_proj: Linear(d_model, d_model)`
  - `output_proj: Linear(d_model, d_model)`
- Two RMSNorm layers:
  - `ln1 = RMSNorm(d_model)`
  - `ln2 = RMSNorm(d_model)`
- SwiGLU feed-forward network:
  - `w1: Linear(d_model, d_ff)`
  - `w3: Linear(d_model, d_ff)`
  - `w2: Linear(d_ff, d_model)`

## Part (a): GPT-2 XL Parameter Count and Memory

For the GPT-2 XL-shaped model, use:

```text
vocab_size = V = 50257
context_length = T = 1024
num_layers = L = 48
d_model = d = 1600
num_heads = 25
d_ff = 6400
```

Parameter count by component:

```text
token embeddings = V * d
LM head          = V * d
attention/layer  = 4 * d^2
FFN/layer        = 3 * d * d_ff
RMSNorm/layer    = 2 * d
final RMSNorm    = d
```

So the total parameter count is:

```text
total params = 2*V*d + L*(4*d^2 + 3*d*d_ff + 2*d) + d
```

For GPT-2 XL:

```text
per-block params      = 40,963,200
embeddings + LM head  = 160,822,400
all Transformer blocks = 1,966,233,600
final RMSNorm         = 1,600
total params          = 2,127,057,600
```

With single-precision floating point, loading just the parameters requires:

```text
2,127,057,600 params * 4 bytes/param = 8,508,230,400 bytes
                                          ~= 8.51 GB
                                          ~= 7.92 GiB
```

**Answer:** This implementation's GPT-2 XL-shaped model has **2,127,057,600 trainable parameters**. In FP32, the parameters alone require about **8.51 GB** of memory, or about **7.92 GiB**.

## Part (b): Matrix Multiplies and Forward-Pass FLOPs

Let:

```text
T = context_length
d = d_model
h = num_heads
d_head = d / h
L = num_layers
V = vocab_size
```

### Per-layer matrix multiplies

For one Transformer layer, the matrix multiplies are:

| Component | Matrix multiply | FLOPs |
|---|---:|---:|
| Q projection | `(T x d) @ (d x d)` | `2*T*d^2` |
| K projection | `(T x d) @ (d x d)` | `2*T*d^2` |
| V projection | `(T x d) @ (d x d)` | `2*T*d^2` |
| Attention scores | per head `(T x d_head) @ (d_head x T)` | `2*T^2*d` total over heads |
| Attention weighted values | per head `(T x T) @ (T x d_head)` | `2*T^2*d` total over heads |
| Output projection | `(T x d) @ (d x d)` | `2*T*d^2` |
| FFN `w1` | `(T x d) @ (d x d_ff)` | `2*T*d*d_ff` |
| FFN `w3` | `(T x d) @ (d x d_ff)` | `2*T*d*d_ff` |
| FFN `w2` | `(T x d_ff) @ (d_ff x d)` | `2*T*d_ff*d` |

The per-layer total is:

```text
8*T*d^2 + 4*T^2*d + 6*T*d*d_ff
```

The final LM head also performs a matrix multiply:

```text
(T x d) @ (d x V) = 2*T*d*V
```

Therefore, the total forward-pass matmul FLOPs are:

```text
L * (8*T*d^2 + 4*T^2*d + 6*T*d*d_ff) + 2*T*d*V
```

### GPT-2 XL numeric FLOPs

For GPT-2 XL:

```text
T = 1024
L = 48
d = 1600
d_ff = 6400
V = 50257
```

Breakdown:

```text
attention linear projections = 1,006,632,960,000 FLOPs
attention quadratic matmuls  =   322,122,547,200 FLOPs
FFN matmuls                  = 3,019,898,880,000 FLOPs
LM head                      =   164,682,137,600 FLOPs
```

Total:

```text
total = 4,513,336,524,800 FLOPs
      ~= 4.51 TFLOPs
```

## Part (c): Which Parts Require the Most FLOPs?

The **feed-forward network** requires the most FLOPs for the GPT-2 XL-shaped model, because each layer uses three large dense projections in the SwiGLU FFN and `d_ff = 4*d_model`. The attention linear projections are the next largest component, while the quadratic attention matmuls are smaller than the FFN and dense attention projections for these GPT-2 settings at `T = 1024`.

## Part (d): GPT-2 Small, Medium, Large, and XL Comparison

For all models below, use:

```text
V = 50257
T = 1024
d_ff = 4*d_model
```

The components are grouped as follows:

```text
attention linear    = Q projection + K projection + V projection + output projection
attention quadratic = QK^T attention scores + attention_weights @ V
FFN                 = w1 + w3 + w2
LM head             = final logits projection
```

| Model | Layers | `d_model` | Heads | Attention linear | Attention quadratic | FFN | LM head | Total FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 small | 12 | 768 | 12 | 57,982,058,496 / 16.58% | 38,654,705,664 / 11.06% | 173,946,175,488 / 49.75% | 79,047,426,048 / 22.61% | 349,630,365,696 |
| GPT-2 medium | 24 | 1024 | 16 | 206,158,430,208 / 19.96% | 103,079,215,104 / 9.98% | 618,475,290,624 / 59.87% | 105,396,568,064 / 10.20% | 1,033,109,504,000 |
| GPT-2 large | 36 | 1280 | 20 | 483,183,820,800 / 21.40% | 193,273,528,320 / 8.56% | 1,449,551,462,400 / 64.20% | 131,745,710,080 / 5.84% | 2,257,754,521,600 |
| GPT-2 XL | 48 | 1600 | 25 | 1,006,632,960,000 / 22.30% | 322,122,547,200 / 7.14% | 3,019,898,880,000 / 66.91% | 164,682,137,600 / 3.65% | 4,513,336,524,800 |

As model size increases, the FFN takes up a larger proportion of total FLOPs, because it scales with `L*d_model*d_ff`, and here `d_ff = 4*d_model`, so it behaves like an `L*d_model^2` term. The LM head takes up a smaller proportion because it scales only with `d_model*vocab_size`, not with the number of layers, and the quadratic attention part also decreases in proportion here because `T` is fixed while `d_model` and `L` increase.

## Verification Script

The numbers above can be reproduced with:

```python
V = 50257
T = 1024
models = [
    ("GPT-2 small", 12, 768, 12, 4 * 768),
    ("GPT-2 medium", 24, 1024, 16, 4 * 1024),
    ("GPT-2 large", 36, 1280, 20, 4 * 1280),
    ("GPT-2 XL", 48, 1600, 25, 4 * 1600),
]

for name, L, d, h, d_ff in models:
    attn_linear = L * 8 * T * d * d
    attn_quadratic = L * 4 * T * T * d
    ffn = L * 6 * T * d * d_ff
    lm_head = 2 * T * d * V
    total = attn_linear + attn_quadratic + ffn + lm_head
    params = 2 * V * d + L * (4 * d * d + 3 * d * d_ff + 2 * d) + d

    print(name)
    print("params:", f"{params:,}")
    print("FP32 memory bytes:", f"{params * 4:,}")
    for label, value in [
        ("attention linear", attn_linear),
        ("attention quadratic", attn_quadratic),
        ("FFN", ffn),
        ("LM head", lm_head),
    ]:
        print(label, f"{value:,}", f"{value / total * 100:.2f}%")
    print("total FLOPs:", f"{total:,}")
    print()
```

## Problem (e): GPT-2 XL FLOPs as Context Length Increases

Using the Problem (e) code in `scripts/transformer_accounting.py`, keeping the GPT-2 XL-shaped model fixed and varying only the context length gives:

| Context length | Params | FP32 memory bytes | Attention linear | Attention quadratic | FFN | LM head | Total FLOPs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 2,127,057,600 | 8,508,230,400 | 1,006,632,960,000 / 22.30% | 322,122,547,200 / 7.14% | 3,019,898,880,000 / 66.91% | 164,682,137,600 / 3.65% | 4,513,336,524,800 |
| 2,048 | 2,127,057,600 | 8,508,230,400 | 2,013,265,920,000 / 20.82% | 1,288,490,188,800 / 13.32% | 6,039,797,760,000 / 62.45% | 329,364,275,200 / 3.41% | 9,670,918,144,000 |
| 4,096 | 2,127,057,600 | 8,508,230,400 | 4,026,531,840,000 / 18.37% | 5,153,960,755,200 / 23.51% | 12,079,595,520,000 / 55.11% | 658,728,550,400 / 3.01% | 21,918,816,665,600 |
| 8,192 | 2,127,057,600 | 8,508,230,400 | 8,053,063,680,000 / 14.87% | 20,615,843,020,800 / 38.07% | 24,159,191,040,000 / 44.62% | 1,317,457,100,800 / 2.43% | 54,145,554,841,600 |
| 16,384 | 2,127,057,600 | 8,508,230,400 | 16,106,127,360,000 / 10.77% | 82,463,372,083,200 / 55.15% | 48,318,382,080,000 / 32.32% | 2,634,914,201,600 / 1.76% | 149,522,795,724,800 |

The parameter count and parameter memory stay constant because the model weights are unchanged. The quadratic attention term grows fastest with context length, so it becomes the largest share of total FLOPs by context length `16,384`.
