from cs336_basics.model import TransformerLM
import torch
import torch.nn as nn
import torch.nn.functional as F


# GPT-2 XL:
vocab_size = 50257
context_length = 1024
d_model = 1600
num_layers = 48
num_heads = 25
d_ff = 6400
rope_theta = 1e4

# gpt2xl = TransformerLM(
#     vocab_size=vocab_size,
#     context_length=context_length,
#     d_model=d_model,
#     num_layers=num_layers,
#     num_heads=num_heads,
#     d_ff=d_ff,
#     rope_theta=rope_theta,
# )

# num_trainable_params = sum(
#     [param.numel() for param in gpt2xl.parameters() if param.requires_grad])

'''
Embedding: d_model * vocab_size
Block: 48 * num_param_per_block
ln_final: d_model
lm_head: d_model * vocab_size
Block:
  ln: d_model
  attn: 4 * d_model * d_model

  ln: d_model
  ffn: d_model * d_ff * 3 
'''

# For derivation, check /plan/transformer_accounting.md.
# 2.1 Billion
# print(f"Number of trainable parameters: {num_trainable_params:,}.")
num_trainable_params_derived = (
    (d_model * vocab_size) +
    48 * (d_model + 4 * d_model * d_model + d_model + d_model * d_ff * 3) +
    d_model +
    d_model * vocab_size
)
print(
    f"Number of trainable parameters via derivation: {num_trainable_params_derived:,}.")
# 8.51 GB
print(
    f"Memory required to load the model: {num_trainable_params_derived*4:,} Bytes.")


'''
FLOPs required from matrix multiplication:
Primitive:
- linear layer: 
    2 * in_features * out_features * context_length
- RMSNorm:
    0 since it does not require matrix multiplication

- Embedding:
  0

- attn:
 q_proj: 2 * d_model * d_model * context_length
 k_proj: 2 * d_model * d_model * context_length
 v_proj: 2 * d_model * d_model * context_length
 out_proj: 2 * d_model * d_model * context_length
 attention_score: 2 * d_model * context_length * context_length
 weighted values: 2 * d_model * context_length * context_length

 - ffn:
   2 * d_model * d_ff * context_lengh * 3
    

Embedding: 0
Block: 48 * num_param_per_block
ln_final: 0
lm_head: 2 * d_model * vocab_size * context_length
Block:
  ln: 0
  attn: (2 * d_model * d_model * context_length)) * 4 + 4 * d_model * context_length ** 2 

  ln: 0
  ffn: 6 * d_model * d_ff * context_length
'''
num_flops = (
    2 * d_model * vocab_size * context_length +  # lm_head
    48 * (
        (2 * d_model * d_model * context_length) * 4 +  # q, j, v proj
        4 * d_model * context_length ** 2 +  # attention score and weighted values
        6 * d_model * d_ff * context_length  # ffn
    )
)


print(f"Number of FLOPs: {num_flops:,}.")  # 4.51 Trillion FLOPs

'''
Problem (d)
'''

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


'''
Problem (e)
'''

V = 50257
models = [
    ("L=1024", 48, 1600, 25, 4 * 1600, 1024),
    ("L=2048", 48, 1600, 25, 4 * 1600, 2048),
    ("L=4096", 48, 1600, 25, 4 * 1600, 4096),
    ("L=8192", 48, 1600, 25, 4 * 1600, 8192),
    ("L=16384", 48, 1600, 25, 4 * 1600, 16384),
]

for name, L, d, h, d_ff, T in models:
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
