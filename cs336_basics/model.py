"""The implementation is written by Wesley Hsieh as a casual side project.
https://wesleyhsieh0806.github.io/

You are free to refer to this implementation or reach out to discuss any questions.
Copying everything directly is not recommended as I don't want to be called out by your Stanford professor.

Meanwhile, a lot of the blocks could refer to Llama3's implementation.
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
Happy learning!
"""
import math
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from jaxtyping import Float, Int, Bool


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        # Row-major ordering (d_out, d_in) to match paper notation.
        self.weight = nn.Parameter(torch.zeros(
            (out_features, in_features), dtype=dtype, device=device))

        # Weight initialization.
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight,
                              std=std,
                              a=(-3*std), b=(3*std))

    def forward(self,
                x: Float[Tensor, "... in_features"]
                ) -> Float[Tensor, "... out_features"]:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(
            (num_embeddings, embedding_dim), dtype=dtype, device=device))

        # Initializaiton.
        nn.init.trunc_normal_(self.weight,
                              std=1.0,
                              a=(-3.0),
                              b=(3.0))

    def forward(self, token_ids: Int[Tensor, "batch seq_len"]
                ) -> Float[Tensor, "batch seq_len embedding_dim"]:
        # Pytorch's advanced indexing feature allows you to do this.
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """RMSNorm layer
        RMSNorm(a_i) = \frac{a_i}{RMS(a_i)}g_i
        RMS(a_i) = torch.sqrt(torch.sum(a_i ** 2, dim=-1) / d_model + eps)
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(
            (d_model), dtype=dtype, device=device))

    def forward(self, x: Float[Tensor, "... d_model"]):
        in_dtype = x.dtype
        dim = x.shape[-1]

        # Convert to float32
        x = x.to(torch.float32)
        mean_square_x = torch.sum(x ** 2, dim=-1, keepdim=True) / dim
        rms_x = torch.sqrt(mean_square_x + self.eps)
        output = x / rms_x * self.weight

        # Convert back to original dtype
        return output.to(in_dtype)


class SiLU(nn.Module):
    """Swish Activation Function"""

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(torch.sigmoid(x))
        return x * torch.sigmoid(x)


class FeedForwardNetwork(nn.Module):
    """SwiGLU feedforward layer"""

    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or (8 * d_model // 3)

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.silu = SiLU()

    def forward(self, x):
        x = self.silu(self.w1(x)) * self.w3(x)
        return self.w2(x)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """A more elegant way is to implement RoPE using complex numbers, where
        multiplication naturally leads to rotation in the complex plane.
        To align with the course requirements, we implement RoPE using sin/cos rotation matrices.

        1. Get all token positions i (from 0 to seq_len - 1)
        2. Get all denominator terms.
        3. Create rotation angle matrices (seq_len, d_k // 2)
        4. Register buffer for sin and cos.
        """
        super().__init__()
        token_positions = torch.arange(max_seq_len, device=device)
        freqs = 1.0 / torch.exp(torch.arange(0, d_k, 2,
                                device=device) / d_k * math.log(theta))
        angles = torch.outer(token_positions, freqs)

        sin = angles.sin()
        cos = angles.cos()

        # Use persistent=False to exclude them from state_dict.
        # (max_seq_len, d_k // 2)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self,
                x: Float[Tensor, "batch seq_len d_k"],
                token_positions: Int[Tensor, "batch seq_len"] | None = None) -> Float[Tensor, "batch seq_len d_k"]:
        """Applying RoPE to the input tensor x."""
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
            token_positions = token_positions.reshape(
                (1,) * (x.ndim - 2) + token_positions.shape)

        sin = self.sin[token_positions]  # (B, seq_len, d_k // 2)
        cos = self.cos[token_positions]

        # Convert x into 2D tensor for rotation.
        x = rearrange(x, "... (d_div_two two) -> ... d_div_two two", two=2)
        x = torch.stack([
            cos * x[..., 0] - sin * x[..., 1],
            sin * x[..., 0] + cos * x[..., 1],
        ], dim=-1)  # (B, seq_len, d_k // 2, 2)
        return rearrange(x, "... d_div_two two -> ... (d_div_two two)")


def softmax(x: Float, dim: int = -1) -> Float:
    # Subtract x with the maximum along a dimension
    max_x = x.max(dim=dim, keepdim=True)[0]
    x = x - max_x
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... seq_len d_k"],
    K: Float[Tensor, "... seq_len d_k"],
    V: Float[Tensor, "... seq_len d_v"],
    mask: Bool[Tensor, "... seq_len seq_len"] = None
) -> Float[Tensor, "... seq_len d_v"]:
    """Scaled dot-product attention."""
    # (... seq_len seq_len)
    attention_score = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])

    # Apply masking.
    if mask is not None:
        attention_score = attention_score.masked_fill(
            ~mask, torch.finfo(attention_score.dtype).min)

    # (... seq_len seq_len)
    weight = softmax(attention_score, dim=-1)
    return weight @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 1024, rope_theta=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

        # Create causal mask.
        self.mask = torch.tril(torch.ones(
            (max_seq_len, max_seq_len), device=device, dtype=torch.bool))

        # Rope will be applied equally to each head.
        self.use_rope = rope_theta is not None
        if self.use_rope:
            self.rope = RotaryPositionalEncoding(
                theta=rope_theta,
                d_k=d_model // num_heads,
                max_seq_len=max_seq_len,
                device=device)

    def forward(self, x: Float[Tensor, "... seq_len d_model"],
                token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]

        # project query, key, value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Convert the shape for multi-head attention
        q = rearrange(
            q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads, d_k=self.d_model // self.num_heads)
        k = rearrange(
            k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads, d_k=self.d_model // self.num_heads)
        v = rearrange(
            v, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads, d_k=self.d_model // self.num_heads)

        # Apply RoPE.
        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = self.mask[:seq_len, :seq_len]

        # Multi-head attention.
        output = scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(
            output, "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.output_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, rope_theta: int):
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, rope_theta)
        self.ln1 = RMSNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
