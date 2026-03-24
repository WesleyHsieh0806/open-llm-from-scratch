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
from jaxtyping import Float, Int

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        # Row-major ordering (d_out, d_in) to match paper notation.
        self.W = nn.Parameter(torch.zeros((out_features, in_features), dtype=dtype, device=device))
        
        # Weight initialization.
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, 
                            std=std, 
                            a=(-3*std), b=(3*std))

    def forward(self,
                x: Float[Tensor, "... in_features"]
        ) -> Float[Tensor, "... out_features"]:
        return x @ self.W.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device)) 

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
        self.gamma = nn.Parameter(torch.ones((d_model), dtype=dtype, device=device))
        
    def forward(self, x: Float[Tensor, "... d_model"]):
        in_dtype = x.dtype
        dim = x.shape[-1]
        
        # Convert to float32
        x = x.to(torch.float32)
        mean_square_x = torch.sum(x ** 2, dim=-1, keepdim=True) / dim
        rms_x =  torch.sqrt(mean_square_x + self.eps)
        output = x / rms_x * self.gamma
        
        # Convert back to original dtype
        return output.to(in_dtype)
    
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
        token_positions= torch.arange(max_seq_len, device=device)
        freqs = 1.0 / torch.exp(torch.arange(0, d_k, 2, device=device) / d_k * math.log(theta))
        angles = torch.outer(token_positions, freqs)

        sin = angles.sin()
        cos = angles.cos()

        self.register_buffer("sin", sin)  # (max_seq_len, d_k // 2)
        self.register_buffer("cos", cos)

    def forward(self, 
                x: Float[Tensor, "batch seq_len d_k"], 
                token_positions: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_k"]:
        """Applying RoPE to the input tensor x."""
        sin = self.sin[token_positions]  # (B, seq_len, d_k // 2)
        cos = self.cos[token_positions]

        # Convert x into 2D tensor for rotation.
        x = rearrange(x, "... (d_div_two two) -> ... d_div_two two", two=2)
        x = torch.stack([
            cos * x[..., 0] - sin * x[..., 1],
            sin * x[..., 0] + cos * x[..., 1],
        ], dim=-1)  # (B, seq_len, d_k // 2, 2)
        return rearrange(x, "... d_div_two two -> ... (d_div_two two)")

        