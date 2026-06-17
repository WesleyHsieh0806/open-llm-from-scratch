import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy_loss(logits: Float[Tensor, "... seq_len vocab_size"], labels: Int[Tensor, "... seq_len"]) -> Float[Tensor, ""]:
    '''
    Parameters:
        logits: (... seq_len vocab_size), unormalized LLM logits 
        labels: (... seq_len), ground truth labels
    Returns:
        loss: 
    '''
    # Normalize logits without modifying the input tensor in-place.
    logits = logits - logits.max(dim=-1, keepdim=True).values
    target_logits = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1))
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    loss = -target_logits + logsumexp
    return loss.mean()



class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, labels):
        return cross_entropy_loss(logits, labels)
