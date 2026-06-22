import math
from collections.abc import Callable, Iterable
from typing import Optional, Tuple

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        # Recomputes the loss.
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                t = state.get("t", 0)
                grad = param.grad
                with torch.no_grad():
                    param.add_(grad, alpha=-lr / math.sqrt(t + 1))
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not (0 <= betas[0] < 1):
            raise ValueError(f'Invalid beta1: {betas[0]}')
        if not (0 <= betas[1] < 1):
            raise ValueError(f'Invalid beta2: {betas[1]}')
        if eps < 0:
            raise ValueError(f'Invalid epsilon: {eps}')
        if weight_decay < 0:
            raise ValueError(f'Invalid weight_decay: {weight_decay}')

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                state["t"] += 1
                t = state['t']
                m = state['m']
                v = state['v']

                with torch.no_grad():
                    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                    param.addcdiv_(m, (v.sqrt() + eps), value=-lr_t)
                    param.add_(param, alpha=-weight_decay * lr)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate

    cosine_term = (1 + math.cos(math.pi * (it - warmup_iters) /
                   (cosine_cycle_iters - warmup_iters))) / 2
    return min_learning_rate + cosine_term * (max_learning_rate - min_learning_rate)


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps=1e-6):
    grads = [param.grad for param in params if param.grad is not None]
    if len(grads) == 0:
        return
    
    # Two-stage norm calculation to avoid a giant flattened tensor.
    grad_norms = torch.stack([torch.linalg.vector_norm(grad.detach(), 2) for grad in grads])
    total_norm = torch.linalg.vector_norm(grad_norms, 2)

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(clip_coef)
