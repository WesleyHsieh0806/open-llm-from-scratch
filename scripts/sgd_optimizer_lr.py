from pathlib import Path
import importlib.util

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER_PATH = REPO_ROOT / "cs336_basics" / "optimizer.py"

spec = importlib.util.spec_from_file_location("cs336_basics_optimizer", OPTIMIZER_PATH)
optimizer_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(optimizer_module)
SGD = optimizer_module.SGD


LEARNING_RATES = [1e1, 1e2, 1e3]
NUM_ITERATIONS = 10


def run_sgd_example(lr: float) -> list[float]:
    torch.manual_seed(0)
    weight = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weight], lr=lr)

    losses = []
    for _ in range(NUM_ITERATIONS):
        opt.zero_grad()
        loss = (weight ** 2).mean()
        losses.append(loss.cpu().item())
        loss.backward()
        opt.step()
    return losses


if __name__ == "__main__":
    all_losses = {lr: run_sgd_example(lr) for lr in LEARNING_RATES}

    header = ["learning rate", *[f"iter {i}" for i in range(NUM_ITERATIONS)]]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for lr, losses in all_losses.items():
        row = [f"{lr:.0e}", *[f"{loss:.6g}" for loss in losses]]
        print("| " + " | ".join(row) + " |")
