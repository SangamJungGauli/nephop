"""Tune a small PyTorch model with NepHOp.

Needs PyTorch (`pip install torch`) and the nephop package
(`maturin develop` inside python/). Run: python examples/pytorch_tune.py
"""

import torch
from torch import nn

from nephop import Kshetra, RandomSampler, Yatra

# A made-up regression task: learn y = sin(3x) from noisy samples.
torch.manual_seed(0)
x = torch.rand(600, 1) * 4 - 2
y = torch.sin(3 * x) + 0.1 * torch.randn_like(x)
x_train, y_train = x[:400], y[:400]
x_val, y_val = x[400:], y[400:]


def validation_loss(params):
    """Train briefly with these settings and return the validation loss."""
    torch.manual_seed(0)  # same start for every Paila, so scores compare fairly
    model = nn.Sequential(
        nn.Linear(1, params["hidden"]), nn.Tanh(), nn.Linear(params["hidden"], 1)
    )
    make = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[params["optimizer"]]
    optimizer = make(model.parameters(), lr=params["lr"])
    batch = params["batch"]

    for _ in range(30):  # epochs
        order = torch.randperm(len(x_train))
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        return nn.functional.mse_loss(model(x_val), y_val).item()


kshetra = (
    Kshetra()
    .log_float("lr", 1e-4, 1e-1)
    .int("batch", 8, 128)
    .int("hidden", 4, 64)
    .choice("optimizer", ["adam", "sgd"])
)

# Ask and tell, so we can print progress after every Paila.
yatra, sampler = Yatra("minimize"), RandomSampler(seed=42)
for step in range(25):
    params = yatra.ask(kshetra, sampler)
    score = validation_loss(params)
    yatra.tell(params, score)
    print(f"{step + 1:2d}  loss={score:.4f}  {params}")

best_params, best_score = yatra.lakshya()
print(f"\nLakshya: loss={best_score:.4f}")
for name, value in sorted(best_params.items()):
    print(f"  {name} = {value}")
