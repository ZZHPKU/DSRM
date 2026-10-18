"""Tanh multilayer perceptron with a linear scalar output."""
import torch
from torch import nn


def make_mlp(input_dim, output_dim, *, hidden_layers=2, width=32,
             activation="tanh", dtype=torch.float32, device="cpu"):
    if activation != "tanh":
        raise ValueError("Only smooth tanh networks are supported")
    if input_dim <= 0 or output_dim <= 0 or hidden_layers < 1 or width <= 0:
        raise ValueError("MLP dimensions must be positive")
    layers = []
    for i in range(hidden_layers):
        layers += [nn.Linear(input_dim if i == 0 else width, width, dtype=dtype, device=device), nn.Tanh()]
    layers.append(nn.Linear(width, output_dim, dtype=dtype, device=device))
    return nn.Sequential(*layers)
