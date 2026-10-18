"""Seeded network construction, independent of the numerical algorithm."""
import copy

import torch

from .terminal import HardTerminalScalar
from .taylor import TaylorScalar


def build_model(problem, config, seed, *, device="cpu", dtype=torch.float32):
    model_type = TaylorScalar if config.backend == "taylor" else HardTerminalScalar
    # Original experiments initialize on CPU in FP32 before choosing training precision.
    terminal = copy.deepcopy(problem.terminal).cpu()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = model_type(d=problem.d, T=problem.T, terminal=terminal,
                           width=config.width, hidden_layers=config.hidden_layers,
                           input_shift=config.input_shift)
    # Nonpersistent fixed data are reconstructed, not rounded through a checkpoint.
    if not terminal.state_dict():
        model.terminal = copy.deepcopy(problem.terminal).cpu()
    return model.to(device=device, dtype=dtype)
