"""Draw logical batches before splitting them into device batches."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Points:
    t: torch.Tensor
    x: torch.Tensor
    sqrt_weight: torch.Tensor

    def __len__(self):
        return len(self.x)

    def slice(self, start, stop):
        return Points(self.t[start:stop], self.x[start:stop], self.sqrt_weight[start:stop])


@dataclass(frozen=True)
class Paths:
    t: torch.Tensor
    x: torch.Tensor

    def __len__(self):
        return len(self.x)

    @property
    def N(self):
        return self.x.shape[1]-1

    def slice(self, start, stop):
        return Paths(self.t, self.x[start:stop])


@dataclass(frozen=True)
class Cells:
    t: torch.Tensor
    x: torch.Tensor
    dx: torch.Tensor
    h: float

    def __len__(self):
        return len(self.x)

    def slice(self, start, stop):
        return Cells(self.t[start:stop], self.x[start:stop], self.dx[start:stop], self.h)


def generator(seed, device="cpu"):
    return torch.Generator(device=device).manual_seed(int(seed))


def occupation(count, d, rng, *, T=1., dtype=torch.float32, terminal=False):
    t = (torch.full((count,), T, device=rng.device, dtype=dtype) if terminal else
         T*torch.rand(count, generator=rng, device=rng.device, dtype=dtype))
    x = t.sqrt()[:, None]*torch.randn(count, d, generator=rng, device=rng.device, dtype=dtype)
    return Points(t, x, torch.ones_like(t))


def paths(count, N, d, rng, *, T=1., dtype=torch.float32, master_N=None):
    master_N = N if master_N is None else master_N
    if N < 1 or master_N % N:
        raise ValueError("master_N must be divisible by N")
    # Division preserves the original T=1 sampling arithmetic.
    dx = torch.randn(count, master_N, d, generator=rng, device=rng.device, dtype=dtype)/math.sqrt(master_N/T)
    x = torch.cat((dx.new_zeros(count, 1, d), dx.cumsum(1)), 1)
    t = torch.linspace(0, T, master_N+1, device=rng.device, dtype=dtype)
    stride = master_N//N
    return Paths(t[::stride], x[:, ::stride])


def cells(count, d, time, h, rng, *, dtype=torch.float32):
    t = torch.full((count,), time, device=rng.device, dtype=dtype)
    x = torch.randn(count, d, generator=rng, device=rng.device, dtype=dtype)*math.sqrt(time)
    dx = torch.empty_like(x).normal_(generator=rng)*math.sqrt(h)
    return Cells(t, x, dx, h)
