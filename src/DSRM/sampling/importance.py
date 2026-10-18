"""Mixture importance sampling for translated Gaussian heat problems."""
import math

import torch

from .brownian import Points


class GaussianMixtureSampler:
    def __init__(self, center, templates, *, device="cpu", dtype=torch.float32):
        self.device, self.dtype = torch.device(device), dtype
        self.center = torch.tensor(center, device=device, dtype=dtype)
        self.templates = torch.tensor(templates, device=device, dtype=dtype)
        if self.center.ndim != 1 or self.templates.ndim != 2 or self.templates.shape[1] != len(center):
            raise ValueError("center and diffusion templates must share a dimension")

    def draw(self, count, rng):
        kw = dict(device=self.device, dtype=self.dtype, generator=rng)
        bins = 2**(23 if self.dtype == torch.float32 else 52)
        t = (torch.randint(bins, (count,), device=self.device, generator=rng).to(self.dtype)+.5)/bins
        z = torch.randn(count, len(self.center), **kw)
        modes = len(self.templates)
        choose = torch.randint(modes, (count,), device=self.device, generator=rng)
        tilted = torch.rand(count, **kw) < .5
        v = t[:, None]
        q = 4+2*(1-t[:, None])*self.templates[choose]
        total = q+2*v
        mean = 2*v*self.center/total
        x = torch.where(tilted[:, None], mean+(v*q/total).sqrt()*z, v.sqrt()*z)
        q_all = 4+2*(1-t[:, None, None])*self.templates[None]
        total_all = q_all+2*v[:, None]
        log_ratio = (.5*torch.log(total_all/q_all)-(x[:, None]-self.center).square()/q_all
                     +self.center.square()/total_all).sum(-1)
        log_mix = torch.logaddexp(log_ratio.logsumexp(-1)-math.log(2*modes),
                                  t.new_full((count,), math.log(.5)))
        return Points(t, x, torch.exp(-.5*log_mix))
