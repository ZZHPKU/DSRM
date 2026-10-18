"""Translated Gaussian terminal and exact anisotropic heat flow."""
import numpy as np
from numpy.polynomial.legendre import leggauss
import torch
from torch import nn
D, S = 10, 1.

def pair(p, q, amp_p, amp_q, center, variance):
    """Gaussian product moments for diagonal covariance, retaining full Hessian."""
    den = 1 + variance[:, None] * (p + q)
    c = variance[:, None] / den
    mu = -center[None] / den
    second = mu * mu + c
    variance_square = 4 * mu * mu * c + 2 * c * c
    fourth = second * second + variance_square
    mass = amp_p * amp_q * np.exp(-.5 * np.log(den).sum(-1)
                                  - .5 * (center * center * (p + q) / den).sum(-1))
    diagonal = (p * p * q * q * fourth - (p * p * q + p * q * q) * second + p * q).sum(-1)
    w = p * q * second
    hessian = mass * (diagonal + w.sum(-1) ** 2 - (w * w).sum(-1))
    return mass, hessian, second, variance_square

def diffusion(alpha):
    return [alpha] + [.5] * (D - 1)


def exact_energies(R, coefficients, order=128):
    t, w = leggauss(order)
    t, w = (t + 1) / 2, w / 2
    p = 1 / (4 + 2 * (1 - t[:, None]) * np.asarray(coefficients))
    amp = np.sqrt(4 * p).prod(-1)
    center = np.array([R] + [0.] * (D - 1))
    mass, hessian, second, _ = pair(p, p, amp, amp, center, S * t)
    return dict(value=float(w @ mass), gradient=float(w @ (mass * (p * p * second).sum(-1))),
                hessian=float(w @ hessian))


class GaussianTerminal(nn.Module):
    d, T = D, 1.

    def __init__(self, config):
        super().__init__()
        self.register_buffer('center', torch.tensor([config['R']] + [0.] * (D - 1), dtype=torch.float64))
        self.scale = config['output_scale']

    def heat(self, t, x, coefficients):
        a = x.new_tensor(coefficients)
        q = 4 + 2 * (1 - t[:, None]) * a[None]
        z = x - self.center
        y = torch.exp(.5 * torch.log(4 / q).sum(-1) - .5 * (z.square() / q).sum(-1)) / self.scale
        b = z / q
        h = y[:, None, None] * (b[:, :, None] * b[:, None, :] - torch.diag_embed(1 / q))
        return y, -y[:, None] * b, h

    def value(self, x):
        return torch.exp(-(x - self.center).square().sum(-1) / 8) / self.scale

    def gradient(self, x):
        return -self.value(x)[:, None] * (x - self.center) / 4

    def hessian(self, x):
        b = (x - self.center) / 4
        return self.value(x)[:, None, None] * (b[:, :, None] * b[:, None, :] - torch.eye(D, device=x.device, dtype=x.dtype) / 4)

    def taylor_tensors(self, x, mode, direction=None):
        z = x-self.center
        y = self.value(x)
        b = z/4
        second = (self.hessian(x) if mode == "dense" else
                  y[:, None]*(b.square()-.25) if mode == "diagonal" else None)
        quadratic = (y*((b*direction).sum(-1).square()-.25*direction.square().sum(-1))
                     if direction is not None else None)
        return y, -y[:, None]*b, second, quadratic


