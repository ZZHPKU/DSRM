"""Sine truth without a time envelope, with reproducible frequency scaling."""
import math

import numpy as np
from scipy.special import roots_legendre
import torch

from ..differentiation.jets import ScalarJet
from ..differentiation.fixed import FixedDerivatives
from torch import nn


def energy_density(parameters, t):
    """Exact Gaussian spatial moments at arbitrary physical times."""
    a, w, omega, b = (parameters[k].detach().cpu().numpy() for k in ("a", "w", "omega", "b"))
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    phase = t[:, None]*omega+b
    minus = np.exp(-.5*t[:, None, None]*np.square(w[:, None]-w[None]).sum(-1))
    plus = np.exp(-.5*t[:, None, None]*np.square(w[:, None]+w[None]).sum(-1))
    cm = minus*np.cos(phase[:, :, None]-phase[:, None, :])
    cp = plus*np.cos(phase[:, :, None]+phase[:, None, :])
    ss, cc = .5*(cm-cp), .5*(cm+cp)
    gram = w@w.T
    amplitude = np.outer(a, a)*float(parameters["scale"])**2/len(a)
    terms = dict(value=ss, gradient=cc*gram, hessian=ss*gram**2,
                 time=cc*omega[:, None]*omega[None, :])
    return {name: (term*amplitude).sum((1, 2)) for name, term in terms.items()}


def exact_energies(parameters, order=256):
    """Exact Gaussian spatial moments and Gauss--Legendre time integration."""
    t, weight = roots_legendre(order)
    return {name: float((weight/2) @ values)
            for name, values in energy_density(parameters, (t+1)/2).items()}


def derive_teacher(parent, temporal_scale):
    """Keep the saved modes; remove (1-t), scale omega, and normalize Hessian energy."""
    if not math.isfinite(temporal_scale) or not 0 < temporal_scale <= 1:
        raise ValueError("Temporal frequency scale must be in (0, 1]")
    parameters = {k: v.detach().clone().to(torch.float64) for k, v in parent["parameters"].items()}
    parameters["omega"] *= temporal_scale
    parameters["scale"].fill_(1.)
    base = exact_energies(parameters)
    if not base["hessian"] > 0:
        raise ValueError("The saved modes have no Hessian energy")
    parameters["scale"].fill_(base["hessian"]**-.5)
    e128, e256 = exact_energies(parameters, 128), exact_energies(parameters, 256)
    differences = {k: abs(e128[k]-e256[k])/e256[k] for k in e256}
    if max(differences.values()) > 1e-10:
        raise ValueError("Time quadrature for teacher normalization did not converge")
    return dict(kind="sine_ungated", seed=parent["seed"], parameters=parameters,
                temporal_frequency_scale=temporal_scale,
                normalization=dict(order128=e128, order256=e256, relative_differences=differences,
                                   scale=float(parameters["scale"])))


class UngatedSineTeacher(FixedDerivatives, nn.Module):
    def __init__(self, parameters):
        super().__init__()
        for name, value in parameters.items():
            self.register_buffer(name, value.detach().clone())
        self.d, self.T = parameters["w"].shape[1], 1.
        self._refresh_derivatives()

    def _derivative_buffers(self):
        amplitude = self.scale*self.a/math.sqrt(len(self.a))
        outer = (self.w[:, :, None]*self.w[:, None, :]).flatten(1)
        return dict(_amplitude=amplitude, _time_modes=amplitude*self.omega,
                    _gradient_modes=amplitude[:, None]*self.w,
                    _hessian_modes=-amplitude[:, None]*outer)

    def forward(self, t, x):
        phase = x@self.w.T+t[:, None]*self.omega+self.b
        return phase.sin()@self._amplitude

    def exact(self, t, x):
        phase = x@self.w.T+t[:, None]*self.omega+self.b
        sine, cosine = phase.sin(), phase.cos()
        return ScalarJet(y=sine@self._amplitude, dt=cosine@self._time_modes,
                         grad=cosine@self._gradient_modes,
                         hessian=(sine@self._hessian_modes).reshape(-1, self.w.shape[1], self.w.shape[1]))


def draw_teacher(seed, d=10, modes=16):
    rng = torch.Generator().manual_seed(seed)
    kw = dict(generator=rng, dtype=torch.float64)
    a = torch.randn(modes, **kw)
    w = torch.randn(modes, d, **kw)
    w = w / w.norm(dim=1, keepdim=True)
    return dict(a=a, w=w, omega=2*torch.rand(modes, **kw)-1,
                b=math.pi*(2*torch.rand(modes, **kw)-1), scale=torch.tensor(1., dtype=torch.float64))



class SineTerminal(FixedDerivatives, nn.Module):
    """Store only g(x)=u*(1,x), reconstructing its precision from the job truth."""

    def __init__(self, parameters):
        super().__init__()
        p = parameters
        self.d, self.T = p["w"].shape[1], 1.
        for name, value in dict(amplitude=p["scale"]*p["a"]/math.sqrt(len(p["a"])),
                                w=p["w"], phase=p["b"]+p["omega"]).items():
            self.register_buffer(name, value.detach().clone(), persistent=False)
        self._refresh_derivatives()

    def _derivative_buffers(self):
        outer = (self.w[:, :, None]*self.w[:, None, :]).flatten(1)
        return dict(_gradient_modes=self.amplitude[:, None]*self.w,
                    _hessian_modes=-self.amplitude[:, None]*outer,
                    _diagonal_modes=-self.amplitude[:, None]*self.w.square())

    def forward(self, x):
        return (x@self.w.T+self.phase).sin()@self.amplitude

    def value(self, x):
        return self.forward(x)

    def gradient(self, x):
        return self.taylor_tensors(x, "none")[1]

    def hessian(self, x):
        return self.taylor_tensors(x, "dense")[2]

    def taylor_tensors(self, x, mode, direction=None):
        phase = x@self.w.T+self.phase
        sine, cosine = phase.sin(), phase.cos()
        second = None
        if mode == "dense":
            second = (sine@self._hessian_modes).reshape(-1, x.shape[1], x.shape[1])
        elif mode == "diagonal":
            second = sine@self._diagonal_modes
        quadratic = (-(sine*self.amplitude*(direction@self.w.T).square()).sum(-1)
                     if direction is not None else None)
        return sine@self.amplitude, cosine@self._gradient_modes, second, quadratic


