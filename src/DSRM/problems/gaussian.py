"""Gaussian-mixture heat reference and exact Brownian occupation energies."""
import math

import numpy as np
from scipy.special import roots_legendre
import torch
from torch import nn

from ..differentiation.jets import ScalarJet
from ..differentiation.fixed import FixedDerivatives


def draw_teacher(seed, d=10, modes=12, *, center_scale=.75,
                 width_minimum=1.4, width_maximum=1.8, diffusivity=.6):
    if min(d, modes, center_scale, width_minimum, diffusivity) <= 0 or width_maximum < width_minimum:
        raise ValueError("Invalid Gaussian-mixture parameters")
    rng = torch.Generator().manual_seed(seed)
    centers = center_scale*torch.randn(modes, d, generator=rng, dtype=torch.float64)
    widths = width_minimum+(width_maximum-width_minimum)*torch.rand(modes, generator=rng, dtype=torch.float64)
    coefficients = (2*torch.randint(2, (modes,), generator=rng)-1).to(torch.float64)/math.sqrt(modes)
    return dict(centers=centers, widths=widths, coefficients=coefficients,
                diffusivity=torch.tensor(diffusivity, dtype=torch.float64),
                scale=torch.tensor(1., dtype=torch.float64))


def energy_density(parameters, times):
    """Integrate products of Gaussian jets analytically in x ~ N(0,t I)."""
    centers, widths, coefficients = (parameters[k].detach().cpu().numpy()
                                    for k in ("centers", "widths", "coefficients"))
    t = np.asarray(times, dtype=float)[:, None, None]
    d = centers.shape[1]
    nu = float(parameters["diffusivity"])
    v = widths[None, :]**2+2*nu*(1-t[:, 0, :])
    vi, vj = v[:, :, None], v[:, None, :]
    ci, cj = centers[None, :, None, :], centers[None, None, :, :]
    variance = t/(1+t*(1/vi+1/vj))
    linear = ci/vi[..., None]+cj/vj[..., None]
    mean = variance[..., None]*linear
    ri, rj = mean-ci, mean-cj
    ri2, rj2 = (ri*ri).sum(-1), (rj*rj).sum(-1)
    inner = (ri*rj).sum(-1)
    log_product = (d/2)*(np.log(widths[None, :, None]**2/vi)
                  + np.log(widths[None, None, :]**2/vj)-np.log1p(t*(1/vi+1/vj)))
    log_product += -.5*((ci*ci).sum(-1)/vi+(cj*cj).sum(-1)/vj)
    log_product += .5*variance*(linear*linear).sum(-1)
    product = np.exp(log_product)*np.outer(coefficients, coefficients)[None]*float(parameters["scale"])**2
    ei, ej = ri2+d*variance, rj2+d*variance
    dot2 = inner**2+2*d*variance*inner+variance*(ri2+rj2+2*inner)+d*(d+2)*variance**2
    norm_product = ri2*rj2+variance*(d*(ri2+rj2)+4*inner)+d*(d+2)*variance**2
    diag_product = (ri*ri*rj*rj).sum(-1)+variance*(ri2+rj2+4*inner)+3*d*variance**2
    common = -ei/(vi**2*vj)-ej/(vi*vj**2)+d/(vi*vj)
    factors = dict(value=np.ones_like(product), gradient=(inner+d*variance)/(vi*vj),
                   hessian=dot2/(vi**2*vj**2)+common,
                   hessian_diagonal=diag_product/(vi**2*vj**2)+common,
                   time=nu**2*(norm_product/(vi**2*vj**2)-d*ei/(vi**2*vj)
                                -d*ej/(vi*vj**2)+d*d/(vi*vj)))
    return {key: (product*factor).sum((1, 2)) for key, factor in factors.items()}


def exact_energies(parameters, order=256, interval=(0., 1.)):
    nodes, weights = roots_legendre(order)
    left, right = interval
    times = left+(nodes+1)*(right-left)/2
    return {key: float(weights@values*(right-left)/2)
            for key, values in energy_density(parameters, times).items()}


def normalize_teacher(parameters):
    parameters = {k: v.detach().clone() for k, v in parameters.items()}
    raw = exact_energies(parameters)
    if raw["hessian"] <= 0:
        raise ValueError("Gaussian truth has no Hessian energy")
    parameters["scale"] /= math.sqrt(raw["hessian"])
    coarse, fine = exact_energies(parameters, 128), exact_energies(parameters, 256)
    differences = {k: abs(coarse[k]/fine[k]-1) for k in fine}
    if max(differences.values()) > 1e-10:
        raise ValueError("Gaussian time quadrature did not converge")
    return parameters, dict(order128=coarse, order256=fine, relative_differences=differences,
                            raw_energy=raw, scale=float(parameters["scale"]))


class GaussianTeacher(FixedDerivatives, nn.Module):
    def __init__(self, parameters):
        super().__init__()
        for key, value in parameters.items():
            self.register_buffer(key, value.detach().clone())
        self.d, self.T = parameters["centers"].shape[1], 1.
        self._refresh_derivatives()

    def _derivative_buffers(self):
        return dict(_widths_squared=self.widths.square(),
                    _amplitude=self.scale*self.coefficients,
                    _identity=torch.eye(self.d, device=self.widths.device, dtype=self.widths.dtype))

    def components(self, t, x):
        v = self._widths_squared[None]+2*self.diffusivity*(1-t[:, None])
        r = x[:, None, :]-self.centers[None]
        atoms = self._amplitude[None]*(self._widths_squared[None]/v).pow(self.d/2)
        return atoms*torch.exp(-.5*r.square().sum(-1)*v.reciprocal()), r, v

    def forward(self, t, x):
        return self.components(t, x)[0].sum(-1)

    def exact(self, t, x):
        atoms, r, v = self.components(t, x)
        inverse = v.reciprocal()
        first = atoms*inverse
        gradient = -torch.einsum("bm,bmd->bd", first, r)
        hessian = torch.einsum("bm,bmi,bmj->bij", atoms*inverse.square(), r, r)
        hessian = hessian-first.sum(-1)[:, None, None]*self._identity
        dt = -self.diffusivity*hessian.diagonal(dim1=-2, dim2=-1).sum(-1)
        return ScalarJet(y=atoms.sum(-1), dt=dt, grad=gradient, hessian=hessian)


class GaussianTerminal(GaussianTeacher):
    def _derivative_buffers(self):
        values = super()._derivative_buffers()
        inverse = self.widths.square().reciprocal()
        values.update(_inverse_variance=inverse, _inverse_variance_squared=inverse.square())
        return values

    def _terminal_components(self, x):
        r = x[:, None, :]-self.centers[None]
        atoms = self._amplitude[None]*torch.exp(-.5*r.square().sum(-1)*self._inverse_variance)
        return atoms, r

    def value(self, x):
        return self._terminal_components(x)[0].sum(-1)

    def gradient(self, x):
        return self.taylor_tensors(x, "none")[1]

    def hessian(self, x):
        return self.taylor_tensors(x, "dense")[2]

    def taylor_tensors(self, x, mode, direction=None):
        """Only form a dense Hessian when requested; project analytically otherwise."""
        if mode not in ("none", "dense", "diagonal"):
            raise ValueError(mode)
        atoms, r = self._terminal_components(x)
        first = atoms*self._inverse_variance
        gradient = -torch.einsum("bm,bmd->bd", first, r)
        second = None
        weighted = atoms*self._inverse_variance_squared
        if mode == "dense":
            second = torch.einsum("bm,bmi,bmj->bij", weighted, r, r)
            second = second-first.sum(-1)[:, None, None]*self._identity
        elif mode == "diagonal":
            second = (weighted[:, :, None]*r.square()).sum(1)-first.sum(-1)[:, None]
        quadratic = None
        if direction is not None:
            projected = (r*direction[:, None, :]).sum(-1)
            quadratic = (weighted*projected.square()).sum(-1)-first.sum(-1)*direction.square().sum(-1)
        return atoms.sum(-1), gradient, second, quadratic

    def identity_payload(self):
        return dict(type="gaussian_mixture_terminal", d=self.d, T=self.T)
