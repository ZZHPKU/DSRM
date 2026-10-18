"""Nonlinear Hessian problems with Gaussian or sine reference solutions."""
import torch

from .base import PDEProblem
from . import gaussian, sine


class ManufacturedProblem(PDEProblem):
    def __init__(self, terminal, reference, energies, *, c, rho, epsilon):
        super().__init__(terminal, reference=reference, reference_energies=energies)
        if c <= 0 or rho < 0 or epsilon <= 0:
            raise ValueError("Require c > 0, rho >= 0, epsilon > 0")
        self.c, self.rho, self.epsilon = c, rho, epsilon

    def hessian_operator(self, hessian):
        return self.c*hessian.diagonal(dim1=-2, dim2=-1).sum(-1) + self.rho*(
            torch.sqrt(self.epsilon**2+hessian.square().sum((-1, -2)))-self.epsilon)

    def source(self, t, x):
        with torch.no_grad():
            truth = self.reference.exact(t, x)
            return -truth.dt-self.hessian_operator(truth.hessian)

    def operator(self, t, x, jet):
        return self.hessian_operator(jet.require_hessian())+self.source(t, x)


def gaussian_problem(*, seed=0, d=10, modes=12, diffusivity=1., c=.5, rho=.15,
                     epsilon=.1, center_scale=.75, width_minimum=1.4, width_maximum=1.8):
    params, normalization = gaussian.normalize_teacher(gaussian.draw_teacher(
        int(seed), d, modes, diffusivity=diffusivity, center_scale=center_scale,
        width_minimum=width_minimum, width_maximum=width_maximum))
    return ManufacturedProblem(gaussian.GaussianTerminal(params), gaussian.GaussianTeacher(params),
                               normalization["order256"], c=c, rho=rho, epsilon=epsilon)


def sine_problem(*, seed=0, d=10, modes=16, temporal_scale=.25, c=.5, rho=.15, epsilon=.1):
    params = sine.derive_teacher(dict(seed=int(seed), parameters=sine.draw_teacher(int(seed), d, modes)),
                                 temporal_scale)
    return ManufacturedProblem(sine.SineTerminal(params["parameters"]),
                               sine.UngatedSineTeacher(params["parameters"]),
                               params["normalization"]["order256"], c=c, rho=rho, epsilon=epsilon)
