"""Anisotropic linear heat problem with a translated Gaussian terminal."""
import math

from torch import nn

from ..differentiation.jets import ScalarJet
from ..differentiation.requests import JetRequest
from .base import PDEProblem
from .translated import GaussianTerminal, diffusion, exact_energies


class HeatReference(nn.Module):
    def __init__(self, terminal, coefficients):
        super().__init__()
        self.terminal, self.coefficients = terminal, coefficients

    def exact(self, t, x):
        y, grad, hessian = self.terminal.heat(t, x, self.coefficients)
        dt = -(hessian.diagonal(dim1=-2, dim2=-1)*x.new_tensor(self.coefficients)).sum(-1)
        return ScalarJet(y=y, dt=dt, grad=grad, hessian=hessian)


class HeatProblem(PDEProblem):
    def __init__(self, terminal, coefficients, energies):
        super().__init__(terminal, reference=HeatReference(terminal, coefficients),
                         reference_energies=energies, value_scale=terminal.scale)
        self.coefficients = tuple(coefficients)

    def operator_request(self):
        return JetRequest(need_hessian_diagonal=True)

    def operator(self, t, x, jet):
        return (jet.require_hessian_diagonal()*x.new_tensor(self.coefficients)).sum(-1)


def translated_heat_problem(*, R=0., alpha=.5, output_scale=None):
    if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(R):
        raise ValueError("Require finite R and positive alpha")
    coefficients = diffusion(alpha)
    if output_scale is None:
        output_scale = math.sqrt(exact_energies(R, [0.]*10)["hessian"])
    terminal = GaussianTerminal(dict(R=R, output_scale=output_scale))
    return HeatProblem(terminal, coefficients, exact_energies(R, coefficients))
