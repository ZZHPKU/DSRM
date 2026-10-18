"""A custom semilinear equation without a reference solution."""
import torch

from DSRM.differentiation import JetRequest
from DSRM.problems import PDEProblem, TerminalCondition


class GaussianTerminal(TerminalCondition):
    def value(self, x):
        return torch.exp(-x.square().sum(-1)/2)


class SemilinearHeat(PDEProblem):
    def __init__(self, d=2, T=1., reaction=.1):
        super().__init__(GaussianTerminal(d, T))
        self.reaction = reaction

    def operator_request(self):
        return JetRequest(need_hessian_diagonal=True)

    def operator(self, t, x, jet):
        return .5*jet.trace+self.reaction*torch.tanh(jet.y)


def make_problem(d=2, T=1., reaction=.1):
    return SemilinearHeat(d, T, reaction)
