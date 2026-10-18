"""Problem interface for u_t + F(t, x, u, Du, D²u) = 0, u(T, x) = g(x)."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

from torch import Tensor, nn

from ..differentiation.jets import ScalarJet, evaluate_scalar_jet
from ..differentiation.requests import JetRequest


class TerminalCondition(nn.Module, ABC):
    """A fixed smooth terminal function; autograd supplies spatial derivatives."""

    def __init__(self, d: int, T: float = 1.):
        super().__init__()
        if not isinstance(d, int) or d < 1 or not math.isfinite(T) or T <= 0:
            raise ValueError("d must be a positive integer and T must be positive")
        self.d, self.T = d, float(T)

    @abstractmethod
    def value(self, x: Tensor) -> Tensor:
        """Return g(x), shaped [batch]."""

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        return self.value(x)

    def gradient(self, x: Tensor) -> Tensor:
        return evaluate_scalar_jet(
            self, x.new_full((len(x),), self.T), x,
            request=JetRequest(need_full_gradient=True), create_graph=False,
        ).require_gradient()

    def hessian(self, x: Tensor) -> Tensor:
        return evaluate_scalar_jet(
            self, x.new_full((len(x),), self.T), x,
            request=JetRequest(need_full_hessian=True), create_graph=False,
        ).require_hessian()


class PDEProblem(nn.Module, ABC):
    """A PDE on R^d with Brownian occupation/path sampling on [0,T].

    References and their exact occupation energies are optional. ``value_scale``
    converts a normalized problem's solution and residual to physical units.
    """

    def __init__(self, terminal: nn.Module, *, reference: nn.Module | None = None,
                 reference_energies: dict[str, float] | None = None,
                 value_scale: float = 1.):
        super().__init__()
        self.terminal = terminal
        self.d, self.T = terminal.d, float(terminal.T)
        if self.d < 1 or not math.isfinite(self.T) or self.T <= 0:
            raise ValueError("Invalid terminal dimension or horizon")
        if not math.isfinite(value_scale) or value_scale <= 0:
            raise ValueError("value_scale must be positive")
        self.reference = reference
        self.reference_energies = reference_energies
        self.value_scale = value_scale

    def operator_request(self) -> JetRequest:
        """Derivatives consumed by the PDE operator (excluding u_t)."""
        return JetRequest.full()

    @abstractmethod
    def operator(self, t: Tensor, x: Tensor, jet: ScalarJet) -> Tensor:
        """Return F(t,x,u,Du,D²u), shaped [batch], retaining parameter gradients."""
