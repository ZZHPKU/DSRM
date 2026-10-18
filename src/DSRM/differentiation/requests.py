from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


DirectionalReduction = Literal["none", "sum"]
HessianEvaluationMode = Literal["none", "diagonal", "dense"]


def _validate_direction_tensor(value: Tensor, *, label: str) -> None:
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError(f"{label} directions must have shape [B,K,d] with positive dimensions")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{label} directions must use a floating dtype")


@dataclass(frozen=True)
class GradientDirectionRequest:
    """A named batch of first directional derivatives.

    ``directions[b, k]`` requests ``D_v u`` for sample ``b`` and direction
    ``v`` indexed by ``k``. Directions are coefficients and are detached by the
    derivative evaluator before differentiation.
    """

    label: str
    directions: Tensor
    reduction: DirectionalReduction = "none"

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("gradient direction label must be non-empty")
        _validate_direction_tensor(self.directions, label=self.label)
        if self.reduction not in {"none", "sum"}:
            raise ValueError("unsupported gradient directional reduction")


@dataclass(frozen=True)
class HessianQuadraticRequest:
    """A named batch of pure second directions ``v^T D_x^2 u v``."""

    label: str
    directions: Tensor
    reduction: DirectionalReduction = "none"

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Hessian direction label must be non-empty")
        _validate_direction_tensor(self.directions, label=self.label)
        if self.reduction not in {"none", "sum"}:
            raise ValueError("unsupported Hessian directional reduction")


@dataclass(frozen=True)
class HessianBilinearRequest:
    """A named batch of bilinear forms ``left^T D_x^2 u right``.

    Bilinear requests intentionally exclude the pure-direction optimization.
    They therefore select dense Hessian evaluation even when both tensors happen
    to contain equal values.
    """

    label: str
    left: Tensor
    right: Tensor
    reduction: DirectionalReduction = "none"

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Hessian direction label must be non-empty")
        _validate_direction_tensor(self.left, label=f"{self.label}.left")
        _validate_direction_tensor(self.right, label=f"{self.label}.right")
        if self.left.shape != self.right.shape:
            raise ValueError("paired Hessian directions must have matching shapes")
        if self.left.dtype != self.right.dtype or self.left.device != self.right.device:
            raise TypeError("paired Hessian directions must share dtype and device")
        if self.reduction not in {"none", "sum"}:
            raise ValueError("unsupported Hessian directional reduction")


HessianDirectionRequest = HessianQuadraticRequest | HessianBilinearRequest


def coordinate_directions(x: Tensor) -> Tensor:
    """Return the coordinate basis as a broadcasted ``[B,d,d]`` view."""
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError("x must have shape [B,d] with positive dimensions")
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return eye.unsqueeze(0).expand(x.shape[0], -1, -1)


def fixed_directions(x: Tensor, directions: Tensor) -> Tensor:
    """Broadcast fixed directions ``[K,d]`` over the sample dimension."""
    if x.ndim != 2 or directions.ndim != 2 or directions.shape[1] != x.shape[1]:
        raise ValueError("expected x=[B,d] and fixed directions=[K,d]")
    if directions.shape[0] <= 0:
        raise ValueError("at least one fixed direction is required")
    value = directions.to(dtype=x.dtype, device=x.device)
    return value.unsqueeze(0).expand(x.shape[0], -1, -1)


def sample_directions(x: Tensor, directions: Tensor) -> Tensor:
    """Canonicalize one direction per sample from ``[B,d]`` to ``[B,1,d]``."""
    if x.ndim != 2 or directions.shape != x.shape:
        raise ValueError("sample directions must have the same [B,d] shape as x")
    return directions.to(dtype=x.dtype, device=x.device).unsqueeze(1)


@dataclass(frozen=True)
class JetRequest:
    need_time_derivative: bool = False
    need_full_gradient: bool = False
    need_hessian_diagonal: bool = False
    need_full_hessian: bool = False
    gradient_directions: tuple[GradientDirectionRequest, ...] = ()
    hessian_directions: tuple[HessianDirectionRequest, ...] = ()

    def __post_init__(self) -> None:
        if self.need_full_hessian and self.need_hessian_diagonal:
            object.__setattr__(self, "need_hessian_diagonal", False)
        gradient_labels = [request.label for request in self.gradient_directions]
        hessian_labels = [request.label for request in self.hessian_directions]
        if len(gradient_labels) != len(set(gradient_labels)):
            raise ValueError("gradient direction labels must be unique")
        if len(hessian_labels) != len(set(hessian_labels)):
            raise ValueError("Hessian direction labels must be unique")

    @classmethod
    def full(
        cls,
        *,
        need_time_derivative: bool = False,
    ) -> "JetRequest":
        return cls(
            need_time_derivative=need_time_derivative,
            need_full_gradient=True,
            need_full_hessian=True,
        )

    @classmethod
    def diagonal_hessian(
        cls,
    ) -> "JetRequest":
        return cls(need_hessian_diagonal=True)

    @classmethod
    def directional_gradient(
        cls,
        *,
        label: str,
        directions: Tensor,
        reduction: DirectionalReduction = "none",
    ) -> "JetRequest":
        return cls(
            gradient_directions=(
                GradientDirectionRequest(
                    label,
                    directions,
                    reduction=reduction,
                ),
            ),
        )

    @classmethod
    def quadratic_hessian(
        cls,
        *,
        label: str,
        directions: Tensor,
        reduction: DirectionalReduction = "none",
    ) -> "JetRequest":
        return cls(
            hessian_directions=(
                HessianQuadraticRequest(
                    label,
                    directions,
                    reduction=reduction,
                ),
            ),
        )

    @classmethod
    def bilinear_hessian(
        cls,
        *,
        label: str,
        left: Tensor,
        right: Tensor,
        reduction: DirectionalReduction = "none",
    ) -> "JetRequest":
        return cls(
            hessian_directions=(
                HessianBilinearRequest(
                    label,
                    left,
                    right,
                    reduction=reduction,
                ),
            ),
        )

    @classmethod
    def fixed_gradient(
        cls,
        x: Tensor,
        directions: Tensor,
        *,
        label: str,
        reduction: DirectionalReduction = "none",
    ) -> "JetRequest":
        return cls.directional_gradient(
            label=label,
            directions=fixed_directions(x, directions),
            reduction=reduction,
        )

    @classmethod
    def fixed_quadratic_hessian(
        cls,
        x: Tensor,
        directions: Tensor,
        *,
        label: str,
        reduction: DirectionalReduction = "none",
    ) -> "JetRequest":
        return cls.quadratic_hessian(
            label=label,
            directions=fixed_directions(x, directions),
            reduction=reduction,
        )

    @classmethod
    def projected_hessian(
        cls,
        x: Tensor,
        basis: Tensor,
        *,
        label: str,
    ) -> "JetRequest":
        """Request all entries of ``basis @ Hessian @ basis.T``.

        The named result has shape ``[B,K*K]`` in row-major pair order for a
        basis with ``K`` rows.
        """
        if basis.ndim != 2 or basis.shape[1] != x.shape[1] or basis.shape[0] <= 0:
            raise ValueError("projected Hessian basis must have shape [K,d]")
        count = basis.shape[0]
        left = (
            basis[:, None, :]
            .expand(count, count, x.shape[1])
            .reshape(count * count, x.shape[1])
        )
        right = (
            basis[None, :, :]
            .expand(count, count, x.shape[1])
            .reshape(count * count, x.shape[1])
        )
        return cls.bilinear_hessian(
            label=label,
            left=fixed_directions(x, left),
            right=fixed_directions(x, right),
        )

    def merged(self, *others: "JetRequest") -> "JetRequest":
        requests = (self, *others)
        gradients = tuple(
            direction
            for request in requests
            for direction in request.gradient_directions
        )
        hessians = tuple(
            direction
            for request in requests
            for direction in request.hessian_directions
        )
        return JetRequest(
            need_time_derivative=any(
                request.need_time_derivative for request in requests
            ),
            need_full_gradient=any(
                request.need_full_gradient for request in requests
            ),
            need_hessian_diagonal=any(
                request.need_hessian_diagonal for request in requests
            ),
            need_full_hessian=any(
                request.need_full_hessian for request in requests
            ),
            gradient_directions=gradients,
            hessian_directions=hessians,
        )

    @property
    def requires_dense_hessian(self) -> bool:
        """Whether exact evaluation must materialize mixed second derivatives."""
        return self.need_full_hessian or any(
            isinstance(request, HessianBilinearRequest)
            for request in self.hessian_directions
        )

    @property
    def needs_spatial_hessian(self) -> bool:
        return (
            self.need_hessian_diagonal
            or self.need_full_hessian
            or bool(self.hessian_directions)
        )

    @property
    def hessian_evaluation_mode(self) -> HessianEvaluationMode:
        if not self.needs_spatial_hessian:
            return "none"
        return "dense" if self.requires_dense_hessian else "diagonal"
