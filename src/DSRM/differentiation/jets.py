from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from .requests import (
    GradientDirectionRequest,
    HessianDirectionRequest,
    HessianEvaluationMode,
    HessianQuadraticRequest,
    JetRequest,
)


def _reduce_directional(value: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return value
    if reduction == "sum":
        return value.sum(dim=-1)
    raise AssertionError(f"unsupported directional reduction: {reduction}")


@dataclass
class ScalarJet:
    """Requested derivatives of a scalar candidate at independent points.

    Dense gradient/Hessian channels are optional. Named directional channels
    are the primary representation used by training objectives that do not need
    the full tensors.
    """

    y: Tensor
    dt: Tensor | None = None
    grad: Tensor | None = None
    hessian_diagonal: Tensor | None = None
    hessian: Tensor | None = None
    gradient_directions: dict[str, Tensor] = field(default_factory=dict)
    hessian_directions: dict[str, Tensor] = field(default_factory=dict)

    @property
    def hessian_evaluation_mode(self) -> HessianEvaluationMode:
        if self.hessian is not None:
            return "dense"
        if self.hessian_diagonal is not None or self.hessian_directions:
            return "diagonal"
        return "none"

    @property
    def trace(self) -> Tensor:
        return self.require_hessian_diagonal().sum(dim=-1)

    def require_gradient(self) -> Tensor:
        if self.grad is None:
            raise RuntimeError("full gradient was not requested")
        return self.grad

    def require_hessian(self) -> Tensor:
        if self.hessian is None:
            raise RuntimeError("full Hessian was not requested")
        return self.hessian

    def require_hessian_diagonal(self) -> Tensor:
        if self.hessian_diagonal is not None:
            return self.hessian_diagonal
        if self.hessian is not None:
            return torch.diagonal(self.hessian, dim1=-2, dim2=-1)
        raise RuntimeError("Hessian diagonal was not requested")

    def gradient_direction(self, label: str) -> Tensor:
        try:
            return self.gradient_directions[label]
        except KeyError as error:
            raise RuntimeError(f"gradient directional channel {label!r} was not requested") from error

    def hessian_direction(self, label: str) -> Tensor:
        try:
            return self.hessian_directions[label]
        except KeyError as error:
            raise RuntimeError(f"Hessian directional channel {label!r} was not requested") from error

    def detached(self) -> "ScalarJet":
        return ScalarJet(
            y=self.y.detach(),
            dt=None if self.dt is None else self.dt.detach(),
            grad=None if self.grad is None else self.grad.detach(),
            hessian_diagonal=(
                None
                if self.hessian_diagonal is None
                else self.hessian_diagonal.detach()
            ),
            hessian=None if self.hessian is None else self.hessian.detach(),
            gradient_directions={
                name: value.detach() for name, value in self.gradient_directions.items()
            },
            hessian_directions={
                name: value.detach() for name, value in self.hessian_directions.items()
            },
        )

    @classmethod
    def from_dense(
        cls,
        *,
        y: Tensor,
        gradient: Tensor,
        hessian: Tensor,
        request: JetRequest,
        dt: Tensor | None = None,
    ) -> "ScalarJet":
        """Project an already available dense jet onto an explicit request."""
        if y.ndim != 1 or gradient.ndim != 2 or hessian.ndim != 3:
            raise ValueError("expected dense jet shapes y=[B], gradient=[B,d], Hessian=[B,d,d]")
        B, d = gradient.shape
        if y.shape[0] != B or hessian.shape != (B, d, d):
            raise ValueError("dense jet batch or spatial dimensions disagree")
        _validate_request(request, batch_size=B, d=d, dtype=y.dtype, device=y.device)
        directional_gradients = {
            item.label: _project_gradient(gradient, item)
            for item in request.gradient_directions
        }
        directional_hessians = {
            item.label: _project_hessian(hessian, item, fuse_single_direction=request.need_full_hessian)
            for item in request.hessian_directions
        }
        uses_dense_hessian = request.needs_spatial_hessian
        return cls(
            y=y,
            dt=dt if request.need_time_derivative else None,
            grad=(
                gradient
                if request.need_full_gradient or uses_dense_hessian
                else None
            ),
            hessian_diagonal=(
                torch.diagonal(hessian, dim1=-2, dim2=-1)
                if request.need_hessian_diagonal
                else None
            ),
            hessian=hessian if uses_dense_hessian else None,
            gradient_directions=directional_gradients,
            hessian_directions=directional_hessians,
        )


@runtime_checkable
class SpatialDirectionalJetModel(Protocol):
    """Scalar-model capability for one lifted diagonal-Hessian traversal."""

    def evaluate_spatial_directional_jet(
        self,
        t: Tensor,
        x: Tensor,
        directions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


@runtime_checkable
class AnalyticScalarJetModel(Protocol):
    """Scalar-model capability that owns its complete jet evaluation."""

    def evaluate_analytic_jet(
        self,
        t: Tensor,
        x: Tensor,
        *,
        request: JetRequest,
        create_graph: bool,
    ) -> ScalarJet: ...


def _grad_or_zero(output: Tensor, variable: Tensor, *, create_graph: bool) -> Tensor:
    if not output.requires_grad:
        return torch.zeros_like(variable)
    result = torch.autograd.grad(
        output.sum(),
        variable,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(variable) if result is None else result


def _validate_request(
    request: JetRequest,
    *,
    batch_size: int,
    d: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    for item in request.gradient_directions:
        if item.directions.shape[0] != batch_size or item.directions.shape[2] != d:
            raise ValueError(
                f"gradient channel {item.label!r} must use directions [B,K,d] "
                f"with B={batch_size}, d={d}"
            )
        if item.directions.dtype != dtype or item.directions.device != device:
            raise TypeError(
                f"gradient channel {item.label!r} directions must match point dtype/device"
            )
    for item in request.hessian_directions:
        directions = (
            item.directions
            if isinstance(item, HessianQuadraticRequest)
            else item.left
        )
        if directions.shape[0] != batch_size or directions.shape[2] != d:
            raise ValueError(
                f"Hessian channel {item.label!r} must use directions [B,K,d] "
                f"with B={batch_size}, d={d}"
            )
        if directions.dtype != dtype or directions.device != device:
            raise TypeError(
                f"Hessian channel {item.label!r} directions must match point dtype/device"
            )


def _project_gradient(gradient: Tensor, request: GradientDirectionRequest) -> Tensor:
    directions = request.directions.detach()
    value = torch.einsum("bd,bkd->bk", gradient, directions)
    return _reduce_directional(value, request.reduction)


def _project_hessian(
    hessian: Tensor, request: HessianDirectionRequest, *, fuse_single_direction: bool = False,
) -> Tensor:
    if isinstance(request, HessianQuadraticRequest):
        left = right = request.directions.detach()
    else:
        left = request.left.detach()
        right = request.right.detach()
    if fuse_single_direction and left.shape[1] == 1:
        # Fuse with full-Hessian consumers; preserve isolated projections' reduction order.
        value = (hessian*left[:, 0, :, None]*right[:, 0, None, :]).sum((-2, -1))[:, None]
    else:
        value = torch.einsum("bkd,bdc,bkc->bk", left, hessian, right)
    return _reduce_directional(value, request.reduction)


def _dense_hessian_from_gradient(
    gradient: Tensor,
    x_leaf: Tensor,
    *,
    create_graph: bool,
) -> Tensor:
    d = x_leaf.shape[1]
    if not gradient.requires_grad:
        return x_leaf.new_zeros(x_leaf.shape[0], d, d)
    coordinate_basis = torch.eye(d, dtype=gradient.dtype, device=gradient.device)
    grad_outputs = coordinate_basis[:, None, :].expand(
        d,
        gradient.shape[0],
        d,
    )
    rows = torch.autograd.grad(
        gradient,
        x_leaf,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
        is_grads_batched=True,
    )[0]
    if rows is None:
        rows = x_leaf.new_zeros(d, x_leaf.shape[0], d)
    hessian = rows.permute(1, 0, 2)
    return 0.5 * (hessian + hessian.transpose(-1, -2))


@torch.enable_grad()
def evaluate_scalar_jet(
    model: nn.Module,
    t: Tensor,
    x: Tensor,
    *,
    request: JetRequest | None = None,
    create_graph: bool = True,
) -> ScalarJet:
    """Evaluate only the full or directional derivative channels requested.

    The points are independent leaves, so ``dt`` is the physical-time partial
    derivative rather than a derivative through the Brownian sampler. The
    default is an explicit full jet for diagnostics; training solvers always
    pass a narrower request.
    """
    if t.ndim != 1 or x.ndim != 2 or t.shape[0] != x.shape[0]:
        raise ValueError("expected t=[B] and x=[B,d]")
    active_request = JetRequest.full(need_time_derivative=True) if request is None else request
    _validate_request(
        active_request,
        batch_size=x.shape[0],
        d=x.shape[1],
        dtype=x.dtype,
        device=x.device,
    )

    if isinstance(model, AnalyticScalarJetModel):
        result = model.evaluate_analytic_jet(
            t.detach(),
            x.detach(),
            request=active_request,
            create_graph=create_graph,
        )
        if not isinstance(result, ScalarJet):
            raise TypeError("model.evaluate_analytic_jet must return ScalarJet")
        return result

    requested_hessian_mode = active_request.hessian_evaluation_mode
    needs_second = requested_hessian_mode != "none"
    use_lifted_directions = (
        needs_second
        and requested_hessian_mode == "diagonal"
        and isinstance(model, SpatialDirectionalJetModel)
    )

    lifted_blocks: list[Tensor] = []
    lifted_slices: dict[int, slice] = {}
    coordinate_slice: slice | None = None
    lifted_width = 0

    def add_lifted_block(key: int, directions: Tensor) -> slice:
        nonlocal lifted_width
        existing = lifted_slices.get(key)
        if existing is not None:
            return existing
        width = int(directions.shape[1])
        result = slice(lifted_width, lifted_width + width)
        lifted_width += width
        lifted_blocks.append(directions.detach())
        lifted_slices[key] = result
        return result

    if use_lifted_directions:
        if active_request.need_hessian_diagonal:
            coordinate_slice = add_lifted_block(
                -1,
                torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1, -1),
            )
        for item in active_request.hessian_directions:
            if not isinstance(item, HessianQuadraticRequest):
                raise AssertionError("bilinear Hessian request reached lifted path")
            add_lifted_block(id(item.directions), item.directions)

    reusable_gradient_keys = set(lifted_slices)
    full_gradient_from_lifted = (
        active_request.need_full_gradient and coordinate_slice is not None
    )
    needs_autograd_gradient = (
        active_request.need_full_gradient and not full_gradient_from_lifted
    ) or any(
        id(item.directions) not in reusable_gradient_keys
        for item in active_request.gradient_directions
    )
    needs_dense_hessian = needs_second and not use_lifted_directions
    needs_first_autograd = needs_dense_hessian or needs_autograd_gradient
    t_leaf = t.detach().clone().requires_grad_(active_request.need_time_derivative)
    x_leaf = x.detach().clone().requires_grad_(needs_first_autograd)

    lifted_first: Tensor | None = None
    lifted_second: Tensor | None = None
    if use_lifted_directions:
        if not lifted_blocks:
            raise AssertionError("lifted Hessian path has no directions")
        assert isinstance(model, SpatialDirectionalJetModel)
        y, lifted_first, lifted_second = model.evaluate_spatial_directional_jet(
            t_leaf,
            x_leaf,
            torch.cat(lifted_blocks, dim=1),
        )
    else:
        y = model(t_leaf, x_leaf)
    if y.shape != (x.shape[0],):
        raise ValueError(f"scalar model must return [B], got {tuple(y.shape)}")
    if lifted_first is not None and lifted_first.shape != (
        x.shape[0],
        lifted_width,
    ):
        raise ValueError("directional evaluator returned an invalid first-derivative shape")
    if lifted_second is not None and lifted_second.shape != (
        x.shape[0],
        lifted_width,
    ):
        raise ValueError("directional evaluator returned an invalid second-derivative shape")

    gradient_internal: Tensor | None = None
    if full_gradient_from_lifted:
        assert lifted_first is not None and coordinate_slice is not None
        gradient_internal = lifted_first[:, coordinate_slice]
    elif needs_first_autograd:
        gradient_internal = _grad_or_zero(
            y,
            x_leaf,
            create_graph=create_graph or needs_dense_hessian,
        )

    dense_hessian: Tensor | None = None
    if needs_dense_hessian:
        assert gradient_internal is not None
        dense_hessian = _dense_hessian_from_gradient(
            gradient_internal,
            x_leaf,
            create_graph=create_graph,
        )

    directional_gradients: dict[str, Tensor] = {}
    for item in active_request.gradient_directions:
        lifted_slice = lifted_slices.get(id(item.directions))
        if lifted_slice is not None:
            assert lifted_first is not None
            directional_gradients[item.label] = _reduce_directional(
                lifted_first[:, lifted_slice],
                item.reduction,
            )
        else:
            assert gradient_internal is not None
            directional_gradients[item.label] = _project_gradient(
                gradient_internal,
                item,
            )

    directional_hessians: dict[str, Tensor] = {}
    for item in active_request.hessian_directions:
        if dense_hessian is not None:
            directional_hessians[item.label] = _project_hessian(
                dense_hessian, item, fuse_single_direction=active_request.need_full_hessian,
            )
        else:
            assert lifted_second is not None
            assert isinstance(item, HessianQuadraticRequest)
            lifted_slice = lifted_slices[id(item.directions)]
            directional_hessians[item.label] = _reduce_directional(
                lifted_second[:, lifted_slice],
                item.reduction,
            )

    hessian_diagonal: Tensor | None = None
    if active_request.need_hessian_diagonal:
        if dense_hessian is not None:
            hessian_diagonal = torch.diagonal(
                dense_hessian,
                dim1=-2,
                dim2=-1,
            )
        else:
            assert lifted_second is not None and coordinate_slice is not None
            hessian_diagonal = lifted_second[:, coordinate_slice]

    dt = (
        _grad_or_zero(y, t_leaf, create_graph=create_graph)
        if active_request.need_time_derivative
        else None
    )
    result = ScalarJet(
        y=y,
        dt=dt,
        grad=(
            gradient_internal
            if active_request.need_full_gradient or needs_dense_hessian
            else None
        ),
        hessian_diagonal=hessian_diagonal,
        hessian=dense_hessian if needs_dense_hessian else None,
        gradient_directions=directional_gradients,
        hessian_directions=directional_hessians,
    )
    return result if create_graph else result.detached()

