from __future__ import annotations
from typing import Any
import torch
from torch import Tensor, nn
from .mlp import make_mlp
from ..problems.base import TerminalCondition

def _activation_value_slope_curvature(
    activation: nn.Module,
    value: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(activation, nn.Tanh):
        raise TypeError("The directional evaluator requires tanh")
    y = torch.tanh(value)
    slope = 1-y.square()
    return y, slope, -2*y*slope


def _mlp_value_and_directional_jet(
    net: nn.Sequential,
    inputs: Tensor,
    input_directions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Propagate value, first directions, and pure second directions together."""
    if inputs.ndim != 2 or input_directions.ndim != 3:
        raise ValueError("expected inputs=[B,n] and directions=[B,K,n]")
    if (
        input_directions.shape[0] != inputs.shape[0]
        or input_directions.shape[2] != inputs.shape[1]
    ):
        raise ValueError("direction batch or input dimension does not match inputs")

    modules = list(net)
    if not modules or not isinstance(modules[-1], nn.Linear):
        raise TypeError("directional Taylor evaluation requires a final Linear layer")
    hidden_modules = modules[:-1]
    if len(hidden_modules) % 2 != 0:
        raise TypeError("MLP hidden modules must alternate Linear and activation")

    value = inputs
    tangent = input_directions.transpose(1, 2)
    hidden_linears: list[nn.Linear] = []
    preactivation_tangents: list[Tensor] = []
    slopes: list[Tensor] = []
    curvatures: list[Tensor] = []
    for index in range(0, len(hidden_modules), 2):
        linear = hidden_modules[index]
        activation = hidden_modules[index + 1]
        if not isinstance(linear, nn.Linear):
            raise TypeError("MLP hidden affine modules must be Linear")
        preactivation = linear(value)
        preactivation_tangent = torch.einsum(
            "oi,bik->bok", linear.weight, tangent
        )
        value, slope, curvature = _activation_value_slope_curvature(
            activation,
            preactivation,
        )
        hidden_linears.append(linear)
        preactivation_tangents.append(preactivation_tangent)
        slopes.append(slope)
        curvatures.append(curvature)
        tangent = slope[:, :, None] * preactivation_tangent

    output = modules[-1]
    if output.out_features != 1:
        raise ValueError("scalar directional Taylor evaluation requires one output")
    scalar_value = output(value).squeeze(-1)
    first_directions = torch.einsum(
        "oi,bik->bok", output.weight, tangent
    ).squeeze(1)

    second_directions = inputs.new_zeros(
        inputs.shape[0], input_directions.shape[1]
    )
    if hidden_linears:
        adjoint = output.weight[0].expand(inputs.shape[0], -1)
        for index in range(len(hidden_linears) - 1, -1, -1):
            local_curvature = adjoint * curvatures[index]
            second_directions = second_directions + (
                local_curvature[:, :, None]
                * preactivation_tangents[index].square()
            ).sum(dim=1)
            if index > 0:
                adjoint = (
                    adjoint * slopes[index]
                ) @ hidden_linears[index].weight
    return scalar_value, first_directions, second_directions


def _spatial_input_directions(t: Tensor, x: Tensor, directions: Tensor) -> Tensor:
    if (
        t.ndim != 1
        or x.ndim != 2
        or directions.ndim != 3
        or directions.shape != (x.shape[0], directions.shape[1], x.shape[1])
    ):
        raise ValueError("expected t=[B], x=[B,d], directions=[B,K,d]")
    time_directions = directions.new_zeros(directions.shape[0], directions.shape[1], 1)
    return torch.cat((time_directions, directions), dim=-1)


class HardTerminalScalar(nn.Module):
    """U(t,x)=(t/T)g(x)+(1-t/T)M(t,x-shift), with one network evaluation."""

    def __init__(
        self,
        *,
        d: int,
        T: float,
        terminal: TerminalCondition,
        hidden_layers: int = 2,
        width: int = 32,
        activation: str = "tanh",
        input_shift: Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if d <= 0 or T <= 0:
            raise ValueError("d and T must be positive")
        if getattr(terminal, "d", None) != d or abs(float(getattr(terminal, "T", T)) - float(T)) > 1e-12:
            raise ValueError("terminal condition does not match model")
        self.d = int(d)
        self.T = float(T)
        self.terminal = terminal
        self.hidden_layers = int(hidden_layers)
        self.width = int(width)
        self.activation = str(activation)
        shift = torch.zeros(d, dtype=dtype, device=device) if input_shift is None else torch.as_tensor(input_shift, dtype=dtype, device=device).detach().clone()
        if shift.shape != (d,):
            raise ValueError("input_shift must have shape [d]")
        self.register_buffer("input_shift", shift)
        if isinstance(terminal, nn.Module):
            terminal.to(device=torch.device(device), dtype=dtype)
        self.net = make_mlp(
            d + 1,
            1,
            hidden_layers=hidden_layers,
            width=width,
            activation=activation,
            dtype=dtype,
            device=device,
        )

    def _time_vector(self, t: Tensor | float, x: Tensor) -> Tensor:
        if not torch.is_tensor(t):
            t = x.new_tensor(float(t))
        t = t.to(device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        elif t.ndim != 1:
            raise ValueError("t must be scalar, [B], or [B,1]")
        if t.shape[0] != x.shape[0]:
            raise ValueError("t and x batch dimensions must match")
        return t

    def network_value(self, t: Tensor | float, x: Tensor) -> Tensor:
        t_vector = self._time_vector(t, x)
        return self.net(torch.cat((t_vector[:, None], x-self.input_shift), dim=-1)).squeeze(-1)

    def forward(self, t: Tensor | float, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[-1] != self.d:
            raise ValueError(f"x must have shape [B,{self.d}]")
        t_vector = self._time_vector(t, x)
        terminal_value = self.terminal.value(x)
        alpha = t_vector / self.T
        return alpha * terminal_value + (1-alpha) * self.network_value(t_vector, x)

    def evaluate_spatial_directional_jet(
        self,
        t: Tensor,
        x: Tensor,
        directions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        t_vector = self._time_vector(t, x)
        detached_directions = directions.detach()
        inputs = torch.cat((t_vector[:, None], x-self.input_shift), dim=-1)
        free_value, free_first, free_second = _mlp_value_and_directional_jet(
            self.net,
            inputs,
            _spatial_input_directions(t_vector, x, detached_directions),
        )
        terminal_value = self.terminal.value(x)
        terminal_gradient = self.terminal.gradient(x)
        terminal_hessian = self.terminal.hessian(x)
        terminal_first = torch.einsum(
            "bd,bkd->bk", terminal_gradient, detached_directions
        )
        terminal_second = torch.einsum(
            "bkd,bdc,bkc->bk",
            detached_directions,
            terminal_hessian,
            detached_directions,
        )
        alpha = t_vector / self.T
        beta = 1-alpha
        return (
            alpha * terminal_value + beta * free_value,
            alpha[:, None] * terminal_first + beta[:, None] * free_first,
            alpha[:, None] * terminal_second + beta[:, None] * free_second,
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "candidate_kind": "state_local_scalar",
            "terminal_policy": "interpolation",
            "jet_source": "graph_directional_taylor_or_autograd_dense",
            "hessian_convention": "pure_quadratic_lifted_or_symmetrized_dense",
            "activation": self.activation,
            "hidden_layers": self.hidden_layers,
            "width": self.width,
            "d": self.d,
            "T": self.T,
        }
        payload["input_shift"] = self.input_shift.detach().cpu().tolist()
        return payload
