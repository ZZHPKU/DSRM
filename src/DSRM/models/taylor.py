"""Exact tanh jets, optionally fused on CUDA, with unchanged state keys.

Frozen nodes can evaluate without parameter graphs. Derivative requests
choose full, diagonal, or directional Hessian information.
"""
import torch
from torch.nn import functional as F
from ..differentiation.jets import ScalarJet
from .terminal import HardTerminalScalar
from ..differentiation.requests import HessianQuadraticRequest


class TanhJetMixin:
    shared_path_jets = True

    def setup_taylor(self):
        self.hard_mapping = True
        self.fused_reduction = False
        self.jets_compiled = False
        self.jet_function = self.jet_tensors

    def enable_cuda_compile(self):
        self.fused_reduction = True
        self.jet_function = torch.compile(self.jet_tensors, fullgraph=True)
        self.jets_compiled = True

    def terminal_tensors(self, x, mode, direction):
        if hasattr(self.terminal, "taylor_tensors"):
            return self.terminal.taylor_tensors(x, mode, direction)
        # Fixed terminals need spatial derivatives, never a reference PDE solution.
        with torch.enable_grad():
            value = self.terminal.value(x)
            gradient = self.terminal.gradient(x)
            hessian = self.terminal.hessian(x) if mode != "none" or direction is not None else None
        second = (hessian if mode == "dense" else
                  hessian.diagonal(dim1=-2, dim2=-1) if mode == "diagonal" else None)
        quadratic = (torch.einsum("bi,bij,bj->b", direction, hessian, direction)
                     if direction is not None else None)
        return value.detach(), gradient.detach(), second, quadratic

    def jet_tensors(self, t, x, need_time, mode, direction, freeze_operator_hessian,
                    only_direction, need_quadratic=True):
        layers = list(self.net)
        value = torch.cat((t[:, None], x-getattr(self, "input_shift", 0.)), -1)
        pre_jets, slopes, curvatures, linears = [], [], [], []
        tangent = None
        start = 0 if need_time else 1
        quadratic_direction = direction if need_quadratic else None
        need_second = mode != "none" or quadratic_direction is not None
        for i in range(0, len(layers)-1, 2):
            linear = layers[i]
            pre = linear(value)
            if tangent is None:
                if only_direction:
                    tangent_pre = F.linear(direction, linear.weight[:, 1:])[:, None, :]
                else:
                    tangent_pre = linear.weight[:, start:].T.unsqueeze(0).expand(len(x), -1, -1)
            else:
                tangent_pre = F.linear(tangent, linear.weight)
            value = torch.tanh(pre)
            slope = 1-value.square()
            if need_second:
                pre_jets.append(tangent_pre)
                slopes.append(slope)
                curvatures.append(-2*value*slope)
                linears.append(linear)
            tangent = tangent_pre*slope[:, None, :]
        output = layers[-1]
        free_y = output(value).squeeze(-1)
        first = F.linear(tangent, output.weight).squeeze(-1)
        free_grad = first[:, 1:] if need_time else first
        free_dt = first[:, 0] if need_time else None
        d = x.shape[1]
        second = (x.new_zeros(len(x), d, d) if mode == "dense" else
                  x.new_zeros(len(x), d) if mode == "diagonal" else None)
        quadratic = x.new_zeros(len(x)) if quadratic_direction is not None else None
        adjoint = output.weight[0].expand(len(x), -1)
        for i in range(len(linears)-1, -1, -1):
            q = adjoint*curvatures[i]
            J = pre_jets[i][:, 1:] if need_time else pre_jets[i]
            if mode == "dense":
                if self.fused_reduction:
                    second = second+(J[:, :, None, :]*J[:, None, :, :]
                                     *q[:, None, None, :]).sum(-1)
                else:
                    second = second+torch.bmm(J*q[:, None, :], J.transpose(1, 2))
            elif mode == "diagonal":
                second = second+(J.square()*q[:, None, :]).sum(-1)
            if quadratic_direction is not None:
                projected = J[:, 0] if only_direction else (J*quadratic_direction[:, :, None]).sum(1)
                quadratic = quadratic+(q*projected.square()).sum(-1)
            if i > 0:
                adjoint = (adjoint*slopes[i])@linears[i].weight
        if self.hard_mapping:
            with torch.no_grad():
                gy, gg, gh, gq = self.terminal_tensors(x, mode, quadratic_direction)
                if only_direction:
                    gg = (gg*direction).sum(-1, keepdim=True)
            alpha = t/self.T
            beta = 1-alpha
            y = alpha*gy+beta*free_y
            dt = (gy-free_y)/self.T+beta*free_dt if need_time else None
            grad = alpha[:, None]*gg+beta[:, None]*free_grad
            if mode == "dense":
                second = alpha[:, None, None]*gh+beta[:, None, None]*second
            elif mode == "diagonal":
                second = alpha[:, None]*gh+beta[:, None]*second
            if quadratic_direction is not None:
                quadratic = alpha*gq+beta*quadratic
        else:
            y, dt, grad = free_y, free_dt, free_grad
        if freeze_operator_hessian and second is not None:
            second = second.detach()
        return y, dt, grad, second, quadratic

    def evaluate_analytic_jet(self, t, x, *, request, create_graph,
                              freeze_operator_hessian=False):
        directional = (len(request.hessian_directions) == 1
                       and isinstance(request.hessian_directions[0], HessianQuadraticRequest)
                       and request.hessian_directions[0].directions.shape[1] == 1)
        mode = ("dense" if request.requires_dense_hessian or (request.hessian_directions and not directional)
                else "diagonal" if request.need_hessian_diagonal else "none")
        item = request.hessian_directions[0] if directional else None
        only_direction = (directional and mode == "none" and not request.need_time_derivative
                          and not request.need_full_gradient
                          and all(g.directions is item.directions for g in request.gradient_directions))
        # Reuse dense Hessians unless their parameter graph is deliberately frozen.
        separate_quadratic = directional and (mode != "dense" or freeze_operator_hessian)
        direction = item.directions[:, 0].detach() if separate_quadratic else None
        if (mode == "none" and not directional and not request.need_time_derivative
                and not request.need_full_gradient and len(request.gradient_directions) == 1
                and request.gradient_directions[0].directions.shape[1] == 1):
            direction = request.gradient_directions[0].directions[:, 0].detach()
            only_direction = True
        # Explicit chain rules need no parameter graph for read-only evaluation.
        with torch.set_grad_enabled(create_graph):
            y, dt, grad, second, quadratic = self.jet_function(
                t, x, request.need_time_derivative, mode, direction,
                freeze_operator_hessian, only_direction, need_quadratic=separate_quadratic)
            if mode == "dense":
                jet = ScalarJet.from_dense(y=y, gradient=grad, hessian=second, dt=dt, request=request)
            else:
                jet = ScalarJet(y=y, dt=dt, grad=grad if request.need_full_gradient else None,
                                hessian_diagonal=second if mode == "diagonal" else None)
                for g in request.gradient_directions:
                    values = grad if only_direction else torch.einsum("bi,bki->bk", grad, g.directions.detach())
                    jet.gradient_directions[g.label] = values.sum(-1) if g.reduction == "sum" else values
            if separate_quadratic:
                jet.hessian_directions[item.label] = quadratic if item.reduction == "sum" else quadratic[:, None]
        return jet if create_graph else jet.detached()


class TaylorScalar(TanhJetMixin, HardTerminalScalar):
    """Hard-terminal tanh network with exact forward spatial derivatives."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.activation != "tanh":
            raise ValueError("Taylor backend requires tanh")
        self.setup_taylor()
