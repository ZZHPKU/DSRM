"""Spatial nodes and piecewise affine reconstruction for backward regression."""
import torch
from torch import nn

from ..differentiation import JetRequest, ScalarJet, evaluate_scalar_jet


class SpatialNode(nn.Module):
    def __init__(self, raw, time):
        super().__init__()
        self.raw, self.time, self.T = raw, time, raw.T
        self.fixed = TerminalSolution(raw.terminal)

    def forward(self, t, x):
        a = self.time/self.T
        free = self.raw.network_value(torch.zeros_like(t), x)
        return a*self.raw.terminal.value(x)+(1-a)*free

    def evaluate_analytic_jet(self, t, x, *, request, create_graph):
        # A wrapper without this method lets the common evaluator differentiate
        # spatial nodes for either model backend, including custom terminals.
        if not hasattr(self.raw, "jet_tensors"):
            return evaluate_scalar_jet(self.forward, t, x, request=request, create_graph=create_graph)
        with torch.set_grad_enabled(create_graph):
            free = self.raw.evaluate_analytic_jet(torch.zeros_like(t), x,
                                                request=request, create_graph=create_graph)
            fixed = self.fixed.evaluate_analytic_jet(t, x, request=request, create_graph=False)
            a = self.time/self.T
            def mix(field):
                value = getattr(free, field)
                return None if value is None else a*getattr(fixed, field)+(1-a)*value
            result = ScalarJet(y=mix("y"), grad=mix("grad"), hessian=mix("hessian"),
                hessian_diagonal=mix("hessian_diagonal"),
                dt=torch.zeros_like(t) if request.need_time_derivative else None,
                gradient_directions={key: a*fixed.gradient_directions[key]+(1-a)*value
                                     for key, value in free.gradient_directions.items()},
                hessian_directions={key: a*fixed.hessian_directions[key]+(1-a)*value
                                    for key, value in free.hessian_directions.items()})
        return result if create_graph else result.detached()


class TerminalSolution(nn.Module):
    def __init__(self, terminal):
        super().__init__()
        self.terminal = terminal

    def forward(self, t, x):
        return self.terminal.value(x)

    def evaluate_analytic_jet(self, t, x, *, request, create_graph):
        with torch.enable_grad():
            y = self.terminal.value(x)
            grad = self.terminal.gradient(x)
            hessian = self.terminal.hessian(x)
        result = ScalarJet.from_dense(y=y, gradient=grad, hessian=hessian,
                                     dt=torch.zeros_like(t), request=request)
        return result if create_graph else result.detached()


class PiecewiseSolution(nn.Module):
    def __init__(self, nodes, terminal, T):
        super().__init__()
        self.nodes = nn.ModuleList(nodes)
        self.terminal = TerminalSolution(terminal)
        self.T, self.N = T, len(nodes)

    def forward(self, t, x):
        indices = (t*self.N/self.T).long().clamp(0, self.N-1)
        out = torch.empty_like(t)
        for i in indices.unique().tolist():
            mask = indices == i
            fraction = (t[mask]-i*self.T/self.N)/(self.T/self.N)
            right = self.terminal if i+1 == self.N else self.nodes[i+1]
            out[mask] = (1-fraction)*self.nodes[i](t[mask], x[mask])+fraction*right(t[mask], x[mask])
        return out

    def evaluate_analytic_jet(self, t, x, *, request, create_graph):
        indices = (t*self.N/self.T).long().clamp(0, self.N-1)
        h = self.T/self.N
        y, dt = torch.empty_like(t), torch.empty_like(t)
        grad, hessian = torch.empty_like(x), x.new_empty(len(x), x.shape[1], x.shape[1])
        for i in indices.unique().tolist():
            mask = indices == i
            tm, xm = t[mask], x[mask]
            right = self.terminal if i+1 == self.N else self.nodes[i+1]
            left_jet = evaluate_scalar_jet(self.nodes[i], tm, xm, request=JetRequest.full(), create_graph=create_graph)
            right_jet = evaluate_scalar_jet(right, tm, xm, request=JetRequest.full(), create_graph=create_graph)
            fraction = (tm-i*h)/h
            y[mask] = (1-fraction)*left_jet.y+fraction*right_jet.y
            grad[mask] = (1-fraction[:, None])*left_jet.grad+fraction[:, None]*right_jet.grad
            hessian[mask] = (1-fraction[:, None, None])*left_jet.hessian+fraction[:, None, None]*right_jet.hessian
            dt[mask] = (right_jet.y-left_jet.y)/h
        result = ScalarJet.from_dense(y=y, gradient=grad, hessian=hessian, dt=dt, request=request)
        return result if create_graph else result.detached()
