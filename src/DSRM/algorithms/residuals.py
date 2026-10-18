"""Second-order stochastic residual and first-order BSDE residual."""
from dataclasses import fields

import torch

from ..differentiation import JetRequest, ScalarJet, evaluate_scalar_jet


def cell_residual(current, following, dx, h, operator, *, family="G2"):
    first = (current.gradient_direction("increment") if "increment" in current.gradient_directions
             else (current.require_gradient()*dx).sum(-1))
    residual = following-current.y-first
    if family == "G2":
        second = (current.hessian_direction("increment") if "increment" in current.hessian_directions
                  else torch.einsum("bi,bij,bj->b", dx, current.require_hessian(), dx))
        residual = residual-.5*second
    elif family != "F1":
        raise ValueError("Residual family must be G2 or F1")
    return residual+h*operator


def node_slice(jet, batch, nodes, start, stop):
    def cut(v):
        return v.reshape(batch, nodes, *v.shape[1:])[:, start:stop].reshape(-1, *v.shape[1:])
    return ScalarJet(**{f.name: ({k: cut(v) for k, v in getattr(jet, f.name).items()}
        if isinstance(getattr(jet, f.name), dict) else None if getattr(jet, f.name) is None
        else cut(getattr(jet, f.name))) for f in fields(jet)})


def path_values(model, problem, bank, *, endpoint, create_graph=True):
    batch, nodes, d = bank.x.shape
    N = nodes-1
    T = float(bank.t[-1]-bank.t[0])
    h = T/N
    t, x = bank.t.expand(batch, nodes).reshape(-1), bank.x.reshape(-1, d)
    jet = evaluate_scalar_jet(model, t, x, request=JetRequest.full(), create_graph=create_graph)
    current, following = node_slice(jet, batch, nodes, 0, N), node_slice(jet, batch, nodes, 1, nodes)
    if endpoint == "NEXT":
        ts, xs, driver = bank.t[1:], bank.x[:, 1:], following
    elif endpoint == "CURRENT":
        ts, xs, driver = bank.t[:-1], bank.x[:, :-1], current
    else:
        raise ValueError("endpoint must be CURRENT or NEXT")
    operator = problem.operator(ts.expand(batch, N).reshape(-1), xs.reshape(-1, d), driver)
    z = cell_residual(current, following.y, (bank.x[:, 1:]-bank.x[:, :-1]).reshape(-1, d), h, operator)
    values = z.reshape(batch, N).square().sum(1)/(T*h)
    return values if create_graph else values.detach()
