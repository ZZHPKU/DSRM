"""Frozen-target backward regression, using G2 or F1 residuals."""
import hashlib

from ..differentiation import JetRequest, evaluate_scalar_jet
from ..differentiation.requests import GradientDirectionRequest, HessianQuadraticRequest
from .residuals import cell_residual


def extra_seed(seed, layer, N):
    label = "os-first-layer-extra" if layer == N-1 else f"os-layer-extra:{layer}"
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "little") % (2**63-1)


def spatial_initial(state, time):
    result = {k: v.detach().clone() for k, v in state.items()}
    result["net.0.bias"].add_(time*result["net.0.weight"][:, 0])
    result["net.0.weight"][:, 0].zero_()
    return result


def regression_values(current_model, target, problem, bank, *, endpoint, family, create_graph=True):
    directions = bank.dx[:, None, :]
    current_request = JetRequest(
        gradient_directions=(GradientDirectionRequest("increment", directions, "sum"),),
        hessian_directions=(HessianQuadraticRequest("increment", directions, "sum"),) if family == "G2" else ())
    operator_request = problem.operator_request()
    if family == "F1":
        operator_request = operator_request.merged(JetRequest(need_hessian_diagonal=True))
    if endpoint == "CURRENT":
        current_request = current_request.merged(operator_request)
    current = evaluate_scalar_jet(current_model, bank.t, bank.x, request=current_request, create_graph=create_graph)
    # The next node is fixed during the complete optimization of this layer.
    following = evaluate_scalar_jet(target, bank.t+bank.h, bank.x+bank.dx,
                                   request=operator_request if endpoint == "NEXT" else JetRequest(), create_graph=False)
    driver = following if endpoint == "NEXT" else current
    t = bank.t+bank.h if endpoint == "NEXT" else bank.t
    x = bank.x+bank.dx if endpoint == "NEXT" else bank.x
    operator = problem.operator(t, x, driver)
    if family == "F1":
        operator = operator-.5*driver.trace
    z = cell_residual(current, following.y, bank.dx, bank.h, operator, family=family)
    return z.square()/bank.h
