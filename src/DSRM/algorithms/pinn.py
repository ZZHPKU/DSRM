"""Pointwise squared strong-form PDE residual."""
from ..differentiation import JetRequest, evaluate_scalar_jet


def point_values(model, problem, bank, *, create_graph=True):
    request = problem.operator_request().merged(JetRequest(need_time_derivative=True))
    jet = evaluate_scalar_jet(model, bank.t, bank.x, request=request, create_graph=create_graph)
    residual = jet.dt+problem.operator(bank.t, bank.x, jet)
    return (residual*bank.sqrt_weight).square()
