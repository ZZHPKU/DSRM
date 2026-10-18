"""Held-out residual, terminal, and optional reference-solution errors."""
import math

import torch

from ..differentiation import JetRequest, evaluate_scalar_jet
from ..parameters.config import ImportanceSampling
from ..sampling import generator, occupation
from ..sampling.importance import GaussianMixtureSampler
from .statistics import moments, spatial_summary


def point_bank(problem, sampling, count, seed, *, device="cpu", dtype=torch.float64):
    rng = generator(seed, device)
    if isinstance(sampling, ImportanceSampling):
        if problem.T != 1. or problem.d != len(sampling.center):
            raise ValueError("Gaussian mixture sampler requires T=1 and matching dimension")
        return GaussianMixtureSampler(sampling.center, sampling.templates, device=device, dtype=dtype).draw(count, rng)
    return occupation(count, problem.d, rng, T=problem.T, dtype=dtype)


def reference_jet(reference, t, x):
    if hasattr(reference, "exact"):
        return reference.exact(t, x)
    return evaluate_scalar_jet(reference, t, x, request=JetRequest.full(), create_graph=False)


def occupation_metrics(model, problem, bank, *, block=2048):
    parts = []
    with_reference = problem.reference is not None
    request = (JetRequest.full(need_time_derivative=True) if with_reference else
               problem.operator_request().merged(JetRequest(need_time_derivative=True)))
    for lo in range(0, len(bank), block):
        b = bank.slice(lo, lo+block)
        jet = evaluate_scalar_jet(model, b.t, b.x, request=request, create_graph=False)
        residual = jet.dt+problem.operator(b.t, b.x, jet)
        factor = b.sqrt_weight.square()*problem.value_scale**2
        if with_reference:
            truth = reference_jet(problem.reference, b.t, b.x)
            rows = torch.stack(((jet.y-truth.y).square(),
                (jet.grad-truth.grad).square().sum(-1),
                (jet.hessian-truth.hessian).square().sum((-1, -2)), residual.square(),
                truth.y.square(), truth.grad.square().sum(-1), truth.hessian.square().sum((-1, -2))), -1)
            parts.append((factor[:, None]*rows).detach().cpu())
        else:
            parts.append((factor*residual.square()).detach().cpu()[:, None])
    values = torch.cat(parts)
    if with_reference:
        summary = spatial_summary(values, problem.reference_energies)
        energy = (problem.reference_energies or {}).get("hessian")
        if energy and energy > 0:
            summary["H_star"] = math.sqrt(energy)
            summary["K_H"] = summary["absolute_errors"]["hessian"]/summary["residual_rms"] if summary["residual_rms"] else None
    else:
        mean, se = moments(values)
        summary = dict(residual_mse=float(mean[0]), residual_rms=float(mean[0].sqrt()),
                       residual_mse_mc_se=float(se[0]))
    return dict(points=len(bank), reference_available=with_reference, **summary)


def terminal_metrics(model, problem, *, count, seed, block=2048):
    param = next(model.parameters())
    bank = occupation(count, problem.d, generator(seed, param.device), T=problem.T,
                      dtype=param.dtype, terminal=True)
    parts = []
    for lo in range(0, count, block):
        b = bank.slice(lo, lo+block)
        jet = evaluate_scalar_jet(model, b.t, b.x, request=JetRequest(need_full_gradient=True), create_graph=False)
        value, grad = problem.terminal.value(b.x), problem.terminal.gradient(b.x)
        parts.append(torch.stack(((jet.y-value).square(), (jet.grad-grad).square().sum(-1)), -1).detach().cpu())
    means, _ = moments(torch.cat(parts)*problem.value_scale**2)
    return dict(points=count, value_mse=float(means[0]), gradient_mse=float(means[1]))


def evaluate(model, problem, config, *, split="test"):
    count = getattr(config.evaluation, split+"_points")
    seed = getattr(config.seeds, split+"_occupation")
    bank = point_bank(problem, config.sampling, count, seed)
    continuous = occupation_metrics(model, problem, bank, block=config.evaluation.block)
    result = dict(split=split, dtype="float64", device="cpu", continuous=continuous)
    if split == "test":
        result["terminal"] = terminal_metrics(model, problem, count=count, seed=config.seeds.test_terminal,
                                                block=config.evaluation.block)
    return result
