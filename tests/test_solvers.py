"""Small numerical contracts; no archived outputs or GPU required."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from DSRM.algorithms.residuals import cell_residual, path_values
from DSRM.differentiation import JetRequest, ScalarJet, evaluate_scalar_jet
from DSRM.models import build_model
from DSRM.parameters import load_config, parse_config
from DSRM.parameters.config import config_dict
from DSRM.problems import PDEProblem, TerminalCondition
from DSRM.runtime import run
from DSRM.runtime.experiment import prepare
from DSRM.runtime.storage import load_torch
from DSRM.sampling import cells, generator, occupation, paths


ROOT = Path(__file__).resolve().parents[1]


class Terminal(TerminalCondition):
    def value(self, x):
        return torch.exp(-x.square().sum(-1)/2)


class NoReference(PDEProblem):
    def __init__(self, T=1.):
        super().__init__(Terminal(2, T))

    def operator(self, t, x, jet):
        return .5*jet.trace+.1*torch.tanh(jet.y)


def small_config(kind="pinn", *, endpoint="NEXT", family="G2"):
    algorithm = dict(kind=kind, batch=5, schedule=[dict(steps=4, lr=.001)])
    if kind == "global":
        algorithm.update(N=2, master_N=4, endpoint=endpoint)
    if kind == "backward":
        algorithm.update(N=3, endpoint=endpoint, family=family, shared_prefix_steps=3,
                         schedule=[dict(steps=2, lr=.001)],
                         layer_overrides={"2": [dict(steps=4, lr=.001)]})
    return parse_config(dict(name="test", problem=dict(factory="examples.custom_problem:make_problem", kwargs={}),
        model=dict(width=5, hidden_layers=2), algorithm=algorithm,
        runtime=dict(dtype="float64", device_batch=3, log_every=100, checkpoint_every=2),
        evaluation=dict(validation_points=8, test_points=9, block=4, validate_every=2,
                        layer_validation_points=7, layer_validate_every=1)))


class Derivatives(unittest.TestCase):
    def test_dense_hessian_projections_match_independent_contractions(self):
        rng = generator(41)
        for count in (1, 3):
            for quadratic in (False, True):
                for reduction in ("none", "sum"):
                    with self.subTest(count=count, quadratic=quadratic, reduction=reduction):
                        # A nonsymmetric, strided input also checks bilinear orientation.
                        hessian = torch.randn(3, 4, 4, generator=rng, dtype=torch.float64).transpose(-1, -2)
                        actual_h = hessian.detach().requires_grad_()
                        expected_h = hessian.detach().clone().requires_grad_()
                        left = torch.randn(3, count, 4, generator=rng, dtype=torch.float64, requires_grad=True)
                        right = left if quadratic else torch.randn(
                            3, count, 4, generator=rng, dtype=torch.float64, requires_grad=True)
                        request = (JetRequest.quadratic_hessian(label="v", directions=left, reduction=reduction)
                            if quadratic else JetRequest.bilinear_hessian(
                                label="v", left=left, right=right, reduction=reduction))
                        dense_request = JetRequest.full().merged(request)
                        actual = ScalarJet.from_dense(y=hessian.new_zeros(3), gradient=hessian.new_zeros(3, 4),
                            hessian=actual_h, request=dense_request).hessian_direction("v")
                        expected = torch.stack([torch.stack([
                            left[b, k].detach() @ expected_h[b] @ right[b, k].detach()
                            for k in range(count)]) for b in range(3)])
                        if reduction == "sum":
                            expected = expected.sum(-1)
                        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
                        actual.square().sum().backward()
                        expected.square().sum().backward()
                        torch.testing.assert_close(actual_h.grad, expected_h.grad, rtol=2e-12, atol=2e-12)
                        self.assertIsNone(left.grad)
                        self.assertIsNone(right.grad)
                        fixed = ScalarJet.from_dense(y=hessian.new_zeros(3), gradient=hessian.new_zeros(3, 4),
                            hessian=hessian, request=request).hessian_direction("v")
                        fixed_reference = torch.einsum("bkd,bdc,bkc->bk", left.detach(), hessian, right.detach())
                        if reduction == "sum":
                            fixed_reference = fixed_reference.sum(-1)
                        torch.testing.assert_close(fixed, fixed_reference, rtol=0, atol=0)
                        self.assertFalse(fixed.requires_grad)
                        fixed_dense = ScalarJet.from_dense(y=hessian.new_zeros(3), gradient=hessian.new_zeros(3, 4),
                            hessian=hessian, request=dense_request).hessian_direction("v")
                        torch.testing.assert_close(fixed_dense, fixed_reference, rtol=2e-12, atol=2e-12)

    def test_reference_jets_against_autograd_and_pde(self):
        from DSRM.problems.manufactured import gaussian_problem, sine_problem
        from DSRM.problems.heat import translated_heat_problem
        problems = (gaussian_problem(d=2, modes=3), sine_problem(d=2, modes=3),
                    translated_heat_problem(R=10., alpha=5.))
        for problem in problems:
            bank = occupation(5, problem.d, generator(31), dtype=torch.float64)
            ref = problem.reference
            actual = ref.exact(bank.t, bank.x)
            # Differentiate the scalar reference independently of its supplied jets.
            fn = lambda t, x: ref.exact(t, x).y
            expected = evaluate_scalar_jet(fn, bank.t, bank.x)
            for field in ("y", "dt", "grad", "hessian"):
                torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=1e-11, atol=1e-12)
            residual = actual.dt+problem.operator(bank.t, bank.x, actual)
            torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=1e-12)

    def test_path_loss_time_normalization(self):
        class TimeOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = torch.nn.Parameter(torch.tensor(1.3, dtype=torch.float64))
            def forward(self, t, x):
                return self.c*(2.5-t)
        class ZeroOperator:
            def operator(self, t, x, jet):
                return torch.zeros_like(t)
        for N in (2, 8):
            for endpoint in ("CURRENT", "NEXT"):
                model = TimeOnly()
                bank = paths(3, N, 2, generator(6), T=2.5, dtype=torch.float64)
                loss = path_values(model, ZeroOperator(), bank, endpoint=endpoint).mean()
                torch.testing.assert_close(loss, model.c.square(), rtol=1e-12, atol=1e-12)
                gradient, = torch.autograd.grad(loss, model.c)
                torch.testing.assert_close(gradient, 2*model.c, rtol=1e-12, atol=1e-12)

    def test_fixed_terminal_keeps_fp64_precision_in_runtime(self):
        from DSRM.problems.manufactured import sine_problem
        problem = sine_problem(seed=97, d=2, modes=3)
        solver = prepare(small_config(), problem)
        x = torch.randn(5, 2, generator=generator(3), dtype=torch.float64)
        t = x.new_ones(5)
        torch.testing.assert_close(solver.model(t, x), problem.terminal.value(x), rtol=0, atol=0)

    def test_taylor_matches_autograd_values_and_parameter_gradients(self):
        torch.set_num_threads(1)
        problem = NoReference(T=2.5)
        config = small_config()
        analytic = build_model(problem, config.model, 9, dtype=torch.float64)
        automatic = build_model(problem, replace(config.model, backend="autograd"), 9, dtype=torch.float64)
        bank = occupation(5, 2, generator(3), T=problem.T, dtype=torch.float64)
        request = JetRequest.full(need_time_derivative=True)
        a = evaluate_scalar_jet(analytic, bank.t, bank.x, request=request)
        b = evaluate_scalar_jet(automatic, bank.t, bank.x, request=request)
        for field in ("y", "dt", "grad", "hessian"):
            torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=2e-12, atol=2e-12)
        for model, jet in ((analytic, a), (automatic, b)):
            (jet.y.square().sum()+jet.dt.square().sum()+jet.grad.square().sum()+jet.hessian.square().sum()).backward()
        for pa, pb in zip(analytic.parameters(), automatic.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-11, atol=2e-12)
        end = analytic(bank.t.new_full((len(bank),), problem.T), bank.x)
        torch.testing.assert_close(end, problem.terminal.value(bank.x), rtol=0, atol=0)

    def test_quadratic_cell_residual_cancels_brownian_increment(self):
        rng = torch.Generator().manual_seed(0)
        x = torch.randn(7, 2, dtype=torch.float64, generator=rng)
        dx = torch.randn(7, 2, dtype=torch.float64, generator=rng)
        h = .125
        H = 2*torch.eye(2, dtype=x.dtype).expand(7, 2, 2)
        current = ScalarJet(y=x.square().sum(-1), grad=2*x, hessian=H)
        z = cell_residual(current, (x+dx).square().sum(-1)-4*h, dx, h, x.new_full((7,), 4.))
        torch.testing.assert_close(z, torch.zeros_like(z), rtol=0, atol=3e-15)

    def test_taylor_first_direction_needs_no_terminal_hessian(self):
        class FirstOrderTerminal(Terminal):
            def hessian(self, x):
                raise AssertionError("A first-order request must not evaluate the terminal Hessian")
        problem = NoReference(T=2.5)
        problem.terminal = FirstOrderTerminal(problem.d, problem.T)
        config = small_config()
        bank = occupation(5, 2, generator(3), T=problem.T, dtype=torch.float64)
        directions = torch.randn(5, 1, 2, generator=generator(4), dtype=torch.float64,
                                 requires_grad=True)
        for reduction in ("none", "sum"):
            with self.subTest(reduction=reduction):
                analytic = build_model(problem, config.model, 9, dtype=torch.float64)
                automatic = build_model(problem, replace(config.model, backend="autograd"), 9, dtype=torch.float64)
                request = JetRequest.directional_gradient(label="v", directions=directions, reduction=reduction)
                a = evaluate_scalar_jet(analytic, bank.t, bank.x, request=request)
                b = evaluate_scalar_jet(automatic, bank.t, bank.x, request=request)
                torch.testing.assert_close(a.y, b.y, rtol=2e-12, atol=2e-12)
                torch.testing.assert_close(a.gradient_direction("v"), b.gradient_direction("v"), rtol=2e-12, atol=2e-12)
                self.assertIsNone(a.grad)
                self.assertIsNone(a.hessian)
                for jet in (a, b):
                    (jet.y.square().sum()+jet.gradient_direction("v").square().sum()).backward()
                for pa, pb in zip(analytic.parameters(), automatic.parameters()):
                    torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-11, atol=2e-12)
                self.assertIsNone(directions.grad)
                detached = evaluate_scalar_jet(analytic, bank.t, bank.x, request=request, create_graph=False)
                torch.testing.assert_close(detached.gradient_direction("v"), a.gradient_direction("v"))
                self.assertFalse(detached.gradient_direction("v").requires_grad)

    def test_taylor_dense_direction_preserves_frozen_hessian_gradients(self):
        problem = NoReference(T=2.5)
        config = small_config()
        bank = occupation(5, 2, generator(3), T=problem.T, dtype=torch.float64)
        directions = torch.randn(5, 1, 2, generator=generator(4), dtype=torch.float64)
        for freeze in (False, True):
            for reduction in ("none", "sum"):
                with self.subTest(freeze=freeze, reduction=reduction):
                    analytic = build_model(problem, config.model, 9, dtype=torch.float64)
                    automatic = build_model(problem, replace(config.model, backend="autograd"), 9, dtype=torch.float64)
                    analytic.fused_reduction = True
                    request = JetRequest.full(need_time_derivative=True).merged(
                        JetRequest.quadratic_hessian(label="v", directions=directions, reduction=reduction))
                    a = analytic.evaluate_analytic_jet(bank.t, bank.x, request=request, create_graph=True,
                                                       freeze_operator_hessian=freeze)
                    b = evaluate_scalar_jet(automatic, bank.t, bank.x, request=request)
                    for field in ("y", "dt", "grad", "hessian"):
                        torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=2e-12, atol=2e-12)
                    torch.testing.assert_close(a.hessian_direction("v"), b.hessian_direction("v"), rtol=2e-12, atol=2e-12)
                    self.assertEqual(a.hessian.requires_grad, not freeze)
                    self.assertTrue(a.hessian_direction("v").requires_grad)
                    if freeze:
                        b.hessian = b.hessian.detach()
                    for jet in (a, b):
                        (jet.y.square().sum()+jet.dt.square().sum()+jet.grad.square().sum()
                         +jet.hessian.square().sum()+jet.hessian_direction("v").square().sum()).backward()
                    for pa, pb in zip(analytic.parameters(), automatic.parameters()):
                        torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-11, atol=2e-12)

    def test_backward_requested_directions_match_dense_loss_and_gradients(self):
        from DSRM.algorithms.optimization import snapshot
        from DSRM.problems.manufactured import sine_problem
        for endpoint in ("CURRENT", "NEXT"):
            for family in ("G2", "F1"):
                for backend in ("taylor", "autograd"):
                    with self.subTest(endpoint=endpoint, family=family, backend=backend):
                        config = small_config("backward", endpoint=endpoint, family=family)
                        config = replace(config, model=replace(config.model, backend=backend))
                        solver = prepare(config, sine_problem(d=2, modes=3))
                        solver.states[solver.layer] = snapshot(solver.model)
                        solver.layer -= 1
                        solver._prepare_layer()
                        bank = cells(5, 2, solver.layer*solver.h, solver.h, generator(13), dtype=torch.float64)
                        actual = solver.values(bank)
                        left = evaluate_scalar_jet(solver.current, bank.t, bank.x, request=JetRequest.full())
                        right = evaluate_scalar_jet(solver.target, bank.t+bank.h, bank.x+bank.dx,
                                                    request=JetRequest.full(), create_graph=False)
                        driver = right if endpoint == "NEXT" else left
                        t, x = (bank.t+bank.h, bank.x+bank.dx) if endpoint == "NEXT" else (bank.t, bank.x)
                        operator = solver.problem.operator(t, x, driver)
                        if family == "F1":
                            operator = operator-.5*driver.trace
                        expected = cell_residual(left, right.y, bank.dx, bank.h, operator, family=family).square()/bank.h
                        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
                        ga = torch.autograd.grad(actual.mean(), solver.model.parameters())
                        gb = torch.autograd.grad(expected.mean(), solver.model.parameters())
                        for a, b in zip(ga, gb):
                            torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-12)

    def test_translation_taylor_matches_original_loss_and_gradients(self):
        for R in (0, 10, 12):
            for alpha in (.5, 5):
                with self.subTest(R=R, alpha=alpha):
                    config = load_config(ROOT/f"configs/paper/translation/03-r0-R{R}-alpha{alpha:g}.json")
                    config = replace(config, algorithm=replace(config.algorithm, batch=17),
                                     runtime=replace(config.runtime, device="cpu", dtype="float64", compile=False))
                    automatic = prepare(config)
                    analytic = prepare(replace(config, model=replace(config.model, backend="taylor")))
                    bank = automatic.draw(automatic.rng)
                    a, b = analytic.values(bank), automatic.values(bank)
                    torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-12)
                    ga = torch.autograd.grad(a.mean(), analytic.model.parameters())
                    gb = torch.autograd.grad(b.mean(), automatic.model.parameters())
                    for pa, pb in zip(ga, gb):
                        torch.testing.assert_close(pa, pb, rtol=2e-10, atol=2e-12)

    def test_master_grid_and_rng_are_paired(self):
        a = paths(3, 2, 2, generator(8), master_N=8, dtype=torch.float64)
        b = paths(3, 8, 2, generator(8), dtype=torch.float64)
        self.assertTrue(torch.equal(a.x, b.x[:, ::4]))
        self.assertTrue(torch.equal(a.t, b.t[::4]))


class Training(unittest.TestCase):
    def test_checkpoint_refuses_changed_custom_problem(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temp:
            run(config, temp, until=1, problem=NoReference(T=1.))
            with self.assertRaisesRegex(ValueError, "same config, problem"):
                run(config, temp, resume=True, problem=NoReference(T=2.))

    def test_chunks_accumulate_one_logical_update(self):
        for kind in ("pinn", "global", "backward"):
            with self.subTest(kind=kind):
                config = small_config(kind)
                full = prepare(replace(config, runtime=replace(config.runtime, device_batch=5)), NoReference())
                chunked = prepare(config, NoReference())
                full.update(); chunked.update()
                for a, b in zip(full.model.parameters(), chunked.model.parameters()):
                    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_resume_matches_uninterrupted_and_unknown_truth_is_optional(self):
        for kind, endpoint, family in (("pinn", "NEXT", "G2"), ("global", "CURRENT", "G2"),
                                      ("global", "NEXT", "G2"), ("backward", "CURRENT", "G2"),
                                      ("backward", "NEXT", "G2"), ("backward", "CURRENT", "F1"),
                                      ("backward", "NEXT", "F1")):
            with self.subTest(kind=kind, endpoint=endpoint, family=family), tempfile.TemporaryDirectory() as temp:
                config = small_config(kind, endpoint=endpoint, family=family)
                full, resumed = Path(temp)/"full", Path(temp)/"resumed"
                expected = run(config, full, problem=NoReference(T=1.5))
                partial = run(config, resumed, until=3, problem=NoReference(T=1.5))
                self.assertEqual(partial["status"], "paused")
                actual = run(config, resumed, resume=True, problem=NoReference(T=1.5))
                self.assertEqual(expected["test"], actual["test"])
                self.assertFalse(actual["test"]["continuous"]["reference_available"])
                self.assertNotIn("relative_errors", actual["test"]["continuous"])
                self.assertLess(actual["test"]["terminal"]["value_mse"], 1e-28)
                a = load_torch(full/"checkpoint.pt")["solver"]
                b = load_torch(resumed/"checkpoint.pt")["solver"]
                for key in a["model"]:
                    self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
                self.assertTrue(torch.equal(a["rng"], b["rng"]))

    def test_resume_at_backward_layer_boundary(self):
        config = small_config("backward")
        with tempfile.TemporaryDirectory() as temp:
            full, resumed = Path(temp)/"full", Path(temp)/"resumed"
            expected = run(config, full, problem=NoReference())
            run(config, resumed, until=4, problem=NoReference())
            actual = run(config, resumed, resume=True, problem=NoReference())
            self.assertEqual(expected["test"], actual["test"])


class Configuration(unittest.TestCase):
    def test_saved_parameters_can_be_loaded_as_a_new_run(self):
        for kind in ("pinn", "global", "backward"):
            config = small_config(kind)
            self.assertEqual(parse_config(json.loads(json.dumps(config_dict(config)))), config)

    def test_paper_catalog_is_self_contained(self):
        index = json.loads((ROOT/"configs/paper/index.json").read_text())
        self.assertEqual(len(index), 138)
        for entry in index:
            config = load_config(ROOT/"configs/paper"/entry["path"])
            self.assertEqual(config.name, entry["name"])
            self.assertGreater(config.total_steps, 0)

    def test_algorithm_rejects_irrelevant_endpoint(self):
        value = json.loads((ROOT/"configs/examples/custom_pinn.json").read_text())
        value["algorithm"]["endpoint"] = "NEXT"
        with self.assertRaises(ValueError):
            parse_config(value)


if __name__ == "__main__":
    unittest.main()
