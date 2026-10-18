"""Timing coverage, excluded work, state preservation, and CUDA boundaries."""
from contextlib import ExitStack, redirect_stdout
import copy
from dataclasses import replace
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from DSRM.algorithms import backward_solver, global_solver
from DSRM.algorithms.optimization import warmup_update
from DSRM.parameters.config import Problem
from DSRM.runtime import experiment
from DSRM.runtime.storage import atomic_torch, load_torch
from DSRM.timing import TrainingTimer, compiled_update
from test_solvers import NoReference, small_config


class Clock:
    def __init__(self):
        self.value = 0.

    def read(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def assert_state(test, actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            assert_state(test, actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            assert_state(test, a, b)
    else:
        test.assertEqual(actual, expected)


class Timing(unittest.TestCase):
    def test_segments_wait_for_gpu_without_synchronizing_every_update(self):
        clock = Clock()
        timer = TrainingTimer("cpu")
        with patch("DSRM.timing.time", SimpleNamespace(perf_counter=clock.read)), \
                patch.object(timer, "synchronize", side_effect=lambda: clock.advance(2)) as sync:
            timer.start()
            clock.advance(5)
            timer.updated(1)
            timer.start()
            clock.advance(7)
            timer.updated(2)
            self.assertEqual(sync.call_count, 1)
            timer.pause()
            self.assertEqual(timer.seconds, 14.)
            with timer.exclude("validation"):
                clock.advance(100)
            self.assertEqual(timer.seconds, 14.)
            timer.start()
            clock.advance(3)
            timer.updated(3)
            timer.pause()
            self.assertEqual(timer.seconds, 19.)
        self.assertTrue(timer.report(3)["complete"])

    def test_coverage_rejects_gaps_and_marks_legacy_prefix_missing(self):
        timer = TrainingTimer.restore("cpu", None, step=3)
        timer.start()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            timer.updated(5)
        timer.updated(4)
        state = timer.state_dict()
        resumed = TrainingTimer.restore("cpu", state, step=4)
        resumed.start()
        resumed.updated(5)
        report = resumed.report(5)
        self.assertEqual(report["start_step"], 3)
        self.assertEqual(report["timed_updates"], 2)
        self.assertFalse(report["complete"])
        with self.assertRaisesRegex(ValueError, "saved training step"):
            TrainingTimer.restore("cpu", state, step=5)
        with self.assertRaisesRegex(ValueError, "timing version"):
            TrainingTimer.restore("cpu", dict(state, version=999), step=4)
        with self.assertRaisesRegex(ValueError, "finite"):
            TrainingTimer.restore("cpu", dict(state, training_seconds=float("nan")), step=4)

    def test_warmup_restores_weights_gradients_adam_and_random_state(self):
        model = torch.nn.Linear(2, 1).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        def operation():
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(3, 2, dtype=torch.float64)).square().mean().backward()
            optimizer.step()
        operation()
        weights = copy.deepcopy(model.state_dict())
        state = copy.deepcopy(optimizer.state_dict())
        gradients = [p.grad.clone() for p in model.parameters()]
        rng = torch.random.get_rng_state().clone()
        warmup_update(model, optimizer, operation)
        assert_state(self, model.state_dict(), weights)
        assert_state(self, optimizer.state_dict(), state)
        assert_state(self, [p.grad for p in model.parameters()], gradients)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_only_sampling_loss_and_adam_are_charged(self):
        for kind in ("pinn", "global", "backward"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp, \
                    redirect_stdout(io.StringIO()), ExitStack() as stack:
                clock = Clock()
                stack.enter_context(patch("DSRM.timing.time", SimpleNamespace(perf_counter=clock.read)))
                def delayed(function, seconds):
                    def call(*args, **kwargs):
                        clock.advance(seconds)
                        return function(*args, **kwargs)
                    return call
                for module, name, seconds in (
                    (experiment, "occupation", 2), (experiment, "paths", 2),
                    (backward_solver, "cells", 2),
                    (global_solver, "update", 5), (backward_solver, "update", 5),
                    (torch.optim.Adam, "step", 3),
                    (experiment, "evaluate", 100), (backward_solver.BackwardSolver, "validate", 100),
                    (experiment, "atomic_torch", 200), (experiment, "append_event", 300)):
                    stack.enter_context(patch.object(module, name, delayed(getattr(module, name), seconds)))
                config = small_config(kind)
                config = replace(config, runtime=replace(config.runtime, device_batch=config.algorithm.batch,
                                                          log_every=1, checkpoint_every=1))
                result = experiment.run(config, temp, problem=NoReference())
                self.assertEqual(result["training_seconds"], 10*config.total_steps)
                report = result["training_timing"]
                self.assertEqual(report["timed_updates"], config.total_steps)
                self.assertTrue(report["complete"])
                for category in ("validation", "checkpoint", "logging", "final_evaluation"):
                    self.assertGreater(report["excluded_seconds"][category], 0)

    def test_updates_after_selected_model_are_all_timed(self):
        original = global_solver.GlobalSolver.observe
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch.object(global_solver.GlobalSolver, "observe", lambda solver, metric: original(solver, 1.)):
            config = small_config()
            result = experiment.run(config, temp, problem=NoReference())
            self.assertEqual(result["selected_step"], 0)
            self.assertEqual(result["step"], config.total_steps)
            self.assertEqual(result["training_timing"]["timed_updates"], config.total_steps)
            self.assertTrue(result["training_timing"]["complete"])

    def test_resume_adds_time_and_never_invents_missing_measurements(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            measured, legacy = Path(temp)/"measured", Path(temp)/"legacy"
            first = experiment.run(config, measured, until=2, problem=NoReference())
            self.assertEqual(first["training_timing"]["timed_updates"], 2)
            self.assertFalse(first["training_timing"]["complete"])
            final = experiment.run(config, measured, resume=True, problem=NoReference())
            self.assertGreater(final["training_seconds"], first["training_seconds"])
            self.assertTrue(final["training_timing"]["complete"])
            experiment.run(config, legacy, until=2, problem=NoReference())
            saved = load_torch(legacy/"checkpoint.pt")
            saved.pop("training_timing")
            atomic_torch(legacy/"checkpoint.pt", saved)
            partial = experiment.run(config, legacy, resume=True, problem=NoReference())
            self.assertEqual(partial["test"], final["test"])
            self.assertEqual(partial["training_timing"]["start_step"], 2)
            self.assertEqual(partial["training_timing"]["timed_updates"], 2)
            self.assertFalse(partial["training_timing"]["complete"])
            saved = load_torch(measured/"checkpoint.pt")
            self.assertEqual(saved["training_timing"]["training_seconds"], final["training_seconds"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaTiming(unittest.TestCase):
    def setUp(self):
        # Formal experiments each own a process; isolate test compilation caches too.
        torch.compiler.reset()

    def config(self, kind, *, graph=False, endpoint="NEXT", family="G2"):
        config = small_config(kind, endpoint=endpoint, family=family)
        return replace(config,
            problem=Problem("DSRM.problems.manufactured:sine_problem", dict(d=2, modes=3)),
            model=replace(config.model, width=8),
            runtime=replace(config.runtime, device="cuda", dtype="float32", compile=True,
                            cuda_graph=graph, checkpoint_every=2))

    def test_compiled_global_timing_preserves_every_update(self):
        for kind, endpoint in (("pinn", "NEXT"), ("global", "NEXT"), ("global", "CURRENT")):
            with self.subTest(kind=kind, endpoint=endpoint):
                torch.compiler.reset()
                config = self.config(kind, endpoint=endpoint)
                # All 138 paper configurations have equal-sized device batches.
                config = replace(config, runtime=replace(config.runtime, device_batch=config.algorithm.batch))
                untimed, timed = experiment.prepare(config), experiment.prepare(config)
                timer = TrainingTimer("cuda")
                for _ in range(config.total_steps):
                    expected = untimed.update()
                    actual = timed.update(timing=timer)
                    self.assertAlmostEqual(actual, expected, delta=1e-7)
                    assert_state(self, timed.state_dict(), untimed.state_dict())
                report = timer.report(config.total_steps)
                self.assertTrue(report["complete"])
                self.assertGreater(report["excluded_seconds"]["warmup"], 0)

    def test_ragged_batches_match_warmed_untimed_control(self):
        config = self.config("global")
        cold, untimed, timed = (experiment.prepare(config) for _ in range(3))
        # The final short chunk can generalize a static graph. Compare both
        # solvers after this compiler transition, without changing their states.
        cold.update()
        timer = TrainingTimer("cuda")
        for _ in range(config.total_steps):
            self.assertEqual(untimed.update(), timed.update(timing=timer))
            assert_state(self, timed.state_dict(), untimed.state_dict())
        self.assertTrue(timer.report(config.total_steps)["complete"])

    def test_graph_timing_preserves_states_across_layers(self):
        for endpoint, family in (("CURRENT", "G2"), ("NEXT", "G2"),
                                 ("CURRENT", "F1"), ("NEXT", "F1")):
            with self.subTest(endpoint=endpoint, family=family):
                torch.compiler.reset()
                config = self.config("backward", graph=True, endpoint=endpoint, family=family)
                untimed, timed = experiment.prepare(config), experiment.prepare(config)
                timer = TrainingTimer("cuda")
                for _ in range(config.total_steps):
                    expected = untimed.update()
                    actual = timed.update(timing=timer)
                    self.assertAlmostEqual(actual, expected, delta=1e-7)
                    assert_state(self, timed.state_dict(), untimed.state_dict())
                report = timer.report(config.total_steps)
                self.assertTrue(report["complete"])
                for category in ("warmup", "validation", "setup"):
                    self.assertGreater(report["excluded_seconds"][category], 0)

    def test_compiled_backward_without_graph_preserves_layer_transitions(self):
        config = self.config("backward")
        untimed, timed = experiment.prepare(config), experiment.prepare(config)
        timer = TrainingTimer("cuda")
        for _ in range(config.total_steps):
            self.assertAlmostEqual(untimed.update(), timed.update(timing=timer), delta=1e-7)
            assert_state(self, timed.state_dict(), untimed.state_dict())
        self.assertTrue(timer.report(config.total_steps)["complete"])

    def test_timed_region_rejects_unexpected_recompilation(self):
        function = torch.compile(lambda x: x.sin(), backend="eager")
        function(torch.zeros(2, device="cuda"))
        with compiled_update(SimpleNamespace(jets_compiled=True), TrainingTimer("cuda")):
            with self.assertRaisesRegex(RuntimeError, "[Rr]ecompil"):
                function(torch.zeros(3, device="cuda"))

    def test_cuda_resume_keeps_full_coverage_and_selected_solution(self):
        for kind, graph in (("pinn", False), ("backward", True)):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
                torch.compiler.reset()
                config = self.config(kind, graph=graph)
                full, resumed = Path(temp)/"full", Path(temp)/"resumed"
                expected = experiment.run(config, full)
                partial = experiment.run(config, resumed, until=3)
                actual = experiment.run(config, resumed, resume=True)
                self.assertEqual(actual["test"], expected["test"])
                self.assertGreater(actual["training_seconds"], partial["training_seconds"])
                self.assertTrue(actual["training_timing"]["complete"])
                self.assertEqual(actual["training_timing"]["timed_updates"], config.total_steps)
                assert_state(self, load_torch(resumed/"checkpoint.pt")["solver"],
                             load_torch(full/"checkpoint.pt")["solver"])


if __name__ == "__main__":
    unittest.main()
