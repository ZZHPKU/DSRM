"""CUDA graph numerical contracts; skipped when CUDA is unavailable."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from DSRM.parameters import parse_config
from DSRM.runtime.experiment import prepare
from DSRM.runtime.storage import atomic_torch, load_torch
from DSRM.sampling import cells, generator


def configuration(endpoint="NEXT", family="G2", chunk=7, graph=True):
    return parse_config(dict(name="cuda-test",
        problem=dict(factory="DSRM.problems.manufactured:sine_problem", kwargs=dict(d=2, modes=3)),
        model=dict(width=8, hidden_layers=2),
        algorithm=dict(kind="backward", batch=7, N=3, endpoint=endpoint, family=family,
                       shared_prefix_steps=3,
                       schedule=[dict(steps=2, lr=.001), dict(steps=2, lr=.0003)]),
        runtime=dict(device="cuda", device_batch=chunk, threads=2, cuda_graph=graph),
        evaluation=dict(layer_validation_points=9, layer_validate_every=2, layer_validation_block=9)))


def assert_states(test, actual, expected, *, atol):
    test.assertEqual(actual.step, expected.step)
    test.assertEqual(actual.layer, expected.layer)
    test.assertTrue(torch.equal(actual.rng.get_state(), expected.rng.get_state()))
    test.assertTrue(torch.equal(actual.extra_rng.get_state(), expected.extra_rng.get_state()))
    for a, b in zip(actual.model.parameters(), expected.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=atol)
    for a, b in zip(actual.optimizer.state.values(), expected.optimizer.state.values()):
        for key in a:
            torch.testing.assert_close(a[key].cpu(), b[key].cpu(), rtol=0, atol=atol)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaBackward(unittest.TestCase):
    def test_graph_matches_eager_across_layers_and_learning_rates(self):
        for endpoint, family in (("CURRENT", "G2"), ("NEXT", "G2"),
                                 ("CURRENT", "F1"), ("NEXT", "F1")):
            with self.subTest(endpoint=endpoint, family=family):
                config = configuration(endpoint, family)
                graph = prepare(config)
                eager = prepare(replace(config, runtime=replace(config.runtime, cuda_graph=False)))
                for _ in range(config.total_steps):
                    actual, expected = graph.update(), eager.update()
                    self.assertAlmostEqual(actual, expected, delta=2e-6)
                    assert_states(self, graph, eager, atol=3e-6)
                self.assertEqual([r["selected_step"] for r in graph.records],
                                 [r["selected_step"] for r in eager.records])

    def test_graph_resume_restores_optimizer_and_sampling(self):
        config = configuration(chunk=3)
        full = prepare(config)
        for _ in range(3):
            full.update()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"checkpoint.pt"
            atomic_torch(path, full.state_dict())
            saved = load_torch(path)
        resumed = prepare(config)
        resumed.load_state_dict(saved)
        for _ in range(3, config.total_steps):
            self.assertAlmostEqual(full.update(), resumed.update(), delta=1e-7)
            assert_states(self, resumed, full, atol=1e-7)

    def test_graph_resume_at_layer_boundary(self):
        config = configuration()
        full = prepare(config)
        for _ in range(4):
            full.update()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"checkpoint.pt"
            atomic_torch(path, full.state_dict())
            resumed = prepare(config)
            resumed.load_state_dict(load_torch(path))
        for _ in range(4, config.total_steps):
            self.assertAlmostEqual(full.update(), resumed.update(), delta=1e-7)
            assert_states(self, resumed, full, atol=1e-7)

    def test_graph_chunks_keep_one_adam_update(self):
        full, chunked = prepare(configuration()), prepare(configuration(chunk=3))
        for _ in range(3):
            self.assertAlmostEqual(full.update(), chunked.update(), delta=2e-6)
            assert_states(self, chunked, full, atol=3e-6)

    def test_graph_reused_sampling_matches_allocated_updates_exactly(self):
        config = configuration()
        reused, allocated = prepare(config), prepare(config)
        self.assertEqual(reused.update(), allocated.update())
        def draw(layer, rng):
            return cells(config.algorithm.batch, allocated.problem.d, layer*allocated.h,
                         allocated.h, rng, dtype=allocated.dtype)
        with patch.object(allocated.accelerator, "sample", side_effect=draw):
            for _ in range(1, config.total_steps):
                self.assertEqual(reused.update(), allocated.update())
                for field in ("t", "x", "dx"):
                    torch.testing.assert_close(getattr(reused.accelerator.bank, field),
                        getattr(allocated.accelerator.bank, field), rtol=0, atol=0)
                assert_states(self, reused, allocated, atol=0)
        self.assertEqual(reused.records, allocated.records)

    def test_graph_scalar_changes_match_unconditional_writes_exactly(self):
        for endpoint, family in (("CURRENT", "G2"), ("NEXT", "G2"),
                                 ("CURRENT", "F1"), ("NEXT", "F1")):
            with self.subTest(endpoint=endpoint, family=family):
                config = configuration(endpoint, family)
                cached, unconditional = prepare(config), prepare(config)
                self.assertEqual(cached.update(), unconditional.update())
                accelerator = unconditional.accelerator
                original = accelerator.update
                def refresh(bank, layer, rate):
                    accelerator.time.fill_(layer*unconditional.h)
                    accelerator.next_time.fill_((layer+1)*unconditional.h)
                    unconditional.optimizer.param_groups[0]["lr"].fill_(rate)
                    return original(bank, layer, rate)
                with patch.object(accelerator, "update", side_effect=refresh):
                    for _ in range(1, config.total_steps):
                        self.assertEqual(cached.update(), unconditional.update())
                        assert_states(self, cached, unconditional, atol=0)
                self.assertEqual(cached.records, unconditional.records)

    def test_graph_sampler_matches_cells_across_shapes_and_precisions(self):
        solver = prepare(configuration())
        solver.update()
        for dtype in (torch.float32, torch.float64):
            for count, d in ((7, 2), (31, 3), (1024, 10)):
                with self.subTest(dtype=dtype, count=count, d=d):
                    reused_rng, allocated_rng = generator(421, "cuda"), generator(421, "cuda")
                    buffer = cells(count, d, 0., solver.h, generator(0, "cuda"), dtype=dtype)
                    # Only the sampler uses these temporary buffers; replay retains its original addresses.
                    with patch.object(solver.accelerator, "bank", buffer):
                        for layer in (2, 1, 0):
                            actual = solver.accelerator.sample(layer, reused_rng)
                            expected = cells(count, d, layer*solver.h, solver.h, allocated_rng, dtype=dtype)
                            for field in ("t", "x", "dx"):
                                torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
                            self.assertTrue(torch.equal(reused_rng.get_state(), allocated_rng.get_state()))

    def test_graph_rejects_nonfinite_updates(self):
        solver = prepare(configuration())
        solver.update()
        with torch.no_grad():
            next(solver.model.parameters()).fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            solver.update()
        self.assertEqual(solver.step, 1)

    def test_graph_rejects_nonfinite_updates_even_when_loss_is_finite(self):
        for channel in ("gradient", "parameter"):
            for bad in (float("nan"), float("inf")):
                with self.subTest(channel=channel, bad=bad):
                    solver = prepare(configuration())
                    parameter = list(solver.model.parameters())[-1]
                    def inject(optimizer, args, kwargs):
                        with torch.no_grad():
                            value = parameter.grad if channel == "gradient" else parameter
                            value.reshape(-1)[-1].fill_(bad)
                    with solver.optimizer.register_step_post_hook(inject):
                        with self.assertRaises(FloatingPointError):
                            solver.update()
                    self.assertTrue(bool(solver.accelerator.result[0].isfinite()))
                    self.assertEqual(solver.step, 0)
                    if channel == "gradient":
                        self.assertTrue(all(bool(p.isfinite().all()) for p in solver.model.parameters()))


if __name__ == "__main__":
    unittest.main()
