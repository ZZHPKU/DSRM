"""Method selection, shared evaluation, and interrupted comparison workflows."""
from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from DSRM.cli import main
from DSRM.parameters import parse_benchmark
from DSRM.runtime import run, run_benchmark


ROOT = Path(__file__).resolve().parents[1]


def small_benchmark(*, reference=False):
    name = "gaussian_comparison.json" if reference else "custom_comparison.json"
    value = json.loads((ROOT/"configs/examples"/name).read_text())
    value["model"].update(width=4, hidden_layers=1)
    for algorithm in value["algorithms"].values():
        algorithm.update(batch=3, schedule=[dict(steps=2, lr=.001)])
        if algorithm["kind"] != "pinn":
            algorithm["N"] = 2
        if algorithm["kind"] == "global":
            algorithm["master_N"] = 2
    value["evaluation"].update(validation_points=5, test_points=6, block=3, validate_every=1,
                               layer_validation_points=4, layer_validation_block=2, layer_validate_every=1)
    return value


class BenchmarkWorkflow(unittest.TestCase):
    def test_shared_settings_and_algorithm_specific_budgets(self):
        benchmark = parse_benchmark(small_benchmark())
        pinn, global_, backward = (benchmark.experiments[k] for k in ("pinn", "g", "br"))
        for case in (global_, backward):
            for field in ("problem", "model", "sampling", "evaluation", "seeds", "runtime"):
                self.assertEqual(getattr(pinn, field), getattr(case, field))
        self.assertEqual([pinn.total_steps, global_.total_steps, backward.total_steps], [2, 2, 4])
        for field, value in (("problem", {}), ("endpoint", "NEXT")):
            bad = small_benchmark()
            bad["algorithms"]["pinn"][field] = value
            with self.assertRaises(ValueError):
                parse_benchmark(bad)
        for label in ("../outside", "", "/absolute"):
            bad = small_benchmark()
            bad["algorithms"] = {label: bad["algorithms"]["pinn"]}
            with self.assertRaises(ValueError):
                parse_benchmark(bad)

    def test_cli_runs_all_methods_and_reports_reference_errors(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()) as log:
            root = Path(temp)
            config = root/"config.json"
            config.write_text(json.dumps(small_benchmark(reference=True)))
            output = root/"results"
            self.assertEqual(main(["benchmark", str(config), "--output", str(output),
                                   "--device", "cpu", "--device-batch", "2"]), 0)
            report = json.loads((output/"comparison.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["methods"], ["pinn", "g", "br"])
            self.assertEqual(report["runtime"]["device_batch"], 2)
            for row in report["experiments"]:
                result = json.loads((output/row["label"]/"result.json").read_text())
                metrics = result["test"]["continuous"]
                self.assertEqual(metrics["points"], 6)
                self.assertEqual(row["residual_rms"], metrics["residual_rms"])
                self.assertEqual(row["training_seconds"], result["training_seconds"])
                self.assertEqual(row["timed_updates"], result["total_steps"])
                self.assertTrue(row["timing_complete"])
                for channel in ("value", "gradient", "hessian"):
                    self.assertEqual(row[channel+"_relative"], metrics["relative_errors"][channel])
                parameters = json.loads((output/row["label"]/"parameters.json").read_text())
                self.assertEqual(parameters["seeds"]["test_occupation"], report["seeds"]["test_occupation"])
                self.assertEqual(parameters["runtime"]["device_batch"], 2)
            with (output/"comparison.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertIn("Hessian rel.", log.getvalue())

    def test_select_and_resume_started_and_unstarted_methods_without_reference(self):
        benchmark = parse_benchmark(small_benchmark())
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            run(benchmark.experiments["pinn"], root/"pinn", until=1)
            report = run_benchmark(benchmark, root, methods=["pinn", "br"], resume=True)
            self.assertEqual(report["status"], "complete")
            self.assertEqual([row["label"] for row in report["experiments"]], ["pinn", "br"])
            self.assertFalse((root/"g").exists())
            for row in report["experiments"]:
                self.assertNotIn("value_relative", row)
                self.assertIn("residual_mse", row)
            with self.assertRaises(FileExistsError):
                run_benchmark(benchmark, root, methods=["pinn"])
            for methods in ([], ["unknown"], ["pinn", "pinn"]):
                with self.assertRaises(ValueError):
                    run_benchmark(benchmark, root/"invalid", methods=methods)
                self.assertFalse((root/"invalid").exists())

    def test_paused_method_stops_the_benchmark(self):
        benchmark = parse_benchmark(small_benchmark())
        def pause(config, output, **kwargs):
            return run(config, output, until=1, **kwargs)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch("DSRM.runtime.benchmark.run", side_effect=pause):
            report = run_benchmark(benchmark, temp)
            self.assertEqual(report["status"], "paused")
            self.assertEqual(len(report["experiments"]), 1)
            self.assertFalse((Path(temp)/"g").exists())
            self.assertNotIn("residual_rms", report["experiments"][0])

    def test_failure_does_not_leave_a_stale_complete_report(self):
        value = copy.deepcopy(small_benchmark())
        value["problem"]["factory"] = "DSRM.problems:missing_factory"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"comparison.csv").write_text("stale-completed-results\n")
            with self.assertRaises(AttributeError):
                run_benchmark(parse_benchmark(value), root)
            report = json.loads((root/"comparison.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["experiments"], [])
            self.assertNotIn("stale", (root/"comparison.csv").read_text())


if __name__ == "__main__":
    unittest.main()
