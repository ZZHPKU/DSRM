"""Published measurements, selection evidence, and the current reproduction entry point."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reproduce_paper as paper
from DSRM.parameters import load_config
from DSRM.parameters.config import BackwardRegression


FINAL = ROOT / "results/final"


class PublishedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = paper.paper_runs()
        cls.manifest = json.loads((FINAL / "manifest.json").read_text())

    def test_complete_inputs_and_results(self):
        self.assertEqual(len(self.cases), 138)
        self.assertEqual({case.name for case in self.cases}, {r["name"] for r in self.manifest["runs"]})
        actual = {str(p.relative_to(FINAL)) for p in FINAL.rglob("*")
                  if p.is_file() and p.name != "manifest.json"}
        self.assertEqual(actual, set(self.manifest["files"]))
        for path, record in self.manifest["files"].items():
            content = (FINAL / path).read_bytes()
            self.assertEqual(len(content), record["bytes"], path)
            self.assertEqual(hashlib.sha256(content).hexdigest(), record["sha256"], path)
        for record in self.manifest["runs"]:
            self.assertEqual(hashlib.sha256((ROOT / record["config"]).read_bytes()).hexdigest(),
                             record["config_sha256"], record["name"])
        for case in self.cases:
            paper.completed_run(case, FINAL / "runs" / case.name)
        counts = {group: len(paper.paper_runs(group)) for group in paper.GROUPS}
        self.assertEqual(counts, {"5.1": 12, "5.2": 60, "5.3": 48, "5.4": 18, "appendix": 12, "all": 138})

    def test_histories_support_every_selected_solution(self):
        hashes = {row["name"]: row["history_uncompressed_sha256"] for row in self.manifest["runs"]}
        for case in self.cases:
            with self.subTest(run=case.name):
                directory = FINAL / "runs" / case.name
                content = gzip.decompress((directory / "history.jsonl.gz").read_bytes())
                self.assertEqual(hashlib.sha256(content).hexdigest(), hashes[case.name])
                events = [json.loads(line) for line in content.splitlines()]
                result = paper.completed_run(case, directory).result
                config = case.config
                total_steps = config.total_steps
                self.assertTrue(all(0 <= event["step"] <= total_steps for event in events))
                training = [event for event in events if event["event"] == "training"]
                self.assertEqual(training[-1]["step"], total_steps)
                self.assertEqual(training[-1]["timed_updates"], total_steps)
                if isinstance(config.algorithm, BackwardRegression):
                    layers = {record["layer"]: record for record in result["layers"]}
                    self.assertEqual(set(layers), set(range(config.algorithm.N)))
                    validations = {layer: [] for layer in layers}
                    for event in events:
                        if event["event"] == "layer_validation":
                            validations[event["layer"]].append(event)
                    offset = 0
                    for layer in reversed(range(config.algorithm.N)):
                        values = validations[layer]
                        budget = config.algorithm.schedules[layer].steps
                        expected = set(range(0, budget + 1, config.evaluation.layer_validate_every)) | {budget}
                        self.assertEqual({value["layer_step"] for value in values}, expected)
                        self.assertEqual(len(values), len(expected))
                        self.assertTrue(all(value["step"] == offset + value["layer_step"] for value in values))
                        selected = min(values, key=lambda value: (value["residual_mse"], value["layer_step"]))
                        self.assertEqual(layers[layer], dict(layer=layer, steps=budget,
                                         selected_step=selected["layer_step"], residual_mse=selected["residual_mse"]))
                        offset += budget
                    completed = [dict(layer=e["layer"], steps=e["steps"], selected_step=e["selected_step"],
                                      residual_mse=e["residual_mse"])
                                 for e in events if e["event"] == "layer_complete"]
                    self.assertEqual(completed, result["layers"])
                else:
                    values = [event for event in events if event["event"] == "validation"]
                    expected = set(range(0, total_steps + 1, config.evaluation.validate_every)) | {total_steps}
                    self.assertEqual({value["step"] for value in values}, expected)
                    self.assertEqual(len(values), len(expected))
                    selected = min(values, key=lambda value: (value["residual_mse"], value["step"]))
                    self.assertEqual(selected["step"], result["selected_step"])

    def test_pilot_matches_retained_schedules(self):
        pilot = json.loads((FINAL / "learning_rate_search.json").read_text())
        self.assertEqual(len(pilot["candidates"]), 72)
        self.assertFalse(pilot["selection"]["test_used"])
        self.assertFalse(pilot["selection"]["holdout_used_for_selection"])
        for case in self.cases:
            if case.family != "gaussian" or "_tuned_" not in case.name:
                continue
            for selection in pilot["selection"]["selected"].values():
                phases = [phase.split(":") for phase in selection["schedule"]]
                expected = [(int(steps), float(rate)) for steps, rate in phases]
                for layer in selection["layers"]:
                    actual = [(phase.steps, phase.lr) for phase in case.config.algorithm.schedules[layer].phases]
                    self.assertEqual(actual, expected, case.name)

    def test_summaries_regenerate_and_use_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "runs").symlink_to(FINAL / "runs", target_is_directory=True)
            report = paper.summarize(self.cases, output)
            self.assertEqual((report["runs"], report["groups"]), (138, 46))
            for name in ("summary.csv", "groups.csv"):
                self.assertEqual((output / name).read_bytes(), (FINAL / name).read_bytes())
        with (FINAL / "groups.csv").open() as handle:
            group = next(row for row in csv.DictReader(handle) if row["family"] == "translation"
                         and float(row["R"]) == 0 and float(row["alpha"]) == 0.5)
        values = [json.loads((FINAL / "runs" / f"03-r{repeat}-R0-alpha0.5" / "result.json").read_text())
                  ["test"]["continuous"]["residual_rms"] for repeat in range(3)]
        average = sum(values) / 3
        sample_sd = (sum((value - average)**2 for value in values) / 2)**0.5
        self.assertAlmostEqual(float(group["residual_rms_mean"]), average)
        self.assertAlmostEqual(float(group["residual_rms_sample_sd"]), sample_sd)


class ReproductionWorkflow(unittest.TestCase):
    def test_current_solver_runs_and_resumes_without_archives(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            cases = []
            for repeat in range(3):
                value = json.loads((ROOT / "configs/paper/gaussian/A_pinn_c0.5_rho0.15.json").read_text())
                value["name"] = f"tiny-{repeat}"
                value["model"].update(width=4, hidden_layers=1)
                value["algorithm"].update(batch=4, schedule=[dict(steps=2, lr=.001)])
                value["runtime"].update(device="cpu", device_batch=4, compile=False, threads=1)
                value["evaluation"].update(validation_points=4, test_points=6, block=3, validate_every=1)
                value["seeds"]["initialization"] = repeat
                path = root / f"{repeat}.json"
                path.write_text(json.dumps(value))
                cases.append(paper.PaperRun(path, load_config(path), "gaussian", "PINN"))
            output = root / "output"
            with patch.object(paper, "paper_runs", return_value=cases):
                self.assertEqual(paper.main(["--output", str(output)]), 0)
                with patch.object(paper.subprocess, "run", side_effect=AssertionError("Completed run restarted")):
                    self.assertEqual(paper.main(["--output", str(output), "--resume"]), 0)
                with self.assertRaises(FileExistsError):
                    paper.main(["--output", str(output)])
            directory = output / "runs" / cases[0].name
            parameters = json.loads((directory / "parameters.json").read_text())
            parameters["seeds"]["initialization"] += 1
            (directory / "parameters.json").write_text(json.dumps(parameters))
            with self.assertRaisesRegex(ValueError, "Recorded parameters differ"):
                paper.completed_run(cases[0], directory)

    def test_published_runs_cannot_be_overwritten(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                paper.main(["--output", str(FINAL)])
        self.assertEqual(raised.exception.code, 2)
