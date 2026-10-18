"""Train the current paper configurations or summarize their completed results."""
import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from DSRM.parameters import Experiment, load_config


GROUPS = {
    "5.1": ("gaussian/?_g2_*_c0.5_rho0.15_*.json",),
    "5.2": ("gaussian/?_pinn_*.json", "gaussian/?_g-*_*.json",
            "gaussian/?_g2_tuned_*.json"),
    "5.3": ("sine/*.json",),
    "5.4": ("translation/*.json",),
    "appendix": ("gaussian/?_f1_tuned_*.json", "gaussian/?_g2_tuned_c0.5_rho0.15_*.json"),
    "all": ("*/*.json",),
}
COMPONENTS = ("value", "gradient", "hessian")
GROUP_FIELDS = ("family", "method", "N", "c", "rho", "R", "alpha", "schedule")
METRICS = (*[f"{c}_relative" for c in COMPONENTS], "residual_mse", "residual_rms",
           "overall_relative", "K_H", "training_seconds")


@dataclass(frozen=True)
class PaperRun:
    path: Path
    config: Experiment
    family: str
    method: str

    @property
    def name(self):
        return self.config.name


@dataclass(frozen=True)
class CompletedRun:
    case: PaperRun
    result: dict


def paper_runs(group="all"):
    catalog = json.loads((ROOT / "configs/paper/index.json").read_text())
    cases = []
    for entry in catalog:
        path = Path(entry["path"])
        if any(path.match(pattern) for pattern in GROUPS[group]):
            full_path = ROOT / "configs/paper" / path
            config = load_config(full_path)
            if config.name != entry["name"]:
                raise ValueError(f"Catalog/config name mismatch: {path}")
            cases.append(PaperRun(full_path, config, entry["family"], entry["method"]))
    if not cases or len({case.name for case in cases}) != len(cases):
        raise ValueError("The paper catalog must contain distinct runs")
    return sorted(cases, key=lambda case: case.name)


def completed_run(case, directory):
    result = json.loads((directory / "result.json").read_text())
    if load_config(directory / "parameters.json") != case.config:
        raise ValueError(f"Recorded parameters differ from the paper configuration: {case.name}")
    if (result["name"] != case.name or result["status"] != "complete"
            or result["step"] != case.config.total_steps
            or result["total_steps"] != case.config.total_steps):
        raise ValueError(f"Incomplete or mismatched result: {case.name}")
    timing = result["training_timing"]
    if (not timing["complete"] or timing["start_step"] != 0
            or timing["end_step"] != case.config.total_steps
            or timing["timed_updates"] != case.config.total_steps
            or timing["budget_updates"] != case.config.total_steps
            or timing["training_seconds"] != result["training_seconds"]):
        raise ValueError(f"Incomplete training timing: {case.name}")
    if result["test"]["continuous"]["points"] != case.config.evaluation.test_points:
        raise ValueError(f"Test budget differs: {case.name}")
    return CompletedRun(case, result)


def summary_row(completed):
    case, result = completed.case, completed.result
    config, metrics = case.config, result["test"]["continuous"]
    problem = config.problem.kwargs
    schedule = "fixed"
    if case.family == "gaussian" and config.algorithm.kind == "backward":
        schedule = "uniform" if "_uniform_" in case.name else "tuned"
    row = dict(name=case.name, family=case.family, method=case.method,
               N=getattr(config.algorithm, "N", None),
               **{key: problem.get(key) for key in ("c", "rho", "R", "alpha")},
               schedule=schedule, status="complete", total_steps=config.total_steps)
    row.update({f"{c}_relative": metrics["relative_errors"][c] for c in COMPONENTS})
    row.update({key: metrics[key] for key in ("residual_mse", "residual_rms", "K_H")})
    row["overall_relative"] = math.sqrt(sum(metrics["error_mse"][c] for c in COMPONENTS)
                                       / sum(metrics["truth_energy_exact"][c] for c in COMPONENTS))
    row.update(training_seconds=result["training_seconds"], elapsed_seconds=result["elapsed_seconds"],
               selected_step=result.get("selected_step"), completed_utc=result["completed_utc"])
    if any(not math.isfinite(row[key]) or row[key] < 0 for key in METRICS):
        raise ValueError(f"Invalid final metric: {case.name}")
    return row


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(cases, output):
    rows = [summary_row(completed_run(case, output / "runs" / case.name)) for case in cases]
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in GROUP_FIELDS)].append(row)
    groups = []
    for key, members in grouped.items():
        if len(members) != 3:
            raise ValueError(f"Expected three statistical units for {key}; found {len(members)}")
        group = dict(zip(GROUP_FIELDS, key))
        group["count"] = len(members)
        for metric in METRICS:
            values = [row[metric] for row in members]
            group[metric + "_mean"] = mean(values)
            group[metric + "_sample_sd"] = stdev(values)
        groups.append(group)
    write_csv(output / "summary.csv", rows)
    write_csv(output / "groups.csv", groups)
    return dict(runs=len(rows), groups=len(groups), output=str(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduction")
    parser.add_argument("--resume", action="store_true", help="Resume checkpoints and skip completed runs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="List configurations and budgets without training")
    mode.add_argument("--summarize-only", action="store_true", help="Read completed runs without training")
    args = parser.parse_args(argv)
    cases = paper_runs(args.group)
    output = args.output.resolve()
    if args.plan:
        print(json.dumps([dict(name=case.name, config=str(case.path.relative_to(ROOT)),
                               updates=case.config.total_steps) for case in cases], indent=2))
        return 0
    if args.summarize_only:
        if args.group != "all":
            parser.error("Summarize all runs together to preserve the complete published reports")
        print(json.dumps(summarize(cases, output), indent=2))
        return 0
    published = (ROOT / "results/final").resolve()
    if output == published or published in output.parents:
        parser.error("Use a new output directory; results/final contains the published measurements")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    for case in cases:
        directory = output / "runs" / case.name
        if args.resume and (directory / "result.json").exists():
            result = json.loads((directory / "result.json").read_text())
            if result.get("status") == "complete":
                completed_run(case, directory)
                continue
        checkpoint = directory / "checkpoint.pt"
        if directory.exists() and any(directory.iterdir()) and not (args.resume and checkpoint.is_file()):
            raise FileExistsError(f"Use --resume with a checkpoint or a fresh output directory: {directory}")
        command = [sys.executable, "-m", "DSRM", "run", str(case.path), "--output", str(directory)]
        if args.resume and checkpoint.is_file():
            command.append("--resume")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        completed_run(case, directory)
    print(json.dumps(summarize(cases, output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
