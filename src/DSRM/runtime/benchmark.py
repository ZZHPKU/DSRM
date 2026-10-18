"""Run named methods sequentially and preserve a comparison after every run."""
from dataclasses import asdict
import csv
from pathlib import Path

from ..evaluation.benchmark import experiment_summary
from .experiment import run
from .storage import atomic_json, lock


def write_comparison(output, report):
    atomic_json(output/"comparison.json", report)
    rows = report["experiments"]
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["label", "method", "status"]
    with (output/"comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(benchmark, output, *, methods=None, resume=False):
    labels = list(benchmark.experiments) if methods is None else list(methods)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Select at least one method, without duplicate labels")
    unknown = set(labels)-set(benchmark.experiments)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}; configured labels: {list(benchmark.experiments)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with lock(output/"benchmark.lock", blocking=False):
        if not resume:
            for label in labels:
                if (output/label/"checkpoint.pt").exists():
                    raise FileExistsError(f"{output/label} already contains a checkpoint; use --resume or a new output directory")
        shared = benchmark.experiments[labels[0]]
        report = dict(name=benchmark.name, status="running", methods=labels,
                      problem=asdict(shared.problem), model=asdict(shared.model), sampling=asdict(shared.sampling),
                      evaluation=asdict(shared.evaluation), seeds=asdict(shared.seeds), runtime=asdict(shared.runtime),
                      experiments=[])
        write_comparison(output, report)
        try:
            for label in labels:
                config, target = benchmark.experiments[label], output/label
                result = run(config, target, resume=resume and (target/"checkpoint.pt").exists())
                report["experiments"].append(experiment_summary(label, config, result))
                if result["status"] != "complete":
                    report["status"] = "paused"
                    write_comparison(output, report)
                    return report
                write_comparison(output, report)
        except Exception:
            report["status"] = "failed"
            write_comparison(output, report)
            raise
        report["status"] = "complete"
        write_comparison(output, report)
        return report
