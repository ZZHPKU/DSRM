"""Run, resume, evaluate, or compare explicitly configured experiments."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from .evaluation import compare_metrics
from .evaluation.benchmark import comparison_table
from .parameters import load_config
from .parameters.benchmark import load_benchmark
from .runtime import evaluate_checkpoint, run
from .runtime.benchmark import run_benchmark
from .runtime.storage import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="DSRM")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("run", help="Train an experiment and evaluate the selected solution")
    train.add_argument("config", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--until", type=int, help="Pause after this absolute update; retain the configured budget")
    train.add_argument("--device")
    train.add_argument("--device-batch", type=int)
    train.add_argument("--threads", type=int)
    benchmark = commands.add_parser("benchmark", help="Run selected methods with shared problem and evaluation settings")
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--methods", nargs="+", help="Labels from algorithms in the configuration; default: all")
    benchmark.add_argument("--resume", action="store_true")
    benchmark.add_argument("--device")
    benchmark.add_argument("--device-batch", type=int)
    benchmark.add_argument("--threads", type=int)
    evaluation = commands.add_parser("evaluate", help="Evaluate a selected saved solution on CPU in FP64")
    evaluation.add_argument("config", type=Path)
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Compare one completed experiment with a local reference")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--run-id", help="Select a record from an archived runs.json")
    compare.add_argument("--rtol", type=float, default=.05)
    compare.add_argument("--atol", type=float, default=1e-8)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command in ("run", "benchmark"):
        overrides = {k: getattr(args, k) for k in ("device", "device_batch", "threads") if getattr(args, k) is not None}
    if args.command == "run":
        config = load_config(args.config)
        config = replace(config, runtime=replace(config.runtime, **overrides))
        result = run(config, args.output, resume=args.resume, until=args.until)
        print(json.dumps(dict(status=result["status"], step=result["step"],
                              training_seconds=result["training_seconds"],
                              timing_complete=result["training_timing"]["complete"],
                              output=str(args.output)), indent=2))
    elif args.command == "benchmark":
        config = load_benchmark(args.config)
        config = replace(config, experiments={label: replace(case, runtime=replace(case.runtime, **overrides))
                                             for label, case in config.experiments.items()})
        result = run_benchmark(config, args.output, methods=args.methods, resume=args.resume)
        print(comparison_table(result["experiments"]))
        print(f"{result['status']}: {args.output/'comparison.json'}; {args.output/'comparison.csv'}")
    elif args.command == "evaluate":
        result = evaluate_checkpoint(load_config(args.config), args.checkpoint)
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        reference = json.loads(args.reference.read_text())
        if args.run_id:
            matches = [r for r in reference if r["run_id"] == args.run_id]
            if len(matches) != 1:
                parser.error("run-id must identify exactly one reference record")
            reference = matches[0]
        candidate = json.loads(args.candidate.read_text())
        if candidate.get("status", "complete") != "complete":
            parser.error("Candidate training is incomplete")
        def metrics(value):
            value = value.get("test", value)
            return value.get("continuous", value)
        result = compare_metrics(metrics(reference), metrics(candidate), rtol=args.rtol, atol=args.atol)
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    return 0
