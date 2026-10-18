"""Run a small selection at full budgets and compare with the published final results."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from reproduce_paper import ROOT, completed_run, paper_runs
from DSRM.evaluation import compare_metrics
from DSRM.parameters import load_config
from DSRM.runtime import run
from DSRM.runtime.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT/"configs/paper/verification.json")
    parser.add_argument("--reference", type=Path, default=ROOT/"results/final",
                        help="Published results directory containing runs/<name>/result.json")
    parser.add_argument("--output", type=Path, default=Path("results/verification"))
    parser.add_argument("--device", help="Override execution device; retain the complete training budget")
    parser.add_argument("--device-batch", type=int, help="Change gradient accumulation chunk, not logical batch")
    parser.add_argument("--rtol", type=float, default=.05)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--plan", action="store_true", help="Check budgets and print selected tasks without training")
    args = parser.parse_args()
    paths = json.loads(args.selection.read_text())["runs"]
    configs = [load_config(args.selection.parent/path) for path in paths]
    catalog = {case.name: case for case in paper_runs()}
    if any(config != catalog[config.name].config for config in configs):
        raise ValueError("Selection must use the supplied final paper configurations")
    reference = {config.name: completed_run(catalog[config.name], args.reference/"runs"/config.name).result
                 for config in configs}
    plan = []
    for config in configs:
        old = reference[config.name]
        p = config.algorithm
        samples = config.total_steps*p.batch*(p.N if p.kind == "global" else 1)
        if (config.total_steps != old["total_steps"]
                or config.evaluation.test_points != old["test"]["continuous"]["points"]):
            raise ValueError(f"Final training or evaluation budget changed: {config.name}")
        plan.append(dict(name=config.name, method=catalog[config.name].method, updates=config.total_steps,
                         samples=samples, test_points=config.evaluation.test_points))
    print(json.dumps(plan, indent=2))
    if args.plan:
        return 0
    rows = []
    for config in configs:
        overrides = {k: v for k, v in dict(device=args.device, device_batch=args.device_batch).items() if v is not None}
        config = replace(config, runtime=replace(config.runtime, **overrides))
        output = args.output/config.name
        try:
            result = run(config, output, resume=(output/"checkpoint.pt").exists())
        except (RuntimeError, ValueError) as exc:
            atomic_json(args.output/"comparison.json", dict(status="incomplete", plan=plan, comparisons=rows,
                        failed_run=config.name, error=str(exc)))
            raise
        if result["status"] != "complete":
            atomic_json(args.output/"comparison.json", dict(status="incomplete", plan=plan, comparisons=rows))
            return 2
        comparison = compare_metrics(reference[config.name]["test"]["continuous"], result["test"]["continuous"],
                                      rtol=args.rtol, atol=args.atol)
        rows.append(dict(name=config.name, **comparison))
        atomic_json(args.output/"comparison.json", dict(status="running", plan=plan, comparisons=rows))
    passed = all(row["passed"] for row in rows)
    atomic_json(args.output/"comparison.json", dict(status="pass" if passed else "failed", plan=plan, comparisons=rows))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
