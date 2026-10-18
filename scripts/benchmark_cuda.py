"""Time disposable updates at the configured batch size; never write training weights."""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

import torch

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--source", type=Path, default=ROOT/"src")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path to preserve measurements")
    if min(args.warmup, args.steps, args.repeats) < 1:
        parser.error("warmup, steps and repeats must be positive")
    sys.path.insert(0, str(args.source.resolve()))
    from DSRM.algorithms.backward_solver import BackwardSolver
    from DSRM.algorithms.optimization import snapshot
    from DSRM.parameters import load_config
    from DSRM.runtime.experiment import environment, prepare, source_identity
    from DSRM.runtime.storage import atomic_json
    config = load_config(args.config)
    if args.optimized:
        if config.algorithm.kind == "backward":
            config = replace(config, runtime=replace(config.runtime, cuda_graph=True))
        else:
            config = replace(config, model=replace(config.model, backend="taylor"),
                             runtime=replace(config.runtime, compile=True))
    solver = prepare(config)
    report = dict(config=str(args.config), optimized=args.optimized, source=source_identity(),
                  environment=environment(config.runtime.device), warmup=args.warmup,
                  steps=args.steps, repeats=args.repeats, stages=[],
                  scope="Disposable full logical updates including periodic layer validation; "
                        "no checkpoints or final evaluation. Interior targets are synthetic initial weights.")
    stages = ("terminal", "interior") if isinstance(solver, BackwardSolver) else ("training",)
    for stage in stages:
        if stage == "interior":
            solver.layer = config.algorithm.N//2
            solver.states[solver.layer+1] = snapshot(solver.model)
            solver.layer_step = 0
            solver.best_state = None
            solver.best_metric, solver.selected_step = math.inf, 0
            solver._prepare_layer()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.warmup):
            solver.update()
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter()-start
        durations, update_durations, validation_durations = [], [], []
        for repeat in range(args.repeats):
            samples = []
            for _ in range(args.steps):
                validation = isinstance(solver, BackwardSolver) and (
                    (solver.layer_step+1) % config.evaluation.layer_validate_every == 0)
                torch.cuda.synchronize()
                start = time.perf_counter()
                loss = solver.update()
                torch.cuda.synchronize()
                duration = time.perf_counter()-start
                samples.append(duration)
                (validation_durations if validation else update_durations).append(duration)
            durations.append(statistics.mean(samples))
            print(stage, repeat+1, "seconds/update", durations[-1], flush=True)
        row = dict(stage=stage, warmup_seconds=warmup_seconds, repeat_seconds_per_update=durations,
                   mean_seconds_per_update=statistics.mean(durations),
                   update_only_seconds=statistics.mean(update_durations),
                   validation_update_seconds=validation_durations,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3, last_loss=loss)
        report["stages"].append(row)
        atomic_json(args.output, report)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
