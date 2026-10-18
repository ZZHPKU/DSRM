"""Experiment orchestration, resumable checkpoints, and held-out evaluation."""
from dataclasses import asdict
from datetime import datetime, timezone
import copy
import hashlib
import inspect
import json
from pathlib import Path
import platform
import signal
import time

import torch

from ..algorithms.backward import spatial_initial
from ..algorithms.backward_solver import BackwardSolver
from ..algorithms.global_solver import GlobalSolver
from ..evaluation.metrics import evaluate
from ..models import PiecewiseSolution, SpatialNode, build_model
from ..parameters.config import BackwardRegression, PINN, ImportanceSampling, config_dict
from ..problems import load_problem
from ..sampling import generator, occupation, paths
from ..sampling.importance import GaussianMixtureSampler
from ..timing import TrainingTimer
from .storage import atomic_json, atomic_torch, load_torch, lock


def environment(device):
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                device=str(device), cuda=torch.version.cuda, threads=torch.get_num_threads(),
                hardware=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else platform.processor(),
                tf32=False, amp=False)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_identity(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def problem_identity(problem):
    """Include custom class source, fixed tensors, and public scalar coefficients."""
    digest = hashlib.sha256()
    for name, module in problem.named_modules():
        digest.update(f"{name}:{type(module).__module__}.{type(module).__qualname__}".encode())
        source = inspect.getsourcefile(type(module))
        if source and Path(source).is_file():
            digest.update(Path(source).read_bytes())
        public = {k: v for k, v in vars(module).items() if not k.startswith("_")
                  and isinstance(v, (str, int, float, bool, tuple, list, dict, type(None)))}
        digest.update(json.dumps(public, sort_keys=True).encode())
    # Include nonpersistent fixed buffers, such as analytic terminal coefficients.
    for name, tensor in list(problem.named_parameters())+list(problem.named_buffers()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prepare(config, problem=None):
    torch.set_num_threads(config.runtime.threads)
    device = torch.device(config.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is accessible in this process")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    problem = copy.deepcopy(problem) if problem is not None else load_problem(config.problem.factory, config.problem.kwargs)
    problem.requires_grad_(False)
    dtype = getattr(torch, config.runtime.dtype)
    p = config.algorithm
    if config.runtime.cuda_graph and (device.type != "cuda" or
            not isinstance(p, BackwardRegression) or config.model.backend != "taylor"):
        raise ValueError("runtime.cuda_graph requires CUDA backward regression with the taylor backend")
    model = build_model(problem, config.model, config.seeds.initialization, dtype=dtype)
    if isinstance(p, BackwardRegression):
        initial = {k: v.float() if v.is_floating_point() else v for k, v in model.state_dict().items()}
        model.load_state_dict(spatial_initial(initial, (p.N-1)*problem.T/p.N))
    model.to(device=device, dtype=dtype)
    problem.to(device=device, dtype=dtype)
    if config.runtime.compile:
        if config.model.backend != "taylor":
            raise ValueError("Compiled model jets require the taylor backend")
        if device.type == "cuda":
            model.enable_cuda_compile()
    if isinstance(p, BackwardRegression):
        return BackwardSolver(model, problem, p, config.seeds, config.evaluation,
                              config.runtime.device_batch, cuda_graph=config.runtime.cuda_graph)
    if isinstance(p, PINN):
        rng = generator(config.seeds.train_occupation, device)
        if isinstance(config.sampling, ImportanceSampling):
            if problem.T != 1 or len(config.sampling.center) != problem.d:
                raise ValueError("Gaussian mixture sampling requires T=1 and matching dimensions")
            proposal = GaussianMixtureSampler(config.sampling.center, config.sampling.templates, device=device, dtype=dtype)
            draw = lambda rng: proposal.draw(p.batch, rng)
        else:
            draw = lambda rng: occupation(p.batch, problem.d, rng, T=problem.T, dtype=dtype)
    else:
        rng = generator(config.seeds.train_path, device)
        draw = lambda rng: paths(p.batch, p.N, problem.d, rng, T=problem.T, dtype=dtype, master_N=p.master_N)
    return GlobalSolver(model, problem, p, rng, draw, config.runtime.device_batch)


def restored_solution(config, state, *, selected=True, problem=None):
    problem = copy.deepcopy(problem) if problem is not None else load_problem(config.problem.factory, config.problem.kwargs)
    problem = problem.cpu().double().requires_grad_(False)
    if isinstance(config.algorithm, BackwardRegression):
        if any(s is None for s in state["states"]):
            raise ValueError("A complete backward solution requires all spatial layers")
        nodes = []
        for i, node_state in enumerate(state["states"]):
            raw = build_model(problem, config.model, config.seeds.initialization, dtype=torch.float64)
            raw.load_state_dict(node_state)
            if hasattr(raw, "hard_mapping"):
                raw.hard_mapping = False
            nodes.append(SpatialNode(raw, i*problem.T/config.algorithm.N))
        model = PiecewiseSolution(nodes, copy.deepcopy(problem.terminal), problem.T)
    else:
        model = build_model(problem, config.model, config.seeds.initialization, dtype=torch.float64)
        weights = state["best_state"] if selected and state["best_state"] is not None else state["model"]
        model.load_state_dict(weights)
    return model.requires_grad_(False), problem


def append_event(path, event):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(event, allow_nan=False)+"\n")


def run(config, output, *, resume=False, until=None, problem=None):
    """Run a fixed budget; ``until`` pauses at an absolute update without changing it.

    Passing a PDEProblem directly supports interactive use. Reuse that same
    problem when resuming or evaluating its checkpoint.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with lock(output/"run.lock", blocking=False):
        return _run(config, output, resume=resume, until=until, problem=problem)


def _run(config, output, *, resume, until, problem):
    path = output/"checkpoint.pt"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use resume or a new output directory")
    if resume and not path.exists():
        raise FileNotFoundError(path)
    setup_started = time.perf_counter()
    solver = prepare(config, problem)
    identity = dict(config=config_identity(config), source=source_identity(),
                    problem=problem_identity(solver.problem),
                    torch=str(torch.__version__), device=config.runtime.device)
    elapsed = 0.
    if resume:
        saved = load_torch(path)
        if saved["identity"] != identity:
            raise ValueError("Resume requires the same config, problem, source, PyTorch version, and device")
        solver.load_state_dict(saved["solver"])
        elapsed = saved["elapsed_seconds"]
        torch.random.set_rng_state(saved["cpu_rng"])
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], config.runtime.device)
        events = output/"history.jsonl"
        if events.exists():
            rows = [json.loads(row) for row in events.read_text().splitlines() if row.strip()]
            events.write_text("".join(json.dumps(row)+"\n" for row in rows if row["step"] <= solver.step))
    else:
        atomic_json(output/"parameters.json", config_dict(config))
        atomic_json(output/"environment.json", environment(config.runtime.device))
    timing = TrainingTimer.restore(config.runtime.device,
        saved.get("training_timing") if resume else None, step=solver.step)
    timing.synchronize()
    timing.excluded_seconds["setup"] += time.perf_counter()-setup_started
    limit = config.total_steps if until is None else min(until, config.total_steps)
    if limit < solver.step or limit < 0:
        raise ValueError("until must not precede the saved step")
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    started = time.perf_counter()

    def save():
        with timing.exclude("checkpoint"):
            atomic_torch(path, dict(identity=identity, solver=solver.state_dict(),
                elapsed_seconds=elapsed+time.perf_counter()-started,
                training_timing=timing.state_dict(),
                cpu_rng=torch.random.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(config.runtime.device).cpu()
                    if torch.device(config.runtime.device).type == "cuda" else None))

    def validate():
        with timing.exclude("validation"):
            model, equation = restored_solution(config, solver.state_dict(), selected=False, problem=problem)
            metrics = evaluate(model, equation, config, split="validation")["continuous"]
            solver.observe(metrics["residual_mse"])
            append_event(output/"history.jsonl", dict(event="validation", step=solver.step, **metrics))

    try:
        if isinstance(solver, GlobalSolver) and solver.best_state is None:
            validate()
        save()
        while solver.step < limit and not stopped:
            loss = solver.update(timing=timing)
            if isinstance(solver, GlobalSolver) and (
                    solver.step % config.evaluation.validate_every == 0 or solver.step == config.total_steps):
                validate()
            if isinstance(solver, BackwardSolver) and solver.events:
                with timing.exclude("logging"):
                    for event in solver.events:
                        append_event(output/"history.jsonl", event)
                    solver.events.clear()
            if solver.step % config.runtime.log_every == 0 or solver.step == limit:
                with timing.exclude("logging"):
                    event = dict(event="training", step=solver.step, loss=loss,
                                 elapsed_seconds=elapsed+time.perf_counter()-started,
                                 training_seconds=timing.seconds,
                                 timed_updates=timing.end_step-timing.start_step)
                    append_event(output/"history.jsonl", event)
                    print(f"{config.name}: {solver.step}/{config.total_steps}, loss={loss:.6g}", flush=True)
            if solver.step % config.runtime.checkpoint_every == 0:
                save()
        save()
    finally:
        timing.pause()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    complete = solver.step == config.total_steps
    result = dict(name=config.name, status="complete" if complete else "paused", step=solver.step,
                  total_steps=config.total_steps, elapsed_seconds=elapsed+time.perf_counter()-started,
                  completed_utc=datetime.now(timezone.utc).isoformat() if complete else None,
                  selection="per_layer_residual" if isinstance(solver, BackwardSolver) else "continuous_residual")
    if complete:
        with timing.exclude("final_evaluation"):
            model, equation = restored_solution(config, solver.state_dict(), problem=problem)
            result["test"] = evaluate(model, equation, config)
        if isinstance(solver, GlobalSolver):
            result["selected_step"] = solver.selected_step
        else:
            result["layers"] = solver.records
    result.update(training_seconds=timing.seconds, training_timing=timing.report(config.total_steps))
    atomic_json(output/"result.json", result)
    return result


def evaluate_checkpoint(config, checkpoint, *, problem=None):
    saved = load_torch(checkpoint)
    if saved["identity"]["config"] != config_identity(config):
        raise ValueError("Checkpoint was produced with a different configuration")
    model, equation = restored_solution(config, saved["solver"], problem=problem)
    return evaluate(model, equation, config)
