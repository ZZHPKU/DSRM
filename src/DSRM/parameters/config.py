"""Parse experiment files into algorithm-specific, immutable parameters."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Literal


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def keys(value, allowed, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    extra = set(value)-set(allowed)
    if extra:
        raise ValueError(f"Unknown {context} fields: {sorted(extra)}")
    return value


@dataclass(frozen=True)
class Phase:
    steps: int
    lr: float

    def __post_init__(self):
        positive_int(self.steps, "phase.steps")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("phase.lr must be positive")


@dataclass(frozen=True)
class Schedule:
    phases: tuple[Phase, ...]

    def __post_init__(self):
        if not self.phases:
            raise ValueError("A schedule must contain at least one phase")

    @property
    def steps(self):
        return sum(p.steps for p in self.phases)

    def rate(self, completed_steps):
        for phase in self.phases:
            if completed_steps < phase.steps:
                return phase.lr
            completed_steps -= phase.steps
        return self.phases[-1].lr


def parse_schedule(value):
    return Schedule(tuple(Phase(**keys(p, ("steps", "lr"), "phase")) for p in value))


@dataclass(frozen=True)
class PINN:
    batch: int
    schedule: Schedule
    kind: Literal["pinn"] = "pinn"


@dataclass(frozen=True)
class GlobalResidual:
    batch: int
    schedule: Schedule
    N: int
    master_N: int
    endpoint: Literal["CURRENT", "NEXT"]
    kind: Literal["global"] = "global"


@dataclass(frozen=True)
class BackwardRegression:
    batch: int
    N: int
    endpoint: Literal["CURRENT", "NEXT"]
    family: Literal["G2", "F1"]
    schedules: tuple[Schedule, ...]
    shared_prefix_steps: int
    kind: Literal["backward"] = "backward"


Algorithm = PINN | GlobalResidual | BackwardRegression


def parse_algorithm(value) -> Algorithm:
    common = ("kind", "batch")
    kind = value["kind"]
    batch = positive_int(value["batch"], "algorithm.batch")
    if kind == "pinn":
        keys(value, (*common, "schedule"), "pinn")
        return PINN(batch, parse_schedule(value["schedule"]))
    if kind not in ("global", "backward"):
        raise ValueError(f"Unknown algorithm: {kind}")
    N = positive_int(value["N"], "algorithm.N")
    endpoint = value["endpoint"]
    if endpoint not in ("CURRENT", "NEXT"):
        raise ValueError("endpoint must be CURRENT or NEXT")
    if kind == "global":
        keys(value, (*common, "schedule", "N", "master_N", "endpoint"), "global")
        master = positive_int(value.get("master_N", N), "master_N")
        if master % N:
            raise ValueError("master_N must be divisible by N")
        return GlobalResidual(batch, parse_schedule(value["schedule"]), N, master, endpoint)
    keys(value, (*common, "N", "endpoint", "family", "schedule", "layer_overrides",
                 "shared_prefix_steps"), "backward")
    family = value.get("family", "G2")
    if family not in ("G2", "F1"):
        raise ValueError("backward.family must be G2 or F1")
    default = parse_schedule(value["schedule"])
    schedules = [default]*N
    for index, phases in value.get("layer_overrides", {}).items():
        i = int(index)
        if str(i) != index or not 0 <= i < N:
            raise ValueError(f"Invalid layer index: {index}")
        schedules[i] = parse_schedule(phases)
    prefix = value.get("shared_prefix_steps", 0)
    if type(prefix) is not int or prefix < 0:
        raise ValueError("shared_prefix_steps must be a nonnegative integer")
    return BackwardRegression(batch, N, endpoint, family, tuple(schedules), prefix)


@dataclass(frozen=True)
class Model:
    width: int = 128
    hidden_layers: int = 3
    backend: Literal["taylor", "autograd"] = "taylor"
    input_shift: tuple[float, ...] | None = None

    def __post_init__(self):
        positive_int(self.width, "model.width")
        positive_int(self.hidden_layers, "model.hidden_layers")
        if self.backend not in ("taylor", "autograd"):
            raise ValueError("model.backend must be taylor or autograd")


@dataclass(frozen=True)
class Runtime:
    device: str = "cpu"
    dtype: Literal["float32", "float64"] = "float32"
    device_batch: int = 128
    threads: int = 1
    checkpoint_every: int = 500
    log_every: int = 100
    compile: bool = False
    cuda_graph: bool = False

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("runtime.dtype must be float32 or float64")
        if type(self.compile) is not bool:
            raise ValueError("runtime.compile must be boolean")
        if type(self.cuda_graph) is not bool:
            raise ValueError("runtime.cuda_graph must be boolean")
        for name in ("device_batch", "threads", "checkpoint_every", "log_every"):
            positive_int(getattr(self, name), f"runtime.{name}")


@dataclass(frozen=True)
class Evaluation:
    validation_points: int = 16384
    test_points: int = 262144
    block: int = 2048
    validate_every: int = 500
    layer_validation_points: int = 16384
    layer_validate_every: int = 100
    layer_validation_block: int = 16384

    def __post_init__(self):
        for name, value in asdict(self).items():
            positive_int(value, f"evaluation.{name}")


@dataclass(frozen=True)
class Seeds:
    initialization: int = 0
    train_occupation: int = 1
    train_path: int = 2
    validation_occupation: int = 3
    validation_path: int = 4
    test_occupation: int = 5
    test_path: int = 6
    test_terminal: int = 7

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f"seed {name} must be an integer in [0, 2**63)")


@dataclass(frozen=True)
class BrownianSampling:
    kind: Literal["brownian"] = "brownian"


@dataclass(frozen=True)
class ImportanceSampling:
    center: tuple[float, ...]
    templates: tuple[tuple[float, ...], ...]
    kind: Literal["gaussian_mixture"] = "gaussian_mixture"

    def __post_init__(self):
        if not self.center or not self.templates or any(len(t) != len(self.center) for t in self.templates):
            raise ValueError("Importance sampler needs matching center/templates")
        if not all(math.isfinite(x) for x in self.center) or any(
                not math.isfinite(x) or x < 0 for t in self.templates for x in t):
            raise ValueError("Importance sampler parameters must be finite; diffusions nonnegative")


@dataclass(frozen=True)
class Problem:
    factory: str
    kwargs: dict


@dataclass(frozen=True)
class Experiment:
    name: str
    problem: Problem
    algorithm: Algorithm
    model: Model
    sampling: BrownianSampling | ImportanceSampling
    runtime: Runtime
    evaluation: Evaluation
    seeds: Seeds

    @property
    def total_steps(self):
        if isinstance(self.algorithm, BackwardRegression):
            return sum(s.steps for s in self.algorithm.schedules)
        return self.algorithm.schedule.steps


def parse_config(value) -> Experiment:
    keys(value, ("name", "problem", "algorithm", "model", "sampling", "runtime", "evaluation", "seeds"), "config")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ValueError("name must be a nonempty string")
    p = keys(value["problem"], ("factory", "kwargs"), "problem")
    if not isinstance(p["factory"], str) or ":" not in p["factory"] or not isinstance(p.get("kwargs", {}), dict):
        raise ValueError("problem requires a module:factory and an optional kwargs object")
    m = dict(value.get("model", {}))
    if m.get("input_shift") is not None:
        m["input_shift"] = tuple(m["input_shift"])
    sampling = value.get("sampling", {"kind": "brownian"})
    if sampling["kind"] == "brownian":
        keys(sampling, ("kind",), "sampling")
        sampler = BrownianSampling()
    elif sampling["kind"] == "gaussian_mixture":
        keys(sampling, ("kind", "center", "templates"), "sampling")
        sampler = ImportanceSampling(tuple(sampling["center"]), tuple(map(tuple, sampling["templates"])))
    else:
        raise ValueError("Unknown sampling kind")
    algorithm = parse_algorithm(value["algorithm"])
    if isinstance(sampler, ImportanceSampling) and not isinstance(algorithm, PINN):
        raise ValueError("Gaussian mixture importance sampling is supported for PINN only")
    seeds = {k: int(v) if isinstance(v, str) and v.isdecimal() else v for k, v in value.get("seeds", {}).items()}
    return Experiment(value["name"], Problem(p["factory"], p.get("kwargs", {})), algorithm,
                      Model(**m), sampler, Runtime(**value.get("runtime", {})),
                      Evaluation(**value.get("evaluation", {})), Seeds(**seeds))


def load_config(path) -> Experiment:
    return parse_config(json.loads(Path(path).read_text()))


def config_dict(config: Experiment) -> dict:
    """Return a JSON representation accepted by load_config, including overrides."""
    result = asdict(config)
    algorithm = result["algorithm"]
    if isinstance(config.algorithm, BackwardRegression):
        schedules = algorithm.pop("schedules")
        default = schedules[0]["phases"]
        algorithm["schedule"] = default
        algorithm["layer_overrides"] = {str(i): s["phases"] for i, s in enumerate(schedules)
                                        if s["phases"] != default}
    else:
        algorithm["schedule"] = algorithm["schedule"]["phases"]
    return result
