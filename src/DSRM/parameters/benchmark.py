"""Expand shared problem and evaluation settings into named experiments."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .config import Experiment, keys, parse_config


@dataclass(frozen=True)
class Benchmark:
    name: str
    experiments: dict[str, Experiment]


def parse_benchmark(value) -> Benchmark:
    keys(value, ("name", "problem", "algorithms", "model", "sampling", "runtime", "evaluation", "seeds"),
         "benchmark")
    if not isinstance(value.get("name"), str) or not value["name"]:
        raise ValueError("benchmark.name must be a nonempty string")
    algorithms = value.get("algorithms")
    if not isinstance(algorithms, dict) or not algorithms:
        raise ValueError("benchmark.algorithms must be a nonempty object of named algorithm settings")
    shared = {k: v for k, v in value.items() if k != "algorithms"}
    experiments = {}
    for label, algorithm in algorithms.items():
        if not isinstance(label, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None:
            raise ValueError("Algorithm labels must start with a letter or digit and contain only letters, digits, _ or -")
        experiments[label] = parse_config(dict(shared, name=f"{value['name']}-{label}", algorithm=algorithm))
    return Benchmark(value["name"], experiments)


def load_benchmark(path) -> Benchmark:
    return parse_benchmark(json.loads(Path(path).read_text()))
