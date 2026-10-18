from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import torch

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = f.name
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = f.name
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@contextmanager
def lock(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
