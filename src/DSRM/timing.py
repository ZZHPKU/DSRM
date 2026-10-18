"""Synchronized wall time for every formal update, with explicit exclusions."""
from contextlib import contextmanager, nullcontext
import math
import time

import torch


class TrainingTimer:
    version = 1
    categories = ("setup", "warmup", "validation", "checkpoint", "logging", "final_evaluation")

    def __init__(self, device, *, step=0):
        if type(step) is not int or step < 0:
            raise ValueError("Timing step must be a nonnegative integer")
        self.device = torch.device(device)
        self.start_step = self.end_step = step
        self.seconds = 0.
        self.excluded_seconds = dict.fromkeys(self.categories, 0.)
        self._started = None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self):
        if self._started is None:
            self.synchronize()
            self._started = time.perf_counter()

    def pause(self):
        if self._started is not None:
            self.synchronize()
            self.seconds += time.perf_counter()-self._started
            self._started = None

    def updated(self, step):
        if self._started is None or step != self.end_step+1:
            raise ValueError("Timed updates must cover consecutive completed steps")
        self.end_step = step

    @contextmanager
    def exclude(self, category):
        if category not in self.excluded_seconds:
            raise ValueError(f"Unknown timing exclusion: {category}")
        self.pause()
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.excluded_seconds[category] += time.perf_counter()-started
        # The next update explicitly restarts the training clock.

    def state_dict(self):
        self.pause()
        return dict(version=self.version, start_step=self.start_step, end_step=self.end_step,
                    training_seconds=self.seconds, excluded_seconds=dict(self.excluded_seconds))

    @classmethod
    def restore(cls, device, state, *, step):
        timer = cls(device, step=step)
        if state is None:
            # Legacy checkpoints never measured the missing prefix.
            return timer
        if state.get("version") != cls.version:
            raise ValueError("Checkpoint uses a different training timing version")
        start, end = state.get("start_step"), state.get("end_step")
        if type(start) is not int or type(end) is not int or not 0 <= start <= end or end != step:
            raise ValueError("Timing coverage does not match the saved training step")
        seconds, excluded = state.get("training_seconds"), state.get("excluded_seconds")
        if not isinstance(excluded, dict) or set(excluded) != set(cls.categories):
            raise ValueError("Invalid excluded timing categories")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in (seconds, *excluded.values())):
            raise ValueError("Saved durations must be finite and nonnegative")
        timer.start_step, timer.end_step, timer.seconds = start, end, float(seconds)
        timer.excluded_seconds = dict(excluded)
        return timer

    def report(self, budget):
        state = self.state_dict()
        if type(budget) is not int or budget < self.end_step:
            raise ValueError("Timing coverage exceeds the prescribed budget")
        state.update(clock="synchronized_wall", timed_updates=self.end_step-self.start_step,
                     budget_updates=budget,
                     complete=self.start_step == 0 and self.end_step == budget)
        return state


def excluding(timer, category):
    return timer.exclude(category) if timer is not None else nullcontext()


def compiled_update(model, timer):
    """A timed update must use warmed code, never silently include recompilation."""
    if timer is None or not getattr(model, "jets_compiled", False):
        return nullcontext()
    if not hasattr(torch.compiler, "set_stance"):
        raise RuntimeError("Compiled training timing requires torch.compiler.set_stance")
    return torch.compiler.set_stance("fail_on_recompile")
