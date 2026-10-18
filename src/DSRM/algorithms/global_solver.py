"""Fixed-budget optimization for PINN and global stochastic residuals."""
import math

from ..parameters.config import PINN
from ..timing import compiled_update, excluding
from .optimization import adam, snapshot, update, warmup_update
from .pinn import point_values
from .residuals import path_values


class GlobalSolver:
    def __init__(self, model, problem, parameters, rng, draw, chunk):
        self.model, self.problem, self.parameters = model, problem, parameters
        self.rng, self.draw, self.chunk = rng, draw, chunk
        self.optimizer = adam(model, parameters.schedule.rate(0))
        self.step, self.selected_step = 0, 0
        self.best_metric, self.best_state = math.inf, None
        self._warmed = False

    def values(self, bank):
        if isinstance(self.parameters, PINN):
            return point_values(self.model, self.problem, bank)
        return path_values(self.model, self.problem, bank, endpoint=self.parameters.endpoint)

    def update(self, *, timing=None):
        if timing is not None:
            timing.start()
        bank = self.draw(self.rng)
        rate = self.parameters.schedule.rate(self.step)
        if timing is not None and getattr(self.model, "jets_compiled", False) and not self._warmed:
            with excluding(timing, "warmup"):
                warmup_update(self.model, self.optimizer,
                    lambda: update(self.model, self.optimizer, bank, self.chunk, self.values, rate))
            self._warmed = True
            timing.start()
        with compiled_update(self.model, timing):
            loss = update(self.model, self.optimizer, bank, self.chunk, self.values, rate)
        self.step += 1
        if timing is not None:
            timing.updated(self.step)
        return loss

    def observe(self, metric):
        if not math.isfinite(metric):
            raise FloatingPointError("Nonfinite validation metric")
        if metric < self.best_metric:
            self.best_metric, self.selected_step = metric, self.step
            self.best_state = snapshot(self.model)

    def state_dict(self):
        return dict(model=snapshot(self.model), optimizer=self.optimizer.state_dict(),
                    rng=self.rng.get_state().cpu(), step=self.step, best_metric=self.best_metric,
                    best_state=self.best_state, selected_step=self.selected_step)

    def load_state_dict(self, state):
        self._warmed = False
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.set_state(state["rng"])
        self.step, self.selected_step = state["step"], state["selected_step"]
        self.best_metric, self.best_state = state["best_metric"], state["best_state"]
