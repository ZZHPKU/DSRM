"""Layerwise optimization and validation; no filesystem or benchmark names."""
import copy
import math

import torch

from ..models.piecewise import SpatialNode, TerminalSolution
from ..sampling import cells, generator
from ..timing import compiled_update, excluding
from .backward import extra_seed, regression_values
from .optimization import adam, snapshot, update, warmup_update


class BackwardSolver:
    def __init__(self, model, problem, parameters, seeds, evaluation, chunk, *, cuda_graph=False):
        self.model, self.problem, self.parameters = model, problem, parameters
        self.seeds, self.evaluation, self.chunk = seeds, evaluation, chunk
        self.device, self.dtype = next(model.parameters()).device, next(model.parameters()).dtype
        self.h = problem.T/parameters.N
        if hasattr(model, "hard_mapping"):
            model.hard_mapping = False
        self.target_raw = copy.deepcopy(model).requires_grad_(False)
        if hasattr(self.target_raw, "jet_function"):
            # Compiled bound functions must refer to the target's own parameters.
            self.target_raw.jet_function = self.target_raw.jet_tensors
            if model.jets_compiled:
                self.target_raw.enable_cuda_compile()
        self.rng = generator(seeds.train_path, self.device)
        self.step, self.layer, self.layer_step = 0, parameters.N-1, 0
        self.states, self.records = [None]*parameters.N, []
        self.best_metric, self.best_state, self.selected_step = math.inf, None, 0
        self.events = []
        self.cuda_graph, self.accelerator = cuda_graph, None
        self._prepare_layer()

    def _prepare_layer(self):
        self._warmed = False
        if self.layer < 0:
            return
        p = self.parameters
        self.current = SpatialNode(self.model, self.layer*self.h)
        if self.layer == p.N-1:
            self.target = TerminalSolution(self.model.terminal)
        else:
            self.target_raw.load_state_dict(self.states[self.layer+1])
            self.target = SpatialNode(self.target_raw, (self.layer+1)*self.h)
        if self.cuda_graph:
            if not hasattr(self, "optimizer"):
                rate = torch.tensor(p.schedules[self.layer].rate(0), device=self.device, dtype=self.dtype)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=rate,
                                                  fused=True, capturable=True)
            else:
                # A captured graph retains these tensors' addresses across layers.
                for state in self.optimizer.state.values():
                    for value in state.values():
                        value.zero_()
        else:
            self.optimizer = adam(self.model, p.schedules[self.layer].rate(0))
        self.extra_rng = generator(extra_seed(self.seeds.train_path, self.layer, p.N), self.device)
        self.validation_bank = cells(self.evaluation.layer_validation_points, self.problem.d,
            self.layer*self.h, self.h, generator(self.seeds.validation_path+self.layer, self.device), dtype=self.dtype)

    def values(self, bank, *, create_graph=True):
        return regression_values(self.current, self.target, self.problem, bank,
            endpoint=self.parameters.endpoint, family=self.parameters.family, create_graph=create_graph)

    def validate(self):
        total = self.validation_bank.x.new_zeros(())
        block = self.evaluation.layer_validation_block
        for start in range(0, len(self.validation_bank), block):
            bank = self.validation_bank.slice(start, start+block)
            total += self.values(bank, create_graph=False).sum()/len(self.validation_bank)
        metric = float(total)/self.h
        if not math.isfinite(metric):
            raise FloatingPointError("Nonfinite layer validation")
        if metric < self.best_metric:
            self.best_metric, self.selected_step = metric, self.layer_step
            self.best_state = snapshot(self.model)
        self.events.append(dict(event="layer_validation", step=self.step, layer=self.layer,
                                layer_step=self.layer_step, residual_mse=metric))

    def update(self, *, timing=None):
        if self.layer < 0:
            raise RuntimeError("Backward solve is already complete")
        p = self.parameters
        if self.best_state is None:
            with excluding(timing, "validation"):
                self.validate()
        schedule = p.schedules[self.layer]
        if timing is not None:
            timing.start()
        rng = self.extra_rng if p.shared_prefix_steps and self.layer_step >= p.shared_prefix_steps else self.rng
        bank = (self.accelerator.sample(self.layer, rng) if self.cuda_graph and self.accelerator is not None else
                cells(p.batch, self.problem.d, self.layer*self.h, self.h, rng, dtype=self.dtype))
        if self.cuda_graph:
            if self.accelerator is None:
                with excluding(timing, "warmup"):
                    from .cuda_backward import CudaBackwardUpdate
                    device = self.device.index if self.device.index is not None else torch.cuda.current_device()
                    with torch.random.fork_rng(devices=[device]):
                        self.accelerator = CudaBackwardUpdate(self, bank)
                if timing is not None:
                    timing.start()
            loss = self.accelerator.update(bank, self.layer, schedule.rate(self.layer_step))
        else:
            rate = schedule.rate(self.layer_step)
            if timing is not None and getattr(self.model, "jets_compiled", False) and not self._warmed:
                with excluding(timing, "warmup"):
                    warmup_update(self.model, self.optimizer,
                        lambda: update(self.model, self.optimizer, bank, self.chunk, self.values, rate))
                self._warmed = True
                timing.start()
            with compiled_update(self.model, timing):
                loss = update(self.model, self.optimizer, bank, self.chunk, self.values, rate)
        self.layer_step += 1
        self.step += 1
        if timing is not None:
            timing.updated(self.step)
        if self.layer_step % self.evaluation.layer_validate_every == 0 or self.layer_step == schedule.steps:
            with excluding(timing, "validation"):
                self.validate()
        if self.layer_step == schedule.steps:
            with excluding(timing, "setup"):
                self.states[self.layer] = self.best_state
                self.records.append(dict(layer=self.layer, steps=self.layer_step,
                    selected_step=self.selected_step, residual_mse=self.best_metric))
                self.events.append(dict(event="layer_complete", step=self.step, **self.records[-1]))
                # Preserve the fixed common sample prefix even for a shorter layer.
                for _ in range(self.layer_step, p.shared_prefix_steps):
                    cells(p.batch, self.problem.d, self.layer*self.h, self.h, self.rng, dtype=self.dtype)
                self.model.load_state_dict(self.best_state)
                self.layer -= 1
                self.layer_step = 0
                self.best_metric, self.best_state, self.selected_step = math.inf, None, 0
                self._prepare_layer()
        return loss

    def state_dict(self):
        return dict(model=snapshot(self.model), optimizer=self.optimizer.state_dict(),
            rng=self.rng.get_state().cpu(), extra_rng=self.extra_rng.get_state().cpu(),
            step=self.step, layer=self.layer, layer_step=self.layer_step,
            states=self.states, records=self.records, best_metric=self.best_metric,
            best_state=self.best_state, selected_step=self.selected_step)

    def load_state_dict(self, state):
        # Optimizer loading may replace tensors retained by a previously captured graph.
        self.accelerator = None
        for key in ("step", "layer", "layer_step", "states", "records", "best_metric", "best_state", "selected_step"):
            setattr(self, key, state[key])
        self.model.load_state_dict(state["model"])
        self._prepare_layer()
        self.optimizer.load_state_dict(state["optimizer"])
        if self.cuda_graph:
            self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"].to(self.device)
        self.rng.set_state(state["rng"])
        self.extra_rng.set_state(state["extra_rng"])
