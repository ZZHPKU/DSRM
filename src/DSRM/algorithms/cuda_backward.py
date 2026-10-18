"""Capture one complete logical BR/F1 update; keep sampling outside the graph."""
import math

import torch

from ..models.piecewise import SpatialNode
from ..sampling.brownian import Cells
from .backward import regression_values


class CudaBackwardUpdate:
    def __init__(self, solver, bank):
        self.solver, self.optimizer = solver, solver.optimizer
        self.bank = Cells(bank.t.clone(), bank.x.clone(), bank.dx.clone(), bank.h)
        self.time = bank.t.new_tensor(solver.layer*solver.h)
        self.next_time = bank.t.new_tensor((solver.layer+1)*solver.h)
        self._layer, self._rate = solver.layer, None
        current = SpatialNode(solver.model, self.time)
        following = SpatialNode(solver.target_raw, self.next_time)
        # At the terminal node the free network has weight zero. Using the same
        # target module at every layer lets one graph serve the complete solve.
        def loss(t, x, dx):
            values = regression_values(current, following, solver.problem, Cells(t, x, dx, bank.h),
                endpoint=solver.parameters.endpoint, family=solver.parameters.family)
            return values.sum()/len(bank)
        self.loss = torch.compile(loss, fullgraph=True, dynamic=False)
        parameters = list(solver.model.parameters())
        weights = [p.detach().clone() for p in parameters]
        states = {p: {key: value.clone() for key, value in state.items()}
                  for p, state in self.optimizer.state.items()}

        def checked_result(total):
            # One compiled reduction covers the loss, every gradient and updated parameter.
            values = torch.cat([total.reshape(1)]+[
                p.grad.reshape(-1) for p in parameters if p.grad is not None]+[
                p.reshape(-1) for p in parameters])
            finite = values.isfinite().all()
            return torch.stack((total, finite.to(total.dtype)))
        checked_result = torch.compile(checked_result, fullgraph=True, dynamic=False)

        def step():
            self.optimizer.zero_grad(set_to_none=True)
            total = self.bank.x.new_zeros(())
            for start in range(0, len(self.bank), solver.chunk):
                part = self.bank.slice(start, start+solver.chunk)
                value = self.loss(part.t, part.x, part.dx)
                value.backward()
                total += value.detach()
            self.optimizer.step()
            return checked_result(total)

        stream = torch.cuda.Stream(device=bank.x.device)
        stream.wait_stream(torch.cuda.current_stream(bank.x.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream(bank.x.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.result = step()
        # Compilation/warmup/capture must not count as training, including on resume.
        with torch.no_grad():
            for p, weight in zip(parameters, weights):
                p.copy_(weight)
            for p, state in self.optimizer.state.items():
                for key, value in state.items():
                    if p in states:
                        value.copy_(states[p][key])
                    else:
                        value.zero_()

    def sample(self, layer, rng):
        time = layer*self.solver.h
        self.bank.t.fill_(time)
        self.bank.x.normal_(generator=rng).mul_(math.sqrt(time))
        self.bank.dx.normal_(generator=rng).mul_(math.sqrt(self.bank.h))
        return self.bank

    def update(self, bank, layer, rate):
        if bank is not self.bank:
            self.bank.t.copy_(bank.t)
            self.bank.x.copy_(bank.x)
            self.bank.dx.copy_(bank.dx)
        # These graph inputs stay constant within a layer / learning-rate phase.
        if layer != self._layer:
            self.time.fill_(layer*self.solver.h)
            self.next_time.fill_((layer+1)*self.solver.h)
            self._layer = layer
        if rate != self._rate:
            self.optimizer.param_groups[0]["lr"].fill_(rate)
            self._rate = rate
        self.graph.replay()
        loss, finite = self.result.tolist()
        if not finite:
            raise FloatingPointError("Nonfinite CUDA graph loss, gradient, or Adam update")
        return loss
