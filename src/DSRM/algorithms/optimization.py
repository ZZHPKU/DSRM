"""One Adam update per logical batch, with weighted gradient accumulation."""
import copy

import torch


def adam(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999), eps=1e-8)


def update(model, optimizer, bank, chunk, values, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    total = bank.x.new_zeros(())
    for start in range(0, len(bank), chunk):
        loss = values(bank.slice(start, start+chunk)).sum()/len(bank)
        loss.backward()
        total += loss.detach()
    finite = [total.isfinite()]+[p.grad.isfinite().all() for p in model.parameters() if p.grad is not None]
    if not bool(torch.stack(finite).all()):
        raise FloatingPointError("Nonfinite loss or gradient")
    optimizer.step()
    if not bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all()):
        raise FloatingPointError("Nonfinite Adam update")
    return float(total)


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def warmup_update(model, optimizer, operation):
    """Compile a real batch without consuming a formal update or changing state."""
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    parameters = list(model.parameters())
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    state = copy.deepcopy(optimizer.state_dict())
    device = parameters[0].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            operation()
        finally:
            model.load_state_dict(weights)
            optimizer.load_state_dict(state)
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
