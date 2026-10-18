"""Nonpersistent derivative coefficients of fixed analytic functions."""


class FixedDerivatives:
    """Rebuild cached coefficients after dtype/device moves or checkpoint loads.

    Derived tensors must not enter checkpoints: old checkpoints remain valid,
    and the cache always follows the precision of its defining buffers.
    """

    def _refresh_derivatives(self):
        for name, value in self._derivative_buffers().items():
            if name in self._buffers:
                # Captured CUDA graphs retain these storage addresses across
                # load_state_dict (including every BR layer's warm start).
                # Replacing the tensor would leave replay reading old storage.
                cached = self._buffers[name]
                if (cached.shape == value.shape and cached.dtype == value.dtype
                        and cached.device == value.device):
                    cached.copy_(value.detach())
                else:
                    setattr(self, name, value.detach())
            else:
                self.register_buffer(name, value.detach(), persistent=False)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._refresh_derivatives()
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._refresh_derivatives()
