# Custom PDEs

A custom problem supplies the terminal data and spatial operator for the
[equation class supported by the package](../README.md). The same definition
works with PINN, G, and BR; training settings belong in configuration files.

## Define the equation

Use [examples/custom_problem.py](../examples/custom_problem.py) as the runnable
template. Copy it to a new module and replace the equation-specific definitions:

```bash
cp examples/custom_problem.py examples/my_problem.py
```

| Interface | Implement |
| --- | --- |
| `TerminalCondition.value(x)` | The terminal function `g(x)` |
| `PDEProblem.operator(t, x, jet)` | The complete spatial operator `F`, including PDE coefficients and any source term |
| `PDEProblem.operator_request()` | A `JetRequest` declaring which spatial derivatives the operator uses |
| `make_problem(...)` | A factory returning the configured `PDEProblem` |

Pass the terminal object to `PDEProblem`'s constructor. It supplies the dimension
`d` and horizon `T`. The operator excludes the time derivative: time treatment
belongs to the solver. Include the PDE's full diffusion operator; the algorithm
handles its own sampling corrections.

Inputs have shapes `t: [B]` and `x: [B,d]`. Terminal and operator values must be
`[B]`, with one scalar per point. The candidate solution's derivatives are
available through `jet`:

| Accessor | Shape | `JetRequest` flag |
| --- | --- | --- |
| `jet.y` | `[B]` | Always available |
| `jet.require_gradient()` | `[B,d]` | `need_full_gradient=True` |
| `jet.trace` | `[B]` | Diagonal or full Hessian |
| `jet.require_hessian_diagonal()` | `[B,d]` | `need_hessian_diagonal=True` or full Hessian |
| `jet.require_hessian()` | `[B,d,d]` | `need_full_hessian=True` |

The default `operator_request()` requests the full gradient and Hessian. A
specialized request can reduce derivative work when the operator only uses a
subset. The solver adds the derivatives required by its residual.

Use smooth PyTorch operations for terminal data; the base class supplies its
gradient and Hessian through autograd. Keep operator expressions differentiable
with respect to `jet` and evaluate each point independently. Store fixed tensor
coefficients with `register_buffer` so device and precision conversions follow
the problem. The factory must return the same fixed equation for the same
arguments, including any randomly generated coefficients.

## Connect the factory

The factory must be importable by the process running the solver. Modules in
`examples/` are importable when running from the repository root; an external
problem can live in an installed Python package.

Copy the supplied comparison configuration:

```bash
cp configs/examples/custom_comparison.json configs/my_comparison.json
```

Give the copy a descriptive `name` and set its `problem` section:

```json
"problem": {
  "factory": "examples.my_problem:make_problem",
  "kwargs": {"d": 2, "T": 1.0}
}
```

`factory` uses `module:function` syntax. `kwargs` contains the arguments accepted
by your factory; add PDE coefficients there. Keep model and training parameters
in their respective configuration sections. The copied file already contains
settings for all three methods; [configuration](configuration.md) explains how
to select methods and change their budgets.

## Solve the problem

```bash
python -m DSRM benchmark configs/my_comparison.json --output results/my-comparison
```

Each completed run evaluates its selected solution. See
[results and checkpoints](configuration.md#results-and-checkpoints) for the
reports, reevaluation, and resume commands.

## Optional reference solution

Pass a reference module as `reference=` to `PDEProblem` to enable error
measurements. It may implement differentiable `forward(t, x)`, or provide
`exact(t, x)` returning a `ScalarJet` with value, gradient, and Hessian. A
reference is independent of the terminal and operator interfaces.

By default, reference energies for relative errors are estimated from evaluation
samples. If known analytically, `reference_energies` may supply positive mean
squared physical value, gradient norm, and Hessian Frobenius norm under uniform
time and Brownian occupation, keyed by `value`, `gradient`, and `hessian`.
Leave it unset if a channel is identically zero. `value_scale`, when used for a
normalized problem, converts solution values and residuals back to physical
units.
