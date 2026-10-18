# Architecture

The package separates the mathematical problem, numerical algorithm, and
experiment execution. Solvers operate on a common problem interface, so the
same equation can be solved by any of the three methods.

## Code organization

| Location | Responsibility |
| --- | --- |
| [`src/DSRM/problems/`](../src/DSRM/problems/) | PDE and terminal interfaces, built-in equations, optional reference solutions |
| [`src/DSRM/sampling/`](../src/DSRM/sampling/) | Training and evaluation samples, importance weights |
| [`src/DSRM/models/`](../src/DSRM/models/) | Networks, terminal mapping, BR node representation and interpolation |
| [`src/DSRM/differentiation/`](../src/DSRM/differentiation/) | Derivative requests, scalar jets, derivative evaluation |
| [`src/DSRM/algorithms/`](../src/DSRM/algorithms/) | Residuals, optimizer updates, checkpoint selection within a solver |
| [`src/DSRM/evaluation/`](../src/DSRM/evaluation/) | Error estimates, statistical summaries, comparison reports |
| [`src/DSRM/parameters/`](../src/DSRM/parameters/) | Parse JSON into algorithm-specific experiment parameters |
| [`src/DSRM/runtime/`](../src/DSRM/runtime/) | Construct and run experiments, handle files and checkpoints |
| [`configs/`](../configs/) | Runnable example and paper configurations |
| [`examples/`](../examples/) | Custom problem implementations |
| [`tests/`](../tests/), [`scripts/`](../scripts/) | Numerical tests and experiment verification |

The CLI passes parsed parameters to the runtime, which constructs the problem,
model, sampler, and solver. Solvers own numerical state; the runtime owns file
access. A multi-method comparison expands shared settings into independent
experiments and runs them through this same execution path.

## Representation and differentiation

The space-time model enforces terminal data with
`v(t,x) = (t/T) g(x) + (1-t/T) M(t,x-s)`, where `s` is an optional input shift.
BR applies the same mapping at each node and interpolates the physical node
functions linearly in time. Its time derivative is defined between nodes.

Networks use tanh activations. Initial weights are drawn on CPU in float32
before conversion to the training precision. BR initializes its first network
from the corresponding time slice, then warm-starts subsequent layers from the
selected next-layer weights. Adam state is reset between layers.

The Taylor backend propagates analytic network derivatives while retaining
parameter gradients. The autograd backend differentiates the represented
function. Network-specific derivative kernels belong to `models/`; generic
requests and scalar-jet evaluation belong to `differentiation/`. The
[custom PDE interface](custom_problems.md) specifies how operators consume jets.

## Sampling and numerical objectives

The Brownian sampler uses `X_0 = 0` and covariance `t I` at time `t`, independently
of the PDE's diffusion coefficients. PINN uses occupation points, G uses paths,
and BR uses local transition pairs. The optional PINN importance sampler carries
weights that recover the same occupation measure.

| Objective | Numerical convention | Source |
| --- | --- | --- |
| PINN | Mean squared continuous PDE residual | [pinn.py](../src/DSRM/algorithms/pinn.py) |
| G | Sum of squared cell residuals divided by `T*h`, averaged over paths | [residuals.py](../src/DSRM/algorithms/residuals.py) |
| BR | Mean squared cell residual divided by `h`; local validation divides by `h²` | [backward.py](../src/DSRM/algorithms/backward.py) |

Here `h = T/N`. These normalizations are part of the learning-rate convention.
The optional F1 control removes the quadratic increment correction and uses
`F - trace(D²u)/2` to retain the diffusion compensation for Brownian sampling.

Logical samples are drawn before splitting into device batches, and their
gradients accumulate into one optimizer update. G can subsample a common master
grid; BR can reserve common draw prefixes for paired experiments. User-facing
settings are defined in [configuration](configuration.md).

## Evaluation and saved state

PINN and G select the smallest continuous validation residual. BR selects each
layer against its frozen target on a fixed local validation bank. Initial and
final iterates are included; test samples and reference errors do not select
checkpoints. Final evaluation reconstructs the selected solution on CPU in
float64, using the same test distribution for every method.

Checkpoints contain model and optimizer states, random-generator states,
progress, and selected weights. BR also stores completed nodes and the active
layer's selection state. Output files and resume requirements are documented in
[configuration](configuration.md#results-and-checkpoints).
