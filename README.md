# Deep Stochastic Residual Methods for Fully Nonlinear Parabolic Equations

DSRM implements three neural PDE solvers and provides a common
framework for comparing their accuracy and computational cost and solving
user-defined equations.

It supports smooth scalar parabolic terminal-value problems on the whole space:

$$
\partial_t u + F(t,x,u,\nabla_xu,D_x^2u)=0,
\qquad u(T,x)=g(x),\qquad x\in\mathbb R^d.
$$

The operator may be nonlinear in the Hessian. Built-in benchmarks include
Gaussian-mixture, sine, and translated heat equations. A reference solution is
optional.

## Methods

- **PINN (physics-informed neural network)** trains one space-time network by
  minimizing the squared PDE residual at sampled points. It directly evaluates
  the time derivative and the spatial derivatives required by the equation.
- **G (global stochastic residual training)** trains one space-time network
  using a second-order stochastic residual between adjacent points on sampled
  paths. All time levels are optimized jointly, with gradients flowing through
  both endpoints of each interval.
- **BR (backward regression)** trains a separate spatial network at each time
  node, proceeding backward from the terminal condition. Each layer uses the
  same residual as G against a frozen next layer. The node solutions are
  interpolated in time.

## Installation

From the repository root:

```bash
conda env create -f environment.yml
conda activate DSRM
```

The Python package and command are both named `DSRM` (case-sensitive).

## Quick start

Run all three methods with the supplied small CPU example:

```bash
python -m DSRM benchmark configs/examples/gaussian_comparison.json --output results/gaussian-comparison
```

The [configuration](configs/examples/gaussian_comparison.json) defines the
problem, methods, and training budgets. The command produces a comparison table
of residuals, available reference errors, and elapsed time. Its small budget is
intended to demonstrate the workflow.

## Documentation

| Guide | Purpose |
| --- | --- |
| [Configuration](docs/configuration.md) | Select methods, set parameters, run experiments, and read results |
| [Custom PDEs](docs/custom_problems.md) | Define an equation and connect it to the solvers |
| [Architecture](docs/architecture.md) | Locate the code and understand numerical conventions |
| [Paper reproduction](docs/reproduction.md) | Run paper experiments and summarize results for tables and figures |
| [Tests](docs/verification.md) | Run the test suite |
