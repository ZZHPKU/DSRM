# Reproducing the paper experiments

This guide maps the paper experiments to the supplied configurations and explains
how to rerun the final experiments and summarize their results for tables and figures.
The [final results](../results/final/) contain all 138 experiments completed on
2026-10-18 with the current solver code. These replace the earlier paper results.
For installation, use the [README](../README.md#installation); for parameter
definitions, see [configuration](configuration.md).

## Experiment map

The [paper catalog](../configs/paper/index.json) contains 138 self-contained
configurations, including the seeds and execution options used for the final runs.
The [manifest](../results/final/manifest.json) records the environment, experiment
list, configuration hashes, and result-file hashes.
The section numbers below retain the paper's experiment organization.

| Material | Location |
| --- | --- |
| Reproduction script | [`scripts/reproduce_paper.py`](../scripts/reproduce_paper.py), using the current `src/DSRM` solver |
| Per-run results | `results/final/runs/<name>/result.json` |
| Configurations and seeds | `configs/paper/` and each run's `parameters.json` |
| Execution environment | Each run's `environment.json` and the manifest |
| Training and validation histories | Each run's `history.jsonl.gz`, losslessly compressed JSON Lines with every original event |
| Learning-rate pilot records | [`learning_rate_search.json`](../results/final/learning_rate_search.json), 72 historical candidates and their selection criteria |

The pilot selected the fixed learning-rate schedules on 2026-10-11; those schedules
were retained in the final experiments. Its records remain identified as pilot
data, separate from the 138 formal runs. The published inputs are text records;
training creates new model checkpoints in the chosen output directory.

| Paper experiment | Configurations and comparison | Runs | Statistical units |
| --- | --- | ---: | --- |
| §5.1: BR learning rates | [Gaussian](../configs/paper/gaussian/), A/B/C, CURRENT/NEXT, uniform/tuned schedules, `(c, rho) = (0.5, 0.15)`, `N = 256` | 12 | Three reference solutions |
| §5.2: nonlinear coefficients | [Gaussian](../configs/paper/gaussian/), A/B/C, PINN, G-C/N, tuned BR-C/N; four coefficient pairs | 60 | Three reference solutions per coefficient pair and method |
| §5.3: time-grid refinement | [Sine](../configs/paper/sine/), S1/S2/S3, G-C/N and BR-C/N, `N = 64, 128, 256, 512` | 48 | Three reference solutions per grid and method |
| §5.4: translation | [Translated heat](../configs/paper/translation/), PINN, `R = 0, 10, 12`, `alpha = 0.5, 5` | 18 | Three training repeats per fixed problem |
| Appendix: first-order control | [Gaussian](../configs/paper/gaussian/), A/B/C, F1-C/N with the tuned BR schedule | 6 | Three reference solutions; reuse six tuned BR controls from §5.1 |

Here C/N denotes CURRENT/NEXT. The four coefficient pairs in §5.2 are
`(0.5, 0.05)`, `(0.5, 0.15)`, `(0.75, 0.15)`, and `(1.25, 0.15)`.
Six tuned BR runs are shared by §5.1 and §5.2, giving 138 distinct runs overall.
Sine filenames use `original`, `seed-1`, and `seed-2` for S1, S2, and S3;
these identify different reference solutions. Translation filenames use
`r0`, `r1`, and `r2` for optimization repeats.

Keep the supplied problems, seeds, schedules, logical batches, validation rules,
and test sizes when reproducing these experiments. In particular, Gaussian
PINN/G use 12,000 updates at `1e-3` followed by 4,000 at `3e-4`.
The JSON configurations encode the complete schedules.

## Run and resume

Run from the repository root in the activated project environment. Paper
configurations default to CUDA and use full training budgets. For one run:

```bash
python -m DSRM run configs/paper/gaussian/A_g-next_c0.5_rho0.15.json --output results/reproduction/runs/A_g-next_c0.5_rho0.15
```

To continue the same run after interruption:

```bash
python -m DSRM run configs/paper/gaussian/A_g-next_c0.5_rho0.15.json --output results/reproduction/runs/A_g-next_c0.5_rho0.15 --resume
```

The reproduction script runs the full catalog sequentially, then writes per-run
and grouped CSV summaries. It starts a separate process for each experiment,
using the checked-out solver and supplied configurations:

```bash
python scripts/reproduce_paper.py --plan
python scripts/reproduce_paper.py --output results/reproduction
python scripts/reproduce_paper.py --output results/reproduction --resume
```

Use `--group 5.1`, `5.2`, `5.3`, `5.4`, or `appendix` to run a section. The appendix
includes its six shared BR controls. Shared runs use the same output directory;
`--resume` skips completed runs after checking their parameters and budgets.
The script does not train inside the published `results/final/` directory.

Use a fresh output directory after changing code or settings; see
[checkpoint compatibility](configuration.md#results-and-checkpoints).

For a three-run comparison against the final measurements:

```bash
python scripts/verify_paper.py --plan
python scripts/verify_paper.py --output results/verification-final
```

This uses the published final results by default and retains the full training
budgets. Its default 5% metric tolerance is a diagnostic comparison, not a
guarantee of agreement on different hardware or software.

## Aggregate and compare

Use completed `result.json` files, which contain the independently tested,
validation-selected solutions. Read metrics from `test.continuous`.
Test errors must not be used to select checkpoints or learning rates.

For each row of a paper comparison, hold its method, coefficients, grid, and
schedule fixed. Compute the arithmetic mean and sample standard deviation
(`ddof = 1`) across the three units specified above. Gaussian/sine variation is
across reference solutions; translation variation is across training repeats.
Only aggregate a group after all three runs are complete.

Regenerate the published summaries directly from the recorded final results:

```bash
python scripts/reproduce_paper.py --summarize-only --output results/final
```

This recreates `summary.csv` (138 runs) and `groups.csv` (46 groups of three).
It checks each run's recorded configuration, completion, test budget, and complete
training timing before aggregation. Use `--output results/reproduction` to
summarize a completed rerun instead. No historical source archive is required.

Multiply relative errors by 100 for percentages. Tables headed `10^3 r^2`
also multiply residual MSE by 1,000. For the translation table, average each
run's `residual_rms` and `K_H` directly; taking the square root of mean MSE or
dividing mean errors gives different statistics.

The sine time-grid figure plots a separate curve for each truth and method.
For each run, compute its combined relative error as

\[
e_{\mathrm{all}} =
\sqrt{\frac{\sum_{q\in\{u,Du,D^2u\}}\mathrm{MSE}_q}
                 {\sum_{q\in\{u,Du,D^2u\}}\lVert q^\star\rVert_\nu^2}}.
\]

The numerator uses `error_mse` and the denominator uses `truth_energy_exact`,
each summed over `value`, `gradient`, and `hessian`. Plot `100 * e_all` against
`N` on logarithmic axes. In each truth panel, the reference curve is
`e_all(G-NEXT, N=64) * sqrt(64/N)`, also expressed as a percentage.

## Timing and reproducibility limits

The final experiments ran on RTX 4070 SUPER with Python `3.12.13`,
PyTorch `2.13.0+cu130`, NumPy `2.5.1`, and SciPy `1.18.1`, using two CPU threads,
FP32 training, and AMP/TF32 disabled. Final evaluation
uses CPU/FP64. Different hardware, PyTorch versions, or accumulation chunks can
change floating-point trajectories, so the saved seeds do not guarantee
bitwise-identical trained results.

Use each result's `training_seconds` for computational-cost comparisons. All
138 final runs have complete measured timing from initialization through their
full update budgets. Grouped means and sample standard deviations are computed
across the three full runs. The measurement includes sampling, source and terminal
evaluation, network differentiation, backpropagation, and Adam; it excludes
compilation, warmup, validation, saving, logging, layer setup, final testing,
and learning-rate pilots. These measured times replace the earlier short-probe
extrapolations.

The runner's `elapsed_seconds` includes validation and checkpointing and has a
different scope. Report the device, software environment, and measured scope with
any new timings. The package environment specifies compatible dependencies;
the versions above record the environment used for the published measurements.
