# Configuration and execution

Experiments use JSON to keep the problem, algorithm, model, sampling,
evaluation, and execution settings separate. Start from the files in
[`configs/examples/`](../configs/examples/); they are the runnable configuration
examples. Parameter definitions and defaults live in
[`parameters/config.py`](../src/DSRM/parameters/config.py).

## Configuration format

A single experiment has an `algorithm` object. A comparison has an `algorithms`
object mapping labels such as `pinn`, `g`, and `br` to separate algorithm
settings. The remaining sections are shared across methods in a comparison.

| Section | Purpose |
| --- | --- |
| `name` | Experiment or comparison name |
| `problem` | Problem factory and constructor arguments; see [custom PDEs](custom_problems.md#connect-the-factory) |
| `algorithm` / `algorithms` | Method, training budget, and method-specific options |
| `model` | Network architecture and derivative backend |
| `sampling` | Sampling policy |
| `evaluation` | Validation and test sizes and intervals |
| `runtime` | Device, precision, batching, and output intervals |
| `seeds` | Initialization and sampling streams |

`name`, `problem`, and `algorithm`/`algorithms` are required. Other sections use
defaults when omitted. Each comparison entry is an independent run with its own
output directory. Labels are case-sensitive, start with a letter or digit,
and contain only letters, digits, `_`, or `-`.

### Algorithm settings

| Method | `kind` | Unit of `batch` | Schedule applies to | Single-run example |
| --- | --- | --- | --- | --- |
| PINN | `pinn` | Space-time points | Global updates | [custom_pinn.json](../configs/examples/custom_pinn.json) |
| G | `global` | Complete paths | Global updates | [custom_global.json](../configs/examples/custom_global.json) |
| BR | `backward` | Local transition pairs | Updates at each layer | [custom_backward.json](../configs/examples/custom_backward.json) |

Every method requires a positive `batch` and a `schedule`: an ordered list of
`{"steps": ..., "lr": ...}` phases. Both values must be positive, and `steps`
is an integer. Total PINN/G updates are the sum of phase steps. Total BR updates
sum the schedules over all layers.

| Additional field | Applies to | Meaning |
| --- | --- | --- |
| `N` | G, BR; required | Number of time intervals; BR has `N` nonterminal networks |
| `endpoint` | G, BR; required | `CURRENT` or `NEXT`: evaluate the complete PDE operator at the current or following sampled node |
| `master_N` | G; default `N` | Sampling grid size, a positive multiple of `N` |
| `family` | BR; default `G2` | `G2` is the quadratic residual; `F1` is the first-order control |
| `layer_overrides` | BR; default `{}` | Map string node indices to complete replacement schedules |
| `shared_prefix_steps` | BR; default `0` | Common draw prefix per layer for paired experiments; zero uses one sequential stream |

BR executes nodes from `N-1` down to `0`. For example, with `N=64`, an override
for `"63"` changes the first executed layer. Each layer otherwise uses the
base schedule. See [numerical conventions](architecture.md) for optimizer state
and loss normalization.

To compare variants, add independently named entries in `algorithms`, such as
`g_current` and `g_next`.

### Model and sampling

| Field | Default | Meaning |
| --- | --- | --- |
| `model.width` | `128` | Hidden-layer width |
| `model.hidden_layers` | `3` | Number of hidden layers |
| `model.backend` | `taylor` | `taylor` or `autograd` derivative evaluation |
| `model.input_shift` | `null` | Optional length-`d` shift for the free network |
| `sampling.kind` | `brownian` | Brownian sampling for all three methods |

PINN also supports `sampling.kind: "gaussian_mixture"`, with `center` and
`templates`, for importance sampling. It requires `T=1` and matching spatial
dimensions. The [translated-heat configurations](../configs/paper/translation/)
provide examples. Sampling and representation details are in
[architecture](architecture.md).

### Evaluation and execution

| Field | Default | Meaning |
| --- | --- | --- |
| `evaluation.validation_points` | `16384` | Continuous validation points for PINN/G |
| `evaluation.validate_every` | `500` | PINN/G validation interval in updates |
| `evaluation.test_points` | `262144` | Final continuous and terminal test sample counts |
| `evaluation.block` | `2048` | Evaluation chunk size |
| `evaluation.layer_validation_points` | `16384` | Local validation pairs per BR layer |
| `evaluation.layer_validate_every` | `100` | BR validation interval within a layer |
| `evaluation.layer_validation_block` | `16384` | BR validation chunk size |
| `runtime.device` | `cpu` | Training device, such as `cpu` or `cuda:0` |
| `runtime.dtype` | `float32` | Training precision: `float32` or `float64` |
| `runtime.device_batch` | `128` | Samples per gradient-accumulation chunk |
| `runtime.threads` | `1` | PyTorch CPU threads |
| `runtime.checkpoint_every` | `500` | Checkpoint interval in total updates |
| `runtime.log_every` | `100` | Training log interval in total updates |
| `runtime.compile` | `false` | Compile model jets on CUDA; requires `taylor` |
| `runtime.cuda_graph` | `false` | Capture BR/F1 loss, backward pass and fused Adam; requires CUDA and `taylor` |

`device_batch` uses the selected algorithm's sample unit and does not change
its logical batch or training budget. CLI options `--device`, `--device-batch`,
and `--threads` override the corresponding runtime fields.

For BR/F1, `cuda_graph` compiles the loss and captures one logical batch with
fused Adam. Sampling remains outside the graph, and graph preparation restores
the weights and optimizer state before training. For the translation PINN,
`model.backend: "taylor"` with `runtime.compile: true` compiles the exact tanh
derivatives. These execution choices preserve the mathematical objective and
budget, but floating-point training trajectories can differ. Resume still
requires the same configuration and source as the saved checkpoint.

### Seeds

| Field in `seeds` | Default | Stream |
| --- | --- | --- |
| `initialization` | `0` | Network initialization |
| `train_occupation`, `train_path` | `1`, `2` | PINN points; G paths or BR pairs |
| `validation_occupation`, `validation_path` | `3`, `4` | Continuous validation; BR local validation |
| `test_occupation`, `test_terminal` | `5`, `7` | Continuous test points; terminal test points |

Shared evaluation settings and seeds give every method the same test samples.
Training samples differ by method. A seed supplied inside `problem.kwargs`
instead controls the definition of a randomly constructed PDE.

## Run experiments

In the commands below, `comparison.json` is your file with `algorithms`, and
`experiment.json` is a file with one `algorithm`. Run from the repository root
when using modules under `examples/`.

```bash
python -m DSRM benchmark comparison.json --output results/study
python -m DSRM benchmark comparison.json --methods g --output results/g-only
python -m DSRM run experiment.json --output results/run
```

`benchmark` runs all entries in configuration order unless `--methods` selects
one or more labels. The report covers the selected entries. Training and final
evaluation run sequentially for each entry.

## Results and checkpoints

Each run saves its files in the directory specified by `--output`:

| File | Contents |
| --- | --- |
| `parameters.json` | Effective experiment settings, including runtime overrides |
| `environment.json` | Software and device information |
| `history.jsonl` | Training and validation records, one JSON object per line |
| `checkpoint.pt` | Saved state for resuming training or reevaluating the solution |
| `result.json` | Run status and, once training is complete, final evaluation results |

`training_seconds` in `result.json` measures synchronized wall time for formal
updates: sampling, evaluation of the known terminal and source, network
differentiation, backpropagation, and Adam. It counts every prescribed update,
including updates after the selected model. Compilation, warmup, CUDA Graph
capture, validation and model selection, checkpoint writes, logging, setup
and layer transitions, and final evaluation are excluded. Learning-rate pilots
are separate experiments and are not included. Validation still runs at its
configured frequency and determines the selected solution. Batch sizes,
precision, and derivative implementations remain those of `parameters.json`.

`training_timing` records the scope version, covered `start_step` and `end_step`,
`timed_updates`, `budget_updates`, and whether coverage is `complete`. Updates
in this range are `(start_step, end_step]`; complete timing requires coverage
from zero through the entire budget. Checkpoints preserve the accumulated
training time and coverage across pauses. A compatible checkpoint without
timing metadata starts a partial measurement at its saved step; missing prefix
times are never estimated or filled with zero. Source compatibility checks
still apply, so checkpoints from a different frozen implementation require
that implementation or a new run. `excluded_seconds` provides diagnostic
durations for auxiliary work; the checkpoint cannot include the duration of
its own write. The existing `elapsed_seconds` retains its broader training-phase
wall-time meaning, including validation and saving, excluding initial setup
and final evaluation.

Compiled training is warmed outside the training clock, with weights, gradients,
Adam state and random state restored before the formal update. Timed compiled
updates require `torch.compiler.set_stance` and fail if new compilation is
needed after warmup. This prevents an unexpected recompile from silently
entering the reported training time.
For unequal-sized gradient-accumulation chunks, compiler warmup can select a
generalized cached kernel earlier than a cold run, with floating-point rounding
differences. Comparisons of timing on/off should use the same warmed compiler
state. The 138 paper configurations all use equal-sized device batches.

For `benchmark`, each method's files are stored in `<output>/<label>/`.
The parent directory also contains `comparison.json` and `comparison.csv`,
which summarize the selected methods. For aggregation and comparison, see
[paper reproduction](reproduction.md#aggregate-and-compare).

Ctrl-C saves the active training state. Resume with `--resume` and the same
effective settings; a benchmark continues existing checkpoints and starts any
selected methods that have not begun. Completed methods are reevaluated without
further optimizer updates. A report is complete only when all selected methods
finish.

A single run can pause at an absolute update with `--until`, retaining its full
configured budget. For example:

```bash
python -m DSRM run experiment.json --output results/resumable --until 10
python -m DSRM run results/resumable/parameters.json --output results/resumable --resume
python -m DSRM evaluate results/resumable/parameters.json results/resumable/checkpoint.pt --output results/resumable/evaluation.json
```

Resume requires unchanged effective configuration, problem, package source,
PyTorch version, and device. Repeat any runtime overrides when resuming a
benchmark. Use a new output directory after changing these inputs. Evaluating
a BR solution requires all of its time layers to be complete.
