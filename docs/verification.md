# Tests

The test suite covers numerical components and experiment workflows. For paper
experiment groups, execution, and statistics, see [paper reproduction](reproduction.md).

Run from the repository root in the activated project environment:

```bash
python -m unittest discover -s tests -v
```

The tests cover derivatives and parameter gradients, terminal mapping, sampling,
gradient accumulation, optional reference solutions, checkpoint/resume, timing coverage
and exclusions, and method comparison reports. CUDA tests additionally check
that timing and compilation warmup preserve training states, selected solutions
and resume behavior; these tests are skipped when CUDA is unavailable.
