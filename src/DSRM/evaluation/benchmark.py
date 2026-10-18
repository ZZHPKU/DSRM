"""Comparable metrics and explicit budgets for a set of PDE solvers."""
from ..parameters.config import GlobalResidual, PINN


def experiment_summary(label, config, result):
    algorithm = config.algorithm
    if isinstance(algorithm, PINN):
        method, unit = "PINN", "points"
    elif isinstance(algorithm, GlobalResidual):
        method, unit = f"G-{algorithm.endpoint}", "paths"
    else:
        method, unit = f"BR-{algorithm.family}-{algorithm.endpoint}", "cells"
    row = dict(label=label, method=method, status=result["status"], step=result["step"],
               total_steps=config.total_steps, batch=algorithm.batch, batch_unit=unit,
               N=None if isinstance(algorithm, PINN) else algorithm.N,
               elapsed_seconds=result["elapsed_seconds"])
    timing = result.get("training_timing")
    row.update(training_seconds=result.get("training_seconds"),
               timed_updates=timing["timed_updates"] if timing else None,
               timing_complete=timing["complete"] if timing else False)
    if result["status"] != "complete":
        return row
    continuous, terminal = result["test"]["continuous"], result["test"]["terminal"]
    row.update(residual_mse=continuous["residual_mse"], residual_rms=continuous["residual_rms"],
               residual_mse_mc_se=continuous["residual_mse_mc_se"],
               terminal_value_mse=terminal["value_mse"], terminal_gradient_mse=terminal["gradient_mse"])
    for name in ("value", "gradient", "hessian"):
        for source, suffix in (("absolute_errors", "rmse"), ("relative_errors", "relative"),
                               ("relative_error_mc_se", "relative_mc_se")):
            if name in continuous.get(source, {}):
                row[f"{name}_{suffix}"] = continuous[source][name]
    return row


def comparison_table(rows):
    columns = (("label", "Method"), ("status", "Status"), ("residual_rms", "Residual RMS"),
               ("value_relative", "Value rel."), ("gradient_relative", "Gradient rel."),
               ("hessian_relative", "Hessian rel."), ("training_seconds", "Training s"),
               ("timing_complete", "Full timing"), ("elapsed_seconds", "Elapsed s"))
    def display(value):
        return "-" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)
    cells = [[title for _, title in columns]]
    cells.extend([[display(row.get(key)) for key, _ in columns] for row in rows])
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    return "\n".join("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in cells)
