"""Compare selected final metrics, with explicit numerical tolerances."""
import math


def compare_metrics(reference, candidate, *, rtol=.05, atol=1e-8):
    if rtol < 0 or atol < 0:
        raise ValueError("Comparison tolerances must be nonnegative")
    rows = []
    pairs = [("residual_mse", reference["residual_mse"], candidate["residual_mse"])]
    for name, expected in reference.get("relative_errors", {}).items():
        actual = candidate.get("relative_errors", {}).get(name)
        pairs.append((name+"_relative", expected, actual))
    for name, expected, actual in pairs:
        tolerance = atol+rtol*abs(expected)
        passed = actual is not None and math.isfinite(actual) and abs(actual-expected) <= tolerance
        rows.append(dict(metric=name, reference=expected, candidate=actual,
                         absolute_difference=None if actual is None else abs(actual-expected),
                         tolerance=tolerance, passed=passed))
    return dict(passed=all(r["passed"] for r in rows), rtol=rtol, atol=atol, metrics=rows)
