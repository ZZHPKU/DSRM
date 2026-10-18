"""Monte Carlo errors with pointwise statistical units."""
import math
import torch

def moments(values):
    if not torch.isfinite(values).all():
        raise FloatingPointError("Non-finite evaluation")
    mean = values.mean(0)
    se = values.std(0, correction=1)/math.sqrt(len(values)) if len(values)>1 else torch.zeros_like(mean)
    return mean, se


def blocks(values, block_size=256):
    """Paired sums and cross-products; never store per-point Hessian tensors."""
    return [dict(count=len(v), sum=v.sum(0), cross=v.T@v)
            for v in values.split(block_size)]


def ratio_with_se(values, numerator, denominator):
    mean, _ = moments(values)
    if mean[denominator] <= 0:
        return None, None
    ratio = torch.sqrt(mean[numerator]/mean[denominator])
    if ratio == 0:
        return 0., 0.
    influence = (values[:, numerator]-ratio.square()*values[:, denominator])/(2*ratio*mean[denominator])
    return float(ratio), float(influence.std()/math.sqrt(len(values))) if len(values)>1 else 0.


def spatial_summary(values, exact=None):
    # columns: value/gradient/Hessian error², residual², three truth energies
    mean, se = moments(values)
    names = ("value", "gradient", "hessian")
    result = dict(absolute_errors=dict(zip(names, mean[:3].sqrt().tolist())),
                  error_mse=dict(zip(names, mean[:3].tolist())),
                  residual_mse=float(mean[3]), residual_rms=float(mean[3].sqrt()),
                  residual_mse_mc_se=float(se[3]), truth_energy_mc=dict(zip(names, mean[4:7].tolist())))
    relative, rel_se = {}, {}
    for i, name in enumerate(names):
        energy = exact[name] if exact is not None else float(mean[4+i])
        if energy > 0:
            if exact is None:
                relative[name], rel_se[name] = ratio_with_se(values, i, 4+i)
            else:
                relative[name] = math.sqrt(float(mean[i])/energy)
                rel_se[name] = float(se[i])/(2*math.sqrt(float(mean[i])*energy)) if mean[i]>0 else 0.
    if relative:
        result.update(relative_errors=relative, relative_error_mc_se=rel_se)
    if exact is not None:
        result["truth_energy_exact"] = exact
        result["integration_check_mc_over_exact"] = {k: float(mean[4+i])/exact[k] for i, k in enumerate(names)}
    return result


