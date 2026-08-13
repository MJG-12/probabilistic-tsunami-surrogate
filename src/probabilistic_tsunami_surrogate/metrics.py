"""Metrics used in the report, evaluated from cached model predictions."""

import numpy as np
from scipy.stats import norm
from scipy.stats import spearmanr
from scipy.stats import t as student_t


def inverse_transform(values, mean, scale, task):
    """Converts standardized predictions to metres or seconds."""
    transformed = values * scale + mean
    if task == "hmax":
        return np.sinh(transformed)
    if task == "arrival":
        return np.expm1(transformed)
    raise ValueError(f"unknown task: {task}")


def _standardize(values, mean, scale, task):
    if task == "hmax":
        transformed = np.arcsinh(values)
    elif task == "arrival":
        transformed = np.log1p(np.maximum(values, 0))
    else:
        raise ValueError(f"unknown task: {task}")
    return (transformed - mean) / scale


def _physical_quantile(probability, location, scale, nu, mean, target_scale, task):
    quantile = student_t.ppf(
        probability,
        df=nu,
        loc=location,
        scale=scale,
    )
    return inverse_transform(quantile, mean, target_scale, task)


def _probability_between(
    lower,
    upper,
    location,
    scale,
    nu,
    mean,
    target_scale,
    task,
):
    lower_z = _standardize(lower, mean, target_scale, task)
    upper_z = _standardize(upper, mean, target_scale, task)
    return student_t.cdf(upper_z, nu, loc=location, scale=scale) - student_t.cdf(
        lower_z,
        nu,
        loc=location,
        scale=scale,
    )


def _task_metrics(data, mean, target_scale, task):
    from scoringrules import crps_t

    target = np.asarray(data["target"], dtype=np.float64).ravel()
    location = np.asarray(data["mu"], dtype=np.float64).ravel()
    scale = np.asarray(data["sigma"], dtype=np.float64).ravel()
    nu = np.asarray(data["nu"], dtype=np.float64).ravel()
    valid = (
        np.isfinite(target)
        & np.isfinite(location)
        & np.isfinite(scale)
        & np.isfinite(nu)
        & (scale > 0)
        & (nu > 1)
    )
    target = target[valid]
    location = location[valid]
    scale = scale[valid]
    nu = nu[valid]
    if not target.size:
        return {"n": 0}

    target_physical = inverse_transform(target, mean, target_scale, task)
    prediction_physical = inverse_transform(location, mean, target_scale, task)
    residual = prediction_physical - target_physical

    nll_z = -student_t.logpdf(target, nu, loc=location, scale=scale)
    if task == "hmax":
        log_transform_derivative = -0.5 * np.log1p(target_physical**2)
    else:
        log_transform_derivative = -np.log1p(target_physical)
    nll_physical = nll_z + np.log(target_scale) - log_transform_derivative
    crps = crps_t(target, nu, location, scale)

    lower_50 = _physical_quantile(
        0.25, location, scale, nu, mean, target_scale, task
    )
    upper_50 = _physical_quantile(
        0.75, location, scale, nu, mean, target_scale, task
    )
    lower_95 = _physical_quantile(
        0.025, location, scale, nu, mean, target_scale, task
    )
    upper_95 = _physical_quantile(
        0.975, location, scale, nu, mean, target_scale, task
    )
    width_95 = upper_95 - lower_95
    interval_score_95 = width_95 + 40 * (
        np.maximum(lower_95 - target_physical, 0)
        + np.maximum(target_physical - upper_95, 0)
    )
    pit = student_t.cdf(target, nu, loc=location, scale=scale)
    has_reference = nu > 2
    reference_scale = np.sqrt((nu[has_reference] - 2) / nu[has_reference])
    reference_crps = crps_t(
        target[has_reference],
        nu[has_reference],
        np.zeros_like(target[has_reference]),
        reference_scale,
    )
    correlation = spearmanr(np.abs(residual), width_95).statistic

    output = {
        "n": int(target.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "nll_z": float(np.mean(nll_z)),
        "nll": float(np.mean(nll_physical)),
        "crps_z": float(np.mean(crps)),
        "reference_crps_z": (
            float(np.mean(reference_crps)) if reference_crps.size else float("nan")
        ),
        "picp_50": float(
            np.mean((target_physical >= lower_50) & (target_physical <= upper_50))
        ),
        "picp_95": float(
            np.mean((target_physical >= lower_95) & (target_physical <= upper_95))
        ),
        "interval_score_95": float(np.mean(interval_score_95)),
        "iw95_mean": float(np.mean(width_95)),
        "iw95_median": float(np.median(width_95)),
        "iw95_iqr": float(
            np.percentile(width_95, 75) - np.percentile(width_95, 25)
        ),
        "pit_mean": float(np.mean(pit)),
        "pit_std": float(np.std(pit)),
        "spearman_abs_error_iw95": float(correlation),
    }

    if task == "hmax":
        bands = np.array([-np.inf, 0, 0.5, 1, 2, 3, 5, 7, 10, np.inf])
        band = np.searchsorted(bands, target_physical, side="left")
        probability = _probability_between(
            bands[band - 1],
            bands[band],
            location,
            scale,
            nu,
            mean,
            target_scale,
            task,
        )
        output["correct_band_probability"] = float(np.mean(probability))
    else:
        for minutes in (2, 5, 10):
            delta = 60 * minutes
            probability = _probability_between(
                np.maximum(target_physical - delta, 0),
                target_physical + delta,
                location,
                scale,
                nu,
                mean,
                target_scale,
                task,
            )
            output[f"time_window_probability_{minutes}min"] = float(
                np.mean(probability)
            )
    return output


def _select_scenarios(task_data, mask):
    return {name: values[mask] for name, values in task_data.items()}


def summarize_predictions(predictions, stats):
    """Produces overall, slip and magnitude metrics for both report tasks."""
    _, _, hmax_mean, hmax_scale, arrival_mean, arrival_scale = stats
    task_stats = {
        "hmax": (hmax_mean, hmax_scale),
        "arrival": (arrival_mean, arrival_scale),
    }
    slips = np.asarray(predictions["slip_id"])
    magnitudes = np.asarray(predictions["magnitude"])
    summary = {}

    for task, (mean, scale) in task_stats.items():
        task_data = predictions[task]
        result = {
            "overall": _task_metrics(task_data, mean, scale, task),
            "by_slip": {
                str(slip): _task_metrics(
                    _select_scenarios(task_data, slips == slip),
                    mean,
                    scale,
                    task,
                )
                for slip in (0, 1)
            },
            "by_magnitude": {
                str(float(magnitude)): _task_metrics(
                    _select_scenarios(task_data, magnitudes == magnitude),
                    mean,
                    scale,
                    task,
                )
                for magnitude in np.unique(magnitudes)
            },
        }
        summary[task] = result
    return summary


def station_metrics(predictions, stats, target_iqr=None):
    """Returns physical error and interval-width metrics by station."""
    _, _, hmax_mean, hmax_scale, arrival_mean, arrival_scale = stats
    task_stats = {
        "hmax": (hmax_mean, hmax_scale),
        "arrival": (arrival_mean, arrival_scale),
    }
    output = {}

    for task, (mean, target_scale) in task_stats.items():
        data = predictions[task]
        target = np.asarray(data["target"], dtype=np.float64)
        location = np.asarray(data["mu"], dtype=np.float64)
        scale = np.asarray(data["sigma"], dtype=np.float64)
        nu = np.asarray(data["nu"], dtype=np.float64)
        valid = np.isfinite(target) & np.isfinite(location) & (scale > 0) & (nu > 1)

        target_physical = inverse_transform(target, mean, target_scale, task)
        prediction_physical = inverse_transform(location, mean, target_scale, task)
        residual = np.where(valid, prediction_physical - target_physical, np.nan)
        lower = _physical_quantile(
            0.025, location, scale, nu, mean, target_scale, task
        )
        upper = _physical_quantile(
            0.975, location, scale, nu, mean, target_scale, task
        )
        width = np.where(valid, upper - lower, np.nan)
        result = {
            "mae": np.nanmean(np.abs(residual), axis=0),
            "rmse": np.sqrt(np.nanmean(residual**2, axis=0)),
            "iw95_median": np.nanmedian(width, axis=0),
        }
        if target_iqr is not None:
            iqr = np.asarray(target_iqr[task], dtype=np.float64)
            floor = max(1e-6, 0.05 * np.nanmedian(iqr))
            denominator = np.where(np.isfinite(iqr) & (iqr > floor), iqr, floor)
            result["mae_over_iqr"] = result["mae"] / denominator
        output[task] = result
    return output


def pit_values(predictions, task, slip=None):
    """Returns finite probability-integral-transform values."""
    data = predictions[task]
    scenario_mask = np.ones(len(predictions["slip_id"]), dtype=bool)
    if slip is not None:
        scenario_mask = np.asarray(predictions["slip_id"]) == slip
    target = np.asarray(data["target"])[scenario_mask].ravel()
    location = np.asarray(data["mu"])[scenario_mask].ravel()
    scale = np.asarray(data["sigma"])[scenario_mask].ravel()
    nu = np.asarray(data["nu"])[scenario_mask].ravel()
    valid = (
        np.isfinite(target)
        & np.isfinite(location)
        & np.isfinite(scale)
        & np.isfinite(nu)
        & (scale > 0)
        & (nu > 0)
    )
    return student_t.cdf(
        target[valid],
        nu[valid],
        loc=location[valid],
        scale=scale[valid],
    )


def qq_diagnostics(predictions, task, max_points=None):
    """Returns PIT-normal Q-Q coordinates and central-slope diagnostics."""
    pit = np.clip(pit_values(predictions, task), 1e-12, 1 - 1e-12)
    residuals = norm.ppf(pit)
    if max_points is not None and residuals.size > max_points:
        indices = np.random.default_rng(0).choice(
            residuals.size,
            size=max_points,
            replace=False,
        )
        residuals = residuals[indices]
    observed = np.sort(residuals)
    probabilities = (np.arange(1, observed.size + 1) - 0.5) / observed.size
    theoretical = norm.ppf(probabilities)
    central = np.abs(theoretical) <= 1
    intercept, slope = np.linalg.lstsq(
        np.column_stack([np.ones(central.sum()), theoretical[central]]),
        observed[central],
        rcond=None,
    )[0]
    return theoretical, observed, {
        "n": int(observed.size),
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "central_intercept": float(intercept),
        "central_slope": float(slope),
    }
