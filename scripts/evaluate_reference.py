"""Reference Student-t CRPS evaluation."""

from pathlib import Path

import numpy as np
from scoringrules import crps_t

from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.data.preprocessing import slip_id_from_sid


def _unit_var_scale(nu: float) -> float:
    if nu <= 2:
        raise ValueError("df (nu) must be > 2")
    return float(np.sqrt((nu - 2.0) / nu))


def _transform_and_standardize(x: np.ndarray, kind: str, mean: float, std: float) -> np.ndarray:
    if kind == "hmax":
        z = np.arcsinh(x)
    elif kind == "arrival_time":
        z = np.log1p(x)
    else:
        raise ValueError("kind must be 'hmax' or 'arrival_time'")
    return (z - mean) / (std + 1e-12)


def _load_target(root: Path, mw: str, sid: str, kind: str) -> np.ndarray:
    fname = "hmax.npy" if kind == "hmax" else "arrival_times.npy"
    arr = np.load(root / mw / sid / fname).astype(np.float64).ravel()
    return arr[np.isfinite(arr)]


def _dataset_baseline_per_task(kind: str, entries, root: Path, mean: float, std: float, nu_hom: float, nu_het: float):
    """
    Returns dict:
      {'overall': mean_CRPS, 'hom': mean_CRPS, 'het': mean_CRPS}
    Overall is computed by scoring each item with its slip's df, then averaging across all.
    """
    s_hom = _unit_var_scale(nu_hom)
    s_het = _unit_var_scale(nu_het)

    totals = {"overall": 0.0, "hom": 0.0, "het": 0.0}
    counts = {"overall": 0,   "hom": 0,   "het": 0  }

    for mw, sid in entries:
        arr = _load_target(root, mw, sid, kind)
        if arr.size == 0:
            continue
        z = _transform_and_standardize(arr, kind if kind=="hmax" else "arrival_time", mean, std)

        g = slip_id_from_sid(mw, sid)
        if g == 0:
            c = crps_t(z, nu_hom, 0.0, s_hom)
            totals["hom"] += float(c.sum()); counts["hom"] += z.size
        elif g == 1:
            c = crps_t(z, nu_het, 0.0, s_het)
            totals["het"] += float(c.sum()); counts["het"] += z.size
        else:
            raise ValueError("slip_id_from_sid must return 0 or 1")

        totals["overall"] += float(c.sum()); counts["overall"] += z.size

    out = {k: (totals[k] / max(counts[k], 1)) for k in totals}
    return out


def evaluate_reference(config):
    """Computes reference Student-t CRPS overall and by slip regime."""
    entries = np.load(config.entries_path, allow_pickle=True)
    test_idx = np.load(config.SPLIT_DIR / "test_idx.npy")
    test_entries = entries[test_idx]
    stats = config.REFERENCE_STATS_DIR
    specifications = {
        "hmax": ("hmax_mean_asinh.npy", "hmax_std_asinh.npy", 8.9936, 5.7138),
        "arrival_time": (
            "arrival_time_mean_log.npy", "arrival_time_std_log.npy", 4.6362, 4.0618
        ),
    }
    results = {}
    for task, (mean_file, std_file, nu_hom, nu_het) in specifications.items():
        results[task] = _dataset_baseline_per_task(
            kind=task,
            entries=test_entries,
            root=config.DATA_ROOT,
            mean=np.load(stats / mean_file),
            std=np.load(stats / std_file),
            nu_hom=nu_hom,
            nu_het=nu_het,
        )
        print(f"{task}: {results[task]}")
    return results


if __name__ == "__main__":
    evaluate_reference(RunConfig())
