"""Validation and summaries for the simulation dataset."""

from pathlib import Path

import numpy as np

from probabilistic_tsunami_surrogate.config import GRID_SHAPE, N_STATIONS
from probabilistic_tsunami_surrogate.data.cleaning import load_station_data
from probabilistic_tsunami_surrogate.data.preprocessing import (
    slip_id_from_sid,
    station_keep_mask,
)


def validate_scenario(
    root,
    entry,
    grid_shape=GRID_SHAPE,
    max_absolute_elevation=None,
    check_station_files=False,
):
    """Returns data-quality issues for one magnitude/scenario pair."""
    magnitude, scenario_id = entry
    directory = Path(root) / str(magnitude) / str(scenario_id)
    issues = []

    eta_path = directory / "external_files" / "eta.npy"
    try:
        eta = np.load(eta_path, mmap_mode="r", allow_pickle=False)
        if eta.shape != tuple(grid_shape):
            raise ValueError(f"expected shape {tuple(grid_shape)}, found {eta.shape}")
        if not np.isfinite(eta).all():
            raise ValueError("contains nonfinite values")
        if (
            max_absolute_elevation is not None
            and np.max(np.abs(eta)) > max_absolute_elevation
        ):
            raise ValueError(
                f"absolute elevation exceeds {max_absolute_elevation} metres"
            )
    except (OSError, ValueError) as error:
        issues.append({"source": "eta", "severity": "error", "detail": str(error)})

    keep = station_keep_mask()
    for filename, source in (
        ("hmax.npy", "hmax"),
        ("arrival_times.npy", "arrival"),
    ):
        try:
            values = np.load(directory / filename, allow_pickle=False)
            if values.shape != (N_STATIONS,):
                raise ValueError(
                    f"expected shape {(N_STATIONS,)}, found {values.shape}"
                )
            valid = np.isfinite(values[keep])
            if not valid.any():
                raise ValueError("contains no finite retained-station values")
            missing = int((~valid).sum())
            if missing:
                issues.append(
                    {
                        "source": source,
                        "severity": "warning",
                        "detail": f"{missing} retained stations are missing",
                    }
                )
        except (OSError, ValueError) as error:
            issues.append(
                {"source": source, "severity": "error", "detail": str(error)}
            )

    if check_station_files:
        for station_index in np.flatnonzero(keep):
            station_number = int(station_index + 1)
            path = directory / "outputs" / f"sta_{station_number:04d}"
            data, flags = load_station_data(path)
            if data is None:
                issues.append(
                    {
                        "source": "station_output",
                        "severity": "error",
                        "station": station_number,
                        "detail": flags.get("error", ", ".join(flags)),
                    }
                )
            elif flags:
                issues.append(
                    {
                        "source": "station_output",
                        "severity": "warning",
                        "station": station_number,
                        "detail": ", ".join(flags),
                    }
                )

    return [
        {
            "magnitude": str(magnitude),
            "scenario": str(scenario_id),
            **issue,
        }
        for issue in issues
    ]


def validate_dataset(root, entries, **kwargs):
    """Returns valid entries and all validation issues."""
    valid_entries = []
    issues = []
    for entry in entries:
        scenario_issues = validate_scenario(root, entry, **kwargs)
        issues.extend(scenario_issues)
        if not any(issue["severity"] == "error" for issue in scenario_issues):
            valid_entries.append(tuple(entry))
    return valid_entries, issues


def station_target_iqr(root, entries, indices=None, min_valid=3):
    """Computes physical target IQR by retained station."""
    entries = np.asarray(entries, dtype=object)
    if indices is not None:
        entries = entries[np.asarray(indices, dtype=int)]
    keep = station_keep_mask()
    values = {"hmax": [], "arrival": []}

    for magnitude, scenario_id in entries:
        directory = Path(root) / str(magnitude) / str(scenario_id)
        values["hmax"].append(np.load(directory / "hmax.npy")[keep])
        values["arrival"].append(np.load(directory / "arrival_times.npy")[keep])

    output = {}
    for task, rows in values.items():
        matrix = np.asarray(rows, dtype=np.float32)
        lower, upper = np.nanpercentile(matrix, [25, 75], axis=0)
        iqr = (upper - lower).astype(np.float32)
        iqr[np.isfinite(matrix).sum(axis=0) < min_valid] = np.nan
        output[task] = iqr
    return output


def summarize_targets(root, entries):
    """Summarizes physical targets overall and by magnitude and slip."""
    keep = station_keep_mask()
    groups = {}
    for magnitude, scenario_id in entries:
        key = (str(magnitude), slip_id_from_sid(magnitude, str(scenario_id)))
        directory = Path(root) / str(magnitude) / str(scenario_id)
        group = groups.setdefault(key, {"hmax": [], "arrival": [], "scenarios": 0})
        group["hmax"].append(np.load(directory / "hmax.npy")[keep])
        group["arrival"].append(np.load(directory / "arrival_times.npy")[keep])
        group["scenarios"] += 1

    summary = {}
    for (magnitude, slip), group in groups.items():
        result = {"scenarios": group["scenarios"]}
        for task in ("hmax", "arrival"):
            values = np.concatenate(group[task])
            values = values[np.isfinite(values)]
            result[task] = {
                "n": int(values.size),
                "median": float(np.median(values)),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            }
        summary[f"Mw{magnitude}_slip{slip}"] = result
    return summary
