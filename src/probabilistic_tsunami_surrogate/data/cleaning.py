import re
from pathlib import Path

import numpy as np

from probabilistic_tsunami_surrogate.config import (
    ARRIVAL_THRESHOLD_METRES,
    BASELINE_WINDOW_SECONDS,
    MIN_STATION_ROWS,
    N_STATIONS,
)


def fix_number_format(token):
    """Repairs numbers such as ``-0.16792338-117``."""
    if re.fullmatch(r"[-+]?\d*\.\d+[-+]\d+", token):
        return re.sub(r"([-+]?\d*\.\d+)([-+]\d+)", r"\1E\2", token)
    return token


def load_station_data(path, min_rows=MIN_STATION_ROWS):
    """Loads one four-column station output, skipping malformed rows."""
    rows = []
    try:
        lines = Path(path).read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
    except OSError as error:
        return None, {"error": str(error)}

    for line in lines:
        tokens = line.split()
        try:
            rows.append([float(token) for token in tokens])
        except ValueError:
            try:
                rows.append([float(fix_number_format(token)) for token in tokens])
            except ValueError:
                continue

    if not rows:
        return None, {"error": "no parseable rows"}

    try:
        data = np.asarray(rows, dtype=np.float32)
    except ValueError:
        return None, {"error": "inconsistent row width"}
    if data.ndim != 2 or data.shape[1] != 4 or len(data) < min_rows:
        return None, {"error": f"unexpected shape {data.shape}"}

    flat_fields = {
        name: True
        for column, name in zip(range(1, 4), ("eta", "u", "v"))
        if np.unique(data[:, column]).size <= 1
    }
    if "eta" in flat_fields:
        return None, flat_fields
    return data, flat_fields


def extract_station_targets(
    data,
    threshold=ARRIVAL_THRESHOLD_METRES,
    baseline_seconds=BASELINE_WINDOW_SECONDS,
):
    """Extracts hmax and first-arrival times from one station series."""
    time = data[:, 0]
    eta = data[:, 1]
    time_step = time[1] - time[0]
    if time_step <= 0:
        raise ValueError("station times must be increasing")

    crossing = np.abs(eta - eta[0]) > threshold
    arrival = time[np.argmax(crossing)] if crossing.any() else np.nan

    baseline_rows = max(1, int(baseline_seconds / time_step))
    baseline = np.nanmean(eta[:baseline_rows])
    adjusted_crossing = np.abs(eta - baseline) > threshold
    adjusted_arrival = (
        time[np.argmax(adjusted_crossing)]
        if adjusted_crossing.any()
        else np.nan
    )
    return np.nanmax(eta), arrival, adjusted_arrival


def build_scenario_targets(scenario_dir, n_stations=N_STATIONS):
    """Builds target arrays from every station file in one scenario."""
    hmax = np.full(n_stations, np.nan, dtype=np.float32)
    arrival = np.full(n_stations, np.nan, dtype=np.float32)
    adjusted_arrival = np.full(n_stations, np.nan, dtype=np.float32)
    issues = []

    for station_index in range(n_stations):
        station_number = station_index + 1
        path = Path(scenario_dir) / "outputs" / f"sta_{station_number:04d}"
        data, flags = load_station_data(path)
        if data is None:
            issues.append({"station": station_number, **flags})
            continue
        hmax[station_index], arrival[station_index], adjusted_arrival[
            station_index
        ] = extract_station_targets(data)

    return hmax, arrival, adjusted_arrival, issues


def save_scenario_targets(scenario_dir, targets, overwrite=False):
    """Writes the three target arrays created for one scenario."""
    hmax, arrival, adjusted_arrival, _ = targets
    scenario_dir = Path(scenario_dir)
    outputs = {
        scenario_dir / "hmax.npy": hmax,
        scenario_dir / "arrival_times.npy": arrival,
        scenario_dir / "arrival_times_avg20s.npy": adjusted_arrival,
    }
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("target arrays already exist")
    for path, values in outputs.items():
        np.save(path, values)
