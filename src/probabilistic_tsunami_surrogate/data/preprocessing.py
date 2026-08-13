from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from probabilistic_tsunami_surrogate.config import (
    EXCLUDED_STATIONS,
    N_STATIONS,
)


def safe_log1p(values):
    """Applies log1p and replaces invalid values with NaN."""
    values = np.asarray(values)
    output = np.full(values.shape, np.nan, dtype=values.dtype)
    valid = np.isfinite(values) & (values >= 0)
    output[valid] = np.log1p(values[valid])
    return output


def safe_arcsinh(values):
    """Applies arcsinh and replaces nonfinite values with NaN."""
    values = np.asarray(values)
    output = np.full(values.shape, np.nan, dtype=values.dtype)
    valid = np.isfinite(values)
    output[valid] = np.arcsinh(values[valid])
    return output


def slip_id_from_sid(magnitude, scenario_id):
    """Returns zero for homogeneous and one for heterogeneous slip."""
    magnitude = float(magnitude)
    if magnitude < 8.9:
        return 0
    if magnitude > 8.9:
        return 1
    return int("_" in scenario_id)


def station_keep_mask():
    """Returns the mask for the fixed 317-station network."""
    removed = np.subtract(EXCLUDED_STATIONS, 1)
    return ~np.isin(np.arange(N_STATIONS), removed)


def fit_preprocessing_stats(
    root,
    indices,
    entries=None,
    eta_sample_ratio=0.1,
    random_state=0,
    arrival_file="arrival_times.npy",
):
    """Fits input and target scalers using training scenarios only."""
    if not 0 < eta_sample_ratio <= 1:
        raise ValueError("eta_sample_ratio must be in (0, 1]")

    root = Path(root)
    if entries is None:
        entries = np.load(root / "entries.npy", allow_pickle=True)
    selected = np.asarray(entries, dtype=object)[np.asarray(indices, dtype=int)]
    keep = station_keep_mask()
    rng = np.random.default_rng(random_state)

    eta_samples = []
    for magnitude, scenario_id in selected:
        eta = np.load(
            root / str(magnitude) / str(scenario_id) / "external_files" / "eta.npy",
            mmap_mode="r",
        )
        sample_size = int(eta.size * eta_sample_ratio)
        if sample_size:
            sample_indices = rng.choice(eta.size, sample_size, replace=False)
            eta_samples.append(np.asarray(eta).ravel()[sample_indices])

    if not eta_samples:
        raise ValueError("eta_sample_ratio selected no values")

    eta_scaler = RobustScaler().fit(np.concatenate(eta_samples).reshape(-1, 1))

    def fit_target(filename, transform):
        scaler = StandardScaler()
        found_values = False
        for magnitude, scenario_id in selected:
            values = np.load(root / str(magnitude) / str(scenario_id) / filename)[keep]
            values = transform(values)
            values = values[np.isfinite(values)]
            if values.size:
                scaler.partial_fit(values.reshape(-1, 1))
                found_values = True
        if not found_values:
            raise ValueError(f"no valid target values found in {filename}")
        return scaler

    hmax_scaler = fit_target("hmax.npy", safe_arcsinh)
    arrival_scaler = fit_target(arrival_file, safe_log1p)

    return (
        eta_scaler.center_.item(),
        eta_scaler.scale_.item(),
        hmax_scaler.mean_.item(),
        hmax_scaler.scale_.item(),
        arrival_scaler.mean_.item(),
        arrival_scaler.scale_.item(),
    )


def discover_entries(root):
    """Finds all magnitude/scenario directory pairs."""
    entries = []
    for magnitude_dir in Path(root).iterdir():
        if not magnitude_dir.is_dir():
            continue
        try:
            float(magnitude_dir.name)
        except ValueError:
            continue
        for scenario_dir in magnitude_dir.glob("M*"):
            if scenario_dir.is_dir():
                entries.append((magnitude_dir.name, scenario_dir.name))
    return sorted(entries, key=lambda entry: (float(entry[0]), entry[1]))


def convert_txt_to_npy(root, dtype=np.float32):
    """Converts text arrays below a data root to NumPy arrays."""
    converted = []
    for path in Path(root).rglob("*.txt"):
        output_path = path.with_suffix(".npy")
        if output_path.exists():
            continue
        try:
            values = np.loadtxt(path, dtype=dtype)
        except ValueError:
            continue
        np.save(output_path, values)
        converted.append(output_path)
    return converted


def load_elevation_vectors(
    root,
    entries,
    stride: int = 4,
    max_absolute_elevation=None,
):
    """Loads valid initial-elevation grids into one flattened matrix."""
    if stride < 1:
        raise ValueError("stride must be at least one")

    root = Path(root)
    accepted = []
    vectors = []
    issues = []

    for magnitude, scenario_id in entries:
        path = root / magnitude / scenario_id / "external_files" / "eta.npy"
        try:
            grid = np.load(path, mmap_mode="r", allow_pickle=False)[::stride, ::stride]
            if not np.isfinite(grid).all():
                raise ValueError("elevation grid contains nonfinite values")
            if (
                max_absolute_elevation is not None
                and np.max(np.abs(grid)) > max_absolute_elevation
            ):
                raise ValueError("elevation exceeds the configured range")
        except (OSError, ValueError) as error:
            issues.append((magnitude, scenario_id, str(error)))
            continue

        accepted.append((magnitude, scenario_id))
        vectors.append(grid.ravel().astype(np.float16))

    if not vectors:
        raise RuntimeError("no valid initial-elevation grids were found")

    matrix = np.stack(vectors).astype(np.float32, copy=False)
    return accepted, matrix, issues
