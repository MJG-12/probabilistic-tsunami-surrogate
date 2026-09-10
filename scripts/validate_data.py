import csv
import json

import numpy as np

from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.data.validation import (
    summarize_targets,
    validate_dataset,
)
from probabilistic_tsunami_surrogate.plotting import plot_data_distributions


def validate_data(config):
    """Checks the saved research catalogue and writes data-quality artifacts."""
    entries = np.load(config.entries_path, allow_pickle=True)
    valid_entries, issues = validate_dataset(
        config.DATA_ROOT,
        entries,
        max_absolute_elevation=config.MAX_ABSOLUTE_ELEVATION,
        check_station_files=config.CHECK_STATION_FILES,
    )
    config.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    np.save(
        config.VALIDATION_DIR / "valid_entries.npy",
        np.asarray(valid_entries, dtype=object),
    )

    fields = ["magnitude", "scenario", "source", "severity", "station", "detail"]
    with (config.VALIDATION_DIR / "issues.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)

    summary = summarize_targets(config.DATA_ROOT, valid_entries)
    with (config.VALIDATION_DIR / "target_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    plot_data_distributions(
        config.DATA_ROOT,
        valid_entries,
        config.VALIDATION_DIR,
    )
    print(
        f"valid={len(valid_entries)}/{len(entries)} issues={len(issues)} "
        f"output={config.VALIDATION_DIR}"
    )


if __name__ == "__main__":
    validate_data(RunConfig())
