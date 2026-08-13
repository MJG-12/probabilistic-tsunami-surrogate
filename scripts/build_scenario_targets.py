import csv

from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.data.cleaning import (
    build_scenario_targets,
    save_scenario_targets,
)
from probabilistic_tsunami_surrogate.data.preprocessing import discover_entries


def build_targets(config):
    """Builds target arrays and records unusable station files."""
    entries = discover_entries(config.DATA_ROOT)
    issues = []
    completed = 0
    skipped = 0

    for magnitude, scenario_id in entries:
        directory = config.DATA_ROOT / magnitude / scenario_id
        targets = build_scenario_targets(directory)
        try:
            save_scenario_targets(directory, targets, config.OVERWRITE_TARGETS)
        except FileExistsError:
            skipped += 1
            continue
        completed += 1
        for issue in targets[3]:
            issues.append({"magnitude": magnitude, "scenario": scenario_id, **issue})

    if issues:
        fields = sorted(set().union(*(issue.keys() for issue in issues)))
        with open(config.DATA_ROOT / "target_issues.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(issues)
    print(
        f"Built {completed} scenarios; skipped {skipped}; "
        f"recorded {len(issues)} station issues"
    )


if __name__ == "__main__":
    build_targets(RunConfig())
