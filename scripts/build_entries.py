import numpy as np

from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.data.preprocessing import prepare_entries


def build_entries(config):
    """Filters, deduplicates and saves the scenario catalogue."""
    entries = prepare_entries(
        config.DATA_ROOT,
        max_absolute_elevation=config.MAX_ABSOLUTE_ELEVATION,
    )
    if not entries:
        raise ValueError("no usable scenarios")
    np.save(config.entries_path, entries, allow_pickle=True)
    print(f"Wrote {len(entries)} entries to {config.entries_path}")


if __name__ == "__main__":
    build_entries(RunConfig())
