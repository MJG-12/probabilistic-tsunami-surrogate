import numpy as np

from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.data.preprocessing import discover_entries


def build_entries(config):
    """Discovers scenarios and saves their canonical order."""
    entries = discover_entries(config.DATA_ROOT)
    np.save(config.entries_path, entries, allow_pickle=True)
    print(f"Wrote {len(entries)} entries to {config.entries_path}")


if __name__ == "__main__":
    build_entries(RunConfig())
