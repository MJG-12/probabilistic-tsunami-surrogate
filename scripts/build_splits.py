import numpy as np

from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.data.splits import (
    make_cross_validation_folds,
    make_train_test_split,
    save_splits,
)


def build_splits(config):
    """Creates and saves one train/validation/test split."""
    entries = np.load(config.entries_path, allow_pickle=True)
    train_idx, test_idx = make_train_test_split(entries)
    cv_splits = make_cross_validation_folds(entries, train_idx)
    save_splits(config.SPLIT_DIR, train_idx, test_idx, cv_splits)
    print(f"train={len(train_idx)} test={len(test_idx)}")


if __name__ == "__main__":
    build_splits(RunConfig())
