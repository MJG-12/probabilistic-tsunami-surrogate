from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from probabilistic_tsunami_surrogate.config import (
    N_CV_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
)


def make_train_test_split(entries, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Creates one magnitude-stratified train/test split."""
    labels = np.asarray([magnitude for magnitude, _ in entries])
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    return next(splitter.split(np.zeros(len(entries)), labels))


def make_cross_validation_folds(
    entries,
    train_idx,
    n_splits=N_CV_FOLDS,
    random_state=RANDOM_STATE,
):
    """Creates magnitude-stratified folds inside the training set."""
    labels = np.asarray([magnitude for magnitude, _ in entries])
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    return [
        (train_idx[fold_train], train_idx[fold_validation])
        for fold_train, fold_validation in splitter.split(
            np.zeros(len(train_idx)),
            labels[train_idx],
        )
    ]


def make_micro_validation_split(
    train_idx,
    entries,
    validation_size=0.075,
    random_state=RANDOM_STATE,
):
    """Carves a magnitude-stratified validation set out of training indices."""
    train_idx = np.asarray(train_idx, dtype=int)
    labels = np.asarray([magnitude for magnitude, _ in entries])[train_idx]
    n_validation = max(1, int(round(len(train_idx) * validation_size)))
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=n_validation,
        random_state=random_state,
    )
    core, validation = next(splitter.split(np.zeros(len(train_idx)), labels))
    return train_idx[core], train_idx[validation]


def save_splits(output_dir, train_idx, test_idx, cv_splits):
    """Writes train, test and cross-validation index arrays."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "test_idx.npy", test_idx)
    for fold, (fold_train, fold_validation) in enumerate(cv_splits):
        np.save(output_dir / f"cv{fold}_train.npy", fold_train)
        np.save(output_dir / f"cv{fold}_val.npy", fold_validation)
