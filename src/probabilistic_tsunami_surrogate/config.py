"""Fixed conventions and editable run settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """Stores local paths and settings for model training and evaluation."""

    DATA_ROOT: Path = Path("data/raw")
    SPLIT_DIR: Path = Path("data/splits")
    RUN_DIR: Path = Path("artifacts/model")
    VALIDATION_DIR: Path = Path("artifacts/data-validation")
    REFERENCE_STATS_DIR: Path = Path("data/stats")
    OVERWRITE_TARGETS: bool = False
    CHECK_STATION_FILES: bool = False
    MAX_ABSOLUTE_ELEVATION: float | None = None

    MAX_EPOCHS: int = 80
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 1e-3
    MICRO_VALIDATION_SIZE: float = 0.075
    PATIENCE: int = 5
    SOFT_KAPPA: float = 5.0
    PLATEAU_THRESHOLD: float = 4.5e-3
    PLATEAU_COOLDOWN: int = 1
    NUM_WORKERS: int | None = None
    DEVICE: str | None = None
    CHECKPOINT: Path | None = None

    @property
    def entries_path(self):
        """Returns the canonical scenario index path."""
        return self.DATA_ROOT / "entries.npy"

    @property
    def evaluation_path(self):
        """Returns the evaluation summary path."""
        return self.RUN_DIR / "evaluation.json"

    @property
    def report_dir(self):
        """Returns the directory for generated report assets."""
        return self.RUN_DIR / "report-assets"


GRID_SHAPE = (992, 2010)
N_STATIONS = 322
EXCLUDED_STATIONS = (28, 83, 152, 81, 48)
N_MODEL_STATIONS = N_STATIONS - len(EXCLUDED_STATIONS)
MIN_STATION_ROWS = 8986
ARRIVAL_THRESHOLD_METRES = 0.005
BASELINE_WINDOW_SECONDS = 20.0

TEST_SIZE = 0.15
N_CV_FOLDS = 5
RANDOM_STATE = 42
