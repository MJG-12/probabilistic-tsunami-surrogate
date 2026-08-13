"""PyTorch dataset for dual-task tsunami prediction."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from probabilistic_tsunami_surrogate.data.preprocessing import (
    safe_arcsinh,
    safe_log1p,
    slip_id_from_sid,
    station_keep_mask,
)


class MakranDataset(Dataset):
    """Loads η, bathymetry, wave height and arrival time for each scenario."""

    def __init__(
        self,
        root,
        indices,
        stats,
        channels=("eta",),
        stride=1,
        arrival_file="arrival_times.npy",
    ):
        self.root = Path(root)
        self.indices = np.asarray(indices, dtype=int)
        self.entries = np.load(self.root / "entries.npy", allow_pickle=True)[self.indices]
        self.channels = tuple(channels)
        self.stride = stride
        self.arrival_file = arrival_file
        (
            self.eta_center,
            self.eta_scale,
            self.hmax_mean,
            self.hmax_scale,
            self.arrival_mean,
            self.arrival_scale,
        ) = stats

        bathymetry = np.load(
            self.root / "additional_files" / "bathy.npy",
            mmap_mode="r",
            allow_pickle=False,
        )[::stride, ::stride].copy()
        bathymetry = torch.as_tensor(bathymetry, dtype=torch.float32)
        self.bathymetry = (bathymetry - bathymetry.mean()) / bathymetry.std()
        self.keep_stations = station_keep_mask()

    def __len__(self):
        return len(self.indices)

    def _load_grid(self, scenario_dir, channel):
        grid = np.load(
            scenario_dir / "external_files" / f"{channel}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        return torch.as_tensor(
            grid[:: self.stride, :: self.stride].astype(np.float32),
            dtype=torch.float32,
        )

    def __getitem__(self, item):
        magnitude, scenario_id = self.entries[item]
        scenario_dir = self.root / str(magnitude) / str(scenario_id)

        grids = [self._load_grid(scenario_dir, channel) for channel in self.channels]
        grids.append(self.bathymetry)
        inputs = torch.stack(grids)
        inputs[0] = (inputs[0] - self.eta_center) / self.eta_scale

        hmax = safe_arcsinh(np.load(scenario_dir / "hmax.npy"))[self.keep_stations]
        arrival = safe_log1p(np.load(scenario_dir / self.arrival_file))[
            self.keep_stations
        ]
        targets = {
            "hmax": torch.as_tensor(
                (hmax - self.hmax_mean) / self.hmax_scale,
                dtype=torch.float32,
            ),
            "arrival": torch.as_tensor(
                (arrival - self.arrival_mean) / self.arrival_scale,
                dtype=torch.float32,
            ),
        }
        metadata = {
            "idx": int(self.indices[item]),
            "slip_id": torch.tensor(
                slip_id_from_sid(float(magnitude), str(scenario_id)),
                dtype=torch.long,
            ),
            "mw": torch.tensor(float(magnitude), dtype=torch.float32),
        }
        return inputs, targets, metadata


def station_pixel_coordinates(stations_path, grid_shape, stride=1):
    """Maps retained station coordinates to model-input pixel coordinates."""
    height, width = grid_shape
    coordinates = np.load(stations_path).astype(np.float32)[station_keep_mask()]
    x_pixels = coordinates[:, 0] / stride
    y_pixels = (height - 1) - coordinates[:, 1] / stride
    pixels = np.stack([x_pixels, y_pixels], axis=1)
    return np.clip(pixels, [0, 0], [width - 1, height - 1]).astype(np.float32)
