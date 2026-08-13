"""Station-aware ResNet for dual-task tsunami prediction."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from probabilistic_tsunami_surrogate.config import N_MODEL_STATIONS


def fourier_features(coordinates, width, height, n_frequencies=16):
    """Encodes aspect-corrected Fourier features of station coordinates."""
    frequencies = (
        2.0
        ** torch.arange(
            n_frequencies,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
    ) * math.pi
    scale_x = (width - 1.0) / max(height - 1.0, 1)
    x = coordinates[:, 0:1] * (frequencies * scale_x)
    y = coordinates[:, 1:2] * frequencies
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


class ResNet18Features(nn.Module):
    """Returns the final spatial map and pooled features from ResNet-18."""

    def __init__(self, in_channels=2):
        super().__init__()
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        features = self.stem(inputs)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        feature_map = self.layer4(features)
        global_features = self.pool(feature_map).flatten(1)
        return feature_map, global_features


class StationHead(nn.Module):
    """Predicts location and log scale for both tasks at each station."""

    def __init__(self, in_features, hidden_features=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_features, 4),
        )

    def forward(self, features, n_stations):
        output = self.net(features).view(-1, n_stations, 4)
        return output.unbind(dim=-1)


class StationAwareResNet(nn.Module):
    """Predicts Student-t parameters for wave height and arrival time."""

    context_pool_sizes = (1, 3, 5)

    def __init__(self, in_channels=2, n_stations=N_MODEL_STATIONS):
        super().__init__()
        self.n_stations = n_stations
        self.backbone = ResNet18Features(in_channels)
        self.global_dropout = nn.Dropout(0.15)

        self.slip_embed = nn.Embedding(2, 4)
        self.alpha_h = nn.Parameter(torch.tensor(0.1))
        self.alpha_t = nn.Parameter(torch.tensor(0.01))

        self.station_emb = nn.Embedding(n_stations, 8)
        self.id_dropout = nn.Dropout(0.1)
        self.register_buffer(
            "station_xy_pixels",
            torch.empty(n_stations, 2),
            persistent=False,
        )
        self._has_station_coordinates = False

        global_features = 2 * (512 + 4)
        local_features = 512 * len(self.context_pool_sizes)
        coordinate_features = 2 + 4 * 16
        self.head = StationHead(
            global_features + local_features + coordinate_features + 8
        )

        self.station_bias_h = nn.Parameter(torch.zeros(n_stations))
        self.station_bias_t = nn.Parameter(torch.zeros(n_stations))
        self.mu_regime_h = nn.Parameter(torch.zeros(2))
        self.mu_regime_t = nn.Parameter(torch.zeros(2))

        self.log_scale_station_h = nn.Parameter(torch.zeros(n_stations))
        self.log_scale_station_t = nn.Parameter(torch.zeros(n_stations))
        self.log_scale_regime_h = nn.Parameter(torch.zeros(2))
        self.log_scale_regime_t = nn.Parameter(torch.zeros(2))

        self.nu_regime_raw_h = nn.Parameter(
            self._raw_nu((8.0, 4.5), minimum=3.0, maximum=15.0)
        )
        self.nu_regime_raw_t = nn.Parameter(
            self._raw_nu((3.0, 2.4), minimum=1.8, maximum=15.0)
        )

    @staticmethod
    def _raw_nu(values, minimum, maximum):
        values = torch.as_tensor(values, dtype=torch.float32)
        fraction = (values - minimum) / (maximum - minimum)
        return torch.logit(fraction)

    @staticmethod
    def _bounded(values, limit):
        return limit * torch.tanh(values)

    @torch.no_grad()
    def set_station_coordinates(self, coordinates):
        """Sets retained-station coordinates in model-input pixels."""
        if coordinates.shape != (self.n_stations, 2):
            raise ValueError(
                f"expected coordinates with shape {(self.n_stations, 2)}"
            )
        self.station_xy_pixels.copy_(coordinates)
        self._has_station_coordinates = True

    def _sampling_grid(self, batch_size, height, width):
        if not self._has_station_coordinates:
            raise RuntimeError("call set_station_coordinates before forward")
        x = 2.0 * self.station_xy_pixels[:, 0] / max(width - 1, 1) - 1.0
        y = 2.0 * self.station_xy_pixels[:, 1] / max(height - 1, 1) - 1.0
        grid = torch.stack([x, y], dim=1).view(1, self.n_stations, 1, 2)
        return grid.expand(batch_size, -1, -1, -1)

    def forward(self, inputs, slip_id):
        batch_size, _, height, width = inputs.shape
        feature_map, global_features = self.backbone(inputs)
        global_features = self.global_dropout(global_features)

        slip_features = self.slip_embed(slip_id.long())
        global_h = torch.cat(
            [global_features, self.alpha_h * slip_features], dim=1
        )
        global_t = torch.cat(
            [global_features, self.alpha_t * slip_features], dim=1
        )
        global_features = torch.cat([global_h, global_t], dim=1)
        global_features = global_features.unsqueeze(1).expand(
            batch_size, self.n_stations, -1
        )

        station_ids = torch.arange(self.n_stations, device=inputs.device)
        station_features = self.id_dropout(self.station_emb(station_ids))
        station_features = station_features.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        sampling_grid = self._sampling_grid(batch_size, height, width)
        normalized_coordinates = sampling_grid[0, :, 0]
        coordinate_features = fourier_features(
            normalized_coordinates,
            width,
            height,
        )
        coordinate_features = torch.cat(
            [normalized_coordinates, coordinate_features], dim=1
        )
        coordinate_features = coordinate_features.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        local_features = []
        for pool_size in self.context_pool_sizes:
            pooled = (
                F.avg_pool2d(
                    feature_map,
                    kernel_size=pool_size,
                    stride=1,
                    padding=pool_size // 2,
                    count_include_pad=False,
                )
                if pool_size > 1
                else feature_map
            )
            sampled = F.grid_sample(
                pooled,
                sampling_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            local_features.append(sampled.squeeze(-1).transpose(1, 2))

        features = torch.cat(
            [
                global_features,
                station_features,
                torch.cat(local_features, dim=2),
                coordinate_features,
            ],
            dim=2,
        )
        mu_h, log_scale_h, mu_t, log_scale_t = self.head(
            features.reshape(batch_size * self.n_stations, -1),
            self.n_stations,
        )

        mu_h = mu_h + self.station_bias_h
        mu_t = mu_t + self.station_bias_t
        mu_h = mu_h + self.mu_regime_h[slip_id].unsqueeze(1)
        mu_t = mu_t + self.mu_regime_t[slip_id].unsqueeze(1)

        log_scale_h = (
            log_scale_h
            + self._bounded(self.log_scale_station_h, 1.6)
            + self._bounded(self.log_scale_regime_h[slip_id].unsqueeze(1), 1.2)
        )
        log_scale_t = (
            log_scale_t
            + self._bounded(self.log_scale_station_t, 0.8)
            + self._bounded(self.log_scale_regime_t[slip_id].unsqueeze(1), 0.8)
        )
        log_scale_h = log_scale_h.clamp(min=-5.0, max=0.9)
        log_scale_t = log_scale_t.clamp(min=-5.0, max=0.35)

        nu_h = 3.0 + torch.sigmoid(
            self.nu_regime_raw_h[slip_id].unsqueeze(1)
        ) * 12.0
        nu_t = 1.8 + torch.sigmoid(
            self.nu_regime_raw_t[slip_id].unsqueeze(1)
        ) * 13.2

        return {
            "hmax": {
                "mu": mu_h,
                "sigma": log_scale_h.exp() + 1e-6,
                "log_sigma": log_scale_h,
                "nu": nu_h,
                "dist": "student_t",
            },
            "arrival": {
                "mu": mu_t,
                "sigma": log_scale_t.exp() + 1e-6,
                "log_sigma": log_scale_t,
                "nu": nu_t,
                "dist": "student_t",
            },
        }
