import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from probabilistic_tsunami_surrogate.model import (  # noqa: E402
    StationAwareResNet,
    fourier_features,
)
from probabilistic_tsunami_surrogate.training import (  # noqa: E402
    initialize_regime_offsets,
    make_optimizer,
)


def test_model_output_shapes_and_student_t_bounds():
    model = StationAwareResNet(n_stations=3)
    model.set_station_coordinates(
        torch.tensor([[0.0, 0.0], [31.0, 31.0], [63.0, 63.0]])
    )
    predictions = model(torch.randn(2, 2, 64, 64), torch.tensor([0, 1]))

    for task in ("hmax", "arrival"):
        assert predictions[task]["mu"].shape == (2, 3)
        assert predictions[task]["sigma"].shape == (2, 3)
        assert (predictions[task]["sigma"] > 0).all()
    assert (predictions["hmax"]["nu"] > 3.0).all()
    assert (predictions["arrival"]["nu"] > 1.8).all()


def test_fourier_features_correct_for_grid_aspect_ratio():
    width, height = 640, 320
    one_pixel_x = 2.0 / (width - 1)
    one_pixel_y = 2.0 / (height - 1)
    coordinates = torch.tensor(
        [[0.0, 0.0], [one_pixel_x, 0.0], [0.0, one_pixel_y]]
    )

    features = fourier_features(coordinates, width, height, n_frequencies=1)

    torch.testing.assert_close(features[1, :2], features[2, 2:])


def test_regime_offsets_and_optimizer_groups():
    model = StationAwareResNet(n_stations=3)
    initialize_regime_offsets(model)
    torch.testing.assert_close(model.log_scale_regime_h, torch.tensor([0.05, 0.15]))
    torch.testing.assert_close(model.log_scale_regime_t, torch.tensor([0.10, 0.06]))
    torch.testing.assert_close(model.mu_regime_h, torch.tensor([0.04, 0.24]))
    torch.testing.assert_close(model.mu_regime_t, torch.tensor([-0.05, -0.04]))

    optimizer = make_optimizer(model, learning_rate=1e-3)
    assert [group["lr"] for group in optimizer.param_groups] == [
        1e-3,
        1e-3,
        5e-4,
        2e-4,
    ]
