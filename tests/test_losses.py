import pytest

torch = pytest.importorskip("torch")

from probabilistic_tsunami_surrogate.losses import multitask_student_t_nll  # noqa: E402


def test_multitask_loss_masks_missing_targets_and_backpropagates():
    location = torch.zeros((1, 3), requires_grad=True)
    predictions = {
        task: {
            "mu": location,
            "sigma": torch.ones((1, 3)),
            "nu": torch.full((1, 1), 4.0),
        }
        for task in ("hmax", "arrival")
    }
    targets = {
        "hmax": torch.tensor([[0.0, float("nan"), 1.0]]),
        "arrival": torch.tensor([[1.0, 0.0, float("nan")]]),
    }

    loss = multitask_student_t_nll(predictions, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(location.grad).all()
