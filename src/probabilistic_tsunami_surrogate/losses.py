"""Loss used to train the dual-task probabilistic model."""

import torch


def multitask_student_t_nll(predictions, targets, hmax_weight=1.0, arrival_weight=1.0):
    """Returns the weighted mean NLL over all observed task/station values."""
    anchor = predictions["hmax"]["mu"]
    numerator = anchor.new_zeros(())
    denominator = anchor.new_zeros(())

    for task, weight in (
        ("hmax", hmax_weight),
        ("arrival", arrival_weight),
    ):
        if weight == 0:
            continue

        target = targets[task]
        observed = ~torch.isnan(target)
        head = predictions[task]
        location = head["mu"][observed]
        scale = head["sigma"].expand_as(target)[observed].clamp_min(1e-12)
        degrees_freedom = head["nu"].expand_as(target)[observed]
        values = target[observed]

        squared_residual = ((values - location) / scale).square()
        nll = (
            torch.lgamma(degrees_freedom / 2)
            - torch.lgamma((degrees_freedom + 1) / 2)
            + 0.5 * torch.log(degrees_freedom * torch.pi)
            + torch.log(scale)
            + 0.5
            * (degrees_freedom + 1)
            * torch.log1p(squared_residual / degrees_freedom)
        )
        numerator = numerator + weight * nll.sum()
        denominator = denominator + weight * observed.sum()

    return numerator / denominator.clamp_min(1)
