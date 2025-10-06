from typing import Any, Callable

import numpy as np
import torch


__global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def global_device() -> torch.device:
    """Get global device for whole codebase

    Returns
    -------
    torch.device
        global device
    """
    global __global_device
    return __global_device


def get_correct_device(self, device: torch.device | str | None) -> torch.device:
    """Get the correct device based on precedence order
    Precedence works as follows:
        argument > global_device

    Parameters
    ----------
    device : torch.device | str | None
        chosen device argument, leave empty to use global

    Returns
    -------
    torch.device
        selected correct device
    """
    device = torch.device(device if device is not None else global_device())
    return device


def line_search(
    cost_fn: Callable, return_history: bool = False
) -> tuple[float, float] | tuple[list, list]:
    """Line search for black-box convex function

    Parameters
    ----------
    cost_fn : Callable
        black-box convex function
    return_history : bool, optional
        return full loss history, by default False

    Returns
    -------
    tuple[float, float] | tuple[list, list]
        return minima and min value
        if return_history is True return instead tested parameters and loss history
    """
    losses = []
    n_points = 100
    f_min = 1e-6
    f_max = 1
    f_test = np.concatenate(
        [np.zeros(1), np.logspace(np.log10(f_min), np.log10(f_max), n_points)]
    )

    decrease = True
    min_loss = np.inf
    f_full = np.array([])

    while decrease:
        for factor in f_test:
            loss = cost_fn(factor)
            losses.append(loss)

        f_full = np.concatenate([f_full, f_test])

        new_min = np.min(losses)
        decrease = new_min < min_loss
        min_loss = new_min

        f_min = f_max
        f_max = f_max * 10
        f_test = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    factor = f_full[np.argmin(losses)]
    min_loss = np.min(losses)

    if return_history:
        return list(f_full), losses
    else:
        return factor, min_loss
