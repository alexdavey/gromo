from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import Any, Iterator, Literal
from warnings import warn

import torch
import torch.utils.data
from torch import nn
from torchmetrics import Metric, classification

from gromo.containers.growing_container import GrowingContainer, GrowingModel
from gromo.modules.constant_module import ConstantModule
from gromo.modules.growing_module import GrowingModule, MergeGrowingModule
from gromo.utils.fixed_point import (
    FixedPointConfig,
    FixedPointUpdateResult,
    TensorMap,
    _solve_fixed_point,
)
from gromo.utils.utils import global_device


class AverageMeter(object):
    """Computes and stores an average"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the meter to initial state."""
        self.sum: torch.Tensor | None = None
        self.count = 0

    def update(self, val: torch.Tensor, n: int = 1):
        """
        Updates the average with a new value.

        Parameters
        ----------
        val : torch.Tensor
            The new value to include in the average.
        n : int, optional
            The number of samples that `val` represents. Default is 1.
        """
        if torch.isfinite(val).all():
            if self.sum is None:
                self.sum = val * n
            else:
                self.sum += val * n
            self.count += n

    def compute(self) -> torch.Tensor:
        """Returns the current average.

        Returns
        -------
        torch.Tensor
            The average of the values seen so far. Returns 0.0 if no values have been
            added.
        """
        if self.count == 0:
            return torch.tensor(0.0)
            # raise ValueError("AverageMeter has no values to compute average")
        else:
            assert self.sum is not None, (
                "Sum should not be None when count is greater than 0"
            )
            return self.sum / self.count


class DummyMetric(Metric):
    """A dummy metric that always returns 0.0."""

    def __init__(self):
        super().__init__()

    def update(self, *_, **__):
        """No-op for updating the metric."""
        return

    def compute(self) -> torch.Tensor:
        """Returns the computed metric value.

        Returns
        -------
        torch.Tensor
            Always returns a tensor with value 0.0 on the device of the metric.
        """
        return torch.tensor(0.0, device=self.device)


def enumerate_dataloader(
    dataloader: torch.utils.data.DataLoader,
    dataloader_seed: int | None = None,
    batch_limit: int | None = None,
    epochs: float | None = None,
) -> Generator[tuple[int, Any]]:
    """
    A generator that yields batches from a dataloader with an optional batch limit.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader to iterate over.
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    batch_limit : int | None, optional
        Maximum number of batches to yield after `epochs` epochs.
        Use -1 for no limit. Default is None.
    epochs : float | None, optional
        Proportion of the dataloader to iterate over.
        Is incompatible with non None `batch_limit`.

    Yields
    ------
    Generator[tuple[int, Any]]
        A generator yielding tuples of (batch_index, batch_data).

    Raises
    ------
    AttributeError
        If `dataloader_seed` is provided but the dataloader does not have a random
        number generator attribute.
    TypeError
        If `epochs` and `batch_limit` are both provided.
    """
    if (epochs is not None) and (batch_limit is not None):
        msg = f"Only one  of `epochs` and `batch_limit` can be provided, but got {epochs=} and {batch_limit=}"
        raise TypeError(msg)
    assert (epochs is None) or (epochs >= 0), "Epochs must be non-negative"
    assert (batch_limit is None) or (batch_limit == -1 or batch_limit >= 0), (
        "Batch limit must be -1 or non-negative"
    )
    if dataloader_seed is not None:
        if hasattr(dataloader, "generator") and isinstance(
            dataloader.generator, torch.Generator
        ):
            dataloader.generator.manual_seed(dataloader_seed)
        else:
            raise AttributeError(
                "The dataloader does not have a 'generator' attribute of type torch.Generator, "
                "so the seed cannot be set."
            )
    if batch_limit is None:
        if epochs is None:
            batch_limit = None
        else:
            batch_limit = int(len(dataloader) * epochs)
    elif batch_limit == -1:
        batch_limit = None
    for i, batch in enumerate(dataloader):
        if batch_limit is not None and i >= batch_limit:
            break
        yield i, batch


@torch.no_grad()
def evaluate_model(
    model: nn.Module | GrowingContainer | GrowingModel,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    use_extended_model: bool = False,
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    mask: dict | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """
    Evaluate the model on a dataloader.

    Parameters
    ----------
    model : nn.Module | GrowingContainer | GrowingModel
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The dataloader for evaluation data.
    loss_function : nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The loss function to use. Must have reduction="mean".
    use_extended_model : bool, optional
        Whether to use the extended model for evaluation. Default is False.
    metrics : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int | None, optional
        Maximum number of batches to evaluate. Use -1 for no limit. Default is None.
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    mask : dict | None, optional
        The mask to use for the extended model. Only used if `use_extended_model` is True.
        Default is None.
    device : torch.device, optional
        Device to use. Default is torch.device("cpu").

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, metrics_value).

    Raises
    ------
    TypeError
        If the model is not an instance of GrowingContainer or GrowingModel when
        `use_extended_model` is True.
    """
    assert (
        not isinstance(loss_function, nn.Module) or loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    # metrics meters
    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()
        metrics = metrics.to(device)

    # prediction function
    if use_extended_model:
        if isinstance(model, GrowingModel):
            predict_fn = lambda x: model.extended_forward(x, mask=mask)
        elif isinstance(model, GrowingContainer):
            predict_fn = lambda x: model.extended_forward(x, mask=mask)[0]
        else:
            raise TypeError(
                "Model must be an instance of GrowingModel or GrowingContainer when use_extended_model is True"
            )
    else:
        predict_fn = lambda x: model(x)

    model.eval()
    for _, (x, y) in enumerate_dataloader(
        dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        x, y = x.to(device), y.to(device)
        y_pred: torch.Tensor = predict_fn(x)
        loss = loss_function(y_pred, y)
        loss_meter.update(loss, x.size(0))
        metrics.update(y_pred, y)

    return loss_meter.compute().item(), metrics.compute().item()


def gradient_descent(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_function: nn.Module,
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device = torch.device("cpu"),
    scheduler_step_granularity: Literal["epoch", "batch"] = "epoch",
) -> tuple[float, float]:
    """
    Train the model on the train_dataloader using classic gradient descent.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for training data.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    scheduler : torch.optim.lr_scheduler.LRScheduler | None, optional
        Learning rate scheduler. Default is None.
    loss_function : nn.Module
        The loss function to use. Must have reduction="mean".
    metrics : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int | None, optional
        Maximum number of batches to train. Use -1 for no limit. Default is None.
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    device : torch.device, optional
        Device to use. Default is torch.device("cpu").
    scheduler_step_granularity : Literal["epoch", "batch"], optional
        Whether to step the scheduler after each epoch (`"epoch"`, default) or each mini-batch (`"batch"`).

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, aux_loss_function_value).
    """
    assert (
        not isinstance(loss_function, nn.Module) or loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    # metrics meters
    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()
        metrics = metrics.to(device)

    model.train()
    for i, (x, y) in enumerate_dataloader(
        train_dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_function(y_pred, y)
        assert loss.isnan().sum() == 0, (
            f"During training of {model}, loss is NaN: {loss}, sample index: {i / len(train_dataloader)}"
        )

        loss.backward()
        optimizer.step()

        # update metrics
        loss_meter.update(loss.detach(), x.size(0))
        metrics.update(y_pred.detach(), y)

        if scheduler is not None and scheduler_step_granularity == "batch":
            scheduler.step()

    if scheduler is not None and scheduler_step_granularity == "epoch":
        scheduler.step()

    return loss_meter.compute().item(), metrics.compute().item()


def _compute_statistics_pass(
    model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device = torch.device("cpu"),
    check_finite: bool = False,
) -> tuple[float, float]:
    """Run one complete statistics pass.

    This helper preserves the behavior of :func:`compute_statistics` while allowing
    the fixed-point workflow to reject non-finite losses before backward propagation.
    """
    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()
        metrics = metrics.to(device)

    model.init_computation()
    model.eval()
    for _, (x, y) in enumerate_dataloader(
        dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        if check_finite and not torch.isfinite(loss).all().item():
            raise FloatingPointError("statistics pass produced a non-finite loss")
        loss.backward()
        model.update_computation()
        loss_meter.update(loss.detach() / x.size(0), x.size(0))
        metrics.update(y_pred.detach(), y)

    return loss_meter.compute().item(), metrics.compute().item()


def compute_statistics(
    model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = nn.MSELoss(reduction="sum"),
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Compute tensor statistics of the model over a dataloader.

    Parameters
    ----------
    model : GrowingContainer
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use.
    loss_function : nn.Module
        The loss function to use. Must have reduction="sum".
    metrics : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int | None, optional
        The maximum number of batches to use. Default is None (no limit).
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    device : torch.device, optional
        The device to use. Default is torch.device("cpu").

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, metrics_value).
    """
    assert not isinstance(loss_function, nn.Module) or loss_function.reduction == "sum", (
        "The loss function should not be averaged over the batch"
    )
    return _compute_statistics_pass(
        model=model,
        dataloader=dataloader,
        loss_function=loss_function,
        metrics=metrics,
        batch_limit=batch_limit,
        dataloader_seed=dataloader_seed,
        device=device,
    )


def _raw_optimal_delta(module: GrowingModule) -> torch.nn.Module | None:
    """Return the stored delta module without invoking specialized properties."""
    key = (
        "_hidden_optimal_delta_layer"
        if isinstance(module, ConstantModule)
        else "optimal_delta_layer"
    )
    return module._modules.get(key)


def _has_pending_growth_update(model: GrowingContainer) -> bool:
    """Return whether the model already contains a proposed growth update."""
    for module in model.modules():
        if not isinstance(module, GrowingModule):
            continue
        if _raw_optimal_delta(module) is not None:
            return True
        if module.extended_input_layer is not None:
            return True
        if module.extended_output_layer is not None:
            return True
    return False


def _collect_optimal_deltas(
    model: GrowingContainer,
) -> tuple[TensorMap, dict[str, GrowingModule]]:
    """Collect nonempty active weight and bias deltas by stable module path."""
    updates: TensorMap = {}
    modules: dict[str, GrowingModule] = {}
    for name, module in model.named_modules():
        if not isinstance(module, GrowingModule) or isinstance(module, ConstantModule):
            continue
        delta_layer = _raw_optimal_delta(module)
        if delta_layer is None:
            continue
        found_parameter = False
        delta_weight = getattr(delta_layer, "weight", None)
        if isinstance(delta_weight, torch.Tensor) and delta_weight.numel() > 0:
            updates[f"{name}.weight"] = delta_weight.detach().clone()
            found_parameter = True
        delta_bias = getattr(delta_layer, "bias", None)
        if isinstance(delta_bias, torch.Tensor) and delta_bias.numel() > 0:
            updates[f"{name}.bias"] = delta_bias.detach().clone()
            found_parameter = True
        if found_parameter:
            modules[name] = module
    return updates, modules


def _capture_base_parameters(
    updates: Mapping[str, torch.Tensor],
    modules: Mapping[str, GrowingModule],
) -> TensorMap:
    """Clone the base parameters corresponding to a fixed-point update map."""
    base: TensorMap = {}
    for key, update in updates.items():
        module_name, parameter_name = key.rsplit(".", 1)
        module = modules[module_name]
        parameter = getattr(module.layer, parameter_name, None)
        if not isinstance(parameter, torch.Tensor):
            raise RuntimeError(f"{key} does not identify a tensor parameter")
        if parameter.shape != update.shape:
            raise RuntimeError(
                f"the update for {key} has shape {tuple(update.shape)}, "
                f"but the parameter has shape {tuple(parameter.shape)}"
            )
        base[key] = parameter.detach().clone()
    return base


@contextmanager
def _temporary_fixed_point_candidate(
    candidate: Mapping[str, torch.Tensor] | None,
    base: Mapping[str, torch.Tensor],
    modules: Mapping[str, GrowingModule],
) -> Iterator[None]:
    """Temporarily install ``W - D`` and always restore immutable base values."""
    if candidate is None:
        yield
        return

    try:
        with torch.no_grad():
            for key, update in candidate.items():
                module_name, parameter_name = key.rsplit(".", 1)
                parameter = getattr(modules[module_name].layer, parameter_name)
                parameter.copy_(base[key] - update)
        yield
    finally:
        with torch.no_grad():
            for key, value in base.items():
                module_name, parameter_name = key.rsplit(".", 1)
                parameter = getattr(modules[module_name].layer, parameter_name)
                parameter.copy_(value)


def _clear_growth_updates(model: GrowingContainer) -> None:
    """Clear partial proposals after a failed high-level growth computation."""
    for module in model.modules():
        if isinstance(module, GrowingModule):
            module.optimal_delta_layer = None
            module.extended_input_layer = None
            module.extended_output_layer = None
            module.parameter_update_decrease = None
            module.eigenvalues_extension = None
            module.delta_raw = None
        elif isinstance(module, MergeGrowingModule) and hasattr(
            module, "parameter_update_decrease"
        ):
            module.parameter_update_decrease = None


def _restore_parameter_gradients(
    gradients: Mapping[torch.nn.Parameter, torch.Tensor | None],
) -> None:
    """Restore parameter gradients captured before statistics computation."""
    for parameter, gradient in gradients.items():
        parameter.grad = None if gradient is None else gradient.detach().clone()


def compute_fixed_point_updates(
    model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    *,
    fixed_point_config: FixedPointConfig | None = None,
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device = torch.device("cpu"),
    optimal_update_kwargs: Mapping[str, Any] | None = None,
) -> FixedPointUpdateResult:
    """Compute a complete growth proposal, optionally at an endpoint fixed point.

    Fixed-point mode solves the joint equation ``D = T(D)`` over the optimal weight
    and bias deltas produced for all active growing layers. Gromo subtracts stored
    deltas when applying them, so each map evaluation recomputes all statistics at
    ``W - D``. Every iteration is therefore a complete dataloader pass.

    The utility restores base parameters, parameter gradients, model mode, and the
    available PyTorch and dataloader RNG states. The final optimal deltas and neuron
    extensions remain installed as proposals; this function never calls
    :meth:`GrowingContainer.apply_change`.

    Deterministic maps require a deterministic dataloader or ``dataloader_seed``.
    Persistent workers and random transforms using external RNGs cannot be fully
    controlled by the state snapshots performed here.

    Parameters
    ----------
    model: GrowingContainer
        Model whose active growing layers should receive proposals.
    dataloader: torch.utils.data.DataLoader
        Data used to recompute gradient statistics on every map evaluation.
    loss_function: nn.Module
        Loss with ``reduction="sum"``.
    fixed_point_config: FixedPointConfig | None
        Solver configuration, or ``None`` for the existing one-pass behavior.
    metrics: Metric | None
        Optional metric evaluated on the terminal statistics pass.
    batch_limit: int | None
        Optional number of batches per statistics pass.
    dataloader_seed: int | None
        Seed used to replay a dataloader with a generator.
    device: torch.device
        Device to which inputs and targets are moved.
    optimal_update_kwargs: Mapping[str, Any] | None
        Options forwarded unchanged to ``model.compute_optimal_updates``.

    Returns
    -------
    FixedPointUpdateResult
        Terminal loss, metric, and optional fixed-point diagnostics.

    Raises
    ------
    RuntimeError
        If a proposal is already pending or the active update set changes.
    ValueError
        If fixed-point mode is requested while optimal deltas are disabled.
    FloatingPointError
        If a map evaluation produces a non-finite loss or update.
    """
    assert not isinstance(loss_function, nn.Module) or loss_function.reduction == "sum", (
        "The loss function should not be averaged over the batch"
    )
    kwargs = dict(optimal_update_kwargs or {})
    if fixed_point_config is not None and kwargs.get("compute_delta", True) is False:
        raise ValueError("fixed-point mode requires compute_delta=True")
    if _has_pending_growth_update(model):
        raise RuntimeError(
            "compute_fixed_point_updates requires a model without a pending growth proposal"
        )

    original_training = model.training
    original_gradients = {
        parameter: (None if parameter.grad is None else parameter.grad.detach().clone())
        for parameter in model.parameters()
    }
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )
    generator = getattr(dataloader, "generator", None)
    generator_state = (
        generator.get_state() if isinstance(generator, torch.Generator) else None
    )

    def reset_iteration_rng() -> None:
        """Replay model-side and dataloader-side randomness for every map call."""
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        if generator_state is not None:
            generator.set_state(generator_state)

    try:
        if fixed_point_config is None:
            reset_iteration_rng()
            loss, metric = _compute_statistics_pass(
                model=model,
                dataloader=dataloader,
                loss_function=loss_function,
                metrics=metrics,
                batch_limit=batch_limit,
                dataloader_seed=dataloader_seed,
                device=device,
                check_finite=True,
            )
            model.compute_optimal_updates(**kwargs)
            result = FixedPointUpdateResult(loss=loss, metric=metric, fixed_point=None)
        else:
            base_parameters: TensorMap = {}
            target_modules: dict[str, GrowingModule] = {}
            latest_metric = 0.0

            def evaluate_map(candidate: TensorMap | None) -> tuple[TensorMap, float]:
                nonlocal base_parameters, target_modules, latest_metric
                reset_iteration_rng()
                with _temporary_fixed_point_candidate(
                    candidate,
                    base_parameters,
                    target_modules,
                ):
                    loss, latest_metric = _compute_statistics_pass(
                        model=model,
                        dataloader=dataloader,
                        loss_function=loss_function,
                        metrics=metrics,
                        batch_limit=batch_limit,
                        dataloader_seed=dataloader_seed,
                        device=device,
                        check_finite=True,
                    )
                    model.compute_optimal_updates(**kwargs)
                    updates, discovered_modules = _collect_optimal_deltas(model)
                    if candidate is None:
                        target_modules = discovered_modules
                        base_parameters = _capture_base_parameters(
                            updates,
                            target_modules,
                        )
                    return updates, loss

            solution = _solve_fixed_point(evaluate_map, fixed_point_config)
            terminal = solution.result.history[-1]
            result = FixedPointUpdateResult(
                loss=terminal.loss,
                metric=latest_metric,
                fixed_point=solution.result,
            )

        model.reset_computation()
        return result
    except Exception:
        try:
            model.reset_computation()
        except (
            AssertionError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as reset_error:
            warn(
                f"Failed to reset growth statistics after an error: {reset_error}",
                RuntimeWarning,
                stacklevel=2,
            )
        _clear_growth_updates(model)
        raise
    finally:
        model.train(original_training)
        _restore_parameter_gradients(original_gradients)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        if generator_state is not None:
            generator.set_state(generator_state)


# backward compatibility
# I could not keep it in utils.py because of circular imports,
# with `global_device` being defined in utils.py and used
# in `growing_container.py`
def evaluate_extended_dataset(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    mask: dict | None = None,
) -> tuple[float, float]:
    """Evaluate extended network on dataset

    Parameters
    ----------
    model : nn.Module
        network to evaluate
    dataloader : torch.utils.data.DataLoader
        dataloader containing the data
    loss_fn : Callable
        loss function for bottleneck calculation
    mask : dict | None, optional
        extension mask for specific nodes and edges, by default None
        example: mask["edges"] for edges and mask["nodes"] for nodes

    Returns
    -------
    tuple[float, float]
        accuracy and loss
    """
    device = global_device()
    _, y = next(iter(dataloader))
    if y.dim() == 1 and model.out_features > 1:
        nb_classes = model.out_features
    else:
        nb_classes = None
    metric = None
    if nb_classes is not None:
        metric = classification.MulticlassAccuracy(model.out_features, average="micro")
    loss, accuracy = evaluate_model(
        model,
        dataloader,
        loss_fn,
        metrics=metric,
        device=device,
        use_extended_model=True,
        mask=mask,
    )
    if metric is None:
        accuracy = -1
    return accuracy, loss


def evaluate_dataset(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: Callable
) -> tuple[float, float]:
    """Evaluate network on dataset

    Parameters
    ----------
    model : nn.Module
        network to evaluate
    dataloader : torch.utils.data.DataLoader
        dataloader containing the data
    loss_fn : Callable
        loss function for bottleneck calculation

    Returns
    -------
    tuple[float, float]
        accuracy and loss
    """
    device = global_device()
    _, y = next(iter(dataloader))
    if y.dim() == 1 and model.out_features > 1:
        nb_classes = model.out_features
    else:
        nb_classes = None
    metric = None
    if nb_classes is not None:
        metric = classification.MulticlassAccuracy(model.out_features, average="micro")
    loss, accuracy = evaluate_model(
        model, dataloader, loss_fn, metrics=metric, device=device
    )
    if metric is None:
        accuracy = -1
    return accuracy, loss
