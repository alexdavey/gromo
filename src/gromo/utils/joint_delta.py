from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as functional
from torch import nn

from gromo.containers.growing_block import GrowingBlock
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.training_utils import enumerate_dataloader


SupportedJointLayer = LinearGrowingModule | Conv2dGrowingModule


@dataclass
class JointBottleneckGapResult:
    """Measurement-only comparison between analytic and joint bottleneck residuals."""

    current_layer_name: str
    previous_layer_name: str
    old_directional_residual: float
    joint_directional_residual: float
    directional_gap: float
    relative_directional_gap: float
    old_bottleneck_norm_sq: float
    joint_bottleneck_norm_sq: float
    residual_gap: float
    normalized_gap: float
    normalized_gap_percent: float
    relative_residual_gap: float
    initial_joint_directional_loss: float
    final_joint_directional_loss: float
    joint_measurement_scale: float
    batches: int


def _normalize_frobenius(
    tensor: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(tensor)
    return tensor / norm.clamp_min(eps), norm


def _new_delta_layer(layer: SupportedJointLayer, init_scale: float) -> nn.Module:
    weight = init_scale * torch.randn_like(layer.weight)
    bias = init_scale * torch.randn_like(layer.bias) if layer.bias is not None else None
    return layer.layer_of_tensor(weight, bias)


def _post_layer_jvp(
    layer: SupportedJointLayer,
    pre_activity: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    if isinstance(layer.post_layer_function, nn.Identity):
        return tangent

    def post_fn(x: torch.Tensor) -> torch.Tensor:
        output = layer.post_layer_function(x)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "post_layer_function must return a Tensor for joint bottleneck "
                "measurement."
            )
        return output

    _, tangent_output = torch.autograd.functional.jvp(
        post_fn,
        (pre_activity.detach(),),
        (tangent,),
        create_graph=True,
        strict=False,
    )
    return tangent_output


def _compute_batch_quantities(
    model: nn.Module,
    previous_layer: SupportedJointLayer,
    current_layer: SupportedJointLayer,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_(True)
    y_pred = model(x)
    task_loss = loss_function(y_pred, y)
    task_loss.backward()

    gradient = current_layer.pre_activity.grad
    if gradient is None:
        raise RuntimeError(
            "current_layer.pre_activity.grad is None. "
            "The current layer pre-activity must receive gradients."
        )

    return (
        previous_layer.input.detach(),
        previous_layer.pre_activity.detach(),
        current_layer.input.detach(),
        gradient.detach(),
    )


def _current_layer_signal_from_previous(
    current_layer: SupportedJointLayer,
    previous_activity_signal: torch.Tensor,
) -> torch.Tensor:
    if isinstance(current_layer, LinearGrowingModule):
        return functional.linear(
            previous_activity_signal,
            current_layer.layer.weight.detach(),
            bias=None,
        )
    if isinstance(current_layer, Conv2dGrowingModule):
        return functional.conv2d(
            previous_activity_signal,
            current_layer.layer.weight.detach(),
            bias=None,
            stride=current_layer.stride,
            padding=current_layer.padding,
            dilation=current_layer.dilation,
        )
    raise TypeError(f"Unsupported current layer type: {type(current_layer)}.")


def _joint_activity_signal(
    previous_layer: SupportedJointLayer,
    current_layer: SupportedJointLayer,
    previous_delta_layer: nn.Module,
    current_delta_layer: nn.Module,
    previous_input: torch.Tensor,
    previous_pre_activity: torch.Tensor,
    current_input: torch.Tensor,
) -> torch.Tensor:
    previous_pre_activity_signal = previous_delta_layer(previous_input)
    previous_activity_signal = _post_layer_jvp(
        previous_layer,
        previous_pre_activity,
        previous_pre_activity_signal,
    )
    current_signal_from_previous = _current_layer_signal_from_previous(
        current_layer,
        previous_activity_signal,
    )
    current_signal = current_delta_layer(current_input)
    return current_signal + current_signal_from_previous


def _directional_loss_for_batch(
    model: nn.Module,
    previous_layer: SupportedJointLayer,
    current_layer: SupportedJointLayer,
    previous_delta_layer: nn.Module,
    current_delta_layer: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eps: float,
) -> torch.Tensor:
    (
        previous_input,
        previous_pre_activity,
        current_input,
        gradient,
    ) = _compute_batch_quantities(
        model,
        previous_layer,
        current_layer,
        x,
        y,
        loss_function,
    )

    gradient_direction, gradient_norm = _normalize_frobenius(gradient, eps)
    if gradient_norm <= eps:
        raise ValueError(
            "Cannot measure a joint bottleneck gap for a zero loss-gradient target."
        )

    joint_signal = _joint_activity_signal(
        previous_layer,
        current_layer,
        previous_delta_layer,
        current_delta_layer,
        previous_input,
        previous_pre_activity,
        current_input,
    )
    joint_direction, _ = _normalize_frobenius(joint_signal, eps)
    return ((gradient_direction - joint_direction) ** 2).sum()


def _mean_joint_directional_loss(
    model: nn.Module,
    previous_layer: SupportedJointLayer,
    current_layer: SupportedJointLayer,
    previous_delta_layer: nn.Module,
    current_delta_layer: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eps: float,
    device: torch.device,
    batch_limit: int | None,
    dataloader_seed: int | None,
) -> torch.Tensor:
    losses = []
    for _, (x, y) in enumerate_dataloader(
        dataloader,
        dataloader_seed=dataloader_seed,
        batch_limit=batch_limit,
    ):
        losses.append(
            _directional_loss_for_batch(
                model,
                previous_layer,
                current_layer,
                previous_delta_layer,
                current_delta_layer,
                x.to(device),
                y.to(device),
                loss_function,
                eps,
            )
        )
    if not losses:
        raise ValueError("No batches were available for joint bottleneck measurement.")
    return torch.stack(losses).mean()


def _accumulate_gap_metrics(
    model: nn.Module,
    previous_layer: SupportedJointLayer,
    current_layer: SupportedJointLayer,
    previous_delta_layer: nn.Module,
    current_delta_layer: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eps: float,
    device: torch.device,
    batch_limit: int | None,
    dataloader_seed: int | None,
) -> dict[str, float | int]:
    old_directional_residual = 0.0
    joint_directional_residual = 0.0
    old_bottleneck_norm_sq = 0.0
    joint_bottleneck_norm_sq = 0.0
    joint_measurement_scale = 0.0
    batches = 0

    for _, (x, y) in enumerate_dataloader(
        dataloader,
        dataloader_seed=dataloader_seed,
        batch_limit=batch_limit,
    ):
        (
            previous_input,
            previous_pre_activity,
            current_input,
            gradient,
        ) = _compute_batch_quantities(
            model,
            previous_layer,
            current_layer,
            x.to(device),
            y.to(device),
            loss_function,
        )

        gradient_direction, gradient_norm = _normalize_frobenius(gradient, eps)
        if gradient_norm <= eps:
            raise ValueError(
                "Cannot measure a joint bottleneck gap for a zero loss-gradient target."
            )

        old_signal = current_layer.optimal_delta_layer(current_input).detach()
        old_direction, _ = _normalize_frobenius(old_signal, eps)

        joint_signal = _joint_activity_signal(
            previous_layer,
            current_layer,
            previous_delta_layer,
            current_delta_layer,
            previous_input,
            previous_pre_activity,
            current_input,
        ).detach()
        joint_direction, _ = _normalize_frobenius(joint_signal, eps)
        alpha_joint = torch.clamp(torch.sum(gradient * joint_direction), min=0.0)

        old_directional_residual += float(
            ((gradient_direction - old_direction) ** 2).sum().detach()
        )
        joint_directional_residual += float(
            ((gradient_direction - joint_direction) ** 2).sum().detach()
        )
        old_bottleneck_norm_sq += float(((gradient - old_signal) ** 2).sum().detach())
        joint_bottleneck_norm_sq += float(
            ((gradient - alpha_joint * joint_direction) ** 2).sum().detach()
        )
        joint_measurement_scale += float(alpha_joint.detach())
        batches += 1

    if batches == 0:
        raise ValueError("No batches were available for joint bottleneck measurement.")

    return {
        "old_directional_residual": old_directional_residual / batches,
        "joint_directional_residual": joint_directional_residual / batches,
        "old_bottleneck_norm_sq": old_bottleneck_norm_sq / batches,
        "joint_bottleneck_norm_sq": joint_bottleneck_norm_sq / batches,
        "joint_measurement_scale": joint_measurement_scale / batches,
        "batches": batches,
    }


def resolve_joint_bottleneck_layers(
    selected_layer: nn.Module,
) -> tuple[SupportedJointLayer, SupportedJointLayer]:
    """Resolve a selected growable layer/block to a previous/current layer pair."""
    if isinstance(selected_layer, GrowingBlock):
        previous_layer = selected_layer.first_layer
        current_layer = selected_layer.second_layer
    else:
        current_layer = selected_layer
        previous_layer = getattr(current_layer, "previous_module", None)

    if not isinstance(previous_layer, (LinearGrowingModule, Conv2dGrowingModule)):
        raise TypeError(
            "selected layer must resolve to a LinearGrowingModule or "
            "Conv2dGrowingModule previous layer."
        )
    if not isinstance(current_layer, (LinearGrowingModule, Conv2dGrowingModule)):
        raise TypeError(
            "selected layer must resolve to a LinearGrowingModule or "
            "Conv2dGrowingModule current layer."
        )
    return previous_layer, current_layer


def _validate_supported_layer_pair(
    previous_layer: SupportedJointLayer,
    current_layer: SupportedJointLayer,
) -> None:
    if isinstance(previous_layer, LinearGrowingModule) and isinstance(
        current_layer, LinearGrowingModule
    ):
        return
    if isinstance(previous_layer, Conv2dGrowingModule) and isinstance(
        current_layer, Conv2dGrowingModule
    ):
        return
    raise TypeError(
        "Only LinearGrowingModule->LinearGrowingModule and "
        "Conv2dGrowingModule->Conv2dGrowingModule pairs are supported."
    )


def _set_requires_grad(
    model: nn.Module,
    requires_grad: bool,
) -> dict[nn.Parameter, bool]:
    original_requires_grad = {}
    for parameter in model.parameters():
        original_requires_grad[parameter] = parameter.requires_grad
        parameter.requires_grad_(requires_grad)
    return original_requires_grad


def _restore_requires_grad(original_requires_grad: dict[nn.Parameter, bool]) -> None:
    for parameter, requires_grad in original_requires_grad.items():
        parameter.requires_grad_(requires_grad)


def compute_joint_bottleneck_gap(
    model: nn.Module,
    previous_layer: SupportedJointLayer,
    current_layer: SupportedJointLayer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    lr: float = 3e-4,
    steps: int = 1000,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    eps: float = 1e-12,
    init_scale: float = 1e-2,
    device: torch.device | None = None,
) -> JointBottleneckGapResult:
    """Measure how much a joint delta can reduce the selected layer bottleneck."""
    _validate_supported_layer_pair(previous_layer, current_layer)
    if current_layer.previous_module is not previous_layer:
        raise ValueError("previous_layer must be current_layer.previous_module.")
    if current_layer.optimal_delta_layer is None:
        raise ValueError("current_layer.optimal_delta_layer must be computed first.")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}.")
    if init_scale <= 0:
        raise ValueError(f"init_scale must be positive, got {init_scale}.")

    if device is None:
        device = current_layer.device

    previous_delta_layer = _new_delta_layer(previous_layer, init_scale).to(device)
    current_delta_layer = _new_delta_layer(current_layer, init_scale).to(device)
    optimizer = torch.optim.Adam(
        list(previous_delta_layer.parameters()) + list(current_delta_layer.parameters()),
        lr=lr,
    )

    previous_store_input = previous_layer.store_input
    previous_store_pre_activity = previous_layer.store_pre_activity
    current_store_input = current_layer.store_input
    current_store_pre_activity = current_layer.store_pre_activity
    model_training = model.training
    original_requires_grad: dict[nn.Parameter, bool] = {}

    try:
        original_requires_grad = _set_requires_grad(model, False)
        previous_layer.store_input = True
        previous_layer.store_pre_activity = True
        current_layer.store_input = True
        current_layer.store_pre_activity = True
        model.eval()

        with torch.enable_grad():
            initial_loss = float(
                _mean_joint_directional_loss(
                    model,
                    previous_layer,
                    current_layer,
                    previous_delta_layer,
                    current_delta_layer,
                    dataloader,
                    loss_function,
                    eps,
                    device,
                    batch_limit,
                    dataloader_seed,
                ).detach()
            )

            for _ in range(steps):
                optimizer.zero_grad()
                loss = _mean_joint_directional_loss(
                    model,
                    previous_layer,
                    current_layer,
                    previous_delta_layer,
                    current_delta_layer,
                    dataloader,
                    loss_function,
                    eps,
                    device,
                    batch_limit,
                    dataloader_seed,
                )
                loss.backward()
                optimizer.step()

            final_loss = float(
                _mean_joint_directional_loss(
                    model,
                    previous_layer,
                    current_layer,
                    previous_delta_layer,
                    current_delta_layer,
                    dataloader,
                    loss_function,
                    eps,
                    device,
                    batch_limit,
                    dataloader_seed,
                ).detach()
            )

            metrics = _accumulate_gap_metrics(
                model,
                previous_layer,
                current_layer,
                previous_delta_layer,
                current_delta_layer,
                dataloader,
                loss_function,
                eps,
                device,
                batch_limit,
                dataloader_seed,
            )
    finally:
        previous_layer.store_input = previous_store_input
        previous_layer.store_pre_activity = previous_store_pre_activity
        current_layer.store_input = current_store_input
        current_layer.store_pre_activity = current_store_pre_activity
        _restore_requires_grad(original_requires_grad)
        model.train(model_training)
        model.zero_grad(set_to_none=True)

    old_directional = float(metrics["old_directional_residual"])
    joint_directional = float(metrics["joint_directional_residual"])
    old_norm_sq = float(metrics["old_bottleneck_norm_sq"])
    joint_norm_sq = float(metrics["joint_bottleneck_norm_sq"])
    directional_gap = old_directional - joint_directional
    residual_gap = old_norm_sq - joint_norm_sq
    normalized_gap = residual_gap / max(old_norm_sq, eps)

    return JointBottleneckGapResult(
        current_layer_name=current_layer.name,
        previous_layer_name=previous_layer.name,
        old_directional_residual=old_directional,
        joint_directional_residual=joint_directional,
        directional_gap=directional_gap,
        relative_directional_gap=directional_gap / max(abs(old_directional), eps),
        old_bottleneck_norm_sq=old_norm_sq,
        joint_bottleneck_norm_sq=joint_norm_sq,
        residual_gap=residual_gap,
        normalized_gap=normalized_gap,
        normalized_gap_percent=100.0 * normalized_gap,
        relative_residual_gap=normalized_gap,
        initial_joint_directional_loss=initial_loss,
        final_joint_directional_loss=final_loss,
        joint_measurement_scale=float(metrics["joint_measurement_scale"]),
        batches=int(metrics["batches"]),
    )
