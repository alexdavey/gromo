r"""
Expressivity bottleneck gap demo
================================

This script follows the standard train/grow rhythm on a small synthetic
``GrowingMLP`` and measures, but does not apply, the joint bottleneck
approximation at each analytic FoGro growth step.

Run from the repository root:

    PYTHONPATH=src python examples/demo_bottleneck_gap.py
"""

import argparse
from pathlib import Path
from time import time
from typing import Any

import matplotlib.pyplot as plt
import torch
from helpers.synthetic_data import MultiSinDataloader

from gromo.containers.growing_mlp import GrowingMLP
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.joint_delta import (
    JointBottleneckGapResult,
    compute_joint_bottleneck_gap,
)
from gromo.utils.training_utils import (
    compute_statistics,
    evaluate_model,
    gradient_descent,
)
from gromo.utils.utils import activation_fn, global_device, set_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train/grow with analytic FoGro and plot the joint "
        "expressivity bottleneck gap."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nb-step", type=int, default=20)
    parser.add_argument("--growth-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-batches", type=int, default=10)
    parser.add_argument("--test-batches", type=int, default=2)
    parser.add_argument("--growth-batch-limit", type=int, default=-1)
    parser.add_argument("--in-features", type=int, default=10)
    parser.add_argument("--out-features", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--number-hidden-layers", type=int, default=2)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--optimizer-lr", type=float, default=1e-2)
    parser.add_argument("--growth-scaling", type=float, default=0.1)
    parser.add_argument("--maximum-added-neurons", type=int, default=1)
    parser.add_argument("--statistical-threshold", type=float, default=1e-3)
    parser.add_argument("--joint-lr", type=float, default=3e-4)
    parser.add_argument("--joint-steps", type=int, default=1000)
    parser.add_argument("--joint-init-scale", type=float, default=1e-2)
    parser.add_argument(
        "--plot-path",
        default="bottleneck_gap.png",
        help="Path where the bottleneck-gap plot is written.",
    )
    return parser.parse_args()


def setup_device(device_name: str) -> torch.device:
    set_device(torch.device(device_name))
    device = global_device()
    print(f"Using device: {device}")
    return device


def create_dataloaders(args: argparse.Namespace, device: torch.device):
    train_loader = MultiSinDataloader(
        nb_sample=args.train_batches,
        batch_size=args.batch_size,
        seed=args.seed,
        in_features=args.in_features,
        out_features=args.out_features,
        device=device,
    )
    test_loader = MultiSinDataloader(
        nb_sample=args.test_batches,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        in_features=args.in_features,
        out_features=args.out_features,
        device=device,
    )
    return train_loader, test_loader


def create_model(args: argparse.Namespace, device: torch.device) -> GrowingMLP:
    model = GrowingMLP(
        in_features=args.in_features,
        out_features=args.out_features,
        hidden_size=args.hidden_size,
        number_hidden_layers=args.number_hidden_layers,
        activation=activation_fn(args.activation),
        use_bias=True,
        device=device,
    )
    print(f"Model:\n{model}")
    return model


def create_optimizer(args: argparse.Namespace, model: torch.nn.Module):
    return torch.optim.SGD(model.parameters(), lr=args.optimizer_lr)


def _selected_linear_pair(
    model: GrowingMLP,
) -> tuple[LinearGrowingModule, LinearGrowingModule]:
    current_layer = model.currently_updated_layer
    if not isinstance(current_layer, LinearGrowingModule) or not isinstance(
        current_layer.previous_module,
        LinearGrowingModule,
    ):
        raise TypeError(
            "The bottleneck-gap demo only supports selected LinearGrowingModule "
            "layers with a LinearGrowingModule previous layer."
        )
    return current_layer.previous_module, current_layer


def perform_growth_step(
    model: GrowingMLP,
    train_loader,
    loss_function_growth: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    logs: dict[str, Any] = {}
    growth_batch_limit = (
        None if args.growth_batch_limit == -1 else args.growth_batch_limit
    )

    initial_loss, _ = compute_statistics(
        model=model,
        dataloader=train_loader,
        loss_function=loss_function_growth,
        batch_limit=growth_batch_limit,
        device=device,
    )
    logs["initial_train_loss"] = initial_loss

    model.compute_optimal_updates(
        numerical_threshold=1e-6,
        statistical_threshold=0.0,
        maximum_added_neurons=args.maximum_added_neurons,
        dtype=torch.float32,
        compute_delta=True,
        use_covariance=True,
        alpha_zero=False,
        omega_zero=False,
        use_projection=True,
        ignore_singular_values=False,
    )
    model.reset_computation()

    selected_update = model.select_best_update()
    logs["selected_update"] = selected_update
    previous_layer, current_layer = _selected_linear_pair(model)

    gap = compute_joint_bottleneck_gap(
        model=model,
        previous_layer=previous_layer,
        current_layer=current_layer,
        dataloader=train_loader,
        loss_function=loss_function_growth,
        lr=args.joint_lr,
        steps=args.joint_steps,
        batch_limit=growth_batch_limit,
        init_scale=args.joint_init_scale,
        device=device,
    )
    logs.update(_gap_to_logs(gap))

    current_layer.sub_select_optimal_added_parameters(
        threshold=args.statistical_threshold,
    )
    logs["added_neurons"] = (
        current_layer.eigenvalues_extension.size(0)
        if current_layer.eigenvalues_extension is not None
        else 0
    )
    current_layer.set_scaling_factor(args.growth_scaling)
    model.apply_change(extension_size=logs["added_neurons"])
    return logs


def _gap_to_logs(gap: JointBottleneckGapResult) -> dict[str, Any]:
    return {
        "bottleneck_gap_layer": gap.current_layer_name,
        "old_directional_residual": gap.old_directional_residual,
        "joint_directional_residual": gap.joint_directional_residual,
        "directional_gap": gap.directional_gap,
        "relative_directional_gap": gap.relative_directional_gap,
        "old_bottleneck_norm_sq": gap.old_bottleneck_norm_sq,
        "joint_bottleneck_norm_sq": gap.joint_bottleneck_norm_sq,
        "residual_gap": gap.residual_gap,
        "normalized_gap": gap.normalized_gap,
        "normalized_gap_percent": gap.normalized_gap_percent,
        "relative_residual_gap": gap.relative_residual_gap,
        "initial_joint_directional_loss": gap.initial_joint_directional_loss,
        "final_joint_directional_loss": gap.final_joint_directional_loss,
        "joint_measurement_scale": gap.joint_measurement_scale,
    }


def perform_training_step(
    model: GrowingMLP,
    train_loader,
    loss_function_train: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    train_loss, _ = gradient_descent(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        loss_function=loss_function_train,
        scheduler=None,
        device=device,
    )
    return {"train_loss": train_loss}


def plot_gap_history(history: list[dict[str, Any]], plot_path: str) -> None:
    growth_rows = [row for row in history if row.get("is_growth_step") == 1]
    if not growth_rows:
        print("No growth steps were run; no bottleneck-gap plot was written.")
        return

    steps = [row["step"] for row in growth_rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(steps, [row["directional_gap"] for row in growth_rows], marker="o")
    axes[0].plot(steps, [row["residual_gap"] for row in growth_rows], marker="o")
    axes[0].set_ylabel("gap")
    axes[0].legend(["directional", "measured residual"])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        steps,
        [row["normalized_gap_percent"] for row in growth_rows],
        marker="o",
    )
    axes[1].set_ylabel("normalized gap (%)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        steps,
        [row["old_directional_residual"] for row in growth_rows],
        marker="o",
    )
    axes[2].plot(
        steps,
        [row["joint_directional_residual"] for row in growth_rows],
        marker="o",
    )
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("directional residual")
    axes[2].legend(["analytic", "joint"])
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = Path(plot_path)
    fig.savefig(output_path)
    print(f"Wrote bottleneck-gap plot to {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = setup_device(args.device)
    train_loader, test_loader = create_dataloaders(args, device)
    model = create_model(args, device)
    optimizer = create_optimizer(args, model)
    loss_function_train = torch.nn.MSELoss(reduction="mean")
    loss_function_growth = torch.nn.MSELoss(reduction="sum")

    history: list[dict[str, Any]] = []
    train_loss, _ = evaluate_model(
        model=model,
        dataloader=train_loader,
        loss_function=loss_function_train,
        device=device,
    )
    test_loss, _ = evaluate_model(
        model=model,
        dataloader=test_loader,
        loss_function=loss_function_train,
        device=device,
    )
    print(f"Step 0/{args.nb_step} | Loss: {train_loss:.4f} ({test_loss:.4f}) [init]")

    for step in range(1, args.nb_step + 1):
        step_start = time()
        is_growth_step = args.growth_every > 0 and step % args.growth_every == 0
        if is_growth_step:
            logs = perform_growth_step(
                model=model,
                train_loader=train_loader,
                loss_function_growth=loss_function_growth,
                args=args,
                device=device,
            )
            logs["epoch_type"] = "growth"
            logs["is_growth_step"] = 1
            optimizer = create_optimizer(args, model)
        else:
            logs = perform_training_step(
                model=model,
                train_loader=train_loader,
                loss_function_train=loss_function_train,
                optimizer=optimizer,
                device=device,
            )
            logs["epoch_type"] = "training"
            logs["is_growth_step"] = 0

        train_loss, _ = evaluate_model(
            model=model,
            dataloader=train_loader,
            loss_function=loss_function_train,
            device=device,
        )
        test_loss, _ = evaluate_model(
            model=model,
            dataloader=test_loader,
            loss_function=loss_function_train,
            device=device,
        )
        logs.update(
            {
                "step": step,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "step_duration": time() - step_start,
            }
        )
        history.append(logs)

        extra = ""
        if logs["is_growth_step"]:
            extra = (
                f" | gap={logs['directional_gap']:.4f}"
                f" residual_gap={logs['residual_gap']:.4f}"
                f" normalized_gap={logs['normalized_gap_percent']:.2f}%"
            )
        print(
            f"Step {step}/{args.nb_step} | "
            f"Loss: {train_loss:.4f} ({test_loss:.4f}) "
            f"[{logs['epoch_type']}]{extra}"
        )

    plot_gap_history(history, args.plot_path)
    print(f"Final model:\n{model}")


if __name__ == "__main__":
    main()
