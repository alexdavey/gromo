"""Fixed-point iteration utilities for optimal-delta updates."""

import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal
from warnings import warn

import torch


TensorMap = dict[str, torch.Tensor]
FixedPointStatus = Literal["converged", "max_iterations"]
FixedPointMap = Callable[[TensorMap | None], tuple[Mapping[str, torch.Tensor], float]]


@dataclass(frozen=True)
class FixedPointConfig:
    """Configuration for relaxed fixed-point iteration."""

    max_iterations: int = 25
    damping: float = 1.0
    atol: float = 1e-8
    rtol: float = 1e-6
    fail_on_nonconvergence: bool = False

    def __post_init__(self) -> None:
        """Validate solver settings."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must lie in (0, 1]")
        if self.atol < 0.0 or self.rtol < 0.0:
            raise ValueError("atol and rtol must be nonnegative")
        if self.atol == 0.0 and self.rtol == 0.0:
            raise ValueError("at least one of atol and rtol must be positive")


@dataclass(frozen=True)
class FixedPointIteration:
    """Diagnostics for one complete fixed-point map evaluation."""

    iteration: int
    loss: float
    candidate_norm: float
    map_norm: float
    residual_norm: float
    relative_residual: float


@dataclass(frozen=True)
class FixedPointResult:
    """Terminal diagnostics for a fixed-point solve."""

    converged: bool
    status: FixedPointStatus
    iterations: int
    residual_norm: float
    relative_residual: float
    update_norm: float
    frozen_update_norm: float
    module_names: tuple[str, ...]
    history: tuple[FixedPointIteration, ...]


@dataclass(frozen=True)
class FixedPointUpdateResult:
    """Loss, metric, and optional diagnostics for a fixed-point update."""

    loss: float
    metric: float
    fixed_point: FixedPointResult | None


@dataclass(frozen=True)
class _FixedPointSolution:
    """Internal solver output including the terminal tensor maps."""

    result: FixedPointResult
    candidate: TensorMap
    update: TensorMap


def _clone_tensor_map(values: Mapping[str, torch.Tensor]) -> TensorMap:
    """Detach and clone every tensor in a named map."""
    return {name: value.detach().clone() for name, value in values.items()}


def _tensor_map_norm(values: Mapping[str, torch.Tensor]) -> float:
    """Compute the joint Frobenius norm without concatenating devices or dtypes."""
    squared_norm = 0.0
    for value in values.values():
        squared_norm += float(torch.linalg.vector_norm(value).double().square().item())
    return math.sqrt(squared_norm)


def _validate_tensor_map(
    values: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Validate finiteness and, when supplied, compatibility with a reference map."""
    if not values:
        raise ValueError("the fixed-point map did not produce any optimal deltas")
    if reference is not None and values.keys() != reference.keys():
        raise RuntimeError(
            "the optimal-delta modules changed during fixed-point iteration: "
            f"expected {tuple(reference)}, got {tuple(values)}"
        )
    for name, value in values.items():
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(
                f"the fixed-point map produced a non-finite tensor for {name}"
            )
        if reference is None:
            continue
        expected = reference[name]
        if value.shape != expected.shape:
            raise RuntimeError(
                f"the shape of {name} changed during fixed-point iteration: "
                f"expected {tuple(expected.shape)}, got {tuple(value.shape)}"
            )
        if value.dtype != expected.dtype or value.device != expected.device:
            raise RuntimeError(
                f"the dtype or device of {name} changed during fixed-point iteration"
            )


def _module_names(keys: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    """Extract stable module paths from ``<module>.<parameter>`` keys."""
    names = {key.rsplit(".", 1)[0] if "." in key else key for key in keys}
    return tuple(sorted(names))


def _solve_fixed_point(
    map_function: FixedPointMap,
    config: FixedPointConfig,
) -> _FixedPointSolution:
    """Solve ``D = T(D)`` with relaxed Picard iteration over named tensors.

    ``map_function(None)`` denotes the first evaluation at the zero candidate. The
    names and tensor metadata returned by that evaluation define the fixed-point
    vector for all subsequent iterations.
    """
    candidate: TensorMap | None = None
    reference: TensorMap | None = None
    history: list[FixedPointIteration] = []
    frozen_update_norm = 0.0
    terminal_update: TensorMap | None = None

    for iteration in range(1, config.max_iterations + 1):
        mapped_raw, loss = map_function(candidate)
        if not math.isfinite(loss):
            raise FloatingPointError("the fixed-point map produced a non-finite loss")

        mapped = _clone_tensor_map(mapped_raw)
        if reference is None:
            _validate_tensor_map(mapped)
            reference = _clone_tensor_map(mapped)
            candidate = {
                name: torch.zeros_like(value) for name, value in reference.items()
            }
            frozen_update_norm = _tensor_map_norm(mapped)
        else:
            _validate_tensor_map(mapped, reference)

        assert candidate is not None
        _validate_tensor_map(candidate, reference)
        residual = {name: mapped[name] - candidate[name] for name in reference}
        candidate_norm = _tensor_map_norm(candidate)
        map_norm = _tensor_map_norm(mapped)
        residual_norm = _tensor_map_norm(residual)
        relative_residual = residual_norm / max(
            candidate_norm,
            map_norm,
            sys.float_info.epsilon,
        )
        history.append(
            FixedPointIteration(
                iteration=iteration,
                loss=loss,
                candidate_norm=candidate_norm,
                map_norm=map_norm,
                residual_norm=residual_norm,
                relative_residual=relative_residual,
            )
        )
        terminal_update = mapped

        if residual_norm <= config.atol or relative_residual <= config.rtol:
            result = FixedPointResult(
                converged=True,
                status="converged",
                iterations=iteration,
                residual_norm=residual_norm,
                relative_residual=relative_residual,
                update_norm=map_norm,
                frozen_update_norm=frozen_update_norm,
                module_names=_module_names(reference),
                history=tuple(history),
            )
            return _FixedPointSolution(
                result=result,
                candidate=_clone_tensor_map(candidate),
                update=mapped,
            )

        candidate = {
            name: (
                candidate[name] + config.damping * (mapped[name] - candidate[name])
            ).detach()
            for name in reference
        }

    assert candidate is not None
    assert terminal_update is not None
    terminal = history[-1]
    result = FixedPointResult(
        converged=False,
        status="max_iterations",
        iterations=config.max_iterations,
        residual_norm=terminal.residual_norm,
        relative_residual=terminal.relative_residual,
        update_norm=terminal.map_norm,
        frozen_update_norm=frozen_update_norm,
        module_names=_module_names(reference),
        history=tuple(history),
    )
    message = (
        "Optimal-delta fixed-point iteration did not converge: "
        f"iterations={result.iterations}, residual={result.residual_norm:.6e}, "
        f"relative_residual={result.relative_residual:.6e}"
    )
    if config.fail_on_nonconvergence:
        raise RuntimeError(message)
    warn(message, RuntimeWarning, stacklevel=2)
    return _FixedPointSolution(
        result=result,
        candidate=_clone_tensor_map(candidate),
        update=_clone_tensor_map(terminal_update),
    )
