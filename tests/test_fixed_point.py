import unittest
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.fixed_point import FixedPointConfig, _solve_fixed_point
from gromo.utils.training_utils import compute_fixed_point_updates, compute_statistics


class TestFixedPointSolver(unittest.TestCase):
    """Tests for the tensor-map fixed-point solver."""

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            FixedPointConfig(max_iterations=0)
        with self.assertRaises(ValueError):
            FixedPointConfig(damping=0.0)
        with self.assertRaises(ValueError):
            FixedPointConfig(damping=1.1)
        with self.assertRaises(ValueError):
            FixedPointConfig(atol=-1.0)
        with self.assertRaises(ValueError):
            FixedPointConfig(atol=0.0, rtol=0.0)

    def test_constant_map(self):
        constant = torch.tensor([2.0, -3.0])

        def map_function(_candidate):
            return {"layer.weight": constant}, 1.0

        solution = _solve_fixed_point(
            map_function,
            FixedPointConfig(max_iterations=3, atol=1e-12, rtol=1e-12),
        )

        self.assertTrue(solution.result.converged)
        self.assertEqual(solution.result.iterations, 2)
        torch.testing.assert_close(solution.update["layer.weight"], constant)
        torch.testing.assert_close(solution.candidate["layer.weight"], constant)

    def test_affine_contraction(self):
        a = 0.25
        b = torch.tensor([3.0])

        def map_function(candidate):
            value = torch.zeros_like(b) if candidate is None else candidate["x"]
            return {"x": a * value + b}, 0.0

        solution = _solve_fixed_point(
            map_function,
            FixedPointConfig(max_iterations=30, atol=1e-10, rtol=1e-7),
        )

        self.assertTrue(solution.result.converged)
        torch.testing.assert_close(
            solution.update["x"],
            b / (1.0 - a),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_damping_is_applied_to_residual(self):
        b = torch.tensor([4.0])

        def oscillatory_map(candidate):
            value = torch.zeros_like(b) if candidate is None else candidate["x"]
            return {"x": -value + b}, 0.0

        solution = _solve_fixed_point(
            oscillatory_map,
            FixedPointConfig(
                max_iterations=3,
                damping=0.5,
                atol=1e-12,
                rtol=1e-12,
            ),
        )

        self.assertTrue(solution.result.converged)
        self.assertEqual(solution.result.iterations, 2)
        torch.testing.assert_close(solution.update["x"], b / 2.0)

    def test_divergent_map_reaches_iteration_limit(self):
        def divergent_map(candidate):
            value = torch.zeros(1) if candidate is None else candidate["x"]
            return {"x": 2.0 * value + 1.0}, 0.0

        with self.assertWarnsRegex(RuntimeWarning, "did not converge"):
            solution = _solve_fixed_point(
                divergent_map,
                FixedPointConfig(max_iterations=4, atol=1e-12, rtol=1e-12),
            )

        self.assertFalse(solution.result.converged)
        self.assertEqual(solution.result.status, "max_iterations")

    def test_nonconvergence_can_raise(self):
        def divergent_map(candidate):
            value = torch.zeros(1) if candidate is None else candidate["x"]
            return {"x": value + 1.0}, 0.0

        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            _solve_fixed_point(
                divergent_map,
                FixedPointConfig(
                    max_iterations=2,
                    fail_on_nonconvergence=True,
                ),
            )

    def test_map_keys_and_nonfinite_values_are_validated(self):
        calls = 0

        def changing_map(_candidate):
            nonlocal calls
            calls += 1
            key = "x" if calls == 1 else "y"
            return {key: torch.ones(1)}, 0.0

        with self.assertRaisesRegex(RuntimeError, "modules changed"):
            _solve_fixed_point(changing_map, FixedPointConfig(max_iterations=2))

        def nonfinite_map(_candidate):
            return {"x": torch.tensor([float("nan")])}, 0.0

        with self.assertRaises(FloatingPointError):
            _solve_fixed_point(nonfinite_map, FixedPointConfig())


class _LinearContainer(GrowingContainer):
    """Minimal container exposing one or two active linear growing layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = False,
        hidden_features: int | None = None,
        fail_after_forwards: int | None = None,
    ):
        super().__init__(in_features=in_features, out_features=out_features)
        self.fail_after_forwards = fail_after_forwards
        self.forward_calls = 0
        if hidden_features is None:
            self.first = LinearGrowingModule(
                in_features,
                out_features,
                use_bias=use_bias,
                name="first",
            )
            self.second = None
            self._growing_layers = [self.first]
        else:
            self.first = LinearGrowingModule(
                in_features,
                hidden_features,
                use_bias=use_bias,
                name="first",
            )
            self.second = LinearGrowingModule(
                hidden_features,
                out_features,
                use_bias=use_bias,
                name="second",
            )
            self._growing_layers = [self.first, self.second]

    def set_growing_layers(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        if (
            self.fail_after_forwards is not None
            and self.forward_calls > self.fail_after_forwards
        ):
            raise RuntimeError("intentional forward failure")
        x = self.first(x)
        return x if self.second is None else self.second(x)

    def extended_forward(
        self,
        x: torch.Tensor,
        mask: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward(x), None

    @property
    def first_order_improvement(self) -> torch.Tensor:
        return torch.tensor(0.0)


class TestComputeFixedPointUpdates(unittest.TestCase):
    """Integration tests for Gromo's fixed-point orchestration."""

    def setUp(self):
        self.original_rng_state = torch.get_rng_state()
        torch.manual_seed(0)

    def tearDown(self):
        torch.set_rng_state(self.original_rng_state)

    @staticmethod
    def _loader(x: torch.Tensor, y: torch.Tensor) -> DataLoader:
        generator = torch.Generator().manual_seed(123)
        return DataLoader(
            TensorDataset(x, y),
            batch_size=len(x),
            generator=generator,
        )

    def test_one_sample_softmax_fixed_point_and_state_restoration(self):
        model = _LinearContainer(1, 10, use_bias=False)
        with torch.no_grad():
            model.first.weight.zero_()
        base_weight = model.first.weight.detach().clone()
        original_gradient = torch.full_like(model.first.weight, 7.0)
        model.first.weight.grad = original_gradient.clone()
        model.train()
        loader = self._loader(torch.ones(1, 1), torch.zeros(1, dtype=torch.long))
        loss = nn.CrossEntropyLoss(reduction="sum")
        cpu_rng_state = torch.get_rng_state()
        generator_state = loader.generator.get_state()

        result = compute_fixed_point_updates(
            model,
            loader,
            loss,
            fixed_point_config=FixedPointConfig(max_iterations=20, rtol=1e-6),
        )

        self.assertIsNotNone(result.fixed_point)
        assert result.fixed_point is not None
        self.assertTrue(result.fixed_point.converged)
        self.assertEqual(result.fixed_point.module_names, ("first",))
        self.assertAlmostEqual(result.fixed_point.frozen_update_norm, 0.9486833, places=5)
        delta = model.first.optimal_delta_layer.weight.detach()
        expected = torch.full_like(delta, 0.7892317 / 9.0)
        expected[0, 0] = -0.7892317
        torch.testing.assert_close(delta, expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(torch.equal(model.first.weight, base_weight))
        self.assertTrue(torch.equal(model.first.weight.grad, original_gradient))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng_state))
        self.assertTrue(torch.equal(loader.generator.get_state(), generator_state))

    def test_bias_and_joint_active_set(self):
        model = _LinearContainer(
            2,
            1,
            hidden_features=2,
            use_bias=True,
        )
        base = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        x = torch.tensor([[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0]])
        y = torch.tensor([[0.5], [-1.0], [2.0]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = compute_fixed_point_updates(
                model,
                self._loader(x, y),
                nn.MSELoss(reduction="sum"),
                fixed_point_config=FixedPointConfig(max_iterations=2),
            )

        assert result.fixed_point is not None
        self.assertEqual(result.fixed_point.module_names, ("first", "second"))
        self.assertIsNotNone(model.first.optimal_delta_layer.bias)
        self.assertIsNotNone(model.second.optimal_delta_layer.bias)
        for name, parameter in model.named_parameters():
            if ".optimal_delta_layer." not in name:
                self.assertTrue(torch.equal(parameter, base[name]))

    def test_one_pass_matches_existing_workflow(self):
        manual = _LinearContainer(2, 1, use_bias=True)
        high_level = _LinearContainer(2, 1, use_bias=True)
        high_level.load_state_dict(manual.state_dict())
        x = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, -2.0]])
        y = torch.tensor([[0.5], [-1.0], [2.0]])
        loader = self._loader(x, y)
        loss = nn.MSELoss(reduction="sum")

        compute_statistics(manual, loader, loss_function=loss)
        manual.compute_optimal_updates()
        result = compute_fixed_point_updates(high_level, loader, loss)

        self.assertIsNone(result.fixed_point)
        torch.testing.assert_close(
            high_level.first.optimal_delta_layer.weight,
            manual.first.optimal_delta_layer.weight,
        )
        torch.testing.assert_close(
            high_level.first.optimal_delta_layer.bias,
            manual.first.optimal_delta_layer.bias,
        )

    def test_pending_update_and_disabled_delta_are_rejected(self):
        model = _LinearContainer(1, 1)
        loader = self._loader(torch.ones(1, 1), torch.zeros(1, 1))
        loss = nn.MSELoss(reduction="sum")
        model.first.optimal_delta_layer = nn.Linear(1, 1, bias=False)

        with self.assertRaisesRegex(RuntimeError, "pending growth proposal"):
            compute_fixed_point_updates(model, loader, loss)

        model.first.optimal_delta_layer = None
        with self.assertRaisesRegex(ValueError, "compute_delta=True"):
            compute_fixed_point_updates(
                model,
                loader,
                loss,
                fixed_point_config=FixedPointConfig(),
                optimal_update_kwargs={"compute_delta": False},
            )

    def test_exception_restores_parameters_and_clears_proposals(self):
        model = _LinearContainer(1, 1, fail_after_forwards=1)
        base_weight = model.first.weight.detach().clone()
        original_gradient = torch.full_like(model.first.weight, 3.0)
        model.first.weight.grad = original_gradient.clone()
        model.train()
        loader = self._loader(torch.ones(1, 1), torch.zeros(1, 1))
        cpu_rng_state = torch.get_rng_state()

        with self.assertRaisesRegex(RuntimeError, "intentional forward failure"):
            compute_fixed_point_updates(
                model,
                loader,
                nn.MSELoss(reduction="sum"),
                fixed_point_config=FixedPointConfig(max_iterations=3),
            )

        self.assertTrue(torch.equal(model.first.weight, base_weight))
        self.assertTrue(torch.equal(model.first.weight.grad, original_gradient))
        self.assertIsNone(model.first.optimal_delta_layer)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng_state))

    def test_nonfinite_loss_clears_partial_state(self):
        model = _LinearContainer(1, 1)
        loader = self._loader(
            torch.tensor([[float("nan")]]),
            torch.zeros(1, 1),
        )
        with self.assertRaises(FloatingPointError):
            compute_fixed_point_updates(
                model,
                loader,
                nn.MSELoss(reduction="sum"),
                fixed_point_config=FixedPointConfig(),
            )
        self.assertIsNone(model.first.optimal_delta_layer)

    def test_batch_norm_running_statistics_are_not_updated(self):
        model = _LinearContainer(2, 2)
        batch_norm = nn.BatchNorm1d(2)
        model.first.post_layer_function = batch_norm
        running_mean = batch_norm.running_mean.detach().clone()
        running_variance = batch_norm.running_var.detach().clone()
        x = torch.tensor([[1.0, -1.0], [2.0, 0.5], [-0.5, 3.0]])
        y = torch.zeros_like(x)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            compute_fixed_point_updates(
                model,
                self._loader(x, y),
                nn.MSELoss(reduction="sum"),
                fixed_point_config=FixedPointConfig(max_iterations=2),
            )

        self.assertTrue(torch.equal(batch_norm.running_mean, running_mean))
        self.assertTrue(torch.equal(batch_norm.running_var, running_variance))


if __name__ == "__main__":
    unittest.main()
