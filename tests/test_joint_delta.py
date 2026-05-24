import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.joint_delta import compute_joint_bottleneck_gap
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase


class TestJointDelta(TorchTestCase):
    def _make_linear_chain(
        self,
    ) -> tuple[nn.Sequential, LinearGrowingModule, LinearGrowingModule, DataLoader]:
        torch.manual_seed(123)
        device = global_device()
        previous_layer = LinearGrowingModule(
            2,
            3,
            use_bias=True,
            device=device,
            name="previous",
        )
        current_layer = LinearGrowingModule(
            3,
            2,
            use_bias=True,
            previous_module=previous_layer,
            device=device,
            name="current",
        )
        model = nn.Sequential(previous_layer, current_layer)
        x = torch.tensor(
            [
                [1.0, -1.0],
                [0.5, 2.0],
                [-1.5, 0.25],
                [2.0, 1.0],
            ],
            device=device,
        )
        y = torch.tensor(
            [
                [0.25, -0.5],
                [-1.0, 0.75],
                [1.5, 0.25],
                [-0.25, 1.0],
            ],
            device=device,
        )
        dataloader = DataLoader(TensorDataset(x, y), batch_size=x.shape[0])
        return model, previous_layer, current_layer, dataloader

    def _prepare_analytic_delta(
        self,
        model: nn.Sequential,
        previous_layer: LinearGrowingModule,
        current_layer: LinearGrowingModule,
        dataloader: DataLoader,
        loss_function: nn.Module,
    ) -> None:
        previous_layer.init_computation()
        current_layer.init_computation()
        for x, y in dataloader:
            model.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward()
            previous_layer.update_computation()
            current_layer.update_computation()
        current_layer.compute_optimal_delta()
        previous_layer.reset_computation()
        current_layer.reset_computation()

    def test_compute_joint_bottleneck_gap_is_measurement_only(self) -> None:
        model, previous_layer, current_layer, dataloader = self._make_linear_chain()
        loss_function = nn.MSELoss(reduction="sum")
        self._prepare_analytic_delta(
            model,
            previous_layer,
            current_layer,
            dataloader,
            loss_function,
        )
        previous_weight = previous_layer.weight.detach().clone()
        current_weight = current_layer.weight.detach().clone()
        analytic_delta_weight = current_layer.optimal_delta_layer.weight.detach().clone()

        result = compute_joint_bottleneck_gap(
            model,
            previous_layer,
            current_layer,
            dataloader,
            loss_function,
            lr=0.05,
            steps=10,
            init_scale=1e-2,
        )

        self.assertLessEqual(
            result.final_joint_directional_loss,
            result.initial_joint_directional_loss,
        )
        self.assertAlmostEqual(
            result.directional_gap,
            result.old_directional_residual - result.joint_directional_residual,
        )
        self.assertAlmostEqual(
            result.residual_gap,
            result.old_bottleneck_norm_sq - result.joint_bottleneck_norm_sq,
        )
        self.assertAlmostEqual(
            result.normalized_gap,
            result.residual_gap / result.old_bottleneck_norm_sq,
        )
        self.assertAlmostEqual(
            result.normalized_gap_percent,
            100.0 * result.normalized_gap,
        )
        self.assertEqual(result.batches, 1)
        self.assertAllClose(previous_layer.weight, previous_weight)
        self.assertAllClose(current_layer.weight, current_weight)
        self.assertAllClose(
            current_layer.optimal_delta_layer.weight,
            analytic_delta_weight,
        )
        self.assertFalse(previous_layer.store_input)
        self.assertFalse(previous_layer.store_pre_activity)
        self.assertFalse(current_layer.store_input)
        self.assertFalse(current_layer.store_pre_activity)

    def test_compute_requires_existing_analytic_delta(self) -> None:
        model, previous_layer, current_layer, dataloader = self._make_linear_chain()

        with self.assertRaises(ValueError):
            compute_joint_bottleneck_gap(
                model,
                previous_layer,
                current_layer,
                dataloader,
                nn.MSELoss(reduction="sum"),
            )

    def test_compute_rejects_non_adjacent_layers(self) -> None:
        model, _previous_layer, current_layer, dataloader = self._make_linear_chain()
        current_layer.optimal_delta_layer = current_layer.layer_of_tensor(
            torch.zeros_like(current_layer.weight),
            torch.zeros_like(current_layer.bias),
        )
        other_previous = LinearGrowingModule(
            2,
            3,
            device=global_device(),
            name="other_previous",
        )

        with self.assertRaises(ValueError):
            compute_joint_bottleneck_gap(
                model,
                other_previous,
                current_layer,
                dataloader,
                nn.MSELoss(reduction="sum"),
            )
