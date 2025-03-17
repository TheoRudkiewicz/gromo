import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.linear_growing_module import LinearGrowingModule


# Create synthetic data
def create_synthetic_data(num_samples=20, input_dim=2, output_dim=1, batch_size=1):
    input_data = torch.randn(num_samples, input_dim)
    output_data = torch.randn(num_samples, output_dim)
    dataset = TensorDataset(input_data, output_data)
    return DataLoader(dataset, batch_size=batch_size)


class Perceptron(GrowingContainer):
    def __init__(self, in_features, hidden_feature, out_features):
        super(Perceptron, self).__init__(in_features, out_features)

        # define the layers
        self.layers = nn.ModuleList(
            [
                LinearGrowingModule(
                    in_features,
                    hidden_feature,
                    name="hidden",
                    post_layer_function=nn.ReLU(),
                ),
                LinearGrowingModule(hidden_feature, out_features, name="output"),
            ]
        )

        # linking the layers
        for i, layer in enumerate(self.layers):
            if i > 0:
                layer.previous_module = self.layers[i - 1]
            if i < len(self.layers) - 1:
                layer.next_module = self.layers[i + 1]

        # set growing layers
        self.set_growing_layers()

    def set_growing_layers(self):
        self.growing_layers = self.layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extended forward pass through the network"""
        x_ext = None
        for layer in self.layers:
            x, x_ext = layer.extended_forward(x, x_ext)
        return x


def gather_statistics(dataloader, model, loss):
    model.init_computation()
    for i, (x, y) in enumerate(dataloader):
        model.zero_grad()
        loss(model(x), y).backward()
        model.update_computation()


class TestGrowingContainer(unittest.TestCase):
    """
    Test the GrowingContainer class. We only test the function calls to
    the growing layers, as the correctness of the computations should be
    tested in the respective growing layer tests.
    """

    def setUp(self):
        input_dim = 2
        output_dim = 1
        hidden_dim = 4
        self.num_samples = 20
        self.batch_size = 4
        self.dataloader = create_synthetic_data(
            self.num_samples, input_dim, output_dim, self.batch_size
        )
        self.model = Perceptron(input_dim, hidden_dim, output_dim)
        self.loss = nn.MSELoss()

    def test_init_computation(self):
        self.model.init_computation()
        for layer in self.model.growing_layers:
            # store input
            self.assertTrue(
                layer.store_input,
                "init_computation was not called on the growing layer",
            )

            # store pre-activity
            self.assertTrue(
                layer.store_pre_activity,
                "init_computation was not called on the growing layer",
            )

            # Check tensors: only tensor_s and tensor_m should be initialized with non-None values.
            # tensor_m_prev and cross_covariance should be None since their shape is not known yet.
            for tensor_name in ["tensor_s", "tensor_m"]:
                self.assertIsNotNone(
                    getattr(layer, tensor_name)._tensor,
                    f"init_computation was not called on the growing layer for {tensor_name}",
                )

    def test_update_computation(self):
        self.model.init_computation()
        for i, (x, y) in enumerate(self.dataloader):
            self.model.zero_grad()
            loss = self.loss(self.model(x), y)
            loss.backward()
            self.model.update_computation()

            for layer in self.model.growing_layers:
                # check number of samples in the tensor statistics
                for tensor_name in ["tensor_s", "tensor_m"]:
                    self.assertEqual(
                        getattr(layer, tensor_name).samples,
                        (i + 1) * self.batch_size,
                        f"update_computation was not called on the growing layer for {tensor_name}",
                    )
                if layer.previous_module is not None:
                    for tensor_name in ["tensor_m_prev", "cross_covariance"]:
                        self.assertEqual(
                            getattr(layer, tensor_name).samples,
                            (i + 1) * self.batch_size,
                            f"update_computation was not called on the growing layer for {tensor_name}",
                        )

    def test_reset_computation(self):
        self.model.reset_computation()

        for layer in self.model.growing_layers:
            # store input
            self.assertFalse(
                layer.store_input,
                "reset_computation was not called on the growing layer",
            )

            # store pre-activity
            self.assertFalse(
                layer.store_pre_activity,
                "reset_computation was not called on the growing layer",
            )

            # Check tensors
            for tensor_name in [
                "tensor_s",
                "tensor_m",
                "tensor_m_prev",
                "cross_covariance",
            ]:
                self.assertIsNone(
                    getattr(layer, tensor_name)._tensor,
                    f"reset_computation was not called on the growing layer for {tensor_name}",
                )

    def test_compute_optimal_updates(self):
        gather_statistics(self.dataloader, self.model, self.loss)
        self.model.compute_optimal_updates()

        for layer in self.model.growing_layers:
            # Check if the optimal updates are computed
            self.assertTrue(
                hasattr(layer, "optimal_delta_layer"),
                "compute_optimal_updates was not called on the growing layer",
            )
            self.assertTrue(
                hasattr(layer, "parameter_update_decrease"),
                "compute_optimal_updates was not called on the growing layer",
            )
            if layer.previous_module is not None:
                self.assertTrue(
                    hasattr(layer, "extended_input_layer"),
                    "compute_optimal_updates was not called on the growing layer",
                )
                self.assertTrue(
                    hasattr(layer.previous_module, "extended_output_layer"),
                    "compute_optimal_updates was not called on the growing layer",
                )

    def test_select_best_update(self):
        # computing the optimal updates
        gather_statistics(self.dataloader, self.model, self.loss)
        self.model.compute_optimal_updates()
        self.assertIsNone(
            self.model.currently_updated_layer_index, "There should be no layer to update"
        )

        # selecting the best update
        self.model.select_best_update()
        self.assertIsNotNone(
            self.model.currently_updated_layer_index, "No layer to update"
        )

        # Check if the optimal updates are computed
        for i, layer in enumerate(self.model.growing_layers):
            if i != self.model.currently_updated_layer_index:
                self.assertIsNone(
                    layer.optimal_delta_layer,
                    "select_best_update was not called on the growing layer",
                )
                self.assertIsNone(
                    layer.parameter_update_decrease,
                    "select_best_update was not called on the growing layer",
                )
                if layer.previous_module is not None:
                    self.assertIsNone(
                        layer.extended_input_layer,
                        "select_best_update was not called on the growing layer",
                    )
                    self.assertIsNone(
                        layer.previous_module.extended_output_layer,
                        "select_best_update was not called on the growing layer",
                    )
            else:
                self.assertIsNotNone(
                    layer.optimal_delta_layer,
                    "select_best_update was not called on the growing layer",
                )
                self.assertIsNotNone(
                    layer.parameter_update_decrease,
                    "select_best_update was not called on the growing layer",
                )
                if layer.previous_module is not None:
                    self.assertIsNotNone(
                        layer.extended_input_layer,
                        "select_best_update was not called on the growing layer",
                    )
                    self.assertIsNotNone(
                        layer.previous_module.extended_output_layer,
                        "select_best_update was not called on the growing layer",
                    )

    def test_apply_change(self):
        gather_statistics(self.dataloader, self.model, self.loss)
        self.model.compute_optimal_updates()
        self.model.select_best_update()
        self.model.currently_updated_layer.scaling_factor = 1.0
        self.model.apply_change()

        self.assertIsNone(self.model.currently_updated_layer_index, "No layer to update")


if __name__ == "__main__":
    unittest.main()
