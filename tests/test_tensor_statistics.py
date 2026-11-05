from unittest import TestCase, main

import torch
from torch.utils.data import DataLoader, TensorDataset

from gromo.utils.tensor_statistic import (
    TensorStatistic,
    TensorStatiticWithEstimationError,
)
from gromo.utils.utils import reset_device, set_device


class TestTensorStatistic(TestCase):
    _tested_class = TensorStatistic

    def test_mean(self):
        set_device("cpu")
        x = None
        n_samples = 0
        f = lambda: (x.sum(dim=0), x.size(0))  # type: ignore
        tensor_statistic = self._tested_class(
            shape=(2, 3), update_function=f, name="Average"
        )
        tensor_statistic_unshaped = self._tested_class(
            shape=None, update_function=f, name="Average-unshaped"
        )

        for t in [tensor_statistic, tensor_statistic_unshaped]:
            self.assertRaises(ValueError, t)

        tensor_statistic.init()
        tensor_statistic_unshaped.init()
        mean_x = torch.zeros((2, 3))
        for n in [1, 5, 8, 15]:
            x = torch.randn(n, 2, 3)
            n_samples += x.size(0)
            mean_x += x.sum(dim=0)
            for t in [tensor_statistic, tensor_statistic_unshaped]:
                t.updated = False
                t.update()
                self.assertTrue(torch.allclose(t(), mean_x / n_samples))
                self.assertEqual(t.samples, n_samples)

                t.update()
                self.assertTrue(torch.allclose(t(), mean_x / n_samples))
                self.assertEqual(t.samples, n_samples)

        x = torch.zeros(1, 3, 4)
        for t in [tensor_statistic, tensor_statistic_unshaped]:
            t.updated = False
            self.assertRaises(AssertionError, t.update)

            t.reset()
            self.assertIsNone(t._tensor)
            self.assertEqual(t.samples, 0)

    def test_dtype(self):
        """The dtype is that of the accumulated tensor, None before any update."""
        t = self._tested_class(
            shape=None,
            update_function=lambda: (torch.zeros(2, dtype=torch.float64), 1),
        )
        self.assertIsNone(t.dtype)
        t.updated = False
        t.update()
        self.assertEqual(t.dtype, torch.float64)

    def _fed(
        self, updates: list[torch.Tensor], accumulation_dtype: torch.dtype | None = None
    ) -> TensorStatistic:
        """A statistic that received ``updates`` in order, one sample each.

        The statistic keeps its first update as its running sum and adds to it in
        place, so it is fed copies.
        """
        pending = iter([u.clone() for u in updates])
        t = self._tested_class(shape=None, update_function=lambda: (next(pending), 1))
        t.init(accumulation_dtype)
        for _ in updates:
            t.updated = False
            t.update()
        return t

    def test_default_accumulation_is_in_the_update_dtype(self):
        """Without an accumulation dtype: an fp32 running sum, exactly as by hand."""
        torch.manual_seed(0)
        updates = [torch.randn(3, 3) for _ in range(20)]
        t = self._fed(updates)
        by_hand = torch.zeros(3, 3)
        for u in updates:
            by_hand += u
        self.assertEqual(t.dtype, torch.float32)
        torch.testing.assert_close(t(), by_hand / len(updates), rtol=0, atol=0)

    def test_float64_accumulation(self):
        """fp64 running sum; the average is that sum over n, rounded once to fp32."""
        torch.manual_seed(0)
        updates = [torch.randn(3, 3) for _ in range(20)]
        t = self._fed(updates, torch.float64)
        by_hand = torch.zeros(3, 3, dtype=torch.float64)
        for u in updates:
            by_hand += u.double()
        self.assertEqual(t.dtype, torch.float64)
        self.assertEqual(t().dtype, torch.float32)
        torch.testing.assert_close(t(), (by_hand / len(updates)).float(), rtol=0, atol=0)

    def test_float64_accumulation_keeps_small_updates(self):
        """1, then 1000 updates of 1e-8.

        The fp32 spacing at 1 is 1.2e-7, so every 1e-8 is rounded away from an
        fp32 sum, which stays at 1. An fp64 sum reaches 1 + 1e-5, and the final
        rounding to fp32 keeps that difference.
        """
        updates = [torch.ones(1)] + [torch.full((1,), 1e-8) for _ in range(1000)]
        n = len(updates)
        torch.testing.assert_close(
            self._fed(updates)(), torch.ones(1) / n, rtol=0, atol=0
        )
        torch.testing.assert_close(
            self._fed(updates, torch.float64)(),
            torch.tensor([(1 + 1e-5) / n]),
            rtol=1e-6,
            atol=0,
        )

    def test_init_without_dtype_restores_the_default(self):
        t = self._tested_class(shape=None, update_function=lambda: (torch.ones(2), 1))
        t.init(torch.float64)
        t.updated = False
        t.update()
        self.assertEqual(t.dtype, torch.float64)
        t.init()
        t.updated = False
        t.update()
        self.assertEqual(t.dtype, torch.float32)

    def tearDown(self) -> None:
        reset_device()


class TestTensorStatiticWithEstimationError(TestTensorStatistic):
    _tested_class = TensorStatiticWithEstimationError

    def setUp(self) -> None:
        set_device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)

    def test_error(self):
        num_batches = 10
        batch_size = 10
        total_samples = torch.Size((num_batches * batch_size,))
        mean = torch.tensor([3.0, 4.0])
        cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])

        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        samples = dist.sample(total_samples)
        dataloader = DataLoader(
            TensorDataset(samples), batch_size=batch_size, shuffle=False
        )

        mean_statistic = TensorStatiticWithEstimationError(
            shape=None,
            update_function=lambda x: (x.sum(dim=0), x.size(0)),
            name="Mean with Error",
        )

        self.assertRaises(AssertionError, mean_statistic.error)
        for i, (batch,) in enumerate(dataloader):
            mean_statistic.updated = False
            mean_statistic.update(x=batch)
            if i == 0:
                self.assertEqual(mean_statistic.error(), float("inf"))
        self.assertEqual(mean_statistic.samples, num_batches * batch_size)
        true_error = torch.norm(mean_statistic() - mean).item() ** 2
        self.assertLessEqual(
            true_error, mean_statistic.error() * 3
        )  # this test pass most of the time, but can fail due to randomness
        # (if no seed is set)

        cov_statistic = TensorStatiticWithEstimationError(
            shape=None,
            update_function=lambda x: (
                (x - mean_statistic()).T @ (x - mean_statistic()),
                x.size(0),
            ),
            name="Covariance with Error",
        )

        for (batch,) in dataloader:
            cov_statistic.updated = False
            cov_statistic.update(x=batch)

        true_error = torch.norm(cov_statistic() - cov).item() ** 2
        self.assertLessEqual(
            true_error, cov_statistic.error() * 2
        )  # this test pass most of the time, but can fail due to randomness
        # (if no seed is set)

    def test_update_count_samples_false(self):
        """update(count_samples=False) returns the raw update without
        counting it as a new batch for the trace-error estimate or the
        sample counter."""
        mean_statistic = TensorStatiticWithEstimationError(
            shape=None,
            update_function=lambda x: (x.sum(dim=0), x.size(0)),
            name="Mean with Error",
        )
        x = torch.randn(5, 2)

        mean_statistic.updated = False
        update, nb_sample = mean_statistic.update(x=x, count_samples=False)

        self.assertTrue(torch.allclose(update, x.sum(dim=0)))
        self.assertEqual(nb_sample, 0)
        self.assertEqual(mean_statistic.samples, 0)
        self.assertEqual(mean_statistic._batches, 0)

    def test_stop_trace_computation(self):
        num_batches = 3
        batch_size = 10
        total_samples = torch.Size((num_batches * batch_size,))
        mean = torch.tensor([3.0, 4.0])
        cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])

        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        samples = dist.sample(total_samples)
        dataloader = DataLoader(
            TensorDataset(samples), batch_size=batch_size, shuffle=False
        )

        mean_statistic = TensorStatiticWithEstimationError(
            shape=None,
            update_function=lambda x: (x.sum(dim=0), x.size(0)),
            name="Mean with Error",
            trace_precision=5.0,
        )

        self.assertRaises(AssertionError, mean_statistic.error)
        for i, (batch,) in enumerate(dataloader):
            mean_statistic.updated = False
            mean_statistic.update(x=batch)

        self.assertFalse(mean_statistic._compute_trace)


if __name__ == "__main__":
    main()
