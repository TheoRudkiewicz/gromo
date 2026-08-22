"""
Tests for GrowRALinear and GrowRAConv2d (module-level).

Tests cover:
- GrowRALinear: init, forward, merge, utilities, FOGRO pipeline
- GrowRAConv2d: init, forward, merge, FOGRO pipeline
- LinearGrowingModule interoperability
- dropout: forward behaviour and extra_repr
- extended_forward with rank > 0
"""

import copy
import math
import unittest
from typing import NamedTuple
from unittest import TestCase

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from gromo.containers.growing_block import GrowingBlock, LinearGrowingBlock
from gromo.growra.container import (
    get_growra_model,
    get_growra_modules,
)
from gromo.growra.module import (
    DEFAULT_GAIN,
    LR_INIT,
    FactorScaling,
    GrowRAConv2d,
    GrowRALinear,
    GrowRANormalization,
    Scaling,
)
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase


try:
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import Linear as _PeftLoraLinear

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


def _linear(*args, **kwargs):
    return nn.Linear(*args, device=global_device(), **kwargs)


def _conv2d(*args, **kwargs):
    return nn.Conv2d(*args, device=global_device(), **kwargs)


def _randn(*args, **kwargs):
    return torch.randn(*args, device=global_device(), **kwargs)


def _ones(*args, **kwargs):
    return torch.ones(*args, device=global_device(), **kwargs)


class TestScaling(TestCase):
    """Direct tests for the Scaling helper module."""

    def test_extended_forward_both_none(self):
        scaling = Scaling(lambda rank: 2.0, lambda: 1)
        x, x_ext = scaling.extended_forward(None, None)
        self.assertIsNone(x)
        self.assertIsNone(x_ext)

    def test_extended_forward_x_only(self):
        scaling = Scaling(lambda rank: 2.0, lambda: 1)
        x, x_ext = scaling.extended_forward(_ones(3), None)
        assert x is not None
        self.assertTrue(torch.allclose(x, _ones(3) * 2.0))
        self.assertIsNone(x_ext)

    def test_extended_forward_both_provided(self):
        scaling = Scaling(lambda rank: 2.0, lambda: 1)
        x, x_ext = scaling.extended_forward(_ones(3), _ones(4))
        assert x is not None and x_ext is not None
        self.assertTrue(torch.allclose(x, _ones(3) * 2.0))
        self.assertTrue(torch.allclose(x_ext, _ones(4) * 2.0))


class TestGrowingGrowraLinearInit(TestCase):
    """Tests for GrowRALinear initialization."""

    def test_basic_init(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4, scaling=0.5)
        self.assertIsInstance(lora, LinearGrowingBlock)
        self.assertEqual(lora.in_features, 10)
        self.assertEqual(lora.out_features, 20)
        self.assertEqual(lora.rank, 4)
        self.assertAlmostEqual(lora.scaling, 0.5)

    def test_init_rank_zero(self):
        linear = _linear(5, 3)
        lora = GrowRALinear(linear, rank=0)
        self.assertEqual(lora.rank, 0)
        self.assertAlmostEqual(lora.scaling, 0.0)

    def test_original_frozen(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        for p in lora.linear.parameters():
            self.assertFalse(p.requires_grad)

    def test_lora_params_trainable(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        params = lora.growra_parameters()
        self.assertTrue(len(params) > 0)
        for p in params:
            self.assertTrue(p.requires_grad)

    def test_scaling_property(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4, scaling=0.5)
        self.assertAlmostEqual(lora.scaling, 0.5)

    def test_scaling_property_larger(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4, scaling=2.0)
        self.assertAlmostEqual(lora.scaling, 2.0)

    def test_scaling_default_rank_invariant(self):
        """Default scaling=1.0 gives scaling=1 regardless of rank."""
        linear = _linear(10, 20)
        for rank in (1, 4, 16):
            lora = GrowRALinear(linear, rank=rank)
            self.assertAlmostEqual(lora.scaling, 1.0, msg=f"rank={rank}")

    def test_scaling_callable(self):
        """Callable scaling is evaluated at the current rank."""
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4, scaling=lambda r: r**-0.5)
        self.assertAlmostEqual(lora.scaling, 4**-0.5, places=5)

    def test_first_second_layer_properties(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        self.assertEqual(lora.first_layer.in_features, 10)
        self.assertEqual(lora.first_layer.out_features, 4)
        self.assertEqual(lora.second_layer.in_features, 4)
        self.assertEqual(lora.second_layer.out_features, 20)

    def test_weight_bias_properties(self):
        linear = _linear(10, 20, bias=True)
        lora = GrowRALinear(linear, rank=2)
        self.assertIs(lora.weight, linear.weight)
        self.assertIs(lora.bias, linear.bias)

    def test_no_bias(self):
        linear = _linear(10, 20, bias=False)
        lora = GrowRALinear(linear, rank=2)
        self.assertIsNone(lora.bias)

    def test_extra_repr(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4, scaling=0.5)
        r = lora.extra_repr()
        self.assertIn("in_features=10", r)
        self.assertIn("out_features=20", r)
        self.assertIn("rank=4", r)
        self.assertIn("scaling=0.5", r)


class TestGrowingGrowraLinearForward(TestCase):
    """Tests for GrowRALinear forward pass."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        x = _randn(5, 10)
        out = lora(x)
        self.assertEqual(out.shape, (5, 20))

    def test_forward_rank_zero_equals_original(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=0)
        x = _randn(5, 10)
        out_lora = lora(x)
        out_orig = linear(x)
        self.assertTrue(torch.allclose(out_lora, out_orig))

    def test_forward_with_nonzero_weights(self):
        """With non-zero B weights, output differs from original."""
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(5, 10)
        out_lora = lora(x)
        out_orig = linear(x)
        self.assertFalse(torch.allclose(out_lora, out_orig))

    def test_forward_3d_input(self):
        """Test with sequence-like input (batch, seq, features)."""
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        x = _randn(3, 7, 10)
        out = lora(x)
        self.assertEqual(out.shape, (3, 7, 20))

    def test_gradient_flows_to_lora_only(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(5, 10)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        # LoRA params should have gradients
        for p in lora.growra_parameters():
            self.assertIsNotNone(p.grad)
        # Frozen params should not
        self.assertIsNone(lora.linear.weight.grad)

    def test_forward_numerical_correctness(self):
        """output == linear(x) + scaling * B(A(x)) exactly."""
        linear = _linear(10, 20)
        sc = 0.5
        lora = GrowRALinear(linear, rank=4, scaling=sc)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(5, 10)
        with torch.no_grad():
            expected = (
                linear(x)
                + sc * (lora.second_layer.weight @ lora.first_layer.weight @ x.T).T
            )
            actual = lora(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))


class TestGrowingGrowraLinearMerge(TestCase):
    """Tests for GrowRALinear merge."""

    def setUp(self):
        torch.manual_seed(42)

    def test_merge_produces_linear(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        merged = lora.merge()
        self.assertIsInstance(merged, nn.Linear)
        self.assertEqual(merged.in_features, 10)
        self.assertEqual(merged.out_features, 20)

    def test_merge_output_matches_forward(self):
        """Merged layer should produce same output as LoRA forward."""
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4, scaling=0.5)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(8, 10)
        with torch.no_grad():
            out_lora = lora(x)
            merged = lora.merge()
            out_merged = merged(x)
        self.assertTrue(
            torch.allclose(out_lora, out_merged, atol=1e-5),
            f"Max diff: {(out_lora - out_merged).abs().max().item()}",
        )

    def test_merge_rank_zero(self):
        linear = _linear(10, 20)
        w_orig = linear.weight.data.clone()
        lora = GrowRALinear(linear, rank=0)
        merged = lora.merge()
        self.assertTrue(torch.allclose(merged.weight, w_orig))

    def test_merge_with_bias(self):
        linear = _linear(10, 20, bias=True)
        b_orig = linear.bias.data.clone()
        lora = GrowRALinear(linear, rank=4)
        merged = lora.merge()
        self.assertIsNotNone(merged.bias)
        self.assertTrue(torch.allclose(merged.bias, b_orig))

    def test_merge_without_bias(self):
        linear = _linear(10, 20, bias=False)
        lora = GrowRALinear(linear, rank=4)
        merged = lora.merge()
        self.assertIsNone(merged.bias)

    def test_merge_does_not_modify_original(self):
        linear = _linear(10, 20)
        original_weight = linear.weight.data.clone()
        lora = GrowRALinear(linear, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        _ = lora.merge()
        self.assertTrue(torch.allclose(linear.weight.data, original_weight))


class TestGrowingGrowraLinearUtilities(TestCase):
    """Tests for GrowRALinear utility methods."""

    def test_lora_parameters(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        params = lora.growra_parameters()
        self.assertTrue(len(params) > 0)
        for p in params:
            self.assertTrue(p.requires_grad)

    def test_reset_adapter(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        lora.reset_adapter()
        self.assertTrue(
            torch.allclose(
                lora.second_layer.weight.data,
                torch.zeros_like(lora.second_layer.weight.data),
            )
        )


# ===================== _matches_target Tests =====================


class TestFOGROGrowthPipeline(TestCase):
    """Test the FOGRO growth pipeline on GrowRALinear."""

    def setUp(self):
        self.in_features = 8
        self.out_features = 6
        self.batch_size = 16

    def _make_lora(self, rank=0):
        linear = _linear(self.in_features, self.out_features)
        return GrowRALinear(linear, rank=rank)

    def test_init_computation(self):
        lora = self._make_lora(rank=0)
        lora.init_computation()
        self.assertTrue(lora.first_layer.store_input)
        self.assertTrue(lora.second_layer.store_pre_activity)

    def test_reset_computation(self):
        lora = self._make_lora(rank=0)
        lora.init_computation()
        lora.reset_computation()
        self.assertFalse(lora.first_layer.store_input)
        self.assertFalse(lora.second_layer.store_pre_activity)

    def test_full_fogro_loop_rank_zero(self):
        """Test full FOGRO loop starting from rank 0."""
        lora = self._make_lora(rank=0)

        lora.init_computation()

        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        output = lora(x)
        loss = (output**2).sum() / 2
        loss.backward()

        lora.update_computation()

        lora.compute_optimal_updates(
            maximum_added_neurons=4,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )

        self.assertIsNotNone(lora.first_layer.extended_output_layer)
        self.assertIsNotNone(lora.second_layer.extended_input_layer)
        self.assertIsNotNone(lora.eigenvalues_extension)

        old_rank = lora.rank
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        lora.apply_change(scaling_factor=1.0, extension_size=2)
        self.assertEqual(lora.rank, old_rank + 2)

        # Verify forward still works
        x2 = _randn(3, self.in_features)
        y = lora(x2)
        self.assertEqual(y.shape, (3, self.out_features))

        lora.reset_computation()

    def test_full_fogro_loop_nonzero_rank(self):
        """Test full FOGRO loop from nonzero rank with full fogro method."""
        lora = self._make_lora(rank=2)

        lora.init_computation()

        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        output = lora(x)
        loss = (output**2).sum() / 2
        loss.backward()
        lora.update_computation()

        lora.compute_optimal_updates(
            maximum_added_neurons=3,
            compute_delta=True,
            use_covariance=True,
            use_projection=True,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=False,
        )

        self.assertIsNotNone(lora.eigenvalues_extension)

        old_rank = lora.rank
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        lora.apply_change(scaling_factor=1.0, extension_size=2)
        self.assertEqual(lora.rank, old_rank + 2)

        lora.reset_computation()

    def test_fogro_tiny_method(self):
        """Test FOGRO growth with TINY method."""
        lora = self._make_lora(rank=2)

        lora.init_computation()
        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        output = lora(x)
        loss = (output**2).sum() / 2
        loss.backward()
        lora.update_computation()

        lora.compute_optimal_updates(
            maximum_added_neurons=3,
            compute_delta=False,
            use_covariance=True,
            use_projection=True,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=False,
        )
        self.assertIsNotNone(lora.eigenvalues_extension)
        lora.reset_computation()

    def test_first_order_improvement(self):
        """Test that first_order_improvement is accessible after compute."""
        lora = self._make_lora(rank=0)
        lora.init_computation()
        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        output = lora(x)
        loss = (output**2).sum() / 2
        loss.backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=4,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        improvement = lora.first_order_improvement
        self.assertIsInstance(improvement, torch.Tensor)

    def test_extended_forward_runs_after_fogro_step(self):
        """After compute_optimal_updates at rank=0, extended_forward uses the elif branch
        (extended_output_layer set) without error and equals forward (scaling=0)."""
        lora = self._make_lora(rank=0)
        lora.init_computation()
        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        output = lora(x)
        loss = (output**2).sum() / 2
        loss.backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=4,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        self.assertIsNotNone(lora.first_layer.extended_output_layer)
        # extension_scaling is 0 by default, so extended_forward == forward
        x_eval = _randn(3, self.in_features)
        out_forward = lora(x_eval)
        out_extended = lora.extended_forward(x_eval)
        self.assertEqual(out_extended.shape, (3, self.out_features))
        self.assertTrue(
            torch.allclose(out_forward, out_extended),
            "With zero extension scaling, extended_forward should equal forward",
        )
        lora.reset_computation()

    def test_apply_change_no_extension(self):
        """apply_change(apply_extension=False) skips the normalization."""
        lora = self._make_lora(rank=0)
        lora.init_computation()
        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        (lora(x) ** 2).sum().backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        rank_before = lora.rank
        lora.apply_change(scaling_factor=1.0, extension_size=2, apply_extension=False)
        self.assertEqual(lora.rank, rank_before)
        lora.reset_computation()

    def test_normalization_noop_without_growth(self):
        """Normalizing is a no-op when nothing is pending."""
        lora = self._make_lora(rank=2)
        self.assertFalse(lora._has_pending_extensions())
        a_before = lora.first_layer.weight.clone()
        b_before = lora.second_layer.weight.clone()
        lora.normalize_optimal_updates(normalization_type="growra")
        self.assertTrue(torch.equal(lora.first_layer.weight, a_before))
        self.assertTrue(torch.equal(lora.second_layer.weight, b_before))

    def test_apply_change_lr_init_override(self):
        """apply_change(lr_init=...) overrides self.lr_init and is consumed
        (not forwarded to the base GrowingBlock.apply_change)."""
        lora = self._make_lora(rank=0)
        lora.init_computation()
        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        (lora(x) ** 2).sum().backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        lora.apply_change(scaling_factor=1.0, extension_size=2, lr_init=0.5)
        self.assertEqual(lora.lr_init, 0.5)
        lora.reset_computation()


class TestEnableDora(TestCase):
    """Tests for enable_dora() called post-construction."""

    def setUp(self):
        torch.manual_seed(0)

    def test_enable_dora_linear_post_construction(self):
        lora = GrowRALinear(_linear(10, 20), rank=2)
        self.assertFalse(lora.use_dora)
        self.assertIsNone(lora.magnitude)
        lora.enable_dora()
        self.assertTrue(lora.use_dora)
        self.assertIsNotNone(lora.magnitude)
        assert lora.magnitude is not None
        self.assertEqual(lora.magnitude.shape[0], 20)
        self.assertTrue(lora.magnitude.requires_grad)

    def test_enable_dora_linear_output_matches_before(self):
        """Enabling DoRA at rank=0 does not change the forward output."""
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=0)
        x = _randn(3, 10)
        with torch.no_grad():
            out_before = lora(x).clone()
        lora.enable_dora()
        with torch.no_grad():
            out_after = lora(x)
        self.assertTrue(torch.allclose(out_before, out_after, atol=1e-6))

    def test_enable_dora_linear_output_unchanged_at_nonzero_rank(self):
        """enable_dora() on an adapter with BA != 0 must not change the output."""
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(3, 10)
        with torch.no_grad():
            out_before = lora(x).clone()
        lora.enable_dora()
        with torch.no_grad():
            out_after = lora(x)
        self.assertTrue(
            torch.allclose(out_before, out_after, atol=1e-5),
            "enable_dora() changed the output when rank > 0 and BA != 0",
        )

    def test_enable_dora_conv_output_unchanged_at_nonzero_rank(self):
        """enable_dora() on Conv2d with BA != 0 must not change the output."""
        conv = _conv2d(3, 8, 3, padding=1)
        lora = GrowRAConv2d(conv, rank=4)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(2, 3, 8, 8)
        with torch.no_grad():
            out_before = lora(x).clone()
        lora.enable_dora()
        with torch.no_grad():
            out_after = lora(x)
        self.assertTrue(
            torch.allclose(out_before, out_after, atol=1e-5),
            "enable_dora() changed the output when rank > 0 and BA != 0",
        )

    def test_enable_dora_conv_post_construction(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=2)
        self.assertFalse(lora.use_dora)
        self.assertIsNone(lora.magnitude)
        lora.enable_dora()
        self.assertTrue(lora.use_dora)
        self.assertIsNotNone(lora.magnitude)
        assert lora.magnitude is not None
        self.assertEqual(lora.magnitude.shape[0], 8)
        self.assertTrue(lora.magnitude.requires_grad)

    def test_enable_dora_linear_idempotent(self):
        """A second enable_dora() call is a no-op: same magnitude tensor."""
        lora = GrowRALinear(_linear(10, 20), rank=2, use_dora=True)
        magnitude_before = lora.magnitude
        lora.enable_dora()
        self.assertIs(lora.magnitude, magnitude_before)

    def test_enable_dora_conv_idempotent(self):
        """A second enable_dora() call is a no-op: same magnitude tensor."""
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=2, use_dora=True)
        magnitude_before = lora.magnitude
        lora.enable_dora()
        self.assertIs(lora.magnitude, magnitude_before)


class TestGrowingGrowraLinearWithLinearGrowingModule(TestCase):
    """Tests that GrowRALinear works with LinearGrowingModule as input."""

    def setUp(self):
        torch.manual_seed(42)

    def test_init_from_linear_growing_module(self):
        lgm = LinearGrowingModule(
            in_features=10, out_features=20, name="test", device=global_device()
        )
        lora = GrowRALinear(lgm, rank=4, scaling=0.5)
        self.assertEqual(lora.in_features, 10)
        self.assertEqual(lora.out_features, 20)
        self.assertEqual(lora.rank, 4)

    def test_forward_shape(self):
        lgm = LinearGrowingModule(
            in_features=10, out_features=20, name="test", device=global_device()
        )
        lora = GrowRALinear(lgm, rank=4)
        x = _randn(5, 10)
        out = lora(x)
        self.assertEqual(out.shape, (5, 20))

    def test_forward_rank_zero(self):
        lgm = LinearGrowingModule(
            in_features=10, out_features=20, name="test", device=global_device()
        )
        lora = GrowRALinear(lgm, rank=0)
        x = _randn(5, 10)
        out_lora = lora(x)
        out_orig = lgm(x)
        self.assertTrue(torch.allclose(out_lora, out_orig))

    def test_frozen(self):
        lgm = LinearGrowingModule(
            in_features=10, out_features=20, name="test", device=global_device()
        )
        GrowRALinear(lgm, rank=4)
        for p in lgm.parameters():
            self.assertFalse(p.requires_grad)

    def test_merge(self):
        lgm = LinearGrowingModule(
            in_features=10, out_features=20, name="test", device=global_device()
        )
        lora = GrowRALinear(lgm, rank=4)
        merged = lora.merge()
        self.assertIsInstance(merged, nn.Linear)
        self.assertEqual(merged.weight.shape, (20, 10))

    def test_apply_growing_lora_on_lgm_model(self):
        """get_growing_lora_model should detect LinearGrowingModule layers."""
        lgm1 = LinearGrowingModule(
            in_features=10, out_features=20, name="l1", device=global_device()
        )
        lgm2 = LinearGrowingModule(
            in_features=20, out_features=5, name="l2", device=global_device()
        )
        model = nn.Sequential(lgm1, nn.ReLU(), lgm2)
        lora_model = get_growra_model(model)
        lora_mods = get_growra_modules(lora_model)
        self.assertEqual(len(lora_mods), 2)
        for m in lora_mods:
            self.assertIsInstance(m, GrowRALinear)

    def test_forward_after_apply_on_lgm_model(self):
        lgm1 = LinearGrowingModule(
            in_features=10, out_features=20, name="l1", device=global_device()
        )
        lgm2 = LinearGrowingModule(
            in_features=20, out_features=5, name="l2", device=global_device()
        )
        model = nn.Sequential(lgm1, nn.ReLU(), lgm2)
        lora_model = get_growra_model(model)
        x = _randn(3, 10)
        out = lora_model(x)
        self.assertEqual(out.shape, (3, 5))


# --------- Tests for GrowRAConv2d ---------


class TestGrowingGrowraConv2dInit(TestCase):
    """Tests for GrowRAConv2d initialization."""

    def test_basic_init(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=4, scaling=0.5)
        self.assertEqual(lora.in_channels, 3)
        self.assertEqual(lora.out_channels, 16)
        self.assertEqual(lora.rank, 4)
        self.assertAlmostEqual(lora.scaling, 0.5)

    def test_init_rank_zero(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=0)
        self.assertEqual(lora.rank, 0)
        self.assertAlmostEqual(lora.scaling, 0.0)

    def test_original_frozen(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=4)
        for p in lora.conv.parameters():
            self.assertFalse(p.requires_grad)

    def test_lora_params_trainable(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=4)
        params = lora.growra_parameters()
        self.assertTrue(len(params) > 0)
        for p in params:
            self.assertTrue(p.requires_grad)

    def test_scaling(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=4, scaling=2.0)
        self.assertAlmostEqual(lora.scaling, 2.0)

    def test_weight_bias_properties(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        lora = GrowRAConv2d(conv, rank=2)
        self.assertIs(lora.weight, conv.weight)
        self.assertIs(lora.bias, conv.bias)

    def test_extra_repr(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=4)
        r = lora.extra_repr()
        self.assertIn("in_channels=3", r)
        self.assertIn("out_channels=16", r)
        self.assertIn("rank=4", r)

    def test_scaling_callable(self):
        """Callable scaling is stored and evaluated at the current rank."""
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=4, scaling=lambda r: 1.0 / r)
        self.assertAlmostEqual(lora.scaling, 0.25)

    def test_grouped_conv_raises(self):
        conv = _conv2d(4, 8, kernel_size=3, padding=1, groups=2)
        with self.assertRaises(ValueError):
            GrowRAConv2d(conv, rank=2)


class TestGrowingGrowraConv2dForward(TestCase):
    """Tests for GrowRAConv2d forward pass."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=4)
        x = _randn(2, 3, 8, 8)
        out = lora(x)
        self.assertEqual(out.shape, (2, 16, 8, 8))

    def test_forward_rank_zero_equals_original(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=0)
        x = _randn(2, 3, 8, 8)
        out_lora = lora(x)
        out_orig = conv(x)
        self.assertTrue(torch.allclose(out_lora, out_orig))

    def test_forward_with_stride(self):
        conv = _conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        lora = GrowRAConv2d(conv, rank=4)
        x = _randn(2, 3, 8, 8)
        out = lora(x)
        self.assertEqual(out.shape, (2, 16, 4, 4))

    def test_gradient_flows_to_lora_only(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(2, 3, 8, 8)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        for p in lora.growra_parameters():
            self.assertIsNotNone(p.grad)
        self.assertIsNone(lora.conv.weight.grad)


class TestGrowingGrowraConv2dMerge(TestCase):
    """Tests for GrowRAConv2d merge."""

    def setUp(self):
        torch.manual_seed(42)

    def test_merge_rank_zero(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=0)
        merged = lora.merge()
        self.assertIsInstance(merged, nn.Conv2d)
        self.assertTrue(torch.allclose(merged.weight, conv.weight))

    def test_merge_preserves_shape(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=4)
        merged = lora.merge()
        self.assertEqual(merged.weight.shape, conv.weight.shape)

    def test_reset_adapter(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        lora.reset_adapter()
        self.assertTrue(
            torch.allclose(
                lora.second_layer.weight.data,
                torch.zeros_like(lora.second_layer.weight.data),
            )
        )


class TestGrowingGrowraConv2dFOGRO(TestCase):
    """Basic FOGRO pipeline test for GrowRAConv2d."""

    def setUp(self):
        torch.manual_seed(42)

    def test_fogro_pipeline_rank_zero(self):
        conv = _conv2d(3, 8, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=0)

        lora.init_computation()

        x = _randn(4, 3, 8, 8)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        lora.update_computation()

        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        lora.apply_change(scaling_factor=1.0, extension_size=2)
        lora.reset_computation()

        self.assertGreater(lora.rank, 0)

    def test_forward_after_growth(self):
        conv = _conv2d(3, 8, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=0)

        lora.init_computation()
        x = _randn(4, 3, 8, 8)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        lora.apply_change(scaling_factor=1.0, extension_size=2)
        lora.reset_computation()

        x2 = _randn(2, 3, 8, 8)
        out = lora(x2)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_apply_change_no_extension(self):
        """apply_change(apply_extension=False) skips the normalization."""
        conv = _conv2d(3, 8, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=0)
        lora.init_computation()
        x = _randn(4, 3, 8, 8)
        lora(x).sum().backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        rank_before = lora.rank
        lora.apply_change(scaling_factor=1.0, extension_size=2, apply_extension=False)
        self.assertEqual(lora.rank, rank_before)
        lora.reset_computation()

    def test_normalization_noop_without_growth(self):
        """Normalizing is a no-op when nothing is pending."""
        conv = _conv2d(3, 8, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=2)
        self.assertFalse(lora._has_pending_extensions())
        a_before = lora.first_layer.weight.clone()
        b_before = lora.second_layer.weight.clone()
        lora.normalize_optimal_updates(normalization_type="growra")
        self.assertTrue(torch.equal(lora.first_layer.weight, a_before))
        self.assertTrue(torch.equal(lora.second_layer.weight, b_before))

    def test_apply_change_lr_init_override(self):
        """apply_change(lr_init=...) overrides self.lr_init and is consumed
        (not forwarded to the base GrowingBlock.apply_change)."""
        conv = _conv2d(3, 8, kernel_size=3, padding=1)
        lora = GrowRAConv2d(conv, rank=0)
        lora.init_computation()
        x = _randn(4, 3, 8, 8)
        lora(x).sum().backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        lora.apply_change(scaling_factor=1.0, extension_size=2, lr_init=0.5)
        self.assertEqual(lora.lr_init, 0.5)
        lora.reset_computation()


# ===================== Dropout Tests =====================


class TestGrowraDropoutLinear(TestCase):
    """Tests for dropout in GrowRALinear."""

    def setUp(self):
        torch.manual_seed(0)

    def test_dropout_stored(self):
        lora = GrowRALinear(_linear(10, 20), rank=4, dropout=0.3)
        self.assertAlmostEqual(lora.dropout.p, 0.3)

    def test_forward_train_mode_is_stochastic(self):
        """In train mode with dropout, two forwards should differ."""
        lora = GrowRALinear(_linear(10, 20), rank=4, dropout=0.9)
        # A seed rank starts as a no-op (zero B), so the adapter path — and
        # with it the dropout applied to its input — is only observable once
        # B is non-zero, as it is after any training.
        nn.init.normal_(lora.second_layer.weight)
        lora.train()
        x = _ones(16, 10)
        out1 = lora(x)
        out2 = lora(x)
        self.assertFalse(torch.allclose(out1, out2))

    def test_forward_eval_mode_is_deterministic(self):
        """In eval mode, dropout is disabled — two forwards must be identical."""
        lora = GrowRALinear(_linear(10, 20), rank=4, dropout=0.9)
        lora.eval()
        x = _ones(16, 10)
        self.assertTrue(torch.allclose(lora(x), lora(x)))

    def test_extra_repr_shows_dropout(self):
        lora = GrowRALinear(_linear(10, 20), rank=2, dropout=0.25)
        self.assertIn("dropout=0.25", lora.extra_repr())

    def test_extra_repr_no_dropout_by_default(self):
        lora = GrowRALinear(_linear(10, 20), rank=2)
        self.assertNotIn("dropout", lora.extra_repr())

    def test_extended_forward_with_nonzero_rank(self):
        """extended_forward with rank > 0 returns base + scaling * lora."""
        lora = GrowRALinear(_linear(10, 20), rank=4)
        x = _randn(3, 10)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (3, 20))

    def test_forward_rank0_store_input(self):
        """rank=0 with store_input=True (after init_computation) uses lora path."""
        lora = GrowRALinear(_linear(10, 20), rank=0, dropout=0.5)
        lora.init_computation()
        x = _randn(4, 10)
        out = lora(x)
        self.assertEqual(out.shape, (4, 20))
        lora.reset_computation()

    def test_explicit_activation_skips_default(self):
        """Passing activation explicitly skips the `if activation is None` branch."""
        lora = GrowRALinear(_linear(10, 20), rank=2, activation=nn.ReLU())
        x = _randn(3, 10)
        self.assertEqual(lora(x).shape, (3, 20))

    def test_extended_forward_rank_zero(self):
        """extended_forward with rank=0 and no growth returns early (line 155)."""
        lora = GrowRALinear(_linear(10, 20), rank=0)
        x = _randn(3, 10)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (3, 20))


class TestGrowraDropoutConv2d(TestCase):
    """Tests for dropout in GrowRAConv2d."""

    def setUp(self):
        torch.manual_seed(0)

    def test_dropout_stored(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, dropout=0.3)
        self.assertAlmostEqual(lora.dropout.p, 0.3)

    def test_forward_train_mode_is_stochastic(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, dropout=0.9)
        # See the GrowRALinear counterpart: a seed rank is a no-op until B is
        # non-zero, which is what makes the dropout observable.
        nn.init.normal_(lora.second_layer.weight)
        lora.train()
        x = _ones(4, 3, 8, 8)
        self.assertFalse(torch.allclose(lora(x), lora(x)))

    def test_forward_eval_mode_is_deterministic(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, dropout=0.9)
        lora.eval()
        x = _ones(4, 3, 8, 8)
        self.assertTrue(torch.allclose(lora(x), lora(x)))

    def test_extra_repr_shows_dropout(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3), rank=2, dropout=0.5)
        self.assertIn("dropout=0.5", lora.extra_repr())

    def test_extra_repr_no_dropout_by_default(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3), rank=2)
        self.assertNotIn("dropout", lora.extra_repr())

    def test_extended_forward_with_nonzero_rank(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4)
        x = _randn(2, 3, 8, 8)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_forward_rank0_store_input(self):
        """rank=0 with store_input=True (after init_computation) uses lora path."""
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=0, dropout=0.5)
        lora.init_computation()
        x = _randn(2, 3, 8, 8)
        out = lora(x)
        self.assertEqual(out.shape, (2, 8, 8, 8))
        lora.reset_computation()

    def test_explicit_activation_skips_default(self):
        """Passing activation explicitly skips the `if activation is None` branch."""
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=2, activation=nn.ReLU())
        x = _randn(2, 3, 8, 8)
        self.assertEqual(lora(x).shape, (2, 8, 8, 8))

    def test_extended_forward_rank_zero(self):
        """extended_forward with rank=0 and no growth returns early (line 348)."""
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=0)
        x = _randn(2, 3, 8, 8)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_wrap_conv2d_growing_module(self):
        """GrowRAConv2d accepts a Conv2dGrowingModule (line 245)."""
        cgm = Conv2dGrowingModule(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,
            name="test_conv",
            device=global_device(),
        )
        lora = GrowRAConv2d(cgm, rank=2)
        self.assertEqual(lora.in_channels, 3)
        self.assertEqual(lora.out_channels, 8)

    def test_merge_conv2d_growing_module(self):
        """merge_lora on a Conv2dGrowingModule-backed LoRA (line 363)."""
        cgm = Conv2dGrowingModule(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,
            name="test_merge",
            device=global_device(),
        )
        lora = GrowRAConv2d(cgm, rank=2)
        merged = lora.merge()
        self.assertIsInstance(merged, nn.Conv2d)
        self.assertEqual(merged.weight.shape[0], 8)

    def test_merge_no_bias(self):
        """merge_lora on conv without bias (line 389->391 False branch)."""
        conv = _conv2d(3, 8, 3, padding=1, bias=False)
        lora = GrowRAConv2d(conv, rank=2)
        merged = lora.merge()
        self.assertIsNone(merged.bias)


class TestDoRALinear(TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_init_matches_base_layer(self):
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=0, use_dora=True)
        x = _randn(4, 10)
        self.assertTrue(torch.allclose(lora(x), linear(x), atol=1e-6))

    def test_explicit_device_skips_default_device_inference(self):
        device = global_device()
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=0, device=device, activation=nn.ReLU())
        self.assertEqual(
            lora.first_layer.weight.device, torch.device(linear.weight.device)
        )

    def test_forward_dora_with_dropout_train_mode(self):
        """DoRA forward's dropout branch: stochastic in train mode, matches
        base-layer output at rank=0 in expectation-free eval mode."""
        lora = GrowRALinear(_linear(10, 20), rank=4, use_dora=True, dropout=0.9)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        lora.train()
        x = _ones(16, 10)
        out1 = lora(x)
        out2 = lora(x)
        self.assertEqual(out1.shape, (16, 20))
        self.assertFalse(torch.allclose(out1, out2))

        lora.eval()
        self.assertTrue(torch.allclose(lora(x), lora(x)))

    def test_extended_forward_dora_rank_zero(self):
        lora = GrowRALinear(_linear(10, 20), rank=0, use_dora=True)
        x = _randn(3, 10)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (3, 20))

    def test_extended_forward_dora_nonzero_rank(self):
        lora = GrowRALinear(_linear(10, 20), rank=4, use_dora=True)
        x = _randn(3, 10)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (3, 20))

    def test_extended_forward_dora_with_directions(self):
        """DoRA extended_forward incorporates computed growth directions."""
        lora = GrowRALinear(_linear(10, 20), rank=2, use_dora=True)
        x = _randn(4, 10)
        lora.init_computation()
        lora.zero_grad()
        lora(x).sum().backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        lora.reset_computation()
        self.assertIsNotNone(lora.first_layer.extended_output_layer)
        with torch.no_grad():
            out_fwd = lora(x)
            out_ext = lora.extended_forward(x)
        self.assertEqual(out_ext.shape, (4, 20))
        self.assertFalse(
            torch.allclose(out_fwd, out_ext),
            "DoRA extended_forward must incorporate growth directions",
        )

    def test_extended_forward_dora_increment_linear_in_extension(self):
        """extended_forward increment must scale linearly with extension magnitude.

        Normalizing against ||W_ext|| causes norm drift when the extension is large,
        making the increment nonlinear. The norm reference must be frozen at ||W_base||.
        """
        torch.manual_seed(0)
        lora = GrowRALinear(_linear(8, 16), rank=2, use_dora=True)
        x = _randn(4, 8)
        lora.init_computation()
        lora.zero_grad()
        lora(x).sum().backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        lora.reset_computation()
        self.assertIsNotNone(lora.first_layer.extended_output_layer)
        B_ext = lora.second_layer.extended_input_layer
        assert B_ext is not None
        # Amplify extension so norm drift is measurable
        B_ext.weight.data *= 20.0
        x_eval = _randn(3, 8)
        with torch.no_grad():
            out_base = lora(x_eval)
            out_ext_1 = lora.extended_forward(x_eval)
            B_ext.weight.data *= 2.0
            out_ext_2 = lora.extended_forward(x_eval)
        delta_1 = out_ext_1 - out_base
        delta_2 = out_ext_2 - out_base
        self.assertTrue(
            torch.allclose(delta_2, 2.0 * delta_1, atol=1e-4),
            "DoRA extended_forward: increment must scale linearly with extension magnitude",
        )

    def test_magnitude_is_trainable(self):
        lora = GrowRALinear(_linear(10, 20), rank=4, use_dora=True)
        self.assertIsNotNone(lora.magnitude)
        assert lora.magnitude is not None
        self.assertTrue(lora.magnitude.requires_grad)
        self.assertTrue(any(p is lora.magnitude for p in lora.growra_parameters()))

    def test_extra_repr_mentions_dora(self):
        lora = GrowRALinear(_linear(10, 20), rank=4, use_dora=True)
        self.assertIn("use_dora=True", lora.extra_repr())

    def test_merge_matches_forward(self):
        lora = GrowRALinear(_linear(10, 20), rank=4, use_dora=True)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        with torch.no_grad():
            assert lora.magnitude is not None
            lora.magnitude.mul_(1.1)
        x = _randn(6, 10)
        with torch.no_grad():
            out_lora = lora(x)
            out_merged = lora.merge()(x)
        self.assertTrue(torch.allclose(out_lora, out_merged, atol=1e-5))

    def test_dora_no_double_gradient_with_store_input(self):
        """With store_input=True, A's gradient must equal the shadow-path gradient only.

        The shadow is super().forward(x) = W0(x) + s*B(A(x)).
        With the fix, the main DoRA path detaches A/B, so the shadow is the sole
        contributor to A.grad.  Without the fix, the main path also contributes,
        making A.grad strictly larger.
        """
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4, use_dora=True)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(5, 10)

        # Reference: shadow-path gradient (call the parent block forward directly,
        # bypassing the DoRA logic entirely).
        lora.zero_grad()
        shadow_out = LinearGrowingBlock.forward(lora, x)
        shadow_out.sum().backward()
        assert lora.first_layer.weight.grad is not None
        grad_A_shadow_only = lora.first_layer.weight.grad.clone()

        # With store_input=True (fixed): only shadow contributes to A.grad
        lora.zero_grad()
        lora.first_layer.store_input = True
        lora(x).sum().backward()
        assert lora.first_layer.weight.grad is not None
        grad_A_with_store = lora.first_layer.weight.grad.clone()
        lora.first_layer.store_input = False

        self.assertTrue(
            torch.allclose(grad_A_shadow_only, grad_A_with_store, atol=1e-5),
            "DoRA+store_input: A gradient differs from shadow-only gradient",
        )

    def test_dora_input_gradient_not_doubled_with_store_input(self):
        """x.grad must equal only the DoRA-path gradient when store_input=True.

        Without the fix, shadow=super().forward(x) back-propagates through x,
        adding (W0 + scaling*B@A)^T·grad on top of eff_w^T·grad.
        With the fix (x.detach()), x.grad comes solely from the DoRA path.
        """
        linear = _linear(10, 20)
        lora = GrowRALinear(linear, rank=4, use_dora=True)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(5, 10).requires_grad_(True)

        lora.first_layer.store_input = True
        lora(x).sum().backward()
        x_grad = x.grad.clone()
        lora.first_layer.store_input = False

        # Expected: gradient of F.linear(x, eff_w, bias).sum() w.r.t. x
        with torch.no_grad():
            weight = linear.weight + lora._delta_weight(detach_adapter=True)
            eff_w = lora.magnitude[:, None] * (weight / lora._weight_norm(weight))
        x2 = x.detach().requires_grad_(True)
        F.linear(x2, eff_w, linear.bias).sum().backward()
        expected = x2.grad.clone()

        self.assertTrue(
            torch.allclose(x_grad, expected, atol=1e-5),
            f"DoRA+store_input: x.grad is doubled by shadow path. "
            f"Max diff: {(x_grad - expected).abs().max().item():.2e}",
        )


class TestDoRAConv2d(TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_init_matches_base_layer(self):
        conv = _conv2d(3, 8, 3, padding=1)
        lora = GrowRAConv2d(conv, rank=0, use_dora=True)
        x = _randn(2, 3, 8, 8)
        self.assertTrue(torch.allclose(lora(x), conv(x), atol=1e-6))

    def test_explicit_device_skips_default_device_inference(self):
        device = global_device()
        conv = _conv2d(3, 8, 3, padding=1)
        lora = GrowRAConv2d(conv, rank=0, device=device, activation=nn.ReLU())
        self.assertEqual(lora.first_layer.weight.device, torch.device(conv.weight.device))

    def test_forward_dora_with_dropout_train_mode(self):
        """DoRA forward's dropout branch: stochastic in train mode, matches
        base-layer output at rank=0 in expectation-free eval mode."""
        lora = GrowRAConv2d(
            _conv2d(3, 8, 3, padding=1), rank=4, use_dora=True, dropout=0.9
        )
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        lora.train()
        x = _ones(4, 3, 8, 8)
        out1 = lora(x)
        out2 = lora(x)
        self.assertEqual(out1.shape, (4, 8, 8, 8))
        self.assertFalse(torch.allclose(out1, out2))

        lora.eval()
        self.assertTrue(torch.allclose(lora(x), lora(x)))

    def test_extended_forward_dora_rank_zero(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=0, use_dora=True)
        x = _randn(2, 3, 8, 8)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_extended_forward_dora_nonzero_rank(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, use_dora=True)
        x = _randn(2, 3, 8, 8)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_extended_forward_dora_with_directions(self):
        """DoRA Conv2d extended_forward incorporates computed growth directions."""
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=2, use_dora=True)
        x = _randn(2, 3, 8, 8)
        lora.init_computation()
        lora.zero_grad()
        lora(x).sum().backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        lora.reset_computation()
        self.assertIsNotNone(lora.first_layer.extended_output_layer)
        with torch.no_grad():
            out_fwd = lora(x)
            out_ext = lora.extended_forward(x)
        self.assertEqual(out_ext.shape, (2, 8, 8, 8))
        self.assertFalse(
            torch.allclose(out_fwd, out_ext),
            "DoRA Conv2d extended_forward must incorporate growth directions",
        )

    def test_magnitude_is_trainable(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, use_dora=True)
        self.assertIsNotNone(lora.magnitude)
        assert lora.magnitude is not None
        self.assertTrue(any(p is lora.magnitude for p in lora.growra_parameters()))

    def test_extra_repr_mentions_dora(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, use_dora=True)
        self.assertIn("use_dora=True", lora.extra_repr())

    def test_merge_matches_forward(self):
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, use_dora=True)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        with torch.no_grad():
            assert lora.magnitude is not None
            lora.magnitude.mul_(0.9)
        x = _randn(2, 3, 8, 8)
        with torch.no_grad():
            out_lora = lora(x)
            out_merged = lora.merge()(x)
        self.assertTrue(torch.allclose(out_lora, out_merged, atol=1e-5))

    def test_dora_no_double_gradient_with_store_input(self):
        """With store_input=True, A's gradient must equal the shadow-path gradient only."""
        from gromo.containers.growing_block import Conv2dGrowingBlock

        conv = _conv2d(3, 8, 3, padding=1)
        lora = GrowRAConv2d(conv, rank=4, use_dora=True)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(2, 3, 8, 8)

        lora.zero_grad()
        Conv2dGrowingBlock.forward(lora, x).sum().backward()
        assert lora.first_layer.weight.grad is not None
        grad_A_shadow_only = lora.first_layer.weight.grad.clone()

        lora.zero_grad()
        lora.first_layer.store_input = True
        lora(x).sum().backward()
        assert lora.first_layer.weight.grad is not None
        grad_A_with_store = lora.first_layer.weight.grad.clone()
        lora.first_layer.store_input = False

        self.assertTrue(
            torch.allclose(grad_A_shadow_only, grad_A_with_store, atol=1e-5),
            "DoRA+store_input: A gradient differs from shadow-only gradient",
        )

    def test_dora_input_gradient_not_doubled_with_store_input(self):
        """x.grad must equal only the DoRA-path gradient when store_input=True.

        Without the fix, shadow=super().forward(x) back-propagates through x,
        adding (W0 + scaling*B@A)^T·grad on top of eff_w^T·grad.
        With the fix (x.detach()), x.grad comes solely from the DoRA path.
        """
        conv = _conv2d(3, 8, 3, padding=1)
        lora = GrowRAConv2d(conv, rank=4, use_dora=True)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(2, 3, 8, 8).requires_grad_(True)

        lora.first_layer.store_input = True
        lora(x).sum().backward()
        x_grad = x.grad.clone()
        lora.first_layer.store_input = False

        orig = lora._conv_base()
        with torch.no_grad():
            weight = orig.weight + lora._delta_weight(detach_adapter=True)
            eff_w = lora.magnitude[:, None, None, None] * (
                weight / lora._weight_norm(weight)
            )
        x2 = x.detach().requires_grad_(True)
        F.conv2d(
            x2, eff_w, orig.bias, orig.stride, orig.padding, orig.dilation, orig.groups
        ).sum().backward()
        expected = x2.grad.clone()

        self.assertTrue(
            torch.allclose(x_grad, expected, atol=1e-5),
            f"DoRA Conv2d+store_input: x.grad is doubled by shadow path. "
            f"Max diff: {(x_grad - expected).abs().max().item():.2e}",
        )

    def test_extended_forward_dora_increment_linear_in_extension(self):
        """Conv2d extended_forward increment must scale linearly with extension magnitude."""
        torch.manual_seed(0)
        lora = GrowRAConv2d(_conv2d(3, 8, 3, padding=1), rank=2, use_dora=True)
        x = _randn(2, 3, 8, 8)
        lora.init_computation()
        lora.zero_grad()
        lora(x).sum().backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        lora.reset_computation()
        self.assertIsNotNone(lora.first_layer.extended_output_layer)
        B_ext = lora.second_layer.extended_input_layer
        assert B_ext is not None
        B_ext.weight.data *= 20.0
        x_eval = _randn(2, 3, 8, 8)
        with torch.no_grad():
            out_base = lora(x_eval)
            out_ext_1 = lora.extended_forward(x_eval)
            B_ext.weight.data *= 2.0
            out_ext_2 = lora.extended_forward(x_eval)
        delta_1 = out_ext_1 - out_base
        delta_2 = out_ext_2 - out_base
        self.assertTrue(
            torch.allclose(delta_2, 2.0 * delta_1, atol=1e-4),
            "DoRA Conv2d extended_forward: increment must scale linearly with extension magnitude",
        )


# ---------------------------------------------------------------------------
# PEFT compatibility helpers
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    """Minimal wrapper so get_peft_model has a named module to target."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.fc = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _find_peft_lora_module(peft_model: nn.Module) -> "_PeftLoraLinear":
    for _, m in peft_model.named_modules():
        if isinstance(m, _PeftLoraLinear):
            return m
    raise AssertionError("No PEFT LoRA Linear found in model")


class TestGrowRAMatchesPEFT(TestCase):
    """GrowRALinear must be mathematically equivalent to PEFT LoRA.

    Both implement the same decomposition::

        output = W_base(x) + (alpha / rank) * B(A(x))

    so given the same base weights and the same A / B matrices the two
    forward passes must produce identical results.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)

    @unittest.skipUnless(HAS_PEFT, "peft is not installed")
    def test_dora_rank0_matches_peft(self):
        """At rank=0, DoRA output equals base layer — same as PEFT DoRA at init."""
        linear = _linear(10, 20)
        growra = GrowRALinear(copy.deepcopy(linear), rank=0, use_dora=True)
        peft_model = get_peft_model(
            _SimpleModel(copy.deepcopy(linear)),
            LoraConfig(
                r=4, lora_alpha=4, target_modules=["fc"], use_dora=True, bias="none"
            ),
        )
        x = _randn(4, 10)
        peft_model.eval()
        growra.eval()
        with torch.no_grad():
            out_growra = growra(x)
            out_peft = peft_model(x)
            out_base = linear(x)
        self.assertTrue(torch.allclose(out_growra, out_base, atol=1e-6))
        self.assertTrue(torch.allclose(out_peft, out_base, atol=1e-6))

    @unittest.skipUnless(HAS_PEFT, "peft is not installed")
    def test_dora_magnitude_init_matches_paper(self):
        """Magnitude is initialized to column norms of W0 (paper eq. 5)."""
        linear = _linear(10, 20)
        growra = GrowRALinear(copy.deepcopy(linear), rank=0, use_dora=True)
        expected = linear.weight.norm(dim=1)
        assert growra.magnitude is not None
        self.assertTrue(
            torch.allclose(growra.magnitude.data, expected, atol=1e-6),
            "magnitude should equal ||W0||_col at rank 0",
        )

    @unittest.skipUnless(HAS_PEFT, "peft is not installed")
    def test_dora_forward_matches_peft_same_weights(self):
        """With identical A, B, magnitude → GrowRA DoRA and PEFT DoRA produce same output."""
        in_f, out_f, rank = 10, 20, 4
        linear = _linear(in_f, out_f)

        peft_model = get_peft_model(
            _SimpleModel(copy.deepcopy(linear)),
            LoraConfig(
                r=rank, lora_alpha=rank, target_modules=["fc"], use_dora=True, bias="none"
            ),
        )
        peft_mod = _find_peft_lora_module(peft_model)
        nn.init.normal_(peft_mod.lora_A["default"].weight)
        nn.init.normal_(peft_mod.lora_B["default"].weight)
        peft_mag = peft_mod.lora_magnitude_vector["default"].weight.data.squeeze()
        A_w = peft_mod.lora_A["default"].weight.data.clone()
        B_w = peft_mod.lora_B["default"].weight.data.clone()

        growra = GrowRALinear(
            copy.deepcopy(linear), rank=rank, scaling=1.0, use_dora=True
        )
        with torch.no_grad():
            growra.first_layer.weight.copy_(A_w)
            growra.second_layer.weight.copy_(B_w)
            assert growra.magnitude is not None
            growra.magnitude.data.copy_(peft_mag)

        x = _randn(5, in_f)
        peft_model.eval()
        growra.eval()
        with torch.no_grad():
            out_peft = peft_model(x)
            out_growra = growra(x)
        self.assertTrue(
            torch.allclose(out_peft, out_growra, atol=1e-5),
            f"Max diff: {(out_peft - out_growra).abs().max().item()}",
        )

    @unittest.skipUnless(HAS_PEFT, "peft is not installed")
    def test_dora_norm_denominator_detached(self):
        """||W+BA||_col is detached from gradient graph (DoRA paper §4.3)."""
        linear = _linear(8, 12)
        lora = GrowRALinear(linear, rank=4, use_dora=True)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(3, 8)
        out = lora(x)
        out.sum().backward()
        assert lora.magnitude is not None
        # With detach: magnitude.grad = sum_x(x * direction) — no norm-denominator term.
        # With no detach: extra coupling term appears, making grad larger in magnitude.
        # We verify by recomputing expected gradient manually.
        with torch.no_grad():
            W = linear.weight + lora._delta_weight()
            W_norm = lora._weight_norm(W).squeeze(1)  # (out,)
            direction = W / W_norm[:, None]  # normalised rows
            # ∂L/∂m_i = sum_j direction[i,j] * sum_batch x[b,j]
            x_sum = x.sum(0)  # (in,)
            expected_mag_grad = (direction * x_sum).sum(1)  # (out,)
        self.assertTrue(
            torch.allclose(lora.magnitude.grad, expected_mag_grad, atol=1e-4),
            "magnitude gradient incorrect — norm denominator may not be detached",
        )

    @unittest.skipUnless(HAS_PEFT, "peft is not installed")
    def test_dora_gradients_match_peft(self):
        """Gradients of A, B and magnitude match PEFT DoRA backward exactly."""
        in_f, out_f, rank = 8, 12, 4
        linear = _linear(in_f, out_f)

        peft_model = get_peft_model(
            _SimpleModel(copy.deepcopy(linear)),
            LoraConfig(
                r=rank, lora_alpha=rank, target_modules=["fc"], use_dora=True, bias="none"
            ),
        )
        peft_mod = _find_peft_lora_module(peft_model)
        torch.manual_seed(7)
        nn.init.normal_(peft_mod.lora_A["default"].weight)
        nn.init.normal_(peft_mod.lora_B["default"].weight)
        A_w = peft_mod.lora_A["default"].weight.data.clone()
        B_w = peft_mod.lora_B["default"].weight.data.clone()
        peft_mag = peft_mod.lora_magnitude_vector["default"].weight.data.squeeze().clone()

        growra = GrowRALinear(
            copy.deepcopy(linear), rank=rank, scaling=1.0, use_dora=True
        )
        with torch.no_grad():
            growra.first_layer.weight.copy_(A_w)
            growra.second_layer.weight.copy_(B_w)
            assert growra.magnitude is not None
            growra.magnitude.data.copy_(peft_mag)

        x = _randn(5, in_f)

        peft_model.train()
        peft_model(x).sum().backward()
        grad_A_peft = peft_mod.lora_A["default"].weight.grad.clone()
        grad_B_peft = peft_mod.lora_B["default"].weight.grad.clone()
        grad_m_peft = (
            peft_mod.lora_magnitude_vector["default"].weight.grad.squeeze().clone()
        )

        growra.train()
        growra(x).sum().backward()

        for name, g_peft, g_growra in [
            ("A", grad_A_peft, growra.first_layer.weight.grad),
            ("B", grad_B_peft, growra.second_layer.weight.grad),
            ("magnitude", grad_m_peft, growra.magnitude.grad),
        ]:
            self.assertTrue(
                torch.allclose(g_peft, g_growra, atol=1e-5),
                f"grad {name} mismatch: max diff {(g_peft - g_growra).abs().max().item():.2e}",
            )

    @unittest.skipUnless(HAS_PEFT, "peft is not installed")
    def test_rank_zero_equals_base(self):
        """At rank=0 GrowRA output equals the frozen base — same as PEFT's init."""
        linear = _linear(10, 20)
        growra = GrowRALinear(copy.deepcopy(linear), rank=0)
        x = _randn(4, 10)
        with torch.no_grad():
            self.assertTrue(torch.allclose(growra(x), linear(x)))

    @unittest.skipUnless(HAS_PEFT, "peft is not installed")
    def test_forward_matches_peft_lora(self):
        """Identical A/B weights → identical forward output."""
        in_f, out_f, rank = 10, 20, 4
        peft_alpha = 4.0  # PEFT lora_alpha; effective scaling = peft_alpha/rank = 1.0
        linear = _linear(in_f, out_f)

        # --- PEFT side ---
        peft_model = get_peft_model(
            _SimpleModel(copy.deepcopy(linear)),
            LoraConfig(r=rank, lora_alpha=peft_alpha, target_modules=["fc"], bias="none"),
        )
        peft_mod = _find_peft_lora_module(peft_model)
        # Non-zero B so the adapter actually contributes to the output
        nn.init.normal_(peft_mod.lora_A["default"].weight)
        nn.init.normal_(peft_mod.lora_B["default"].weight)

        # --- GrowRA side — same frozen base weights, scaling = peft_alpha/rank ---
        growra = GrowRALinear(copy.deepcopy(linear), rank=rank, scaling=peft_alpha / rank)
        with torch.no_grad():
            growra.first_layer.weight.copy_(
                peft_mod.lora_A["default"].weight.to(growra.first_layer.weight.device)
            )
            growra.second_layer.weight.copy_(
                peft_mod.lora_B["default"].weight.to(growra.second_layer.weight.device)
            )

        x = _randn(5, in_f)
        peft_model.eval()
        growra.eval()
        with torch.no_grad():
            out_peft = peft_model(x)
            out_growra = growra(x)

        self.assertTrue(
            torch.allclose(out_peft, out_growra, atol=1e-5),
            f"Max diff: {(out_peft - out_growra).abs().max().item()}",
        )

    @unittest.skipUnless(HAS_PEFT, "peft is not installed")
    def test_scaling_matches(self):
        """alpha / rank scaling is identical between GrowRA and PEFT."""
        in_f, out_f, rank = 8, 16, 2
        peft_alpha = 8.0  # effective scaling = peft_alpha/rank = 4.0
        linear = _linear(in_f, out_f)

        peft_model = get_peft_model(
            _SimpleModel(copy.deepcopy(linear)),
            LoraConfig(r=rank, lora_alpha=peft_alpha, target_modules=["fc"], bias="none"),
        )
        peft_mod = _find_peft_lora_module(peft_model)
        nn.init.ones_(peft_mod.lora_A["default"].weight)
        nn.init.ones_(peft_mod.lora_B["default"].weight)

        growra = GrowRALinear(copy.deepcopy(linear), rank=rank, scaling=peft_alpha / rank)
        with torch.no_grad():
            growra.first_layer.weight.copy_(
                peft_mod.lora_A["default"].weight.to(growra.first_layer.weight.device)
            )
            growra.second_layer.weight.copy_(
                peft_mod.lora_B["default"].weight.to(growra.second_layer.weight.device)
            )

        x = _randn(3, in_f)
        peft_model.eval()
        growra.eval()
        with torch.no_grad():
            out_peft = peft_model(x)
            out_growra = growra(x)

        self.assertTrue(
            torch.allclose(out_peft, out_growra, atol=1e-5),
            f"Max diff: {(out_peft - out_growra).abs().max().item()}",
        )

    @unittest.skipUnless(HAS_PEFT, "peft is not installed")
    def test_rslora_forward_matches_peft(self):
        """RSLoRA scaling (1/sqrt(r)) matches PEFT use_rslora=True."""
        in_f, out_f, rank = 8, 12, 4
        linear = _linear(in_f, out_f)

        peft_model = get_peft_model(
            _SimpleModel(copy.deepcopy(linear)),
            LoraConfig(
                r=rank,
                lora_alpha=1,
                use_rslora=True,
                target_modules=["fc"],
                bias="none",
            ),
        )
        peft_mod = _find_peft_lora_module(peft_model)
        torch.manual_seed(3)
        nn.init.normal_(peft_mod.lora_A["default"].weight)
        nn.init.normal_(peft_mod.lora_B["default"].weight)
        A_w = peft_mod.lora_A["default"].weight.data.clone()
        B_w = peft_mod.lora_B["default"].weight.data.clone()

        growra = GrowRALinear(copy.deepcopy(linear), rank=rank, scaling=lambda r: r**-0.5)
        with torch.no_grad():
            growra.first_layer.weight.copy_(A_w)
            growra.second_layer.weight.copy_(B_w)

        x = _randn(5, in_f)
        peft_model.eval()
        growra.eval()
        with torch.no_grad():
            out_peft = peft_model(x)
            out_growra = growra(x)
        self.assertTrue(
            torch.allclose(out_peft, out_growra, atol=1e-5),
            f"RSLoRA max diff: {(out_peft - out_growra).abs().max().item()}",
        )

    @unittest.skipUnless(HAS_PEFT, "peft is not installed")
    def test_dora_rslora_forward_and_gradients_match_peft(self):
        """DoRA + RSLoRA: forward and A/B/magnitude gradients match PEFT."""
        in_f, out_f, rank = 8, 12, 4
        linear = _linear(in_f, out_f)

        peft_model = get_peft_model(
            _SimpleModel(copy.deepcopy(linear)),
            LoraConfig(
                r=rank,
                lora_alpha=1,
                use_rslora=True,
                use_dora=True,
                target_modules=["fc"],
                bias="none",
            ),
        )
        peft_mod = _find_peft_lora_module(peft_model)
        torch.manual_seed(5)
        nn.init.normal_(peft_mod.lora_A["default"].weight)
        nn.init.normal_(peft_mod.lora_B["default"].weight)
        A_w = peft_mod.lora_A["default"].weight.data.clone()
        B_w = peft_mod.lora_B["default"].weight.data.clone()
        peft_mag = peft_mod.lora_magnitude_vector["default"].weight.data.squeeze().clone()

        growra = GrowRALinear(
            copy.deepcopy(linear), rank=rank, scaling=lambda r: r**-0.5, use_dora=True
        )
        with torch.no_grad():
            growra.first_layer.weight.copy_(A_w)
            growra.second_layer.weight.copy_(B_w)
            assert growra.magnitude is not None
            growra.magnitude.data.copy_(peft_mag)

        x = _randn(5, in_f)
        peft_model.eval()
        growra.eval()
        with torch.no_grad():
            out_peft = peft_model(x)
            out_growra = growra(x)
        self.assertTrue(
            torch.allclose(out_peft, out_growra, atol=1e-5),
            f"DoRA+RSLoRA forward max diff: {(out_peft - out_growra).abs().max().item()}",
        )

        peft_model.train()
        peft_model(x).sum().backward()
        grad_A_peft = peft_mod.lora_A["default"].weight.grad.clone()
        grad_B_peft = peft_mod.lora_B["default"].weight.grad.clone()
        grad_m_peft = (
            peft_mod.lora_magnitude_vector["default"].weight.grad.squeeze().clone()
        )

        growra.train()
        growra(x).sum().backward()

        for name, g_peft, g_growra in [
            ("A", grad_A_peft, growra.first_layer.weight.grad),
            ("B", grad_B_peft, growra.second_layer.weight.grad),
            ("magnitude", grad_m_peft, growra.magnitude.grad),
        ]:
            assert g_growra is not None
            self.assertTrue(
                torch.allclose(g_peft, g_growra, atol=1e-5),
                f"DoRA+RSLoRA grad {name} mismatch: max diff {(g_peft - g_growra).abs().max().item():.2e}",
            )


# ===================== Deepcopy Tests =====================


def _grow_linear(lora: GrowRALinear, added_rank: int) -> None:
    """One FOGRO step on a GrowRALinear; grows rank by added_rank."""
    lora.init_computation()
    x = _randn(4, lora.in_features)
    lora.zero_grad()
    (lora(x) ** 2).sum().backward()
    lora.update_computation()
    lora.compute_optimal_updates(
        maximum_added_neurons=added_rank + 1,
        compute_delta=False,
        use_covariance=True,
        use_projection=False,
        alpha_zero=False,
        omega_zero=False,
        ignore_singular_values=True,
        use_fisher=True,
    )
    lora.sub_select_optimal_added_parameters(keep_neurons=added_rank)
    lora.apply_change(scaling_factor=1.0, extension_size=added_rank)
    lora.reset_computation()


class TestDeepCopy(TestCase):
    """Verify deepcopy produces correctly wired GrowRA modules."""

    def test_linear_deepcopy_rewires_scaling_rank_getter(self):
        """After deepcopy, growing the copy must not affect _scaling.rank_getter of the original."""
        linear = _linear(8, 16)
        original = GrowRALinear(linear, rank=0, scaling=2.0)
        _grow_linear(original, added_rank=2)

        copied = copy.deepcopy(original)
        original_rank_before = original.rank

        # Grow only the copy
        _grow_linear(copied, added_rank=2)

        # Original is unchanged
        self.assertEqual(original.rank, original_rank_before)
        # Copy grew
        self.assertEqual(copied.rank, original_rank_before + 2)

        # _scaling.rank_getter must read EACH module's own rank
        self.assertEqual(original._scaling.rank_getter(), original.rank)
        self.assertEqual(copied._scaling.rank_getter(), copied.rank)

        # Forward and merge must be correct for the copy
        x = _randn(4, 8)
        out = copied(x)
        self.assertEqual(out.shape, (4, 16))
        merged = copied.merge()
        self.assertTrue(torch.allclose(out, merged(x), atol=1e-5))

    def test_linear_deepcopy_rewires_scaling_fn(self):
        """After deepcopy, _scaling.scaling_fn must be the copy's scaling_fn."""
        linear = _linear(8, 16)
        original = GrowRALinear(linear, rank=0, scaling=3.0)
        copied = copy.deepcopy(original)

        # Both scaling_fns are consistent within their own module
        self.assertIs(copied._scaling.scaling_fn, copied.scaling_fn)
        self.assertIs(original._scaling.scaling_fn, original.scaling_fn)

    def test_conv_deepcopy_rewires_scaling_rank_getter(self):
        """deepcopy of GrowRAConv2d correctly rewires _scaling.rank_getter."""
        conv = _conv2d(4, 8, 3, padding=1)
        original = GrowRAConv2d(conv, rank=0, scaling=2.0)
        copied = copy.deepcopy(original)

        self.assertEqual(copied._scaling.rank_getter(), copied.rank)
        self.assertIs(copied._scaling.scaling_fn, copied.scaling_fn)


# ===================== Fisher-mode Tests =====================


def _fisher_grow(lora: GrowRALinear, added_rank: int, n_batches: int = 4) -> None:
    """One Fisher FOGRO step: accumulate statistics then grow by added_rank."""
    lora.init_computation()
    for _ in range(n_batches):
        x = _randn(8, lora.in_features)
        lora.zero_grad()
        (lora(x) ** 2).sum().backward()
        lora.update_computation()
    lora.compute_optimal_updates(
        maximum_added_neurons=added_rank + 2,
        compute_delta=False,
        use_covariance=True,
        use_projection=False,
        alpha_zero=False,
        omega_zero=False,
        ignore_singular_values=True,
        use_fisher=True,
    )
    lora.sub_select_optimal_added_parameters(keep_neurons=added_rank)
    lora.apply_change(scaling_factor=1.0, extension_size=added_rank)
    lora.reset_computation()


class TestGrowRAFisher(TestCase):
    """End-to-end tests for the Fisher-mode growth pipeline.

    All tests use ``use_fisher=True`` in ``compute_optimal_updates`` and
    verify that the pipeline runs correctly and produces non-trivial results.

    At rank 0 the adapter path ``B @ A @ x`` is identically zero, but
    ``second_layer.pre_activity.grad`` receives the upstream gradient (it is
    retained via ``retain_grad()`` during ``init_computation``). This means
    the Fisher covariance and ``tensor_m_prev`` are both well-defined and
    non-zero even before any rank has been added.
    """

    SEED = 0

    def setUp(self) -> None:
        torch.manual_seed(self.SEED)

    def _make_lora(self, in_f: int = 8, out_f: int = 16, rank: int = 0) -> GrowRALinear:
        linear = _linear(in_f, out_f)
        return GrowRALinear(linear, rank=rank)

    # ------------------------------------------------------------------
    # Fisher statistics population
    # ------------------------------------------------------------------

    def test_fisher_covariance_non_zero_at_rank_zero(self):
        """covariance_loss_gradient() is non-zero at rank 0 after one backward pass.

        The upstream gradient flows into second_layer.pre_activity (the
        zero-valued adapter output) because retain_grad() is called during
        init_computation(). The outer product of this gradient gives a
        positive-semidefinite matrix that captures output-space curvature.
        """
        lora = self._make_lora()
        lora.init_computation()
        x = _randn(8, lora.in_features)
        lora.zero_grad()
        (lora(x) ** 2).sum().backward()
        lora.update_computation()

        cov = lora.second_layer.covariance_loss_gradient()
        self.assertEqual(cov.shape, (lora.out_features, lora.out_features))
        self.assertGreater(cov.norm().item(), 0.0)

        lora.reset_computation()

    def test_fisher_covariance_accumulates_over_batches(self):
        """covariance_loss_gradient is updated by each backward and is non-trivially non-zero.

        We collect two fixed batches, run backward for each, and verify the
        accumulated covariance is non-zero and differs from a single-batch run
        (i.e. the statistic genuinely reflects more than one observation).
        """
        lora_one = self._make_lora()
        lora_two = self._make_lora()

        torch.manual_seed(self.SEED + 7)
        x1 = _randn(8, lora_one.in_features)
        x2 = _randn(8, lora_one.in_features)

        # One-batch reference
        lora_one.init_computation()
        lora_one.zero_grad()
        (lora_one(x1) ** 2).sum().backward()
        lora_one.update_computation()
        cov_one = lora_one.second_layer.covariance_loss_gradient().detach().clone()
        lora_one.reset_computation()

        # Two-batch accumulation
        lora_two.init_computation()
        lora_two.zero_grad()
        (lora_two(x1) ** 2).sum().backward()
        lora_two.update_computation()
        lora_two.zero_grad()
        (lora_two(x2) ** 2).sum().backward()
        lora_two.update_computation()
        cov_two = lora_two.second_layer.covariance_loss_gradient().detach().clone()
        lora_two.reset_computation()

        # Both non-zero
        self.assertGreater(cov_one.norm().item(), 0.0)
        self.assertGreater(cov_two.norm().item(), 0.0)
        # Two-batch covariance reflects more information: it is not identical to the
        # single-batch result (different random draws → different result).
        self.assertFalse(
            torch.allclose(cov_one, cov_two),
            "accumulated covariance must differ from single-batch covariance",
        )

    # ------------------------------------------------------------------
    # Fisher growth pipeline
    # ------------------------------------------------------------------

    def test_fisher_pipeline_from_rank_zero(self):
        """Full Fisher FOGRO loop from rank 0: rank grows and output shape is correct."""
        lora = self._make_lora()
        _fisher_grow(lora, added_rank=2)

        self.assertEqual(lora.rank, 2)
        out = lora(_randn(4, lora.in_features))
        self.assertEqual(out.shape, (4, lora.out_features))

    def test_fisher_eigenvalues_positive(self):
        """Fisher growth produces strictly positive eigenvalues."""
        lora = self._make_lora()
        lora.init_computation()
        for _ in range(4):
            x = _randn(8, lora.in_features)
            lora.zero_grad()
            (lora(x) ** 2).sum().backward()
            lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=4,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )

        eigs = lora.second_layer.eigenvalues_extension
        self.assertIsNotNone(eigs)
        self.assertGreater(eigs.shape[0], 0, "at least one neuron must be selected")
        self.assertTrue((eigs > 0).all(), f"all eigenvalues must be positive, got {eigs}")
        lora.reset_computation()

    def test_fisher_pipeline_from_nonzero_rank(self):
        """Fisher FOGRO from rank > 0: rank grows and forward matches merge."""
        lora = self._make_lora()
        _grow_linear(lora, added_rank=2)
        self.assertEqual(lora.rank, 2)

        _fisher_grow(lora, added_rank=2)
        self.assertEqual(lora.rank, 4)

        x = _randn(4, lora.in_features)
        out = lora(x)
        self.assertEqual(out.shape, (4, lora.out_features))
        merged = lora.merge()
        self.assertTrue(
            torch.allclose(out, merged(x), atol=1e-5),
            "forward must match merge after Fisher growth",
        )

    def test_fisher_forward_unchanged_from_original_at_rank_zero(self):
        """At rank 0 the GrowRA adapter is a no-op: output equals the frozen linear."""
        linear = _linear(8, 16)
        lora = GrowRALinear(linear, rank=0)
        x = _randn(4, 8)
        with torch.no_grad():
            self.assertTrue(torch.allclose(lora(x), linear(x)))

    # ------------------------------------------------------------------
    # Cross-layer score ordering
    # ------------------------------------------------------------------

    def test_kfac_eigenvalues_scale_invariant(self):
        """K-FAC eigenvalues are invariant to input scale.

        With full K-FAC (use_covariance=True, use_fisher=True) the score matrix
        is M = F_s^{-1/2} @ grad_W @ C^{-1/2}. Scaling inputs by SCALE also
        scales C by SCALE^2, grad_W by SCALE^2, and F_s by SCALE^2, so the
        SCALE factors cancel and M is unchanged. Two adapters fed x vs SCALE*x
        with identical samples must therefore produce identical eigenvalues.
        """
        torch.manual_seed(self.SEED)
        SCALE = 0.05

        linear = _linear(8, 8)
        lora_big = GrowRALinear(copy.deepcopy(linear), rank=0)
        lora_small = GrowRALinear(copy.deepcopy(linear), rank=0)

        for lora in (lora_big, lora_small):
            lora.init_computation()

        for _ in range(6):
            x = _randn(16, 8)
            lora_big.zero_grad()
            (lora_big(x) ** 2).sum().backward()
            lora_big.update_computation()
            lora_small.zero_grad()
            (lora_small(x * SCALE) ** 2).sum().backward()
            lora_small.update_computation()

        lora_big.compute_optimal_updates(
            maximum_added_neurons=4,
            use_covariance=True,
            use_fisher=True,
            ignore_singular_values=True,
            compute_delta=False,
        )
        lora_small.compute_optimal_updates(
            maximum_added_neurons=4,
            use_covariance=True,
            use_fisher=True,
            ignore_singular_values=True,
            compute_delta=False,
        )

        eigs_big = lora_big.second_layer.eigenvalues_extension
        eigs_small = lora_small.second_layer.eigenvalues_extension

        self.assertIsNotNone(eigs_big)
        self.assertIsNotNone(eigs_small)
        self.assertTrue(
            torch.allclose(eigs_big, eigs_small, atol=1e-4, rtol=1e-3),
            f"K-FAC eigenvalues must be scale-invariant: big={eigs_big}, small={eigs_small}",
        )

        for lora in (lora_big, lora_small):
            lora.reset_computation()


# ============ GrowRA §A.5 normalization: current behaviour ============


def _rank_norms(weight: torch.Tensor, rank_dim: int) -> torch.Tensor:
    """L2 norm of each rank, reducing every axis except ``rank_dim``.

    ``rank_dim`` is 0 for the A factor (``(k, fan_in, ...)``) and 1 for the B
    factor (``(fan_out, k, ...)``): the rank axis is the only axis that is not
    a fan axis, for both GrowRALinear and GrowRAConv2d.
    """
    dims = tuple(d for d in range(weight.dim()) if d != rank_dim)
    return (weight**2).sum(dim=dims).sqrt()


def _per_rank_vectors(weight: torch.Tensor, rank_dim: int) -> torch.Tensor:
    """Flatten ``weight`` to ``(k, -1)``, one row per rank."""
    return weight.transpose(0, rank_dim).flatten(1)


class _Case(NamedTuple):
    """One layer type under test, with the shapes its extensions must have."""

    block: GrowRALinear | GrowRAConv2d
    x: torch.Tensor
    keep: int
    fan_out: int
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]


class TestGrowRANormalizationTargets(TorchTestCase):
    """Pin the numerical result of the GrowRA paper Section A.5 rescale.

    The ``"growra"`` strategy rescales each new A row to unit norm and each new
    B column to ``sqrt(fan_out * lr_init)``.  Nothing asserted this before;
    these tests exist so the planned pre-merge refactor of that rescale has a
    numerical contract to preserve.
    """

    lr_init = 0.05

    def _grow(self, block, x, keep, scaling_factor=1.0, **kwargs):
        """Run one growth step and return what the merge consumed.

        Returns ``(old_rank, pending_a, pending_b, sigma)``, captured after
        sub-selection and before ``apply_change`` -- the only window in which
        the new ranks exist as separate, addressable tensors.
        """
        options = dict(
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=False,
            use_fisher=True,
        )
        options.update(kwargs)

        block.init_computation()
        block.zero_grad()
        (block(x) ** 2).sum().backward()
        block.update_computation()
        block.compute_optimal_updates(maximum_added_neurons=keep + 1, **options)
        block.sub_select_optimal_added_parameters(keep_neurons=keep)

        pending_a = block.first_layer.extended_output_layer.weight.detach().clone()
        pending_b = block.second_layer.extended_input_layer.weight.detach().clone()
        sigma = block.eigenvalues_extension.detach().clone()

        old_rank = block.rank
        block.apply_change(
            scaling_factor=scaling_factor, extension_size=keep, lr_init=self.lr_init
        )
        block.reset_computation()
        return old_rank, pending_a, pending_b, sigma

    @staticmethod
    def _new_ranks(block, old_rank):
        """The slices of the merged weights holding the freshly added ranks."""
        return (
            block.first_layer.weight[old_rank:],
            block.second_layer.weight[:, old_rank:],
        )

    def _make_linear(self):
        """Return a rank-0 GrowRALinear plus everything the assertions need.

        ``a_shape`` / ``b_shape`` spell out the rank-axis convention: the rank
        is dim 0 of A and dim 1 of B, every other axis being a fan axis.
        """
        torch.manual_seed(0)
        block = GrowRALinear(_linear(8, 6), rank=0)
        keep, fan_out = 3, 6
        return _Case(block, _randn(32, 8), keep, fan_out, (keep, 8), (fan_out, keep))

    def _make_conv(self):
        """Same for GrowRAConv2d; B keeps a 1x1 kernel, which is why fan_out is dim 0."""
        torch.manual_seed(0)
        block = GrowRAConv2d(_conv2d(3, 5, kernel_size=3, padding=1), rank=0)
        keep, fan_out = 2, 5
        return _Case(
            block,
            _randn(4, 3, 8, 8),
            keep,
            fan_out,
            (keep, 3, 3, 3),
            (fan_out, keep, 1, 1),
        )

    def _both(self):
        yield "linear", self._make_linear
        yield "conv2d", self._make_conv

    # ---- test 1: the paper A.5 targets ----

    def test_growra_normalization_targets(self):
        """Each new A row has norm 1; each new B column norm sqrt(fan_out * lr)."""
        for name, factory in self._both():
            with self.subTest(layer=name):
                case = factory()
                old_rank, _, _, _ = self._grow(case.block, case.x, case.keep)
                self.assertEqual(case.block.rank, old_rank + case.keep)

                a_new, b_new = self._new_ranks(case.block, old_rank)
                self.assertShapeEqual(a_new, case.a_shape, "new A ranks")
                # ``fan_out = b_shape[0]`` is the true fan-out only because the
                # B convolution has a 1x1 kernel; b_shape pins that too.
                self.assertShapeEqual(b_new, case.b_shape, "new B ranks")

                a_norms = _rank_norms(a_new, rank_dim=0)
                self.assertAllClose(
                    a_norms,
                    torch.ones_like(a_norms),
                    atol=1e-5,
                    message="new A rows must have unit norm",
                )

                b_norms = _rank_norms(b_new, rank_dim=1)
                target = (case.fan_out * self.lr_init) ** 0.5
                self.assertAllClose(
                    b_norms,
                    torch.full_like(b_norms, target),
                    atol=1e-5,
                    message=f"new B columns must have norm sqrt(fan_out * lr) = {target}",
                )

    def test_growra_normalization_targets_follow_lr_init(self):
        """The B target tracks ``lr_init``: quadrupling it doubles the column norm."""
        norms = {}
        for lr_init in (0.05, 0.20):
            self.lr_init = lr_init
            case = self._make_linear()
            old_rank, _, _, _ = self._grow(case.block, case.x, case.keep)
            _, b_new = self._new_ranks(case.block, old_rank)
            norms[lr_init] = _rank_norms(b_new, rank_dim=1)
        self.assertAllClose(
            norms[0.20],
            2 * norms[0.05],
            atol=1e-5,
            message="B norms must scale with sqrt(lr_init)",
        )

    # ---- test 2: it is a normalization, not an initialization ----

    def test_normalization_preserves_direction(self):
        """The rescale changes magnitudes only: every new rank keeps its direction."""
        for name, factory in self._both():
            with self.subTest(layer=name):
                case = factory()
                old_rank, pending_a, pending_b, _ = self._grow(
                    case.block, case.x, case.keep
                )
                a_new, b_new = self._new_ranks(case.block, old_rank)
                self.assertShapeEqual(pending_a, case.a_shape, "pending A extension")
                self.assertShapeEqual(pending_b, case.b_shape, "pending B extension")

                for factor, rank_dim, before, after in (
                    ("A", 0, pending_a, a_new),
                    ("B", 1, pending_b, b_new),
                ):
                    cosine = F.cosine_similarity(
                        _per_rank_vectors(before, rank_dim),
                        _per_rank_vectors(after, rank_dim),
                        dim=1,
                    )
                    self.assertAllClose(
                        cosine,
                        torch.ones_like(cosine),
                        atol=1e-5,
                        message=f"{factor} directions changed during normalization",
                    )
                    # ... and the rescale is a strictly positive stretch.
                    ratio = _rank_norms(after, rank_dim) / _rank_norms(before, rank_dim)
                    self.assertTrue((ratio > 0).all(), f"{factor} ratio={ratio.tolist()}")

    def test_normalization_actually_rescales(self):
        """Guard against a vacuous test 2: the magnitudes really do change."""
        case = self._make_linear()
        old_rank, pending_a, pending_b, _ = self._grow(case.block, case.x, case.keep)
        a_new, b_new = self._new_ranks(case.block, old_rank)
        for factor, rank_dim, before, after in (
            ("A", 0, pending_a, a_new),
            ("B", 1, pending_b, b_new),
        ):
            before_norms = _rank_norms(before, rank_dim)
            after_norms = _rank_norms(after, rank_dim)
            self.assertFalse(
                torch.allclose(before_norms, after_norms, atol=1e-3),
                f"{factor} was already at target, test 2 would be vacuous: "
                f"{before_norms.tolist()} vs {after_norms.tolist()}",
            )

    # ---- test 5a: where the singular values actually live ----

    def test_b_norms_track_sqrt_sigma_without_fisher(self):
        """Before normalization, ``||B[:, i]|| == sqrt(sigma_i)`` -- but only without E.

        ``omega = sqrt(s)[:, None] * v`` with orthonormal rows of ``v``, so the
        relation is exact.  It stops being exact as soon as the empirical
        Fisher preconditioner is applied (``omega @ E^{-1/2}``), which is what
        ``use_fisher=True`` does -- see the companion test below.
        """
        for name, factory in self._both():
            with self.subTest(layer=name):
                case = factory()
                _, _, pending_b, sigma = self._grow(
                    case.block,
                    case.x,
                    case.keep,
                    use_fisher=False,
                    ignore_singular_values=False,
                )
                self.assertShapeEqual(pending_b, case.b_shape, "pending B extension")
                self.assertShapeEqual(sigma, (case.keep,), "singular values")
                ratio = _rank_norms(pending_b, rank_dim=1) / sigma.sqrt()
                self.assertAllClose(
                    ratio,
                    torch.ones_like(ratio),
                    atol=1e-5,
                    message="||B col|| / sqrt(sigma) must be 1 without Fisher",
                )

    def test_b_norms_do_not_track_sqrt_sigma_with_fisher(self):
        """With ``use_fisher=True`` -- the GrowRA default -- the relation breaks.

        ``E^{-1/2}`` acts on the B side exactly as ``S^{-1/2}`` acts on the A
        side.  Pinned because it is the reason a test may not assert
        ``||B[:, i]|| ~ sqrt(sigma_i)`` on the default growth path.
        """
        case = self._make_linear()
        _, _, pending_b, sigma = self._grow(
            case.block, case.x, case.keep, use_fisher=True, ignore_singular_values=False
        )
        ratio = _rank_norms(pending_b, rank_dim=1) / sigma.sqrt()
        self.assertGreater(
            (ratio - 1).abs().max().item(),
            0.1,
            f"expected the Fisher preconditioner to detach ||B col|| from "
            f"sqrt(sigma), got ratio={ratio.tolist()}",
        )

    def test_a_norms_do_not_track_sqrt_sigma(self):
        """The A side never carries ``sqrt(sigma_i)`` exactly: ``S^{-1/2}`` is in the way."""
        case = self._make_linear()
        _, pending_a, _, sigma = self._grow(
            case.block, case.x, case.keep, use_fisher=False, ignore_singular_values=False
        )
        ratio = _rank_norms(pending_a, rank_dim=0) / sigma.sqrt()
        self.assertFalse(
            torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3),
            f"||A row|| / sqrt(sigma) unexpectedly equals 1: {ratio.tolist()}",
        )

    # ---- the guard the vectorized rewrite has to keep ----

    def test_zero_ranks_survive_normalization(self):
        """A rank that is exactly zero stays zero instead of turning into NaN.

        The per-rank ``if norm > 0`` guards became a single masked division, so
        this pins the case produced by ``alpha_zero`` / ``omega_zero`` and by a
        zero singular value.  Both factors are checked at once: a zero A row and
        a zero B column, alongside ranks that must still reach their target.
        """
        zero_a, zero_b = 1, 2
        for name, block, fan_out in (
            ("linear", GrowRALinear(_linear(8, 6), rank=0), 6),
            ("conv2d", GrowRAConv2d(_conv2d(3, 5, kernel_size=3, padding=1), rank=0), 5),
        ):
            with self.subTest(layer=name):
                block.lr_init = self.lr_init
                block.allocate_layer_extensions(extension_size=4)
                assert isinstance(
                    block.first_layer.extended_output_layer, type(block.downsample)
                )
                assert isinstance(
                    block.second_layer.extended_input_layer, type(block.downsample)
                )
                a_ext = block.first_layer.extended_output_layer.weight
                b_ext = block.second_layer.extended_input_layer.weight
                with torch.no_grad():
                    nn.init.normal_(a_ext)
                    nn.init.normal_(b_ext)
                    a_ext[zero_a].zero_()
                    b_ext[:, zero_b].zero_()

                block.normalize_optimal_updates(normalization_type="growra")

                a_norms = _rank_norms(a_ext, rank_dim=0)
                b_norms = _rank_norms(b_ext, rank_dim=1)
                self.assertFalse(a_norms.isnan().any(), f"NaN in A: {a_norms.tolist()}")
                self.assertFalse(b_norms.isnan().any(), f"NaN in B: {b_norms.tolist()}")

                target = (fan_out * self.lr_init) ** 0.5
                expected_a = torch.ones_like(a_norms)
                expected_a[zero_a] = 0.0
                expected_b = torch.full_like(b_norms, target)
                expected_b[zero_b] = 0.0
                self.assertAllClose(
                    a_norms, expected_a, atol=1e-5, message="zeroed A row must stay zero"
                )
                self.assertAllClose(
                    b_norms,
                    expected_b,
                    atol=1e-5,
                    message="zeroed B column must stay zero",
                )

    # ---- the move itself: pre-merge must equal post-merge ----

    @staticmethod
    def _legacy_post_merge_apply_change(
        block, extension_size, lr_init, scaling_factor=1.0
    ):
        """Reproduce the pre-N1 sequence: merge first, *then* rescale the merged slice.

        ``GrowingBlock.apply_change`` is called unbound so that ``GrowRABlock``'s
        override -- which now normalizes before merging -- is bypassed.  The
        rescale below is the pre-N1 body verbatim, per-element loops included.
        """
        block.lr_init = lr_init
        old_rank = block.rank
        GrowingBlock.apply_change(
            block,
            extension_size=extension_size,
            scaling_factor=scaling_factor,
            apply_delta=False,
        )
        if block.rank <= old_rank:
            return
        with torch.no_grad():
            a_new = block.first_layer.weight[old_rank:]
            for i in range(a_new.shape[0]):
                n = a_new[i].norm()
                if n > 0:
                    a_new[i].div_(n)
            b_new = block.second_layer.weight[:, old_rank:]
            target = (b_new.shape[0] * lr_init) ** 0.5
            for i in range(b_new.shape[1]):
                n = b_new[:, i].norm()
                if n > 0:
                    b_new[:, i].mul_(target / n)

    def _grown_weights(self, factory, legacy, **grow_kwargs):
        """Grow one block and return its merged (A, B); ``legacy`` picks the order."""
        case = factory()
        block, keep = case.block, case.keep
        options = dict(
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=False,
            use_fisher=True,
        )
        options.update(grow_kwargs)
        block.init_computation()
        block.zero_grad()
        (block(case.x) ** 2).sum().backward()
        block.update_computation()
        block.compute_optimal_updates(maximum_added_neurons=keep + 1, **options)
        block.sub_select_optimal_added_parameters(keep_neurons=keep)
        if legacy:
            self._legacy_post_merge_apply_change(block, keep, self.lr_init)
        else:
            block.apply_change(
                extension_size=keep,
                scaling_factor=1.0,
                apply_delta=False,
                lr_init=self.lr_init,
            )
        block.reset_computation()
        return (
            block.first_layer.weight.detach().clone(),
            block.second_layer.weight.detach().clone(),
        )

    def test_pre_merge_equals_post_merge(self):
        """Normalizing before the merge gives exactly the pre-N1 weights.

        This is the whole justification for moving the rescale: the merge is a
        concatenation and the rescale is elementwise on the new ranks, so with
        ``scaling_factor=1`` they commute.

        Asserted to ``1e-6``, not bit-for-bit.  It *was* bit-for-bit while the
        rescale divided A and multiplied B, mirroring the per-element loops it
        replaced.  Unifying the two factors into one ``_scale_factor`` gave that
        up deliberately: no single operation reproduces both sides, and
        ``mul_(target / norm)`` differs from ``div_(norm)`` by an ulp on every
        tensor.  The tolerance below is two orders of magnitude above that drift
        and still far under anything that could pass for a real difference.
        """
        for name, factory in self._both():
            for fisher in (True, False):
                for ignore_sv in (True, False):
                    with self.subTest(layer=name, fisher=fisher, ignore_sv=ignore_sv):
                        kw = dict(use_fisher=fisher, ignore_singular_values=ignore_sv)
                        new = self._grown_weights(factory, legacy=False, **kw)
                        old = self._grown_weights(factory, legacy=True, **kw)
                        for factor, a, b in zip("AB", new, old, strict=True):
                            self.assertAllClose(
                                a,
                                b,
                                atol=1e-6,
                                message=f"{factor} differs from the post-merge result",
                            )

    def test_scaling_factor_respected(self):
        """``scaling_factor`` now survives normalization -- it could not before.

        Post-merge, rescaling to a fixed target erased whatever ``apply_change``
        had multiplied in.  Pre-merge, gamma applies on top: each side of the
        merged extension scales by gamma, so the product ``B @ A`` scales by
        gamma squared.
        """
        gamma = 3.0
        for name, factory in self._both():
            with self.subTest(layer=name):
                case = factory()
                old_rank, _, _, _ = self._grow(
                    case.block, case.x, case.keep, scaling_factor=gamma
                )
                a_new, b_new = self._new_ranks(case.block, old_rank)

                a_norms = _rank_norms(a_new, rank_dim=0)
                self.assertAllClose(
                    a_norms,
                    torch.full_like(a_norms, gamma),
                    atol=1e-5,
                    message="new A rows must be gamma times the unit-norm target",
                )
                b_norms = _rank_norms(b_new, rank_dim=1)
                target = gamma * (case.fan_out * self.lr_init) ** 0.5
                self.assertAllClose(
                    b_norms,
                    torch.full_like(b_norms, target),
                    atol=1e-5,
                    message="new B columns must be gamma times sqrt(fan_out * lr)",
                )

    def test_apply_change_without_extensions_is_a_noop(self):
        """``apply_change`` with nothing pending must not raise.

        The old ``rank <= old_rank`` guard absorbed this case silently.  Pre-merge
        there is no rank to compare, and the primitives step 4 will route through
        raise on a missing extension, so ``_has_pending_extensions`` has to carry
        that tolerance explicitly.
        """
        for name, factory in self._both():
            with self.subTest(layer=name):
                block = factory().block
                self.assertFalse(block._has_pending_extensions())
                rank_before = block.rank
                a_before = block.first_layer.weight.clone()
                b_before = block.second_layer.weight.clone()

                block.apply_change(extension_size=0, apply_delta=False)

                self.assertEqual(block.rank, rank_before)
                self.assertTrue(torch.equal(block.first_layer.weight, a_before))
                self.assertTrue(torch.equal(block.second_layer.weight, b_before))


class TestGrowRANormalizationStrategies(TestGrowRANormalizationTargets):
    """The ``normalization=`` selector: joint granularity and the B-side targets.

    Inherits the step-1 fixtures (``_make_linear`` / ``_make_conv`` / ``_grow``)
    so every strategy is exercised on the same blocks the default is pinned on.
    """

    def _grow_with(self, factory, normalization, loss="quadratic", **kwargs):
        """Grow one block under ``normalization``; return pending and merged tensors.

        ``loss`` picks the objective used to accumulate the statistics.  It
        matters more than it looks: the default ``"quadratic"`` makes
        ``dL/dout = 2*out``, an exact linear function of the layer input, so the
        whitened SVD target is orthogonal and **every singular value comes out
        exactly 1** whenever ``use_fisher=True``.  Any test about singular values
        must pass ``loss="tanh"`` (or turn Fisher off) or it measures nothing.
        """
        case = factory()
        block, keep = case.block, case.keep
        options = dict(
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=False,
            use_fisher=True,
        )
        options.update(kwargs)
        block.init_computation()
        block.zero_grad()
        out = block(case.x)
        objective = (out**2).sum() if loss == "quadratic" else torch.tanh(out).sum()
        objective.backward()
        block.update_computation()
        block.compute_optimal_updates(maximum_added_neurons=keep + 1, **options)
        block.sub_select_optimal_added_parameters(keep_neurons=keep)

        pending_a = block.first_layer.extended_output_layer.weight.detach().clone()
        pending_b = block.second_layer.extended_input_layer.weight.detach().clone()
        sigma = block.eigenvalues_extension.detach().clone()
        old_rank = block.rank
        block.apply_change(
            extension_size=keep,
            scaling_factor=1.0,
            apply_delta=False,
            lr_init=self.lr_init,
            normalization=normalization,
        )
        a_new, b_new = self._new_ranks(block, old_rank)
        block.reset_computation()
        return dict(
            case=case,
            pending_a=pending_a,
            pending_b=pending_b,
            sigma=sigma,
            a_new=a_new.detach().clone(),
            b_new=b_new.detach().clone(),
        )

    # ---- test 5: joint preserves relative magnitudes ----

    def test_joint_preserves_relative_magnitudes(self):
        """``growra_joint`` keeps each factor's rank norms proportional to their own.

        Deliberately compared against the *pre-normalization* norms on both
        sides, not against ``sqrt(sigma_i)``: A carries an extra ``S^{-1/2}``
        factor and B an extra ``E^{-1/2}`` one, so a ``sqrt(sigma)`` assertion
        would fail for the wrong reason whenever Fisher is on -- which is the
        default (see ``test_b_norms_do_not_track_sqrt_sigma_with_fisher``).
        """
        for name, factory in self._both():
            with self.subTest(layer=name):
                got = self._grow_with(factory, "growra_joint")
                for factor, rank_dim, before, after in (
                    ("A", 0, got["pending_a"], got["a_new"]),
                    ("B", 1, got["pending_b"], got["b_new"]),
                ):
                    ratio = _rank_norms(after, rank_dim) / _rank_norms(before, rank_dim)
                    self.assertAllClose(
                        ratio,
                        torch.full_like(ratio, ratio[0].item()),
                        atol=1e-5,
                        message=f"{factor}: joint must apply one scalar, got {ratio.tolist()}",
                    )
                    # ... and the spread it preserves must be a real spread.
                    spread = _rank_norms(before, rank_dim)
                    self.assertGreater(
                        (spread.max() / spread.min()).item(),
                        1.01,
                        f"{factor}: pre-normalization norms are already uniform, "
                        f"so this test would pass vacuously: {spread.tolist()}",
                    )

    def test_joint_hits_the_aggregate_target(self):
        """``growra_joint`` meets the per-rank targets in the mean: same variance."""
        for name, factory in self._both():
            with self.subTest(layer=name):
                got = self._grow_with(factory, "growra_joint")
                keep, fan_out = got["case"].keep, got["case"].fan_out
                a_frob = torch.linalg.vector_norm(got["a_new"])
                b_frob = torch.linalg.vector_norm(got["b_new"])
                self.assertAllClose(
                    a_frob,
                    torch.tensor(keep**0.5, device=a_frob.device),
                    atol=1e-5,
                    message="||A||_F must be sqrt(k), i.e. mean square row norm 1",
                )
                self.assertAllClose(
                    b_frob,
                    torch.tensor(
                        (keep * fan_out * self.lr_init) ** 0.5, device=b_frob.device
                    ),
                    atol=1e-5,
                    message="||B||_F must be sqrt(k * fan_out * lr)",
                )

    # ---- test 5b: the flag the default granularity nullifies ----

    def test_per_rank_makes_ignore_singular_values_a_noop(self):
        """``growra`` erases ``ignore_singular_values``; ``growra_joint`` does not.

        With per-rank normalization each direction is rescaled to the same fixed
        norm, which divides out the ``sqrt(sigma_i)`` the flag controls -- so the
        two settings produce the same weights and any ablation over the flag
        measures nothing.  Joint normalization keeps the ratios and the flag bites.
        """
        for name, factory in self._both():
            for strategy, should_match in (("growra", True), ("growra_joint", False)):
                with self.subTest(layer=name, normalization=strategy):
                    on = self._grow_with(
                        factory, strategy, loss="tanh", ignore_singular_values=True
                    )
                    off = self._grow_with(
                        factory, strategy, loss="tanh", ignore_singular_values=False
                    )
                    # Guard the guard: with the default quadratic loss every
                    # singular value is exactly 1 under use_fisher=True, and the
                    # flag would be a no-op for reasons that have nothing to do
                    # with normalization.  See _grow_with's docstring.
                    spread = off["sigma"]
                    self.assertGreater(
                        (spread.max() / spread.min()).item(),
                        1.05,
                        f"singular values are degenerate ({spread.tolist()}), so this "
                        f"test cannot distinguish the two settings",
                    )
                    self.assertFalse(
                        torch.allclose(on["pending_b"], off["pending_b"], atol=1e-6),
                        "the flag did not change the pending extensions at all",
                    )
                    diff = (on["b_new"] - off["b_new"]).abs().max().item()
                    if should_match:
                        # eps-level, not bit-exact: (c*x)/||c*x|| != x/||x|| exactly.
                        self.assertLess(
                            diff,
                            1e-6,
                            f"per-rank normalization should have erased the flag, "
                            f"but B differs by {diff:.3e}",
                        )
                    else:
                        self.assertGreater(
                            diff,
                            1e-4,
                            f"joint normalization should preserve the flag's effect, "
                            f"but B differs by only {diff:.3e}",
                        )

    # ---- test 6: the two granularities agree on uniform input ----

    def test_joint_and_per_rank_agree_when_uniform(self):
        """With equal rank norms the two granularities coincide -- joint generalizes."""
        for name, factory in self._both():
            with self.subTest(layer=name):
                results = {}
                for strategy in ("growra", "growra_joint"):
                    case = factory()
                    block = case.block
                    block.lr_init = self.lr_init
                    block.allocate_layer_extensions(extension_size=case.keep)
                    a_ext = block.first_layer.extended_output_layer.weight
                    b_ext = block.second_layer.extended_input_layer.weight
                    with torch.no_grad():
                        torch.manual_seed(7)
                        nn.init.normal_(a_ext)
                        nn.init.normal_(b_ext)
                        # Force every rank to the same norm on both sides.
                        a_ext.div_(
                            _rank_norms(a_ext, 0).view(-1, *([1] * (a_ext.dim() - 1)))
                        )
                        shape_b = [1] * b_ext.dim()
                        shape_b[1] = b_ext.shape[1]
                        b_ext.div_(_rank_norms(b_ext, 1).view(shape_b))
                    block.normalize_optimal_updates(normalization_type=strategy)
                    results[strategy] = (a_ext.detach().clone(), b_ext.detach().clone())
                for factor, per_rank, joint in zip(
                    "AB", results["growra"], results["growra_joint"], strict=True
                ):
                    self.assertAllClose(
                        per_rank,
                        joint,
                        atol=1e-5,
                        message=f"{factor}: granularities must agree on uniform norms",
                    )

    # ---- test 7: the B-side targets ----

    def test_zero_b(self):
        """``zero_b`` zeroes B exactly and leaves A at the ``growra`` result."""
        for name, factory in self._both():
            with self.subTest(layer=name):
                zeroed = self._grow_with(factory, "zero_b")
                default = self._grow_with(factory, "growra")
                self.assertTrue(
                    torch.equal(zeroed["b_new"], torch.zeros_like(zeroed["b_new"])),
                    "B must be exactly zero, not merely small",
                )
                self.assertTrue(
                    torch.equal(zeroed["a_new"], default["a_new"]),
                    "zero_b must leave the A side bit-identical to growra",
                )

    # ---- test 8: opting out ----

    def test_normalization_none_keeps_optimal_magnitudes(self):
        """``normalization=None`` merges the SVD magnitudes untouched."""
        for name, factory in self._both():
            with self.subTest(layer=name):
                got = self._grow_with(factory, None)
                self.assertTrue(
                    torch.equal(got["a_new"], got["pending_a"]),
                    "A must reach the merged weights unmodified",
                )
                self.assertTrue(
                    torch.equal(got["b_new"], got["pending_b"]),
                    "B must reach the merged weights unmodified",
                )

    # ---- test 9: eigenvalue bookkeeping ----

    def test_eigenvalues_extension_updated(self):
        """Joint updates ``eigenvalues_extension``; per-rank deliberately does not.

        The joint path goes through ``scale_layer_extension``, which scales the
        singular values by ``(c_A * c_B) ** exponent`` with the exponent chosen by
        ``_first_order_uses_squared_singular_values`` -- the one thing
        ``ignore_singular_values`` still controls.  Both regimes are asserted.
        """
        for ignore_sv in (True, False):
            exponent = 1.0 if ignore_sv else 0.5
            with self.subTest(ignore_singular_values=ignore_sv):
                case = self._make_linear()
                block, keep = case.block, case.keep
                block.init_computation()
                block.zero_grad()
                (block(case.x) ** 2).sum().backward()
                block.update_computation()
                block.compute_optimal_updates(
                    maximum_added_neurons=keep + 1,
                    compute_delta=False,
                    use_covariance=True,
                    use_projection=False,
                    alpha_zero=False,
                    omega_zero=False,
                    ignore_singular_values=ignore_sv,
                    use_fisher=True,
                )
                block.sub_select_optimal_added_parameters(keep_neurons=keep)
                block.lr_init = self.lr_init
                sigma_before = block.eigenvalues_extension.detach().clone()
                a_before = _rank_norms(
                    block.first_layer.extended_output_layer.weight, 0
                ).clone()
                b_before = _rank_norms(
                    block.second_layer.extended_input_layer.weight, 1
                ).clone()

                block.normalize_optimal_updates(normalization_type="growra_joint")

                a_after = _rank_norms(block.first_layer.extended_output_layer.weight, 0)
                b_after = _rank_norms(block.second_layer.extended_input_layer.weight, 1)
                scale_a = (a_after[0] / a_before[0]).item()
                scale_b = (b_after[0] / b_before[0]).item()
                expected = sigma_before * (scale_a * scale_b) ** exponent
                self.assertAllClose(
                    block.eigenvalues_extension,
                    expected,
                    atol=1e-5,
                    rtol=1e-4,
                    message=f"eigenvalues must scale by (c_A * c_B)**{exponent}",
                )

    def test_eigenvalues_extension_updated_per_rank(self):
        """Per-rank normalization keeps the singular values consistent too.

        Returning the applied scale from ``_scale_factor`` made this fall out for
        free: the update broadcasts, so it is per rank when the scales are, and
        global when they are not.  Previously only the both-joint strategy
        maintained the invariant and the rest silently did not.
        """
        case = self._make_linear()
        block, keep = case.block, case.keep
        block.init_computation()
        block.zero_grad()
        (block(case.x) ** 2).sum().backward()
        block.update_computation()
        block.compute_optimal_updates(
            maximum_added_neurons=keep + 1,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=False,
            use_fisher=True,
        )
        block.sub_select_optimal_added_parameters(keep_neurons=keep)
        block.lr_init = self.lr_init
        before = block.eigenvalues_extension.detach().clone()
        a_before = _rank_norms(block.first_layer.extended_output_layer.weight, 0).clone()
        b_before = _rank_norms(block.second_layer.extended_input_layer.weight, 1).clone()

        block.normalize_optimal_updates(normalization_type="growra")

        a_after = _rank_norms(block.first_layer.extended_output_layer.weight, 0)
        b_after = _rank_norms(block.second_layer.extended_input_layer.weight, 1)
        scales = (a_after / a_before) * (b_after / b_before)
        # ignore_singular_values=False above, so the singular values were applied
        # to the weights and the exponent is 1/2.
        self.assertAllClose(
            block.eigenvalues_extension,
            before * scales**0.5,
            atol=1e-5,
            rtol=1e-4,
            message="per-rank normalization must rescale each singular value",
        )

    # ---- delegation and escape hatch ----

    def test_base_strategy_still_reachable(self):
        """An unknown ``normalization_type`` falls through to the base dispatch.

        Grown from rank 2 rather than 0: ``gradmax_normalization`` needs existing
        neurons to derive its reference norm, and skips with a warning without them.
        """
        keep = 2
        block = GrowRALinear(_linear(8, 6), rank=2)
        with torch.no_grad():
            nn.init.normal_(block.second_layer.weight)
        case = _Case(block, _randn(32, 8), keep, 6, (keep, 8), (6, keep))
        block.init_computation()
        block.zero_grad()
        (block(case.x) ** 2).sum().backward()
        block.update_computation()
        block.compute_optimal_updates(
            maximum_added_neurons=keep + 1,
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=False,
            use_fisher=True,
        )
        block.sub_select_optimal_added_parameters(keep_neurons=keep)
        before = block.second_layer.extended_input_layer.weight.detach().clone()
        block.normalize_optimal_updates(normalization_type="gradmax_normalization")
        after = block.second_layer.extended_input_layer.weight
        self.assertFalse(
            torch.allclose(before, after, atol=1e-6),
            "the base strategy should have rescaled the extension",
        )

    def test_callable_normalization(self):
        """A callable is invoked with the block and bypasses every named strategy."""
        seen = []

        def custom(block):
            seen.append(block)
            block._extension_weight(block.second_layer.extended_input_layer).fill_(0.25)

        got = self._grow_with(self._make_linear, custom)
        self.assertEqual(len(seen), 1)
        self.assertTrue(
            torch.equal(got["b_new"], torch.full_like(got["b_new"], 0.25)),
            "the callable's effect must reach the merged weights",
        )

    # ---- the two axes, varied independently ----

    def _variances(self, factory, normalization):
        """Merged-in new-rank variances, as ``(var(A), var(B))``."""
        got = self._grow_with(factory, normalization, loss="tanh")
        return (
            got["a_new"].var(unbiased=False).item(),
            got["b_new"].var(unbiased=False).item(),
            got,
        )

    def test_zero_b_joint(self):
        """``zero_b_joint``: A normalized jointly, B zeroed.

        The combination the flat names could not express -- ``zero_b`` pins A to
        per-rank.  Zeroing is granularity-independent, so only the A side differs.
        """
        for name, factory in self._both():
            with self.subTest(layer=name):
                joint = self._grow_with(factory, "zero_b_joint", loss="tanh")
                per_rank = self._grow_with(factory, "zero_b", loss="tanh")
                self.assertTrue(
                    torch.equal(joint["b_new"], torch.zeros_like(joint["b_new"])),
                    "B must be exactly zero",
                )
                # A: one scalar for the whole factor, so the ranks keep their spread.
                ratio = _rank_norms(joint["a_new"], 0) / _rank_norms(
                    joint["pending_a"], 0
                )
                self.assertAllClose(
                    ratio,
                    torch.full_like(ratio, ratio[0].item()),
                    atol=1e-5,
                    message="zero_b_joint must scale A by a single factor",
                )
                # ... and that is genuinely different from zero_b's per-rank A.
                self.assertFalse(
                    torch.allclose(joint["a_new"], per_rank["a_new"], atol=1e-4),
                    "zero_b_joint and zero_b must differ on the A side",
                )

    def test_growra_joint_gain_variance_target(self):
        """``growra_joint_gain``: ``var(B_ij) = gain**2 * lr_init / fan_out``.

        Unlike the default B rule, this makes the *column* norm ``gain*sqrt(lr)``
        rather than ``sqrt(fan_out*lr)``, so the adapter's output scale is what
        stays fixed as ``fan_out`` varies.
        """
        for name, factory in self._both():
            with self.subTest(layer=name):
                var_a, var_b, got = self._variances(factory, "growra_joint_gain")
                fan_out = got["case"].fan_out
                # torch convention: the gain squares into the variance, as
                # torch.nn.init.calculate_gain defines it.
                expected_b = DEFAULT_GAIN**2 * self.lr_init / fan_out
                self.assertAlmostEqual(
                    var_b,
                    expected_b,
                    delta=0.02 * expected_b,
                    msg=f"var(B) should be gain*lr/fan_out = {expected_b}",
                )
                # A keeps the usual target, so the two axes really are independent.
                # fan_in spans every axis of A but the rank one: C_in * kH * kW
                # for a conv, in_features for a linear.
                fan_in = math.prod(got["case"].a_shape[1:])
                self.assertAlmostEqual(
                    var_a,
                    1.0 / fan_in,
                    delta=0.05 / fan_in,
                    msg="var(A) must stay 1/fan_in whatever B does",
                )

    def test_gain_is_configurable(self):
        """``gain`` multiplies the target variance, so ``||B||_F`` goes as its sqrt."""
        norms = {}
        for gain in (1.0, 2.0):
            spec = GrowRANormalization(
                a=FactorScaling(granularity="joint"),
                b=FactorScaling(granularity="joint", gain=gain, multiplier=LR_INIT),
            )
            got = self._grow_with(self._make_linear, spec, loss="tanh")
            norms[gain] = torch.linalg.vector_norm(got["b_new"]).item()
        # ||B||_F is linear in the gain, the variance quadratic.
        self.assertAlmostEqual(
            norms[2.0] / norms[1.0],
            2.0,
            delta=1e-4,
            msg=f"doubling the gain must double ||B||_F, got {norms}",
        )

    def test_spec_accepted_directly(self):
        """A ``GrowRANormalization`` is accepted wherever its name is."""
        by_name = self._grow_with(self._make_linear, "growra", loss="tanh")
        by_spec = self._grow_with(
            self._make_linear,
            GrowRANormalization(
                a=FactorScaling(),
                b=FactorScaling(multiplier=LR_INIT, fan_normalized=False),
            ),
            loss="tanh",
        )
        self.assertTrue(torch.equal(by_name["a_new"], by_spec["a_new"]))
        self.assertTrue(torch.equal(by_name["b_new"], by_spec["b_new"]))

    def test_a_granularity_independent_of_b_rule(self):
        """The A side depends only on ``a_granularity``, never on what B does."""
        per_rank = [
            self._grow_with(self._make_linear, n, loss="tanh")["a_new"]
            for n in ("growra", "zero_b")
        ]
        for other in per_rank[1:]:
            self.assertTrue(
                torch.equal(per_rank[0], other),
                "per-rank A must be identical across B rules",
            )
        joint = [
            self._grow_with(self._make_linear, n, loss="tanh")["a_new"]
            for n in ("growra_joint", "growra_joint_gain", "zero_b_joint")
        ]
        for other in joint[1:]:
            self.assertTrue(
                torch.equal(joint[0], other),
                "joint A must be identical across B rules",
            )
        self.assertFalse(
            torch.allclose(per_rank[0], joint[0], atol=1e-4),
            "the two granularities must actually differ",
        )

    def test_unknown_spec_type_raises(self):
        """A value that is neither a spec, a known name, nor callable is rejected."""
        case = self._make_linear()
        with self.assertRaises(TypeError):
            case.block.normalize_optimal_updates(normalization_type=42)
