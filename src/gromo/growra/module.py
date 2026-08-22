"""
GrowRA module classes for gromo.

Defines :class:`GrowRALinear` and :class:`GrowRAConv2d`, which wrap
a frozen original layer and add a trainable low-rank adaptation::

    output = original(x) + scaling * B(A(x))

where A and B are growing modules from gromo.  The rank starts at 0 and grows
via the FOGRO pipeline (see :mod:`gromo.growra.container`).

.. note::

    GrowRA modules are primarily designed and tested with ``use_fisher=True``
    in :meth:`~gromo.modules.growing_module.GrowingModule.compute_optimal_updates`.
    Other growth strategies (covariance-based, projection-based, etc.) are
    available but receive less testing in the GrowRA context.
"""

import copy
import warnings
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from gromo.containers.growing_block import (
    Conv2dGrowingBlock,
    GrowingBlock,
    LinearGrowingBlock,
)
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from gromo.modules.growing_module import SupportsExtendedForward
from gromo.modules.linear_growing_module import LinearGrowingModule


# Types accepted as the "original layer" for linear GrowRA
_LinearLayerType = nn.Linear | LinearGrowingModule
# Types accepted as the "original layer" for conv GrowRA
_Conv2dLayerType = nn.Conv2d | Conv2dGrowingModule


class Scaling(nn.Module, SupportsExtendedForward):
    """Scale a tensor before residual addition and during extended forward."""

    def __init__(
        self,
        scaling_fn: Callable[[int], float],
        rank_getter: Callable[[], int],
    ) -> None:
        super().__init__()
        self.scaling_fn = scaling_fn
        self.rank_getter = rank_getter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaling to ``x``; returns ``x`` unchanged when rank is 0."""
        scale = 1.0 if self.rank_getter() == 0 else self.get_scaling(self.rank_getter())
        return x * scale

    def get_scaling(self, rank: int) -> float:
        """Return the scaling factor for the given rank."""
        return self.scaling_fn(rank)

    def extended_forward(
        self,
        x: torch.Tensor | None,
        x_ext: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply scaling to both ``x`` and ``x_ext`` using at least rank 1."""
        scale = self.get_scaling(max(1, self.rank_getter()))
        if x is not None:
            x = x * scale
        if x_ext is not None:
            x_ext = x_ext * scale
        return x, x_ext


def _deepcopy_growra_module(self, memo: dict):
    """``__deepcopy__`` for GrowRA modules.

    ``deepcopy`` treats callables (lambdas) as atomic, so the ``rank_getter``
    and ``scaling_fn`` stored on ``_scaling`` would still reference the
    *original* module after a plain copy.  This override re-wires them to the
    newly-created copy before returning it.
    """
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
        setattr(result, k, copy.deepcopy(v, memo))
    result._scaling.scaling_fn = result.scaling_fn
    result._scaling.rank_getter = lambda: result.rank
    return result


class GrowRABlock(GrowingBlock):
    """Behaviour shared by :class:`GrowRALinear` and :class:`GrowRAConv2d`.

    The two blocks differ only in tensor rank, never in structure.  In both, the
    rank axis is dim 0 of the A factor (``(rank, fan_in, ...)``) and dim 1 of the
    B factor (``(fan_out, rank, ...)``), every remaining axis being a fan axis.
    Reducing "every axis except the rank axis" therefore expresses the paper
    Section A.5 rescale once instead of once per layer type.

    Subclasses provide ``__init__`` and the layer-type-specific weight handling
    (``weight``, ``_weight_norm``, ``_delta_weight``, ``forward``, ``merge``, ...);
    everything that only needs ``first_layer`` / ``second_layer`` lives here.
    This is a base class, not a mixin, so that overrides of ``GrowingBlock``
    methods -- ``apply_change`` here, ``normalize_optimal_updates`` later -- are
    genuine overrides with compatible signatures rather than a second, unrelated
    definition of the same name.
    """

    # Set by the subclass ``__init__``; declared here for the methods below.
    lr_init: float
    use_dora: bool
    magnitude: nn.Parameter | None
    scaling_fn: Callable[[int], float]
    # The frozen original layer, which the subclasses also alias as ``self.linear``
    # / ``self.conv``.  Narrowed from ``GrowingBlock.downsample: nn.Module`` so
    # that ``weight`` and ``bias`` below resolve to real attributes instead of
    # going through ``nn.Module.__getattr__``.
    downsample: _LinearLayerType | _Conv2dLayerType

    @property
    def weight(self) -> torch.Tensor:
        """Original weight (read-only)."""
        return self.downsample.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """Original bias (read-only)."""
        return self.downsample.bias

    @property
    def rank(self) -> int:
        """Current rank: the number of hidden neurons of the adapter."""
        return self.hidden_neurons

    @property
    def scaling(self) -> float:
        """Effective scaling factor applied to the adapter output."""
        if self.rank == 0:
            return 0.0
        return self.scaling_fn(self.rank)

    # Layer-type-specific hooks used by the shared methods below.  They must be
    # declared here and not merely implemented by the subclasses: on an
    # ``nn.Module`` any undeclared attribute resolves through ``__getattr__`` to
    # ``Tensor | Module``, which makes calling it a type error.

    def _weight_norm(self, weight: torch.Tensor) -> torch.Tensor:
        """Return the per-output-unit norm of ``weight``, clamped away from zero.

        Parameters
        ----------
        weight : torch.Tensor
            Effective weight of the wrapped layer.

        Returns
        -------
        torch.Tensor
            One norm per output unit, shaped to broadcast over ``weight``.

        Raises
        ------
        NotImplementedError
            Always; subclasses must provide the layer-type-specific reduction.
        """
        raise NotImplementedError

    def _delta_weight(self, detach_adapter: bool = False) -> torch.Tensor:
        """Return the adapter contribution ``scaling * B @ A`` as a weight delta.

        Parameters
        ----------
        detach_adapter : bool
            If ``True``, detach A and B so no gradient flows through the delta.

        Returns
        -------
        torch.Tensor
            Delta with the same shape as the wrapped layer's weight.

        Raises
        ------
        NotImplementedError
            Always; subclasses must provide the layer-type-specific product.
        """
        raise NotImplementedError

    @staticmethod
    def _rank_norms(weight: torch.Tensor, rank_dim: int) -> torch.Tensor:
        """Return the L2 norm of each rank, reducing every axis but ``rank_dim``.

        Parameters
        ----------
        weight : torch.Tensor
            A or B factor, of any tensor rank.
        rank_dim : int
            Axis carrying the ranks: 0 for A, 1 for B.

        Returns
        -------
        torch.Tensor
            One norm per rank, of shape ``(weight.shape[rank_dim],)``.
        """
        dims = tuple(d for d in range(weight.dim()) if d != rank_dim)
        return torch.linalg.vector_norm(weight, dim=dims)

    @classmethod
    def _rank_norms_for_scaling(cls, weight: torch.Tensor, rank_dim: int) -> torch.Tensor:
        """Per-rank norms shaped to broadcast over ``weight``, with zeros mapped to one.

        A rank that is exactly zero -- from ``alpha_zero`` / ``omega_zero``, or
        from a zero singular value -- must come out of normalization still zero
        rather than NaN.  Mapping its norm to one achieves that for both factors:
        dividing zero by one leaves zero, and multiplying zero by any target
        leaves zero.

        Parameters
        ----------
        weight : torch.Tensor
            A or B factor.
        rank_dim : int
            Axis carrying the ranks: 0 for A, 1 for B.

        Returns
        -------
        torch.Tensor
            Norms of shape ``(1, ..., rank, ..., 1)``, broadcastable over ``weight``.
        """
        norms = cls._rank_norms(weight, rank_dim)
        norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        broadcast_shape = [1] * weight.dim()
        broadcast_shape[rank_dim] = weight.shape[rank_dim]
        return norms.view(broadcast_shape)

    @torch.no_grad()
    def _normalize_new_ranks(self, old_rank: int) -> None:
        """Rescale the newly added ranks per GrowRA paper Section A.5.

        This is a *normalization*, not an initialization: every value it touches
        was computed by ``compute_optimal_updates`` and only its magnitude
        changes, never its direction.

        - Each new A row is rescaled so ``var(A_ij) = 1/fan_in`` (linear Kaiming,
          there being no nonlinearity between A and B), giving ``||A_row|| = 1``.
        - Each new B column is rescaled so ``var(B_ij) = lr_init`` (paper Section
          A.5: "scale B to have variance eta"), giving
          ``||B_col|| = sqrt(fan_out * lr_init)``.

        Parameters
        ----------
        old_rank : int
            Rank of the adapter before the extension was merged; the ranks from
            ``old_rank`` onwards are the new ones.
        """
        if self.rank <= old_rank:
            return
        else:
            # The two factors are divided and multiplied respectively rather
            # than sharing one operation: that is bit-for-bit what the per-rank
            # loops this replaces did, so the refactor cannot move any number.
            a_new = self.first_layer.weight[old_rank:]
            a_new.div_(self._rank_norms_for_scaling(a_new, 0))

            b_new = self.second_layer.weight[:, old_rank:]
            # ``fan_out = b_new.shape[0]`` only holds because every axis after
            # the rank axis is a singleton (a 1x1 kernel for GrowRAConv2d);
            # otherwise the per-rank norm would span fan_out * kH * kW entries.
            assert all(size == 1 for size in b_new.shape[2:]), (
                f"B must have a 1x1 kernel to derive fan_out, got {tuple(b_new.shape)}"
            )
            target = (b_new.shape[0] * self.lr_init) ** 0.5
            b_new.mul_(target / self._rank_norms_for_scaling(b_new, 1))

    def apply_change(
        self,
        extension_size: int | None = None,
        scaling_factor: float | torch.Tensor | None = None,
        apply_delta: bool = True,
        apply_extension: bool = True,
        *,
        lr_init: float | None = None,
        reinit_extension: bool = True,
    ) -> None:
        """Apply the pending change, then normalize the ranks it added.

        The first four parameters are those of ``GrowingBlock.apply_change`` and
        are forwarded unchanged; the two keyword-only ones are GrowRA-specific
        and consumed here, the base method rejecting them.

        Parameters
        ----------
        extension_size : int | None
            Number of new ranks to merge, see ``GrowingBlock.apply_change``.
        scaling_factor : float | torch.Tensor | None
            Scaling applied to the update, see ``GrowingBlock.apply_change``.
        apply_delta : bool
            Whether to apply the optimal delta, see ``GrowingBlock.apply_change``.
        apply_extension : bool
            Whether to merge the extension, see ``GrowingBlock.apply_change``.
        lr_init : float | None
            Overrides ``self.lr_init``, the variance targeted for the new B
            columns by ``_normalize_new_ranks``.  ``None`` keeps the current value.
        reinit_extension : bool
            Whether the newly grown ranks are rescaled by
            ``_normalize_new_ranks`` (paper A.5 variance-eta normalization).
            ``False`` keeps the optimal magnitudes produced by the base
            ``apply_change`` (legacy <=819a23c behaviour).
        """
        if lr_init is not None:
            self.lr_init = lr_init
        old_rank = self.rank
        super().apply_change(
            extension_size=extension_size,
            scaling_factor=scaling_factor,
            apply_delta=apply_delta,
            apply_extension=apply_extension,
        )
        if reinit_extension and apply_extension:
            self._normalize_new_ranks(old_rank)

    def enable_dora(self) -> None:
        """Enable DoRA magnitude reparameterization."""
        if self.use_dora:
            return
        self.use_dora = True
        with torch.no_grad():
            magnitude = self._weight_norm(self.weight + self._delta_weight()).reshape(-1)
        self.magnitude = nn.Parameter(magnitude.clone())

    def reset_adapter(self) -> None:
        """Reset adapter to zero output."""
        nn.init.kaiming_uniform_(self.first_layer.weight)
        nn.init.zeros_(self.second_layer.weight)

    def growra_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable adapter parameters (A and B layers).

        Returns
        -------
        list[nn.Parameter]
            Adapter parameters, including the DoRA magnitude when enabled.
        """
        params = list(self.first_layer.parameters()) + list(
            self.second_layer.parameters()
        )
        if self.use_dora and self.magnitude is not None:
            params.append(self.magnitude)
        return params

    __deepcopy__ = _deepcopy_growra_module


class GrowRALinear(GrowRABlock, LinearGrowingBlock):
    """GrowRA block for nn.Linear (or LinearGrowingModule).

    The decomposition is ``W_original + scaling * B @ A`` where A and B
    are the two internal layers of this ``LinearGrowingBlock``. The frozen
    original layer is used as the residual/downsample path.

    Parameters
    ----------
    linear : nn.Linear | LinearGrowingModule
        Original linear layer (will be frozen).
    rank : int
        Initial rank. Default 0 (no adaptation).
    scaling : float | Callable[[int], float]
        Scaling factor applied to the adapter output. A float gives a fixed
        scaling regardless of rank (default ``1.0``, rank-invariant). A
        callable receives the current rank and returns the scaling factor —
        useful for rank-adaptive schedules such as RSLoRA
        (``lambda r: r ** -0.5``).
    dropout : float
        Dropout probability applied to the input before the adapter path.
        Disabled (``p=0.0``) by default.
    use_dora : bool
        If ``True``, use DoRA-style magnitude reparameterization on top of the
        adapter direction update.
    target_rank : int | None
        Target rank for the growing block.
    activation : torch.nn.Module | None
        Activation between A and B. Default ``nn.Identity()``.
    device : torch.device | None
        Device for parameters.
    name : str
        Name for the growing block.
    lr_init : float
        Initial learning rate hint stored on the module. Default ``1e-3``.
    """

    def __init__(
        self,
        linear: nn.Linear | LinearGrowingModule,
        rank: int = 0,
        scaling: float | Callable[[int], float] = 1.0,
        dropout: float = 0.0,
        use_dora: bool = False,
        target_rank: int | None = None,
        activation: torch.nn.Module | None = None,
        device: torch.device | None = None,
        name: str = "growra_block",
        lr_init: float = 1e-3,
    ):
        linear.requires_grad_(False)
        self._raw_scaling: float | Callable[[int], float] = scaling
        if callable(scaling):
            self.scaling_fn: Callable[[int], float] = scaling
        else:
            _s = float(scaling)
            self.scaling_fn = lambda _: _s
        if device is None:
            device = linear.weight.device

        if activation is None:
            activation = nn.Identity()

        dropout_module = nn.Dropout(p=dropout)

        _scaling = Scaling(self.scaling_fn, lambda: self.rank)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Initializing zero-element tensors is a no-op",
                category=UserWarning,
            )
            super().__init__(
                in_features=linear.in_features,
                out_features=linear.out_features,
                hidden_features=rank,
                target_hidden_features=target_rank,
                activation=activation,
                pre_activation=dropout_module,
                pre_addition_function=_scaling,
                name=name,
                kwargs_layer={"use_bias": False},
                downsample=linear,
                device=device,
            )
        self._scaling = _scaling
        self.linear = linear
        self.lr_init = lr_init
        self.dropout = dropout_module
        self.use_dora = False
        self.magnitude: nn.Parameter | None = None
        if use_dora:
            self.enable_dora()

    def _weight_norm(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.norm(dim=1, keepdim=True).clamp_min(torch.finfo(weight.dtype).eps)

    def _delta_weight(self, detach_adapter: bool = False) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros_like(self.linear.weight)
        A = (
            self.first_layer.weight.detach()
            if detach_adapter
            else self.first_layer.weight
        )
        B = (
            self.second_layer.weight.detach()
            if detach_adapter
            else self.second_layer.weight
        )
        return self.scaling * (B @ A)

    def _effective_weight(self) -> torch.Tensor:
        weight = self.linear.weight + self._delta_weight()
        if not self.use_dora:
            return weight
        assert self.magnitude is not None
        return self.magnitude[:, None] * (weight / self._weight_norm(weight).detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ``original(x) + scaling * B(A(x))``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(..., out_features)``.
        """
        if not self.use_dora:
            return super().forward(x)
        gathering = bool(self.first_layer.store_input)
        assert self.magnitude is not None
        delta = self._delta_weight(detach_adapter=gathering)
        weight = self.linear.weight + delta
        mns = self.magnitude / self._weight_norm(weight).detach().squeeze(1)
        if self.training and self.dropout.p > 0:
            # PEFT parity: base path sees clean x; dropped input flows through
            # the DoRA-scaled direction with a (mns - 1) * W0 correction on x_d.
            x_d = self.dropout(x)
            base_clean = F.linear(x, self.linear.weight, self.linear.bias)
            base_dropped = F.linear(x_d, self.linear.weight)
            delta_dropped = F.linear(x_d, delta)
            out = base_clean + (mns - 1) * base_dropped + mns * delta_dropped
        else:
            eff_w = mns[:, None] * weight
            out = F.linear(x, eff_w, self.linear.bias)
        if gathering:
            # x is detached so the shadow's backward does not reach the layer
            # input; only A/B get their gradients through this path.
            shadow = super().forward(x.detach())
            out = out + (shadow - shadow.detach())
        return out

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extended forward including computed optimal growth directions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
        """
        if self.use_dora:
            assert self.magnitude is not None
            A_ext = self.first_layer.extended_output_layer
            B_ext = self.second_layer.extended_input_layer
            if A_ext is None or B_ext is None:
                return F.linear(x, self._effective_weight(), self.linear.bias)
            ext_scaling = self._scaling.get_scaling(max(1, self.rank))
            W = self.linear.weight + self._delta_weight()
            W_base_norm = self._weight_norm(W).detach()
            W = W + ext_scaling * (B_ext.weight @ A_ext.weight)
            W_dora = self.magnitude[:, None] * (W / W_base_norm)
            return F.linear(x, W_dora, self.linear.bias)
        return super().extended_forward(x)

    def merge(self) -> nn.Linear:
        """Merge adapter into the original layer.

        Returns
        -------
        nn.Linear
            New linear layer with merged weights.
        """
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.linear.bias is not None,
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self._effective_weight())
            if self.linear.bias is not None:
                merged.bias.copy_(self.linear.bias)
        return merged

    def extra_repr(self) -> str:
        """Return extra representation string."""
        dropout_p = self.dropout.p
        scaling_str = "adaptive" if callable(self._raw_scaling) else self._raw_scaling
        s = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, scaling={scaling_str}"
        )
        if self.use_dora:
            s += ", use_dora=True"
        if dropout_p > 0.0:
            s += f", dropout={dropout_p}"
        return s


class GrowRAConv2d(GrowRABlock, Conv2dGrowingBlock):
    """GrowRA wrapper for nn.Conv2d (or Conv2dGrowingModule).

    The decomposition is ``Conv_original(x) + scaling * B(A(x))``
    where A and B are ``Conv2dGrowingModule`` instances composed via a
    ``Conv2dGrowingBlock``. The hidden channels equal the rank and
    start at 0 by default.

    Parameters
    ----------
    conv : nn.Conv2d | Conv2dGrowingModule
        Original convolution layer (will be frozen).
    rank : int
        Initial rank (hidden channels). Default 0.
    scaling : float | Callable[[int], float]
        Scaling factor applied to the adapter output. A float gives a fixed
        scaling regardless of rank (default ``1.0``, rank-invariant). A
        callable receives the current rank and returns the scaling factor.
    dropout : float
        Dropout probability applied to the input before the adapter path.
        Disabled (``p=0.0``) by default.
    use_dora : bool
        If ``True``, use DoRA-style magnitude reparameterization on top of the
        adapter direction update.
    target_rank : int | None
        Target rank for the growing block.
    activation : torch.nn.Module | None
        Activation between A and B. Default ``nn.Identity()``.
    device : torch.device | None
        Device for parameters.
    name : str
        Name for the growing block.
    lr_init : float
        Initial learning rate hint stored on the module. Default ``1e-3``.

    Raises
    ------
    ValueError
        If ``conv`` uses ``groups > 1``. Grouped convolutions are not supported
        because the LoRA adapter assumes a standard (groups=1) weight layout.
    """

    def __init__(
        self,
        conv: nn.Conv2d | Conv2dGrowingModule,
        rank: int = 0,
        scaling: float | Callable[[int], float] = 1.0,
        dropout: float = 0.0,
        use_dora: bool = False,
        target_rank: int | None = None,
        activation: torch.nn.Module | None = None,
        device: torch.device | None = None,
        name: str = "growra_conv_block",
        lr_init: float = 1e-3,
    ):
        if isinstance(conv, Conv2dGrowingModule):
            underlying = conv.layer
        else:
            underlying = conv
        if underlying.groups > 1:
            raise ValueError(
                f"GrowRAConv2d does not support grouped convolutions (groups={underlying.groups}). "
                "The LoRA adapter assumes a standard (groups=1) convolution."
            )
        if device is None:
            device = underlying.weight.device
        self.in_channels: int = int(underlying.in_channels)
        self.out_channels: int = int(underlying.out_channels)
        self._raw_scaling: float | Callable[[int], float] = scaling
        if callable(scaling):
            self.scaling_fn: Callable[[int], float] = scaling
        else:
            _s = float(scaling)
            self.scaling_fn = lambda _: _s

        conv.requires_grad_(False)

        if activation is None:
            activation = nn.Identity()

        dropout_module = nn.Dropout(p=dropout)
        _scaling = Scaling(self.scaling_fn, lambda: self.rank)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Initializing zero-element tensors is a no-op",
                category=UserWarning,
            )
            super().__init__(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                hidden_channels=rank,
                target_hidden_channels=target_rank,
                activation=activation,
                pre_activation=dropout_module,
                pre_addition_function=_scaling,
                name=name,
                kwargs_first_layer={
                    "use_bias": False,
                    "kernel_size": underlying.kernel_size,
                    "stride": underlying.stride,
                    "padding": underlying.padding,
                    "dilation": underlying.dilation,
                },
                kwargs_second_layer={
                    "use_bias": False,
                    "kernel_size": 1,
                    "stride": 1,
                    "padding": 0,
                },
                downsample=conv,
                device=device,
            )
        self._scaling = _scaling
        self.conv = conv
        self.dropout = dropout_module
        self.use_dora = False
        self.magnitude: nn.Parameter | None = None
        self.lr_init = lr_init
        if use_dora:
            self.enable_dora()

    def _conv_base(self) -> nn.Conv2d:
        if isinstance(self.conv, Conv2dGrowingModule):
            return self.conv.layer
        return self.conv

    def _weight_norm(self, weight: torch.Tensor) -> torch.Tensor:
        return (
            weight.flatten(1)
            .norm(dim=1, keepdim=True)
            .clamp_min(torch.finfo(weight.dtype).eps)[:, None, None]
        )

    def _delta_weight(self, detach_adapter: bool = False) -> torch.Tensor:
        orig = self._conv_base()
        if self.rank == 0:
            return torch.zeros_like(orig.weight)
        a_w = (
            self.first_layer.weight.detach()
            if detach_adapter
            else self.first_layer.weight
        )
        b_w = (
            self.second_layer.weight.detach()
            if detach_adapter
            else self.second_layer.weight
        )
        assert b_w.shape[-2:] == (1, 1), (
            f"_delta_weight assumes 1x1 B kernel, got {b_w.shape}"
        )
        b_mat = b_w.squeeze(-1).squeeze(-1)
        a_flat = a_w.view(a_w.shape[0], -1)
        delta_flat = b_mat @ a_flat
        return self.scaling * delta_flat.view_as(orig.weight)

    def _effective_weight(self) -> torch.Tensor:
        orig = self._conv_base()
        weight = orig.weight + self._delta_weight()
        if not self.use_dora:
            return weight
        assert self.magnitude is not None
        return self.magnitude[:, None, None, None] * (
            weight / self._weight_norm(weight).detach()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ``conv(x) + scaling * B(A(x))``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(N, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(N, C_out, H_out, W_out)``.
        """
        if not self.use_dora:
            return super().forward(x)
        orig = self._conv_base()
        gathering = bool(self.first_layer.store_input)
        assert self.magnitude is not None
        delta = self._delta_weight(detach_adapter=gathering)
        weight = orig.weight + delta
        norm = self._weight_norm(weight).detach()
        conv_kw = dict(
            stride=orig.stride,
            padding=orig.padding,
            dilation=orig.dilation,
            groups=orig.groups,
        )
        if self.training and self.dropout.p > 0:
            x_d = self.dropout(x)
            mns = self.magnitude.reshape(1, -1, 1, 1) / norm.reshape(1, -1, 1, 1)
            base_clean = F.conv2d(x, orig.weight, orig.bias, **conv_kw)
            base_dropped = F.conv2d(x_d, orig.weight, None, **conv_kw)
            delta_dropped = F.conv2d(x_d, delta, None, **conv_kw)
            out = base_clean + (mns - 1) * base_dropped + mns * delta_dropped
        else:
            eff_w = self.magnitude[:, None, None, None] * (weight / norm)
            out = F.conv2d(x, eff_w, orig.bias, **conv_kw)
        if gathering:
            # x is detached so the shadow's backward does not reach the layer
            # input; only A/B get their gradients through this path.
            shadow = super().forward(x.detach())
            out = out + (shadow - shadow.detach())
        return out

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extended forward including computed optimal growth directions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
        """
        if self.use_dora:
            assert self.magnitude is not None
            orig = self._conv_base()
            A_ext = self.first_layer.extended_output_layer
            B_ext = self.second_layer.extended_input_layer
            if A_ext is None or B_ext is None:
                return F.conv2d(
                    x,
                    self._effective_weight(),
                    orig.bias,
                    orig.stride,
                    orig.padding,
                    orig.dilation,
                    orig.groups,
                )
            ext_scaling = self._scaling.get_scaling(max(1, self.rank))
            W = orig.weight + self._delta_weight()
            W_base_norm = self._weight_norm(W).detach()
            assert B_ext.weight.shape[-2:] == (1, 1), (
                f"B_ext kernel must be 1x1, got {B_ext.weight.shape}"
            )
            b_ext_mat = B_ext.weight.squeeze(-1).squeeze(-1)
            a_ext_flat = A_ext.weight.view(A_ext.weight.shape[0], -1)
            W = W + ext_scaling * (b_ext_mat @ a_ext_flat).view_as(orig.weight)
            W_dora = self.magnitude[:, None, None, None] * (W / W_base_norm)
            return F.conv2d(
                x,
                W_dora,
                orig.bias,
                orig.stride,
                orig.padding,
                orig.dilation,
                orig.groups,
            )
        return super().extended_forward(x)

    def merge(self) -> nn.Conv2d:
        """Merge adapter into the original convolution layer.

        Note: merging is only exact when both A and B use the same
        kernel_size, stride, padding, and dilation as the original.

        Returns
        -------
        nn.Conv2d
            New conv layer with merged weights.
        """
        orig = self._conv_base()
        merged = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            dilation=orig.dilation,
            groups=orig.groups,
            bias=orig.bias is not None,
            device=orig.weight.device,
            dtype=orig.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self._effective_weight())
            if orig.bias is not None:
                merged.bias.copy_(orig.bias)
        return merged

    def extra_repr(self) -> str:
        """Return extra representation string."""
        dropout_p = self.dropout.p
        scaling_str = "adaptive" if callable(self._raw_scaling) else self._raw_scaling
        s = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"rank={self.rank}, scaling={scaling_str}"
        )
        if self.use_dora:
            s += ", use_dora=True"
        if dropout_p > 0.0:
            s += f", dropout={dropout_p}"
        return s


# Union type for any GrowRA adapter
_GrowRATypes = (GrowRALinear, GrowRAConv2d)
