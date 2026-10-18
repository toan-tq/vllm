# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from typing import Any

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._ops import OpOverload

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform

from ..vllm_inductor_pass import VllmFusionPatternMatcherPass, VllmPatternReplacement
from .matcher_utils import (
    MatcherCustomOp,
    MatcherGeluTanhAndMul,
    MatcherQuantFP8,
    MatcherSiluAndMul,
)
from .rms_quant_fusion import QUANT_OPS, empty_bf16, empty_fp32, empty_i32

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

SILU_MUL_OP = torch.ops._C.silu_and_mul.default
GELU_TANH_MUL_OP = torch.ops._C.gelu_tanh_and_mul.default

FUSED_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.silu_and_mul_quant.default,  # noqa: E501
}
silu_and_mul_nvfp4_quant_supported = current_platform.is_cuda() and hasattr(
    torch.ops._C, "silu_and_mul_nvfp4_quant"
)
if silu_and_mul_nvfp4_quant_supported:
    FUSED_OPS[kNvfp4Dynamic] = torch.ops._C.silu_and_mul_nvfp4_quant.default  # noqa: E501

if current_platform.is_cuda_alike():
    FUSED_OPS[kFp8Dynamic128Sym] = torch.ops._C.silu_and_mul_per_block_quant.default
    FUSED_OPS[kFp8Dynamic64Sym] = torch.ops._C.silu_and_mul_per_block_quant.default

# GeGLU (tanh approximation, e.g. Gemma 2/3/4) counterparts of FUSED_OPS. The
# ops come from vLLM's own csrc, so guard for builds that ship other kernels
# (XPU imports this pass too).
gelu_tanh_and_mul_quant_supported = hasattr(torch.ops._C, "gelu_tanh_and_mul_quant")
GELU_TANH_FUSED_OPS: dict[QuantKey, OpOverload] = {}
if gelu_tanh_and_mul_quant_supported:
    GELU_TANH_FUSED_OPS[kFp8StaticTensorSym] = (
        torch.ops._C.gelu_tanh_and_mul_quant.default
    )
    if current_platform.is_cuda_alike():
        GELU_TANH_FUSED_OPS[kFp8Dynamic128Sym] = (
            torch.ops._C.gelu_tanh_and_mul_per_block_quant.default
        )
        GELU_TANH_FUSED_OPS[kFp8Dynamic64Sym] = (
            torch.ops._C.gelu_tanh_and_mul_per_block_quant.default
        )


class ActivationQuantPattern(VllmPatternReplacement):
    """Base class for Activation+Quant fusions.
    Should not be used directly.

    The gated activation defaults to SiLU+Mul; pass an `act_mul_matcher` and
    the matching `fused_ops` table for another activation (GELU-tanh+Mul).
    """

    def __init__(
        self,
        quant_key: QuantKey,
        act_mul_matcher: MatcherCustomOp | None = None,
        fused_ops: dict[QuantKey, OpOverload] | None = None,
    ) -> None:
        self.quant_key = quant_key
        self.quant_dtype = quant_key.dtype
        if fused_ops is None:
            fused_ops = FUSED_OPS

        assert self.quant_key in QUANT_OPS, (
            f"unsupported quantization scheme {self.quant_key}"
        )
        self.QUANT_OP = QUANT_OPS[self.quant_key]

        assert self.quant_key in fused_ops, (
            f"unsupported fusion scheme {self.quant_key}"
        )
        self.FUSED_OP = fused_ops[self.quant_key]

        self.act_mul_matcher = (
            act_mul_matcher if act_mul_matcher is not None else MatcherSiluAndMul()
        )

    def empty_quant(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs = {
            "dtype": self.quant_dtype,
            "device": current_platform.device_type,
            **kwargs,
        }
        return torch.empty(*args, **kwargs)


class SiluMulFp8StaticQuantPattern(ActivationQuantPattern):
    """Fusion for ActMul+Fp8StaticQuant Pattern (SiluMul by default)."""

    def __init__(
        self,
        act_mul_matcher: MatcherCustomOp | None = None,
        fused_ops: dict[QuantKey, OpOverload] | None = None,
    ) -> None:
        super().__init__(kFp8StaticTensorSym, act_mul_matcher, fused_ops)
        self.quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)

    def get_inputs(self) -> list[torch.Tensor]:
        scale = self.quant_matcher.inputs()[1]
        return [
            *self.act_mul_matcher.inputs(),  # input
            scale,
        ]

    @property
    def pattern(self):
        def _pattern(
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            result_act_mul = self.act_mul_matcher(input)
            result_quant = self.quant_matcher(result_act_mul, scale)
            return result_quant[0]

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            d = input.shape[-1] // 2
            output_shape = input.shape[:-1] + (d,)
            result = torch.empty(
                output_shape, device=input.device, dtype=self.quant_dtype
            )
            at = auto_functionalized(
                self.FUSED_OP, result=result, input=input, scale=scale
            )
            return at[1]

        return _replacement


class SiluMulNvfp4QuantPattern(ActivationQuantPattern):
    """Fusion for SiluMul+Nvfp4Quant Pattern."""

    def __init__(self) -> None:
        super().__init__(kNvfp4Dynamic)

    def get_inputs(self) -> list[torch.Tensor]:
        result = self.empty_quant(5, 32)
        output_scale = empty_i32(128, 4)
        input_ = empty_bf16(5, 64)
        scale = empty_fp32(1, 1)
        return [result, output_scale, input_, scale]

    @property
    def pattern(self):
        def _pattern(
            result: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_silu_mul = self.act_mul_matcher(input)
            at = auto_functionalized(
                self.QUANT_OP,
                input=result_silu_mul,
                input_scale=scale,
                is_sf_swizzled_layout=True,
                output=result,
                output_scale=output_scale,
            )
            return at[1], at[2]

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            result: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                result_block_scale=output_scale,
                input=input,
                input_global_scale=scale,
            )
            return at[1], at[2]

        return _replacement


class SiluMulBlockQuantPattern(ActivationQuantPattern):
    """Fusion for ActMul+BlockQuant (FP8 dynamic per-group) Pattern.
    Supports group_size 128 and 64 via QuantKey.
    Parameterized on is_scale_transposed for different scale layouts.
    """

    def __init__(
        self,
        quant_key: QuantKey,
        is_scale_transposed: bool = False,
        is_e8m0: bool = False,
        is_tma_aligned: bool = False,
        match_aiter: bool = False,
        act_mul_matcher: MatcherCustomOp | None = None,
        fused_ops: dict[QuantKey, OpOverload] | None = None,
    ) -> None:
        super().__init__(quant_key, act_mul_matcher, fused_ops)
        self.quant_matcher = MatcherQuantFP8(
            quant_key,
            has_col_major_scales=is_scale_transposed,
            is_e8m0=is_e8m0,
            is_tma_aligned=is_tma_aligned,
        )
        self.group_size = quant_key.scale.group_shape[1]
        self.is_scale_transposed = is_scale_transposed
        self.is_e8m0 = is_e8m0
        self.is_tma_aligned = is_tma_aligned

    def get_inputs(self) -> list[torch.Tensor]:
        scale = self.quant_matcher.empty_f32(1, 1)
        return self.act_mul_matcher.inputs() + [scale]

    @property
    def pattern(self):
        def _pattern(
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            silu_out = self.act_mul_matcher(input)
            result = torch.empty(
                silu_out.shape,
                device=silu_out.device,
                dtype=self.quant_dtype,
            )
            assert scale is not None
            finfo = torch.finfo(self.quant_dtype)
            _, result, scale = auto_functionalized(
                self.quant_matcher.QUANT_OP,
                input=silu_out,
                output_q=result,
                output_s=scale,
                group_size=self.group_size,
                eps=1e-10,
                fp8_min=finfo.min,
                fp8_max=finfo.max,
                scale_ue8m0=self.is_e8m0,
                dummy_is_scale_transposed=self.is_scale_transposed,
                dummy_is_tma_aligned=self.is_tma_aligned,
            )
            return result, scale

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            d = input.shape[-1] // 2
            output_shape = input.shape[:-1] + (d,)
            result = torch.empty(
                output_shape, device=input.device, dtype=self.quant_dtype
            )
            if self.is_scale_transposed:
                scale = torch.empty(
                    (d // self.group_size, input.shape[0]),
                    device=input.device,
                    dtype=torch.float32,
                ).permute(-1, -2)
            else:
                scale = torch.empty(
                    (input.shape[0], d // self.group_size),
                    device=input.device,
                    dtype=torch.float32,
                )
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                input=input,
                scales=scale,
                group_size=self.group_size,
                scale_ub=None,
                is_scale_transposed=self.is_scale_transposed,
            )
            return at[1], at[2]

        return _replacement


class ActivationQuantFusionPass(VllmFusionPatternMatcherPass):
    """This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "activation_quant_fusion_pass")

        # (activation matcher, fused op table); GELU-tanh covers GeGLU models
        # such as Gemma 2/3/4.
        act_variants: list[tuple[MatcherCustomOp, dict[QuantKey, OpOverload]]] = [
            (MatcherSiluAndMul(), FUSED_OPS),
        ]
        # On ROCm the native GeluAndMul falls back to the exact GELU, which the
        # tanh kernels must not replace; only match the custom op there.
        if gelu_tanh_and_mul_quant_supported and (
            not current_platform.is_rocm() or GeluAndMul.enabled()
        ):
            act_variants.append((MatcherGeluTanhAndMul(), GELU_TANH_FUSED_OPS))

        for act_matcher, fused_ops in act_variants:
            self.register(SiluMulFp8StaticQuantPattern(act_matcher, fused_ops))

        if silu_and_mul_nvfp4_quant_supported:
            self.register(SiluMulNvfp4QuantPattern())

        if current_platform.is_cuda():
            for (
                (act_matcher, fused_ops),
                quant_key,
                is_scale_transposed,
                is_e8m0,
                is_tma_aligned,
            ) in itertools.product(
                act_variants,
                [kFp8Dynamic128Sym, kFp8Dynamic64Sym],
                [False, True],
                [True, False],
                [False, True],
            ):
                self.register(
                    SiluMulBlockQuantPattern(
                        quant_key,
                        is_scale_transposed=is_scale_transposed,
                        is_e8m0=is_e8m0,
                        is_tma_aligned=is_tma_aligned,
                        act_mul_matcher=act_matcher,
                        fused_ops=fused_ops,
                    )
                )

        self.dump_patterns(config, self.pm_pass)
