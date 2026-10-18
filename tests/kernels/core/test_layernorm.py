# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.quant_utils import FP8_DTYPE
from tests.kernels.utils import fp8_allclose, fp8_ulp_distance, opcheck
from vllm import ir
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx90a

    on_mi250 = on_gfx90a()
else:
    on_mi250 = False

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [8, 768, 769, 5120, 5125, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True] if not on_mi250 else [True]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


def _rms_norm_tolerance(dtype: torch.dtype) -> dict[str, float]:
    return ir.ops.rms_norm.get_tolerance(dtype)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale
    residual = x.new_empty_strided(x.size(), x.stride()) if add_residual else None
    if residual is not None:
        residual.normal_(std=scale)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    if residual is not None:
        opcheck(
            torch.ops._C.fused_add_rms_norm,
            (x, residual, layer.weight.data, layer.variance_epsilon),
        )
    else:
        opcheck(
            torch.ops._C.rms_norm, (out, x, layer.weight.data, layer.variance_epsilon)
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_weightless(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size, has_weight=False).to(dtype=dtype)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    residual = torch.randn_like(x) if add_residual else None

    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    tol = _rms_norm_tolerance(dtype)
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], **tol)
        torch.testing.assert_close(out[1], ref_out[1], **tol)
    else:
        torch.testing.assert_close(out, ref_out, **tol)

    if residual is not None:
        opcheck(
            torch.ops._C.fused_add_rms_norm,
            (x, residual, None, layer.variance_epsilon),
        )
    else:
        opcheck(
            torch.ops._C.rms_norm,
            (out, x, None, layer.variance_epsilon),
        )


@pytest.mark.parametrize("num_tokens", [7, 4096])
@pytest.mark.parametrize("hidden_size", [768, 769, 5120, 8192])
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm_fp32_weight(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    device: str,
    strided_input: bool,
) -> None:
    """The kernels accept an fp32 weight on fp16/bf16 activations, which is
    what GemmaRMSNorm passes (`weight.float() + 1`), and match the native op."""
    set_random_seed(0)
    torch.set_default_device(device)
    eps = 1e-6
    weight = torch.empty(hidden_size, dtype=torch.float32).normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x_base = torch.randn(num_tokens, last_dim, dtype=dtype) * scale
    x = x_base[..., :hidden_size]
    assert x.is_contiguous() != strided_input

    if add_residual:
        residual = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        ref_out, ref_residual = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
            x, residual.clone(), weight, eps
        )
        x_kernel = x_base.clone()[..., :hidden_size]
        residual_kernel = residual.clone()
        torch.ops._C.fused_add_rms_norm(x_kernel, residual_kernel, weight, eps)
        torch.testing.assert_close(x_kernel, ref_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(residual_kernel, ref_residual, atol=1e-2, rtol=1e-2)
        opcheck(
            torch.ops._C.fused_add_rms_norm, (x_kernel, residual_kernel, weight, eps)
        )
    else:
        ref_out = ir.ops.rms_norm.impls["native"].impl_fn(x, weight, eps)
        out = torch.empty(x.shape, dtype=dtype)
        torch.ops._C.rms_norm(out, x, weight, eps)
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
        opcheck(torch.ops._C.rms_norm, (out, x, weight, eps))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_scale", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
def test_fused_rms_norm_quant(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    quant_scale: float,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x_base = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x_base[..., :hidden_size]
    assert x.is_contiguous() != strided_input

    x *= scale
    if add_residual:
        residual = torch.randn_like(x) * scale
        residual_fused = residual.clone()
    else:
        residual = residual_fused = None

    out_norm = torch.empty_like(x)
    out_quant = torch.empty_like(x, dtype=FP8_DTYPE)
    out_quant_fused = torch.empty_like(out_quant)

    quant_scale_t = torch.tensor(quant_scale, dtype=torch.float32)

    if add_residual:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6
        )

        # Unfused kernel is in-place so it goes second
        # Also use a separate clone of x to avoid modifying the input
        x_unfused_base = x_base.clone()
        x_unfused = x_unfused_base[..., :hidden_size]
        assert x_unfused.is_contiguous() != strided_input
        torch.ops._C.fused_add_rms_norm(x_unfused, residual, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(
            out_quant, x_unfused.contiguous(), quant_scale_t
        )

        torch.accelerator.synchronize()
        torch.testing.assert_close(residual_fused, residual, atol=1e-2, rtol=1e-2)
        opcheck(
            torch.ops._C.fused_add_rms_norm_static_fp8_quant,
            (out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6),
        )
    else:
        torch.ops._C.rms_norm_static_fp8_quant(
            out_quant_fused, x, weight, quant_scale_t, 1e-6
        )

        torch.ops._C.rms_norm(out_norm, x, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(out_quant, out_norm, quant_scale_t)

        opcheck(
            torch.ops._C.rms_norm_static_fp8_quant,
            (out_quant_fused, x, weight, quant_scale_t, 1e-6),
        )

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        if on_gfx950() and dtype == torch.float16 and not add_residual:
            # Fusion may round normalized FP16 across an E4M3 boundary on gfx950.
            assert fp8_allclose(out_quant_fused, out_quant, rtol=0.125, atol=2e-3)
            assert int(fp8_ulp_distance(out_quant_fused, out_quant).max()) <= 1
        else:
            # Fused and unfused FP8 paths can land on opposite sides of an E4M3
            # tie; tolerate a tiny number of isolated fp8 outliers on ROCm.
            ulp = fp8_ulp_distance(out_quant, out_quant_fused)
            max_outliers = ulp.numel() // 100_000 + 8
            num_outliers = int((ulp > 0).sum().item())
            assert num_outliers <= max_outliers, (
                f"FP8 quant mismatch: {num_outliers} fp8 outliers "
                f"(allowed {max_outliers})"
            )
    else:
        torch.testing.assert_close(
            out_quant.to(dtype=torch.float32),
            out_quant_fused.to(dtype=torch.float32),
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@torch.inference_mode()
def test_rms_norm_rejects_other_weight_dtypes(
    default_vllm_config, add_residual: bool
) -> None:
    """Only a matching or fp32 weight is accepted; other mixes raise instead of
    silently reading the weight with the wrong element size."""
    torch.set_default_device(CUDA_DEVICES[0])
    x = torch.randn(4, 64, dtype=torch.float16)
    weight = torch.ones(64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="float32"):
        if add_residual:
            torch.ops._C.fused_add_rms_norm(x, torch.randn_like(x), weight, 1e-6)
        else:
            torch.ops._C.rms_norm(torch.empty_like(x), x, weight, 1e-6)


@pytest.mark.parametrize("num_tokens", [7, 4096])
@pytest.mark.parametrize("hidden_size", [768, 769, 5120])
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("quant_scale", [0.01, 1.0])
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_fused_rms_norm_quant_fp32_weight(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    quant_scale: float,
    device: str,
) -> None:
    """Static FP8 RMSNorm+quant with an fp32 (GemmaRMSNorm-style) weight matches
    the unfused native norm followed by static_scaled_fp8_quant."""
    set_random_seed(0)
    torch.set_default_device(device)
    eps = 1e-6
    weight = torch.empty(hidden_size, dtype=torch.float32).normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
    quant_scale_t = torch.tensor(quant_scale, dtype=torch.float32)
    out_quant = torch.empty_like(x, dtype=FP8_DTYPE)
    out_quant_fused = torch.empty_like(out_quant)

    if add_residual:
        residual = torch.randn_like(x) * scale
        residual_fused = residual.clone()
        ref_norm, ref_residual = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
            x, residual, weight, eps
        )
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out_quant_fused, x.clone(), residual_fused, weight, quant_scale_t, eps
        )
        torch.testing.assert_close(residual_fused, ref_residual, atol=1e-2, rtol=1e-2)
    else:
        ref_norm = ir.ops.rms_norm.impls["native"].impl_fn(x, weight, eps)
        torch.ops._C.rms_norm_static_fp8_quant(
            out_quant_fused, x, weight, quant_scale_t, eps
        )
    torch.ops._C.static_scaled_fp8_quant(out_quant, ref_norm, quant_scale_t)

    # Fused and unfused paths may land on opposite sides of an E4M3 tie for a
    # handful of elements; bound the number of 1-ULP outliers.
    ulp = fp8_ulp_distance(out_quant, out_quant_fused)
    assert int(ulp.max()) <= 1
    assert int((ulp > 0).sum().item()) <= ulp.numel() // 100_000 + 8


@torch.inference_mode()
def test_gemma_rms_norm_mixed_input_weight_dtype(default_vllm_config) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = CUDA_DEVICES[0]
    torch.set_default_device(device)

    num_tokens, hidden_size = 32, 1024
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    layer = GemmaRMSNorm(hidden_size, eps=1e-6).to(device=device)
    layer.weight.data.normal_(mean=0.0, std=0.1)

    # Gemma uses fp32 weight parameter while activations can be bf16.
    assert layer.weight.dtype == torch.float32
    out = layer(x)

    x_fp32 = x.float()
    weight_fp32 = layer.weight.data.float() + 1.0
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    ref = (x_fp32 * torch.rsqrt(variance + layer.variance_epsilon) * weight_fp32).to(
        x.dtype
    )

    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
