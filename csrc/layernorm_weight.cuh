#pragma once

/**
 * Helpers for RMSNorm kernels whose weight dtype may differ from the
 * activation dtype. GemmaRMSNorm-style layers (Gemma 2/3, Qwen3-Next,
 * Qwen3.5) apply an fp32 `1 + w` weight to fp16/bf16 activations.
 */

#include <type_traits>

#include "quantization/vectorization.cuh"

namespace vllm {

// Normalized activation times weight, rounded the way the unfused reference
// (vllm.ir.ops.rms_norm followed by a quant op) rounds it: a same-dtype
// weight multiplies in scalar_t, an fp32 weight multiplies in fp32 and is
// rounded to scalar_t once.
template <typename scalar_t, typename weight_t>
__device__ __forceinline__ scalar_t norm_mul_weight(float x_norm, weight_t w) {
  if constexpr (std::is_same_v<scalar_t, weight_t>) {
    return static_cast<scalar_t>(x_norm) * w;
  } else {
    return static_cast<scalar_t>(x_norm * static_cast<float>(w));
  }
}

// Load VEC_SIZE consecutive weights starting at `weight` as floats. Groups of
// at most 16 bytes use one vector load; wider groups (an fp32 weight read at
// an fp16/bf16 vector width) use scalar loads, which keeps the alignment
// requirement at 16 bytes.
template <typename weight_t, int VEC_SIZE>
__device__ __forceinline__ void load_weight_as_float(
    const weight_t* __restrict__ weight, float (&out)[VEC_SIZE]) {
  if constexpr (sizeof(weight_t) * VEC_SIZE <= 16) {
    const auto vec =
        *reinterpret_cast<const vec_n_t<weight_t, VEC_SIZE>*>(weight);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      out[j] = static_cast<float>(vec.val[j]);
    }
  } else {
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      out[j] = static_cast<float>(weight[j]);
    }
  }
}

}  // namespace vllm
