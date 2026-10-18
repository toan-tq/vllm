# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import time

import torch

from vllm import ir
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed


@torch.inference_mode()
@default_vllm_config()
def main(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int = 0,
    do_profile: bool = False,
    num_warmup_iters: int = 5,
    num_iters: int = 100,
    gemma: bool = False,
    ir_impl: str = "vllm_c",
) -> None:
    set_random_seed(seed)
    torch.set_default_device("cuda")

    if gemma:
        # fp32 (1 + w) weight on `dtype` activations.
        layer = GemmaRMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=0.0, std=0.1)
    else:
        layer = RMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # Pin the IR implementation and verify the dispatch picked it, so the
    # number below is attributed to the right kernel.
    priority = [ir_impl] if ir_impl == "native" else [ir_impl, "native"]
    stack = contextlib.ExitStack()
    stack.enter_context(ir.ops.rms_norm.set_priority(priority))
    stack.enter_context(ir.ops.fused_add_rms_norm.set_priority(priority))
    weight = layer.weight.float() + 1.0 if gemma else layer.weight.data
    eps = layer.variance_epsilon
    if add_residual:
        provider = ir.ops.fused_add_rms_norm.dispatch(x, residual, weight, eps).provider
    else:
        provider = ir.ops.rms_norm.dispatch(x, weight, eps).provider
    if provider != ir_impl:
        raise RuntimeError(
            f"requested IR impl {ir_impl!r} but dispatch selected {provider!r}"
        )
    print(f"IR impl: {provider} (weight dtype {weight.dtype}, input dtype {dtype})")

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.accelerator.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            layer(x, residual)
        torch.accelerator.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=num_warmup_iters, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=num_iters, profile=False)
    print(f"Kernel running time: {latency * 1000000:.3f} us")
    stack.close()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the layernorm kernel.")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--add-residual", action="store_true")
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="half"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num-warmup-iters", type=int, default=5)
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations. "
        "If --profile is set, this number is ignored",
    )
    parser.add_argument(
        "--gemma",
        action="store_true",
        help="Benchmark GemmaRMSNorm (fp32 weight on --dtype activations)",
    )
    parser.add_argument(
        "--ir-impl",
        type=str,
        choices=["vllm_c", "native", "oink"],
        default="vllm_c",
        help="IR implementation to benchmark; the run aborts if dispatch "
        "falls back to another one",
    )

    args = parser.parse_args()
    print(args)

    main(
        num_tokens=args.num_tokens,
        hidden_size=args.hidden_size,
        add_residual=args.add_residual,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        do_profile=args.profile,
        num_warmup_iters=args.num_warmup_iters,
        num_iters=args.num_iters,
        gemma=args.gemma,
        ir_impl=args.ir_impl,
    )
