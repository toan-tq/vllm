# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare dense MLA decode kernels on identical shapes.

Arms:
  triton: vLLM ``decode_attention_fwd`` as called by ``TritonMLAImpl.forward_mqa``
          (multi-token queries are flattened with ``repeat_interleave``).
  fa2:    FlashInfer ``BatchMLAPagedAttentionWrapper(backend="fa2")``.

Reports CUPTI kernel time (cold L2, CUDA graph when capturable), effective
bandwidth assuming the latent cache is read exactly once, that bandwidth as a
fraction of a measured device-to-device copy, and TFLOPS.

Example:
    python benchmarks/kernels/benchmark_mla_decode_backends.py \
        --num-heads 16 20 32 64 128 --batch-sizes 1 8 32 \
        --kv-lens 2048 8192 32768 65536 --q-lens 1 4

"""

import argparse
import math
import statistics

import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

KV_LORA_RANK = 512
ROPE_DIM = 64
HEAD_DIM = KV_LORA_RANK + ROPE_DIM
WARMUP = 10


def bench_us(fn) -> tuple[float, bool]:
    for _ in range(WARMUP):
        fn()
    torch.accelerator.synchronize()
    try:
        times = bench_gpu_time_with_cupti(fn, use_cuda_graph=True)
        graph = True
    except Exception:  # noqa: BLE001
        times = bench_gpu_time_with_cupti(fn, use_cuda_graph=False)
        graph = False
    return statistics.median(times) * 1e3, graph


def measure_copy_gbps() -> float:
    src = torch.empty(1 << 30, dtype=torch.uint8)
    dst = torch.empty_like(src)
    us, _ = bench_us(lambda: dst.copy_(src))
    return 2 * src.numel() / (us * 1e3)


def make_case(batch, q_len, num_heads, kv_len, page_size, kv_dtype):
    blocks_per_seq = math.ceil(kv_len / page_size)
    num_blocks = batch * blocks_per_seq
    cache = torch.randn(num_blocks, page_size, HEAD_DIM, dtype=torch.bfloat16)
    if kv_dtype == "fp8":
        cache = cache.to(torch.float8_e4m3fn)
    block_table = torch.randperm(num_blocks, dtype=torch.int32).view(
        batch, blocks_per_seq
    )
    seq_lens = torch.full((batch,), kv_len, dtype=torch.int32)
    q = torch.randn(batch * q_len, num_heads, HEAD_DIM, dtype=torch.bfloat16)
    return q, cache, block_table, seq_lens


def reference(q, cache, block_table, seq_lens, q_len, scale):
    """FP32 causal reference: query row t of a request sees kv_len - q_len + t + 1."""
    out = torch.empty(q.shape[0], q.shape[1], KV_LORA_RANK, dtype=torch.float32)
    for b in range(block_table.shape[0]):
        kv_len = int(seq_lens[b])
        kv = cache[block_table[b]].reshape(-1, HEAD_DIM)[:kv_len].float()
        for t in range(q_len):
            row = b * q_len + t
            visible = kv_len - q_len + t + 1
            s = q[row].float() @ kv[:visible].T * scale
            out[row] = torch.softmax(s, dim=-1) @ kv[:visible, :KV_LORA_RANK]
    return out


def make_triton_runner(q, cache, block_table, seq_lens, q_len, scale, sm_count):
    rows, num_heads = q.shape[0], q.shape[1]
    page_size = cache.shape[1]
    if q_len > 1:
        block_table = block_table.repeat_interleave(q_len, dim=0)
        offsets = torch.arange(1 - q_len, 1, dtype=seq_lens.dtype)
        seq_lens = (seq_lens.unsqueeze(1) + offsets).flatten().clamp(min=0)
    num_kv_splits = _compute_num_kv_splits(int(seq_lens.max()), sm_count)
    kv = cache.unsqueeze(2)
    kv_c = kv[..., :KV_LORA_RANK]
    o = torch.zeros(rows, num_heads, KV_LORA_RANK, dtype=torch.bfloat16)
    lse = torch.zeros(rows, num_heads, dtype=torch.bfloat16)
    logits = torch.empty(
        rows, num_heads, num_kv_splits, KV_LORA_RANK + 1, dtype=torch.float32
    )
    one = torch.tensor(1.0, dtype=torch.float32)

    def run():
        decode_attention_fwd(
            q,
            kv,
            kv_c,
            o,
            lse,
            block_table,
            seq_lens,
            logits,
            num_kv_splits,
            scale,
            page_size,
            k_scale=one,
            v_scale=one,
            is_mla=True,
        )

    return run, o


def make_fa2_runner(q, cache, block_table, seq_lens, q_len, scale, contiguous_kv):
    import flashinfer

    batch, blocks_per_seq = block_table.shape
    num_heads = q.shape[1]
    page_size = cache.shape[1]
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32) * q_len
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32) * blocks_per_seq
    kv_indices = block_table.flatten().contiguous()
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        seq_lens,
        num_heads,
        KV_LORA_RANK,
        ROPE_DIM,
        page_size,
        q_len > 1,
        scale,
        q.dtype,
        cache.dtype,
    )
    q_nope = q[..., :KV_LORA_RANK].contiguous()
    q_pe = q[..., KV_LORA_RANK:].contiguous()
    ckv, kpe = cache[..., :KV_LORA_RANK], cache[..., KV_LORA_RANK:]
    if contiguous_kv:
        ckv, kpe = ckv.contiguous(), kpe.contiguous()
    o = torch.empty(q.shape[0], num_heads, KV_LORA_RANK, dtype=torch.bfloat16)

    def run():
        wrapper.run(q_nope=q_nope, q_pe=q_pe, ckv_cache=ckv, kpe_cache=kpe, out=o)

    return run, o


def bench_case(backend, case, q_len, scale, sm_count, ref):
    q, cache, block_table, seq_lens = case
    row = {}
    if backend == "triton":
        run, out = make_triton_runner(
            q, cache, block_table, seq_lens, q_len, scale, sm_count
        )
        run()
    else:
        try:
            run, out = make_fa2_runner(
                q, cache, block_table, seq_lens, q_len, scale, False
            )
            run()
        except Exception:  # noqa: BLE001
            # Fall back to separate CKV/KPE tensors if strided views are rejected.
            run, out = make_fa2_runner(
                q, cache, block_table, seq_lens, q_len, scale, True
            )
            run()
            row["note"] = "contiguous ckv/kpe copy"
    torch.accelerator.synchronize()
    if ref is not None:
        err = (out.float() - ref).abs().max().item()
        row["rel_err"] = err / ref.abs().max().item()
    row["us"], row["graph"] = bench_us(run)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-heads", type=int, nargs="+", default=[16, 20, 32, 128])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--q-lens", type=int, nargs="+", default=[1])
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--kv-dtype", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--backends", nargs="+", default=["triton", "fa2"])
    parser.add_argument(
        "--check-max-tokens",
        type=int,
        default=65536,
        help="Skip the FP32 reference when batch * kv_len exceeds this.",
    )
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    copy_gbps = measure_copy_gbps()
    print(
        f"{props.name} sm_{props.major}{props.minor}, "
        f"{props.multi_processor_count} SMs, copy bandwidth {copy_gbps:.0f} GB/s"
    )

    # Absorbed MLA scale for a 128-dim NoPE head; the value only affects numerics.
    scale = 1.0 / math.sqrt(128 + ROPE_DIM)
    elem = 1 if args.kv_dtype == "fp8" else 2
    rows = []
    for num_heads in args.num_heads:
        for batch in args.batch_sizes:
            for kv_len in args.kv_lens:
                for q_len in args.q_lens:
                    case = make_case(
                        batch, q_len, num_heads, kv_len, args.page_size, args.kv_dtype
                    )
                    q = case[0]
                    ref = None
                    if batch * kv_len <= args.check_max_tokens:
                        ref = reference(*case, q_len, scale)
                    kv_bytes = batch * kv_len * HEAD_DIM * elem
                    qo_bytes = q.numel() * 2 + q.shape[0] * num_heads * KV_LORA_RANK * 2
                    flops = (
                        2
                        * batch
                        * q_len
                        * num_heads
                        * kv_len
                        * (HEAD_DIM + KV_LORA_RANK)
                    )
                    for backend in args.backends:
                        row = dict(
                            backend=backend,
                            heads=num_heads,
                            batch=batch,
                            kv_len=kv_len,
                            q_len=q_len,
                            kv_dtype=args.kv_dtype,
                        )
                        try:
                            row.update(
                                bench_case(
                                    backend,
                                    case,
                                    q_len,
                                    scale,
                                    props.multi_processor_count,
                                    ref,
                                )
                            )
                            gbps = (kv_bytes + qo_bytes) / (row["us"] * 1e3)
                            row["gbps"] = gbps
                            row["pct_copy_bw"] = 100 * gbps / copy_gbps
                            row["tflops"] = flops / (row["us"] * 1e6)
                        except Exception as e:  # noqa: BLE001
                            row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                        rows.append(row)
                    del case, q, ref
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
