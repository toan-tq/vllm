# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare dense MLA decode attention kernels on identical paged shapes.

Arms (all read one latent cache ``[num_blocks, page_size, 576]``):

* ``triton``: vLLM ``decode_attention_fwd`` (stage1 + stage2) with the same
  arguments ``TritonMLAImpl.forward_mqa`` passes; multi-token rows are
  expanded with ``repeat_interleave`` exactly as in ``triton_mla.py``.
  ``--split-policies graph`` uses ``_compute_num_kv_splits(max_model_len)``,
  which is what a FULL CUDA graph captured by vLLM replays for every decode
  step; ``eager`` uses the real KV length (piecewise / eager steps).
* ``triton_fused``: ``triton_mla_fused_merge.py`` prototype (UNTESTED ON GPU).
* ``fa2``: FlashInfer ``BatchMLAPagedAttentionWrapper(backend="fa2")``.

Timing: FlashInfer CUPTI, cold L2, inside a CUDA graph when the arm can be
captured. Kernel-only: the ``torch.cat`` and ``torch.zeros`` that
``forward_mqa`` adds are measured by ``bench_mla_decode_layer_chain.py``.

Example:
    .venv/bin/python benchmarks/kernels/mla_decode_sm12x/bench_mla_decode_kernels.py \
        --num-heads 16 20 --batch-sizes 1 8 --kv-lens 8192 32768 --q-lens 1 2

"""

import argparse
import math
import os
import statistics
import sys

import pandas as pd
import torch

from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

KV_LORA_RANK = 512
ROPE_DIM = 64
HEAD_DIM = KV_LORA_RANK + ROPE_DIM
WARMUP = 10
REL_TOL = 2e-2


def env_summary() -> str:
    props = torch.cuda.get_device_properties(0)
    parts = [
        f"{props.name} sm_{props.major}{props.minor}",
        f"{props.multi_processor_count} SMs",
        f"torch {torch.__version__} (CUDA {torch.version.cuda})",
    ]
    for mod in ("triton", "flashinfer", "vllm"):
        try:
            parts.append(f"{mod} {__import__(mod).__version__}")
        except Exception as e:  # noqa: BLE001
            parts.append(f"{mod} unavailable ({type(e).__name__})")
    return ", ".join(parts)


def graph_capturable(fn) -> tuple[bool, str]:
    """Probe capture separately so a failure cannot poison the timed run."""
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graph.replay()
        torch.accelerator.synchronize()
        return True, ""
    except Exception as e:  # noqa: BLE001
        torch.accelerator.synchronize()
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def bench_us(fn) -> tuple[float, bool, str]:
    from flashinfer.testing import bench_gpu_time_with_cupti

    for _ in range(WARMUP):
        fn()
    torch.accelerator.synchronize()
    use_graph, why = graph_capturable(fn)
    times = bench_gpu_time_with_cupti(fn, use_cuda_graph=use_graph)
    return statistics.median(times) * 1e3, use_graph, why


def measure_copy_gbps() -> float:
    src = torch.empty(1 << 30, dtype=torch.uint8)
    dst = torch.empty_like(src)
    us, _, _ = bench_us(lambda: dst.copy_(src))
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


def gather_kv(cache, block_ids):
    # Index through uint8 so FP8 caches never depend on FP8 index kernels.
    if cache.dtype == torch.float8_e4m3fn:
        rows = cache.view(torch.uint8)[block_ids.long()].view(torch.float8_e4m3fn)
    else:
        rows = cache[block_ids.long()]
    return rows.reshape(-1, HEAD_DIM).float()


def reference(q, cache, block_table, seq_lens, q_len, scale):
    """FP32 causal reference: query row t of a request sees kv_len - q_len + t + 1."""
    out = torch.empty(q.shape[0], q.shape[1], KV_LORA_RANK, dtype=torch.float32)
    for b in range(block_table.shape[0]):
        kv_len = int(seq_lens[b])
        kv = gather_kv(cache, block_table[b])[:kv_len]
        for t in range(q_len):
            row = b * q_len + t
            visible = kv_len - q_len + t + 1
            s = q[row].float() @ kv[:visible].T * scale
            out[row] = torch.softmax(s, dim=-1) @ kv[:visible, :KV_LORA_RANK]
    return out


def expand_rows(block_table, seq_lens, q_len):
    """Mirror TritonMLAImpl.forward_mqa (causal) for uniform multi-token rows."""
    if q_len == 1:
        return block_table, seq_lens
    block_table = block_table.repeat_interleave(q_len, dim=0)
    offsets = torch.arange(1 - q_len, 1, dtype=seq_lens.dtype)
    seq_lens = (seq_lens.unsqueeze(1) + offsets).flatten().clamp(min=0)
    return block_table, seq_lens


def make_triton_runner(case, q_len, scale, num_kv_splits):
    q, cache, block_table, seq_lens = case
    rows, num_heads = q.shape[0], q.shape[1]
    page_size = cache.shape[1]
    block_table, seq_lens = expand_rows(block_table, seq_lens, q_len)
    kv = cache.unsqueeze(2)
    kv_c = kv[..., :KV_LORA_RANK]
    o = torch.zeros(rows, num_heads, KV_LORA_RANK, dtype=q.dtype)
    lse = torch.zeros(rows, num_heads, dtype=q.dtype)
    logits = torch.empty(
        rows, num_heads, num_kv_splits, KV_LORA_RANK + 1, dtype=torch.float32
    )
    one = torch.ones(1, dtype=torch.float32)

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


def make_fused_runner(case, q_len, scale, max_model_len, sm_count, block_h):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import triton_mla_fused_merge as proto

    q, cache, block_table, seq_lens = case
    rows, num_heads = q.shape[0], q.shape[1]
    block_table, seq_lens = expand_rows(block_table, seq_lens, q_len)
    max_splits = proto.choose_max_splits(
        rows, num_heads, block_h, sm_count, max_model_len, 256
    )
    ws = proto.MLADecodeWorkspace.create(rows, num_heads, max_splits, block_h)
    q_nope = q[..., :KV_LORA_RANK].contiguous()
    q_pe = q[..., KV_LORA_RANK:]
    o = torch.empty(rows, num_heads, KV_LORA_RANK, dtype=q.dtype)
    lse = torch.empty(rows, num_heads, dtype=torch.float32)

    def run():
        proto.mla_decode_fused(
            q_nope,
            q_pe,
            cache,
            block_table,
            seq_lens,
            scale,
            o,
            lse,
            ws,
            max_splits,
            block_h=block_h,
        )

    return run, o, max_splits


def make_fa2_runner(case, q_len, scale):
    """FlashInfer 0.7.2 planned wrapper, canonical CSR metadata, packed inputs.

    Packed ``query``/``kv_cache`` are split into zero-copy views for FA2
    (``_contracts.py: _resolve_structural_mla_input``); ``plan`` runs once,
    outside capture, so ``run`` is graph-capturable without
    ``use_cuda_graph=True`` as long as the plan is never redone.
    """
    import flashinfer.mla as fi_mla

    q, cache, block_table, seq_lens = case
    batch, blocks_per_seq = block_table.shape
    num_heads = q.shape[1]
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = fi_mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32) * q_len
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32) * blocks_per_seq
    kv_indices = block_table.flatten().contiguous()
    plan_kwargs = dict(
        num_heads=num_heads,
        head_dim_ckv=KV_LORA_RANK,
        head_dim_kpe=ROPE_DIM,
        page_size=cache.shape[1],
        causal=q_len > 1,
        sm_scale=scale,
        q_data_type=q.dtype,
        kv_data_type=cache.dtype,
    )
    # FP8 KV plans select scale_mode="kv-per-tensor" and need run-time scales.
    scales = (
        dict(ckv_scale=1.0, kpe_scale=1.0) if cache.dtype == torch.float8_e4m3fn else {}
    )
    o = torch.empty(q.shape[0], num_heads, KV_LORA_RANK, dtype=q.dtype)
    if hasattr(fi_mla, "MLAPlanMetadata"):
        metadata = fi_mla.MLAPlanMetadata.csr(
            qo_indptr, kv_indptr, kv_indices, seq_lens
        )
        wrapper.plan(metadata=metadata, **plan_kwargs)

        def run():
            wrapper.run(query=q, kv_cache=cache, out=o, **scales)

    else:  # pre-0.7 FlashInfer: legacy flat plan and split tensors
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=seq_lens,
            **plan_kwargs,
        )
        q_nope = q[..., :KV_LORA_RANK].contiguous()
        q_pe = q[..., KV_LORA_RANK:].contiguous()
        ckv, kpe = cache[..., :KV_LORA_RANK], cache[..., KV_LORA_RANK:]

        def run():
            wrapper.run(
                q_nope=q_nope, q_pe=q_pe, ckv_cache=ckv, kpe_cache=kpe, out=o, **scales
            )

    return run, o


def fa2_fp8_unsupported_reason(kv_dtype) -> str | None:
    if kv_dtype != "fp8":
        return None
    cap = torch.cuda.get_device_capability()
    if cap[0] == 9:
        return None
    # flashinfer 0.7.2 _fa_common.py:_validate_generated_fa_plan rejects FP8
    # KV off SM90; flashinfer PR #6987 (open) adds SM89 single-token decode.
    return f"FA2 FP8 KV needs SM90 in flashinfer 0.7.2 (sm_{cap[0]}{cap[1]})"


def bench_arm(arm, case, args, ref, q_len, scale, sm_count):
    row = {}
    if arm.startswith("triton:"):
        policy = arm.split(":", 1)[1]
        kv_max = int(case[3].max()) if policy == "eager" else args.max_model_len
        splits = _compute_num_kv_splits(kv_max, sm_count)
        run, out = make_triton_runner(case, q_len, scale, splits)
        row["splits"] = splits
    elif arm == "triton_fused":
        run, out, splits = make_fused_runner(
            case, q_len, scale, args.max_model_len, sm_count, args.fused_block_h
        )
        row["splits"] = splits
        row["note"] = "UNTESTED prototype; max splits shown"
    else:
        reason = fa2_fp8_unsupported_reason(args.kv_dtype)
        if reason is not None:
            row["error"] = reason
            return row
        run, out = make_fa2_runner(case, q_len, scale)
    run()
    torch.accelerator.synchronize()
    if ref is not None:
        err = (out.float() - ref).abs().max().item()
        row["rel_err"] = err / ref.abs().max().item()
        row["ok"] = row["rel_err"] <= REL_TOL
    row["us"], row["graph"], why = bench_us(run)
    if why:
        row["note"] = f"no graph: {why}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-heads", type=int, nargs="+", default=[16, 20, 64, 128])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--q-lens", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="vLLM's default KV block size (CacheConfig.DEFAULT_BLOCK_SIZE).",
    )
    parser.add_argument("--kv-dtype", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["triton", "fa2"],
        choices=["triton", "fa2", "triton_fused"],
    )
    parser.add_argument(
        "--split-policies",
        nargs="+",
        default=["graph", "eager"],
        choices=["graph", "eager"],
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=202752,
        help="Drives the 'graph' split policy (GLM-4.7-Flash default context).",
    )
    parser.add_argument(
        "--qk-head-dim",
        type=int,
        default=256,
        help="qk_nope + qk_rope of the model; only sets the softmax scale.",
    )
    parser.add_argument("--fused-block-h", type=int, default=16)
    parser.add_argument(
        "--check-max-tokens",
        type=int,
        default=65536,
        help="Skip the FP32 reference when batch * kv_len exceeds this.",
    )
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    copy_gbps = measure_copy_gbps()
    print(env_summary())
    print(f"device-to-device copy bandwidth: {copy_gbps:.0f} GB/s")
    graph_splits = _compute_num_kv_splits(args.max_model_len, sm_count)
    print(
        f"FULL-graph TRITON_MLA splits for max_model_len={args.max_model_len}: "
        f"{graph_splits}"
    )

    arms = []
    for backend in args.backends:
        if backend == "triton":
            arms += [f"triton:{p}" for p in args.split_policies]
        else:
            arms.append(backend)

    scale = 1.0 / math.sqrt(args.qk_head_dim)
    elem = 1 if args.kv_dtype == "fp8" else 2
    rows = []
    for num_heads in args.num_heads:
        for batch in args.batch_sizes:
            for kv_len in args.kv_lens:
                for q_len in args.q_lens:
                    case = make_case(
                        batch, q_len, num_heads, kv_len, args.page_size, args.kv_dtype
                    )
                    ref = None
                    if batch * kv_len <= args.check_max_tokens:
                        ref = reference(*case, q_len, scale)
                    kv_bytes = batch * kv_len * HEAD_DIM * elem
                    q_rows = batch * q_len
                    qo_bytes = q_rows * num_heads * (HEAD_DIM + KV_LORA_RANK) * 2
                    flops = 2 * q_rows * num_heads * kv_len * (HEAD_DIM + KV_LORA_RANK)
                    for arm in arms:
                        row = dict(
                            arm=arm,
                            heads=num_heads,
                            batch=batch,
                            kv_len=kv_len,
                            q_len=q_len,
                            kv_dtype=args.kv_dtype,
                            page=args.page_size,
                        )
                        try:
                            row.update(
                                bench_arm(arm, case, args, ref, q_len, scale, sm_count)
                            )
                            if "us" in row:
                                gbps = (kv_bytes + qo_bytes) / (row["us"] * 1e3)
                                row["gbps"] = gbps
                                row["pct_copy_bw"] = 100 * gbps / copy_gbps
                                row["tflops"] = flops / (row["us"] * 1e6)
                        except Exception as e:  # noqa: BLE001
                            torch.accelerator.synchronize()
                            row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                        rows.append(row)
                        print(row)
                    del case, ref
                    torch.accelerator.empty_cache()

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if "ok" in df and (df["ok"] == False).any():  # noqa: E712
        print(f"WARNING: rows above rel_err {REL_TOL} -- do not trust their timing.")
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
