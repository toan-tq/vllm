# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prototype dense MLA decode kernel: one launch, graph-safe dynamic splits.

UNTESTED ON GPU. Written without access to a GPU or nvcc; run ``--test`` first
and treat every number from ``--bench`` as unconfirmed until the tests pass.
Only checked offline: Triton 3.8.0 compiles it for sm_89/sm_120/sm_121 with
JIT-like divisibility hints (BF16, BLOCK_H=16, 4 warps: 59392 B smem at 1
stage, 63488 B at 2; BLOCK_H=32, 8 warps: 77824/81920 B; FP8 BLOCK_H=16, 1
stage: 88064 B; no register spills) and the PTX holds the expected
``griddepcontrol``, ``fence.acq_rel.gpu`` and single-thread
``atom.global.gpu.acq_rel.add``.

Differences from vLLM's ``decode_attention_fwd`` (stage1 + stage2), each one a
candidate fusion for TRITON_MLA:

* ``q_nope`` and ``q_pe`` come in through separate pointers, so the per-layer
  ``torch.cat`` in ``TritonMLAImpl.forward_mqa`` is not needed.
* Every output row is written (empty rows get ``o = 0``, ``lse = -inf``), so
  ``o`` and ``lse`` can be ``torch.empty`` instead of ``torch.zeros``.
* The split count is decided on the device from ``seq_lens``:
  ``n_split = clamp(cdiv(seq_len, min_tokens_per_split), 1, max_splits)``.
  ``max_splits`` only sizes the grid, so a CUDA graph captured with the
  worst-case ``max_model_len`` no longer replays the worst-case split count.
* ``fused_merge=True``: the CTA that finishes last for a (row, head block)
  merges the partial states (atomic counter, self-resetting), so a decode step
  needs one launch for any split count. ``fused_merge=False`` keeps a separate
  merge kernel; with ``launch_pdl=True`` (SM90+) both kernels use programmatic
  dependent launch.

The last-CTA merge reads ``n_split * BLOCK_H * 513 * 4`` bytes from one SM, so
it only wins when ``n_split`` is small; ``--bench`` measures the crossover.

Example:
    .venv/bin/python benchmarks/kernels/mla_decode_sm12x/triton_mla_fused_merge.py \
        --test --bench

"""

import argparse
import math
import statistics
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

D_CKV = 512
D_KPE = 64


@triton.jit
def _fence_acq_rel_gpu():
    tl.inline_asm_elementwise(
        "fence.acq_rel.gpu; // dummy $0",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _num_splits(seq_len, max_splits, MIN_TOKENS: tl.constexpr):
    if MIN_TOKENS > 0:
        n = tl.minimum(tl.cdiv(seq_len, MIN_TOKENS), max_splits)
        n = tl.maximum(n, 1)
    else:
        n = tl.maximum(seq_len * 0 + max_splits, 1)
    return n


@triton.jit
def _mla_split_kernel(
    QN,
    QP,
    KV,
    BT,
    SL,
    MID,
    CNT,
    Out,
    LSE,
    sm_scale,
    k_scale_ptr,
    stride_qn_b,
    stride_qn_h,
    stride_qp_b,
    stride_qp_h,
    stride_kv_page,
    stride_kv_tok,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_o_b,
    stride_o_h,
    stride_lse_b,
    max_splits,
    n_head_blocks,
    NUM_HEADS: tl.constexpr,
    HEAD_CKV: tl.constexpr,
    HEAD_KPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MIN_TOKENS: tl.constexpr,
    FUSED_MERGE: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    b = tl.program_id(0)
    hb = tl.program_id(1)
    s = tl.program_id(2)

    # Metadata is produced before the graph, so it may be read before the
    # PDL wait; q and the KV cache are read only after it.
    seq_len = tl.load(SL + b)
    n_split = _num_splits(seq_len, max_splits, MIN_TOKENS)

    if s < n_split:
        kv_per_split = tl.cdiv(seq_len, n_split)
        start = s * kv_per_split
        end = tl.minimum(start + kv_per_split, seq_len)

        heads = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = heads < NUM_HEADS
        offs_d = tl.arange(0, HEAD_CKV)
        offs_p = tl.arange(0, HEAD_KPE)
        b64 = b.to(tl.int64)

        if USE_GDC:
            tl.extra.cuda.gdc_wait()

        q = tl.load(
            QN + b64 * stride_qn_b + heads[:, None] * stride_qn_h + offs_d[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )
        qpe = tl.load(
            QP + b64 * stride_qp_b + heads[:, None] * stride_qp_h + offs_p[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )
        ks = tl.load(k_scale_ptr)

        e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, HEAD_CKV], dtype=tl.float32)

        for start_n in tl.range(start, end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < end
            page = tl.load(
                BT + b64 * stride_bt_b + offs_n // PAGE_SIZE,
                mask=mask_n,
                other=0,
            ).to(tl.int64)
            kv_off = page * stride_kv_page + (offs_n % PAGE_SIZE) * stride_kv_tok
            k = tl.load(
                KV + kv_off[None, :] + offs_d[:, None],
                mask=mask_n[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            kpe = tl.load(
                KV + kv_off[None, :] + HEAD_CKV + offs_p[:, None],
                mask=mask_n[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            if kpe.dtype.is_fp8():
                kpe = (kpe.to(tl.float32) * ks).to(qpe.dtype)
            qk = tl.dot(q, k.to(q.dtype))
            qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(q.dtype), tl.trans(k.to(q.dtype)))
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        if USE_GDC:
            tl.extra.cuda.gdc_launch_dependents()

        o_ptrs = Out + b64 * stride_o_b + heads[:, None] * stride_o_h + offs_d[None, :]
        lse_ptrs = LSE + b64 * stride_lse_b + heads
        if n_split == 1:
            has = e_sum > 0.0
            out = tl.where(has[:, None], acc / e_sum[:, None], 0.0)
            lse = tl.where(has, e_max + tl.log(e_sum), float("-inf"))
            tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=mask_h[:, None])
            tl.store(lse_ptrs, lse, mask=mask_h)
        else:
            mid_row = b64 * stride_mid_b + heads * stride_mid_h
            if end > start:
                mid_s = mid_row + s * stride_mid_s
                tl.store(
                    MID + mid_s[:, None] + offs_d[None, :],
                    acc / e_sum[:, None],
                    mask=mask_h[:, None],
                )
                tl.store(MID + mid_s + HEAD_CKV, e_max + tl.log(e_sum), mask=mask_h)
            if FUSED_MERGE:
                # threadFenceReduction pattern: every thread fences its own
                # partial stores, the CTA barrier orders them before thread
                # 0's acq_rel atomic, and the last arriver fences again before
                # reading the other CTAs' partials (bypassing L1 with .cg).
                _fence_acq_rel_gpu()
                tl.debug_barrier()
                cnt_ptr = CNT + b64 * n_head_blocks + hb
                arrived = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
                if arrived == n_split - 1:
                    _fence_acq_rel_gpu()
                    tl.atomic_xchg(cnt_ptr, 0, sem="relaxed", scope="gpu")
                    m_f = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
                    s_f = tl.zeros([BLOCK_H], dtype=tl.float32)
                    a_f = tl.zeros([BLOCK_H, HEAD_CKV], dtype=tl.float32)
                    for sp in range(0, n_split):
                        sp_start = sp * kv_per_split
                        sp_end = tl.minimum(sp_start + kv_per_split, seq_len)
                        if sp_end > sp_start:
                            mid_sp = mid_row + sp * stride_mid_s
                            tv = tl.load(
                                MID + mid_sp[:, None] + offs_d[None, :],
                                mask=mask_h[:, None],
                                other=0.0,
                                cache_modifier=".cg",
                            )
                            tlse = tl.load(
                                MID + mid_sp + HEAD_CKV,
                                mask=mask_h,
                                other=float("-inf"),
                                cache_modifier=".cg",
                            )
                            n_m = tl.maximum(tlse, m_f)
                            old_scale = tl.exp(m_f - n_m)
                            w = tl.exp(tlse - n_m)
                            a_f = a_f * old_scale[:, None] + w[:, None] * tv
                            s_f = s_f * old_scale + w
                            m_f = n_m
                    tl.store(
                        o_ptrs,
                        (a_f / s_f[:, None]).to(Out.dtype.element_ty),
                        mask=mask_h[:, None],
                    )
                    tl.store(lse_ptrs, m_f + tl.log(s_f), mask=mask_h)


@triton.jit
def _mla_merge_kernel(
    MID,
    SL,
    Out,
    LSE,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_o_b,
    stride_o_h,
    stride_lse_b,
    max_splits,
    HEAD_CKV: tl.constexpr,
    MIN_TOKENS: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    seq_len = tl.load(SL + b)
    n_split = _num_splits(seq_len, max_splits, MIN_TOKENS)
    # Rows with a single split were finalized by the split kernel.
    if n_split > 1:
        if USE_GDC:
            tl.extra.cuda.gdc_wait()
        kv_per_split = tl.cdiv(seq_len, n_split)
        offs_d = tl.arange(0, HEAD_CKV)
        mid_row = b.to(tl.int64) * stride_mid_b + h * stride_mid_h
        m_f = float("-inf")
        s_f = 0.0
        a_f = tl.zeros([HEAD_CKV], dtype=tl.float32)
        for sp in range(0, n_split):
            sp_start = sp * kv_per_split
            sp_end = tl.minimum(sp_start + kv_per_split, seq_len)
            if sp_end > sp_start:
                mid_sp = mid_row + sp * stride_mid_s
                tv = tl.load(MID + mid_sp + offs_d)
                tlse = tl.load(MID + mid_sp + HEAD_CKV)
                n_m = tl.maximum(tlse, m_f)
                old_scale = tl.exp(m_f - n_m)
                w = tl.exp(tlse - n_m)
                a_f = a_f * old_scale + w * tv
                s_f = s_f * old_scale + w
                m_f = n_m
        b64 = b.to(tl.int64)
        tl.store(
            Out + b64 * stride_o_b + h * stride_o_h + offs_d,
            (a_f / s_f).to(Out.dtype.element_ty),
        )
        tl.store(LSE + b64 * stride_lse_b + h, m_f + tl.log(s_f))


@dataclass
class MLADecodeWorkspace:
    """Partial states ``[rows, heads, max_splits, 513]`` (vLLM's attn_logits
    layout) and the self-resetting merge counters."""

    mid: torch.Tensor
    counters: torch.Tensor

    @classmethod
    def create(cls, rows, num_heads, max_splits, block_h=16, device="cuda"):
        mid = torch.empty(
            rows, num_heads, max_splits, D_CKV + 1, dtype=torch.float32, device=device
        )
        n_hb = triton.cdiv(num_heads, block_h)
        counters = torch.zeros(rows, n_hb, dtype=torch.int32, device=device)
        return cls(mid, counters)

    def reset(self) -> None:
        self.counters.zero_()


def choose_max_splits(
    rows: int,
    num_heads: int,
    block_h: int,
    sm_count: int,
    max_seq_len: int,
    min_tokens_per_split: int,
    ctas_per_sm: int = 2,
) -> int:
    """Grid split dimension, fixed at launch (and at graph capture).

    Bounded by occupancy (``rows`` is fixed for a captured graph) and by the
    longest sequence; the kernel then uses fewer splits for shorter rows.
    """
    n_hb = triton.cdiv(num_heads, block_h)
    by_len = max(1, triton.cdiv(max_seq_len, max(min_tokens_per_split, 1)))
    by_occupancy = max(1, triton.cdiv(ctas_per_sm * sm_count, rows * n_hb))
    return min(by_len, by_occupancy)


def mla_decode_fused(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    o: torch.Tensor,
    lse: torch.Tensor,
    workspace: MLADecodeWorkspace,
    max_splits: int,
    *,
    k_scale: torch.Tensor | None = None,
    min_tokens_per_split: int = 256,
    block_h: int = 16,
    block_n: int = 32,
    num_warps: int | None = None,
    num_stages: int = 1,
    fused_merge: bool = True,
    launch_pdl: bool = False,
) -> None:
    """Decode attention over a paged latent cache ``[num_blocks, page, 576]``.

    Args:
        q_nope: ``[rows, heads, 512]`` absorbed query, last dim contiguous.
        q_pe: ``[rows, heads, 64]`` rotated query, last dim contiguous.
        kv_cache: ``[num_blocks, page_size, 576]`` latent cache (BF16/FP16/FP8).
        block_table: ``[rows, max_blocks]`` int32, one row per query row.
        seq_lens: ``[rows]`` int32 KV extent per query row.
        sm_scale: softmax scale applied to ``q @ k``.
        o: ``[rows, heads, 512]`` output, fully written.
        lse: ``[rows, heads]`` natural-log LSE, fully written.
        workspace: partial states and counters (counters must start at 0).
        max_splits: grid split dimension (see ``choose_max_splits``).
        k_scale: per-tensor FP8 KV scale (``None`` means 1.0).
        min_tokens_per_split: device-side split granularity; 0 always uses
            ``max_splits`` like vLLM's stage1.
        block_h: query heads per CTA (16 matches vLLM; 32 covers 20 heads).
        block_n: KV tokens per iteration.
        num_warps: warps per CTA (default 8 when ``block_h >= 32``, else 4).
        num_stages: Triton pipelining depth (1 is safe on ~100 KB SMs).
        fused_merge: merge partials in the last CTA instead of a 2nd kernel.
        launch_pdl: programmatic dependent launch (SM90+ only).

    """
    rows, num_heads, d_ckv = q_nope.shape
    assert d_ckv == D_CKV and q_pe.shape == (rows, num_heads, D_KPE)
    assert kv_cache.dim() == 3 and kv_cache.shape[-1] == D_CKV + D_KPE
    assert q_nope.stride(-1) == 1 and q_pe.stride(-1) == 1
    assert kv_cache.stride(-1) == 1
    assert block_table.dtype == torch.int32 and seq_lens.dtype == torch.int32
    assert block_table.shape[0] == rows and seq_lens.shape[0] == rows
    assert o.shape == (rows, num_heads, D_CKV) and o.stride(-1) == 1
    assert lse.shape == (rows, num_heads) and lse.stride(-1) == 1
    assert block_h >= 16 and block_h & (block_h - 1) == 0
    n_hb = triton.cdiv(num_heads, block_h)
    mid = workspace.mid
    assert mid.shape[0] >= rows and mid.shape[1] >= num_heads
    assert mid.shape[2] >= max_splits and mid.shape[3] == D_CKV + 1
    assert workspace.counters.shape[0] >= rows
    assert workspace.counters.shape[1] == n_hb
    assert workspace.counters.is_contiguous()
    if kv_cache.element_size() == 1 and (block_h > 16 or num_stages > 1):
        # Offline Triton 3.8.0 compile (sm_89/120/121): FP8 needs 106496 B
        # for BLOCK_H=32 or 2 stages; ~100 KB GPUs allow 101376 B per block.
        props = torch.cuda.get_device_properties(q_nope.device)
        limit = getattr(props, "shared_memory_per_block_optin", 0)
        if limit and limit < 106496:
            raise ValueError(
                f"FP8 KV with block_h={block_h}, num_stages={num_stages} exceeds "
                f"{limit} B shared memory; use block_h=16, num_stages=1."
            )
    if k_scale is None:
        k_scale = torch.ones(1, dtype=torch.float32, device=q_nope.device)
    if num_warps is None:
        num_warps = 8 if block_h >= 32 else 4
    page_size = kv_cache.shape[1]
    extra = {"launch_pdl": True} if launch_pdl else {}

    _mla_split_kernel[(rows, n_hb, max_splits)](
        q_nope,
        q_pe,
        kv_cache,
        block_table,
        seq_lens,
        mid,
        workspace.counters,
        o,
        lse,
        sm_scale,
        k_scale,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        block_table.stride(0),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        max_splits,
        n_hb,
        NUM_HEADS=num_heads,
        HEAD_CKV=D_CKV,
        HEAD_KPE=D_KPE,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        PAGE_SIZE=page_size,
        MIN_TOKENS=min_tokens_per_split,
        FUSED_MERGE=fused_merge,
        USE_GDC=launch_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra,
    )
    if not fused_merge:
        _mla_merge_kernel[(rows, num_heads)](
            mid,
            seq_lens,
            o,
            lse,
            mid.stride(0),
            mid.stride(1),
            mid.stride(2),
            o.stride(0),
            o.stride(1),
            lse.stride(0),
            max_splits,
            HEAD_CKV=D_CKV,
            MIN_TOKENS=min_tokens_per_split,
            USE_GDC=launch_pdl,
            num_warps=4,
            **extra,
        )


# --------------------------------------------------------------------------
# Tests and timing (GPU required).
# --------------------------------------------------------------------------


def _pdl_supported() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_arch_support_pdl()


def _make_inputs(seq_lens, num_heads, page_size, kv_dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = len(seq_lens)
    max_len = max(max(seq_lens), 1)
    blocks_per_row = triton.cdiv(max_len, page_size)
    num_blocks = rows * blocks_per_row
    cache = torch.randn(
        num_blocks, page_size, D_CKV + D_KPE, generator=gen, device="cuda"
    ).to(torch.bfloat16)
    if kv_dtype == "fp8":
        cache = cache.to(torch.float8_e4m3fn)
    perm = torch.randperm(num_blocks, generator=gen, device="cuda")
    block_table = perm.to(torch.int32).view(rows, blocks_per_row).contiguous()
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    q = torch.randn(rows, num_heads, D_CKV + D_KPE, generator=gen, device="cuda")
    q = q.to(torch.bfloat16)
    return q, cache, block_table, lens


def _vllm_reference(q, cache, block_table, seq_lens, sm_scale, sm_count):
    """Reference: vLLM decode_attention_fwd, eager split heuristic."""
    from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

    rows, num_heads, _ = q.shape
    splits = _compute_num_kv_splits(int(seq_lens.max().item()), sm_count)
    kv = cache.unsqueeze(2)
    o = torch.zeros(rows, num_heads, D_CKV, dtype=q.dtype, device=q.device)
    lse = torch.zeros(rows, num_heads, dtype=torch.float32, device=q.device)
    logits = torch.empty(
        rows, num_heads, splits, D_CKV + 1, dtype=torch.float32, device=q.device
    )
    one = torch.ones(1, dtype=torch.float32, device=q.device)
    decode_attention_fwd(
        q,
        kv,
        kv[..., :D_CKV],
        o,
        lse,
        block_table,
        seq_lens,
        logits,
        splits,
        sm_scale,
        cache.shape[1],
        k_scale=one,
        v_scale=one,
        is_mla=True,
    )
    return o, lse


def _check(o, lse, o_ref, lse_ref, seq_lens, tol):
    nonempty = seq_lens > 0
    errors = []
    if nonempty.any():
        diff = (o[nonempty].float() - o_ref[nonempty].float()).abs().max().item()
        scale = o_ref[nonempty].float().abs().max().item() + 1e-6
        if diff / scale > tol:
            errors.append(f"o rel err {diff / scale:.3e} > {tol}")
        ldiff = (lse[nonempty].float() - lse_ref[nonempty].float()).abs().max()
        if ldiff.item() > 1e-2:
            errors.append(f"lse abs err {ldiff.item():.3e}")
    empty = ~nonempty
    if empty.any():
        if not bool((o[empty] == 0).all()):
            errors.append("empty rows: o != 0")
        if not bool(torch.isneginf(lse[empty].float()).all()):
            errors.append("empty rows: lse != -inf")
    return errors


def run_tests(sm_count: int) -> bool:
    torch.manual_seed(0)
    sm_scale = 1.0 / math.sqrt(256)
    cases = [
        # seq_lens, heads, block_h, min_tokens, fused, kv dtype
        ([1, 33, 1000, 0], 20, 16, 256, True, "bf16"),
        ([1, 33, 1000, 0], 20, 32, 256, True, "bf16"),
        ([1, 33, 1000, 0], 20, 16, 256, False, "bf16"),
        ([8192, 5000], 16, 16, 512, True, "bf16"),
        ([8192, 5000], 16, 16, 0, False, "bf16"),
        ([300, 2048, 17], 64, 16, 64, True, "bf16"),
        ([4097, 2], 128, 16, 256, True, "bf16"),
    ]
    if torch.cuda.get_device_capability() >= (8, 9):
        cases.append(([2000, 31, 0], 20, 16, 256, True, "fp8"))
    pdl = _pdl_supported()
    ok = True
    for seq_lens, heads, block_h, min_tok, fused, kv_dtype in cases:
        q, cache, bt, lens = _make_inputs(seq_lens, heads, 16, kv_dtype)
        rows = len(seq_lens)
        max_splits = choose_max_splits(
            rows, heads, block_h, sm_count, max(seq_lens), max(min_tok, 64)
        )
        ws = MLADecodeWorkspace.create(rows, heads, max_splits, block_h)
        o = torch.empty(rows, heads, D_CKV, dtype=q.dtype, device="cuda")
        lse = torch.empty(rows, heads, dtype=torch.float32, device="cuda")
        try:
            o_ref, lse_ref = _vllm_reference(q, cache, bt, lens, sm_scale, sm_count)
        except Exception as e:  # noqa: BLE001
            # e.g. vllm#60172: FP8 TRITON_MLA needs 102400 B smem at 2 stages.
            print(f"SKIP lens={seq_lens} kv={kv_dtype}: reference failed: {e!r:.200}")
            continue
        for use_pdl in sorted({False, pdl}):
            o.fill_(float("nan"))
            lse.fill_(float("nan"))
            mla_decode_fused(
                q[..., :D_CKV],
                q[..., D_CKV:],
                cache,
                bt,
                lens,
                sm_scale,
                o,
                lse,
                ws,
                max_splits,
                min_tokens_per_split=min_tok,
                block_h=block_h,
                fused_merge=fused,
                launch_pdl=use_pdl,
            )
            torch.accelerator.synchronize()
            errors = _check(o, lse, o_ref, lse_ref, lens, tol=2e-2)
            if fused and int(ws.counters.abs().sum().item()) != 0:
                errors.append("merge counters were not reset")
            name = (
                f"lens={seq_lens} H={heads} block_h={block_h} min_tok={min_tok} "
                f"fused={fused} kv={kv_dtype} pdl={use_pdl} splits<={max_splits}"
            )
            print(("PASS " if not errors else "FAIL ") + name, *errors)
            ok &= not errors
    ok &= _graph_replay_test(sm_count, sm_scale)
    return ok


def _make_runner(
    q_nope, q_pe, cache, bt, lens, sm_scale, o, lse, ws, max_splits, **kwargs
):
    def run():
        mla_decode_fused(
            q_nope, q_pe, cache, bt, lens, sm_scale, o, lse, ws, max_splits, **kwargs
        )

    return run


def _graph_replay_test(sm_count: int, sm_scale: float) -> bool:
    """Capture once, change seq_lens in place, replay: splits must follow."""
    heads, block_h, min_tok = 20, 16, 256
    first, second = [8192, 64, 3000], [100, 8000, 0]
    longest = [max(a, b) for a, b in zip(first, second)]
    q, cache, bt, lens = _make_inputs(longest, heads, 16, "bf16")
    rows = len(first)
    max_splits = choose_max_splits(rows, heads, block_h, sm_count, 8192, min_tok)
    ws = MLADecodeWorkspace.create(rows, heads, max_splits, block_h)
    o = torch.empty(rows, heads, D_CKV, dtype=q.dtype, device="cuda")
    lse = torch.empty(rows, heads, dtype=torch.float32, device="cuda")
    lens.copy_(torch.tensor(first, dtype=torch.int32))
    run = _make_runner(
        q[..., :D_CKV],
        q[..., D_CKV:],
        cache,
        bt,
        lens,
        sm_scale,
        o,
        lse,
        ws,
        max_splits,
        min_tokens_per_split=min_tok,
        block_h=block_h,
    )
    run()  # compile outside capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    ok = True
    for values in (first, second, first):
        lens.copy_(torch.tensor(values, dtype=torch.int32))
        graph.replay()
        torch.accelerator.synchronize()
        o_ref, lse_ref = _vllm_reference(q, cache, bt, lens, sm_scale, sm_count)
        errors = _check(o, lse, o_ref, lse_ref, lens, tol=2e-2)
        status = "PASS" if not errors else "FAIL"
        print(f"{status} graph replay lens={values}", *errors)
        ok &= not errors
    return ok


def _make_vllm_forward_mqa(q_nope, q_pe, cache, bt, lens, sm_scale, splits):
    """What TritonMLAImpl.forward_mqa launches: cat, two zero-fills, stage1/2."""
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

    rows, heads, _ = q_nope.shape
    kv = cache.unsqueeze(2)
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    logits = torch.empty(rows, heads, splits, D_CKV + 1, device="cuda")

    def run():
        qc = torch.cat((q_nope, q_pe), dim=-1)
        o = torch.zeros(rows, heads, D_CKV, dtype=q_nope.dtype, device="cuda")
        lse = torch.zeros(rows, heads, dtype=q_nope.dtype, device="cuda")
        decode_attention_fwd(
            qc,
            kv,
            kv[..., :D_CKV],
            o,
            lse,
            bt,
            lens,
            logits,
            splits,
            sm_scale,
            cache.shape[1],
            k_scale=one,
            v_scale=one,
            is_mla=True,
        )

    return run


def _bench_arms(args, sm_count, heads, batch, kv_len, pdl):
    from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits

    sm_scale = 1.0 / math.sqrt(256)
    q, cache, bt, lens = _make_inputs([kv_len] * batch, heads, 16, "bf16")
    q_nope = q[..., :D_CKV].contiguous()
    q_pe = q[..., D_CKV:]
    arms = {}
    for policy, max_len in (("graph", args.max_model_len), ("eager", kv_len)):
        splits = _compute_num_kv_splits(max_len, sm_count)
        arms[f"vllm forward_mqa {policy} splits={splits}"] = _make_vllm_forward_mqa(
            q_nope, q_pe, cache, bt, lens, sm_scale, splits
        )
    for block_h in args.block_h:
        max_splits = choose_max_splits(
            batch, heads, block_h, sm_count, args.max_model_len, args.min_tokens
        )
        ws = MLADecodeWorkspace.create(batch, heads, max_splits, block_h)
        o = torch.empty(batch, heads, D_CKV, dtype=q.dtype, device="cuda")
        lse = torch.empty(batch, heads, dtype=torch.float32, device="cuda")
        for fused in (True, False):
            for use_pdl in sorted({False, pdl}):
                mode = "fused" if fused else "split+merge"
                name = f"proto bh={block_h} {mode}{' pdl' if use_pdl else ''}"
                arms[f"{name} max_splits={max_splits}"] = _make_runner(
                    q_nope,
                    q_pe,
                    cache,
                    bt,
                    lens,
                    sm_scale,
                    o,
                    lse,
                    ws,
                    max_splits,
                    min_tokens_per_split=args.min_tokens,
                    block_h=block_h,
                    fused_merge=fused,
                    launch_pdl=use_pdl,
                )
    return arms


def run_bench(args, sm_count: int) -> None:
    import pandas as pd
    from flashinfer.testing import bench_gpu_time_with_cupti

    pdl = _pdl_supported()
    rows_out = []
    for heads in args.num_heads:
        for batch in args.batch_sizes:
            for kv_len in args.kv_lens:
                arms = _bench_arms(args, sm_count, heads, batch, kv_len, pdl)
                kv_bytes = batch * kv_len * (D_CKV + D_KPE) * 2
                for name, fn in arms.items():
                    for _ in range(5):
                        fn()
                    torch.accelerator.synchronize()
                    times = bench_gpu_time_with_cupti(fn, use_cuda_graph=True)
                    us = statistics.median(times) * 1e3
                    row = dict(
                        arm=name,
                        heads=heads,
                        batch=batch,
                        kv_len=kv_len,
                        us=us,
                        kv_gbps=kv_bytes / (us * 1e3),
                    )
                    rows_out.append(row)
                    print(row)
                del arms
                torch.accelerator.empty_cache()
    df = pd.DataFrame(rows_out)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if args.csv:
        df.to_csv(args.csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--num-heads", type=int, nargs="+", default=[20])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--block-h", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=202752)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    torch.set_default_device("cuda")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if not (args.test or args.bench):
        args.test = True
    if args.test and not run_tests(sm_count):
        raise SystemExit("Prototype tests failed; do not trust --bench numbers.")
    if args.bench:
        run_bench(args, sm_count)


if __name__ == "__main__":
    main()
