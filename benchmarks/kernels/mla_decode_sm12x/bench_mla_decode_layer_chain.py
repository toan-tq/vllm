# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time one dense-MLA decode layer (TRITON_MLA path) as a single CUDA graph.

The chain mirrors ``MultiHeadLatentAttentionWrapper.forward`` +
``MLAAttention.forward_impl`` + ``TritonMLAImpl.forward_mqa`` with real vLLM
ops, from the hidden state to the ``o_proj`` output:

    q_lora models (GLM-4.7-Flash): fused_qkv_a GEMM -> q_a/kv_a RMSNorm ->
    q_b GEMM -> RoPE(q_pe, k_pe) -> concat_and_cache_mla -> W_UK bmm ->
    torch.cat -> torch.zeros(o), torch.zeros(lse) -> [row expansion for
    q_len > 1] -> stage1 -> stage2 -> W_UV bmm -> o_proj GEMM.
    no-q_lora models (DeepSeek-V2-Lite): kv_a GEMM -> kv_a RMSNorm -> q GEMM
    -> same tail.

Variants (each is checked against ``base`` before it is timed):

    no_memset     o/lse via torch.empty (stage2 writes every element).
    q_copy        bmm writes into a [T, H, 576] buffer, q_pe copied (no cat;
                  same kernel count, fewer bytes).
    absorbed_q    W_UK folded into the q projection weight: one GEMM emits
                  the 576-wide MQA query, no bmm and no cat.
    fused_norms   vllm.models.common.ops.fused_q_kv_rmsnorm (q_lora only).
    rope_cache    ops.concat_and_cache_mla_rope_fused (RoPE + cache write).
    hoisted_rows  row expansion done once per step, not per layer (q_len > 1).
    split_rope    q-only and k-only RoPE, serial (control for two_stream).
    two_stream    KV path (kv_a norm, k RoPE, cache write) on a second stream
                  overlapping the q path inside the same graph.
    pdl           stage1/stage2 launched with programmatic dependent launch
                  (SM90+; same kernel bodies behind a griddepcontrol.wait).
    proto_attn    triton_mla_fused_merge.py prototype (UNTESTED ON GPU).
    combined      no_memset + absorbed_q + fused_norms + hoisted_rows +
                  two_stream (+ pdl when supported).

``--split-policy graph`` reproduces what vLLM's FULL decode graph replays:
``num_kv_splits`` from ``max_model_len``, not from the real KV length.

Example:
    cd benchmarks/kernels/mla_decode_sm12x
    ../../../.venv/bin/python bench_mla_decode_layer_chain.py \
        --preset glm47flash --batch-sizes 1 8 --kv-lens 8192 32768 --q-lens 1 2

"""

import argparse
import math
import os
import statistics
import sys
from dataclasses import dataclass, field

import pandas as pd
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_grouped_kernel_stage1,
    _fwd_kernel_stage2,
    _page_stride,
    decode_attention_fwd,
)

KV_LORA = 512
ROPE = 64
LATENT = KV_LORA + ROPE
EPS = 1e-5

# Shapes: transformers 5.20.2 Glm4MoeLiteConfig defaults (num_attention_heads=20,
# qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256, kv_lora_rank=512,
# q_lora_rank=768, hidden_size=2048, num_hidden_layers=47,
# max_position_embeddings=202752). DeepSeek-V2-Lite values are from its model
# card (not verified offline here).
PRESETS = {
    "glm47flash": dict(
        heads=20, nope=192, v_head=256, q_lora=768, hidden=2048, layers=47,
        max_model_len=202752,
    ),
    "dsv2lite": dict(
        heads=16, nope=128, v_head=128, q_lora=0, hidden=2048, layers=27,
        max_model_len=163840,
    ),
}  # fmt: skip


@dataclass(frozen=True)
class Variant:
    name: str
    memset: bool = True
    q_mode: str = "cat"  # cat | copy | absorbed
    fused_norms: bool = False
    rope_cache: bool = False
    hoist_rows: bool = False
    split_rope: bool = False
    two_stream: bool = False
    pdl: bool = False
    attn: str = "vllm"  # vllm | proto
    bitwise: bool = True


@dataclass
class Shape:
    heads: int
    nope: int
    v_head: int
    q_lora: int
    hidden: int
    batch: int
    q_len: int
    kv_len: int
    page: int
    splits: int
    o_proj: bool
    extra: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return self.batch * self.q_len

    @property
    def qk_head(self) -> int:
        return self.nope + ROPE


# --------------------------------------------------------------------------
# PDL launch of vLLM's own stage1/stage2 bodies (SM90+).
# --------------------------------------------------------------------------


@triton.jit
def _stage1_pdl(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kpbs,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vpbs,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    k_scale,
    v_scale,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    _fwd_grouped_kernel_stage1(
        Q,
        K_Buffer,
        V_Buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        Att_Out,
        stride_req_to_tokens_b,
        stride_qbs,
        stride_qh,
        stride_buf_kpbs,
        stride_buf_kbs,
        stride_buf_kh,
        stride_buf_vpbs,
        stride_buf_vbs,
        stride_buf_vh,
        stride_mid_ob,
        stride_mid_oh,
        stride_mid_os,
        k_scale,
        v_scale,
        kv_group_num=kv_group_num,
        q_head_num=q_head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=PAGE_SIZE,
        logit_cap=logit_cap,
        Lk=Lk,
        Lv=Lv,
        IS_MLA=True,
    )


@triton.jit
def _stage2_pdl(
    Mid_O,
    o,
    lse,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    _fwd_kernel_stage2(
        Mid_O,
        o,
        lse,
        B_Seqlen,
        stride_mid_ob,
        stride_mid_oh,
        stride_mid_os,
        stride_obs,
        stride_oh,
        stride_lse_bs,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
    )


def decode_attention_fwd_pdl(
    q, kv, kv_c, o, lse, bt, seq_lens, logits, splits, sm_scale, page, k_scale
):
    """Same launch parameters as _decode_grouped_att_m_fwd (MLA, CUDA)."""
    rows, heads = q.shape[0], q.shape[1]
    block_h = 16
    kv_group_num = heads // kv.shape[-2]
    grid = (rows, triton.cdiv(heads, min(block_h, kv_group_num)), splits)
    _stage1_pdl[grid](
        q,
        kv,
        kv_c,
        sm_scale,
        bt,
        seq_lens,
        logits,
        bt.stride(0),
        q.stride(0),
        q.stride(1),
        _page_stride(kv, page),
        kv.stride(-3),
        kv.stride(-2),
        _page_stride(kv_c, page),
        kv_c.stride(-3),
        kv_c.stride(-2),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        k_scale,
        k_scale,
        kv_group_num=kv_group_num,
        q_head_num=heads,
        BLOCK_DMODEL=KV_LORA,
        BLOCK_DPE=ROPE,
        BLOCK_DV=KV_LORA,
        BLOCK_N=32,
        BLOCK_H=block_h,
        NUM_KV_SPLITS=splits,
        PAGE_SIZE=page,
        logit_cap=0.0,
        Lk=LATENT,
        Lv=KV_LORA,
        num_warps=4,
        num_stages=2,
        launch_pdl=True,
    )
    _stage2_pdl[(rows, heads)](
        logits,
        o,
        lse,
        seq_lens,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=splits,
        BLOCK_DV=KV_LORA,
        Lv=KV_LORA,
        num_warps=4,
        num_stages=2,
        launch_pdl=True,
    )


# --------------------------------------------------------------------------
# The layer.
# --------------------------------------------------------------------------


def expand_rows(block_table, seq_lens, q_len):
    """TritonMLAImpl.forward_mqa, causal branch (triton_mla.py)."""
    if q_len == 1:
        return block_table, seq_lens
    block_table = block_table.repeat_interleave(q_len, dim=0)
    offsets = torch.arange(1 - q_len, 1, device=seq_lens.device, dtype=seq_lens.dtype)
    seq_lens = (seq_lens.unsqueeze(1) + offsets).flatten().clamp(min=0)
    return block_table, seq_lens


class MLADecodeLayer:
    def __init__(self, s: Shape, seed: int = 0):
        self.s = s
        g = torch.Generator(device="cuda").manual_seed(seed)

        def w(*shape, fan_in):
            t = torch.randn(*shape, generator=g, device="cuda") / math.sqrt(fan_in)
            return t.to(torch.bfloat16)

        T, H = s.rows, s.heads
        self.h = w(T, s.hidden, fan_in=1)
        if s.q_lora:
            self.W_a = w(s.q_lora + LATENT, s.hidden, fan_in=s.hidden)
            self.w_q_ln = 1 + 0.1 * w(s.q_lora, fan_in=1)
            q_in = s.q_lora
        else:
            self.W_kva = w(LATENT, s.hidden, fan_in=s.hidden)
            q_in = s.hidden
        self.w_kv_ln = 1 + 0.1 * w(KV_LORA, fan_in=1)
        self.W_q = w(H * s.qk_head, q_in, fan_in=q_in)
        self.W_UK_T = w(H, s.nope, KV_LORA, fan_in=s.nope)
        self.W_UV = w(H, KV_LORA, s.v_head, fan_in=KV_LORA)
        self.W_o = w(s.hidden, H * s.v_head, fan_in=H * s.v_head)
        self.W_abs = self._absorb_weight()
        self.scale = 1.0 / math.sqrt(s.qk_head)

        blocks_per_seq = triton.cdiv(s.kv_len, s.page)
        num_blocks = s.batch * blocks_per_seq
        self.kv_cache = torch.randn(
            num_blocks, s.page, LATENT, generator=g, device="cuda"
        ).to(torch.bfloat16)
        perm = torch.randperm(num_blocks, generator=g, device="cuda")
        self.block_table = perm.to(torch.int32).view(s.batch, blocks_per_seq)
        self.seq_lens = torch.full((s.batch,), s.kv_len, dtype=torch.int32)
        pos = torch.arange(s.q_len, device="cuda") + (s.kv_len - s.q_len)
        self.positions = pos.repeat(s.batch).to(torch.int64)
        req = torch.arange(s.batch, device="cuda").repeat_interleave(s.q_len)
        pages = self.block_table[req, self.positions // s.page].to(torch.int64)
        self.slot_mapping = pages * s.page + self.positions % s.page
        inv_freq = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device="cuda") / ROPE))
        freqs = torch.outer(torch.arange(s.kv_len + 1, device="cuda"), inv_freq)
        self.cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)
        self.k_scale = torch.ones(1, dtype=torch.float32)

        self.logits = torch.empty(T, H, s.splits, KV_LORA + 1, dtype=torch.float32)
        self.kv_c_buf = torch.empty(T, KV_LORA, dtype=torch.bfloat16)
        self.kv_lora_buf = torch.empty(T, LATENT, dtype=torch.bfloat16)
        self.bt_rows, self.sl_rows = expand_rows(
            self.block_table, self.seq_lens, s.q_len
        )
        self.aux = torch.cuda.Stream()
        self.proto = None

    def _absorb_weight(self) -> torch.Tensor:
        """[H*576, q_in]: per head (W_UK_T[h]^T @ W_q_nope[h]) stacked on W_q_pe[h]."""
        s = self.s
        wq = self.W_q.float().view(s.heads, s.qk_head, -1)
        nope = torch.einsum("hpl,hpi->hli", self.W_UK_T.float(), wq[:, : s.nope])
        return (
            torch.cat((nope, wq[:, s.nope :]), dim=1)
            .reshape(-1, wq.shape[-1])
            .to(torch.bfloat16)
        )

    # -- pieces ------------------------------------------------------------

    def _rms(self, x, weight, out=None):
        out = torch.empty_like(x) if out is None else out
        ops.rms_norm(out, x, weight, EPS)
        return out

    def _rope(self, x, key=None):
        ops.rotary_embedding(self.positions, x, key, ROPE, self.cos_sin, False)

    def _q_gemm(self, v, q_in):
        s = self.s
        if v.q_mode == "absorbed":
            q = F.linear(q_in, self.W_abs).view(s.rows, s.heads, LATENT)
            return q, q[..., KV_LORA:]
        q = F.linear(q_in, self.W_q).view(s.rows, s.heads, s.qk_head)
        return q, q[..., s.nope :]

    def _absorb(self, v, q, q_pe):
        """Returns the MQA query as a tensor or a (ql_nope, q_pe) tuple."""
        s = self.s
        if v.q_mode == "absorbed":
            return q
        q_nope = q[..., : s.nope].transpose(0, 1)
        if v.q_mode == "copy":
            buf = q.new_empty(s.rows, s.heads, LATENT)
            torch.bmm(q_nope, self.W_UK_T, out=buf[..., :KV_LORA].transpose(0, 1))
            buf[..., KV_LORA:].copy_(q_pe)
            return buf
        ql = q.new_empty(s.rows, s.heads, KV_LORA)
        torch.bmm(q_nope, self.W_UK_T, out=ql.transpose(0, 1))
        return (ql, q_pe)

    def _cache_write(self, kv_c, k_pe):
        ops.concat_and_cache_mla(
            kv_c,
            k_pe.squeeze(1),
            self.kv_cache,
            self.slot_mapping,
            "auto",
            self.k_scale,
        )

    def _rope_and_cache(self, v, q_pe, k_pe, kv_c):
        if v.rope_cache:
            ops.concat_and_cache_mla_rope_fused(
                self.positions, q_pe, k_pe.squeeze(1), kv_c, self.cos_sin, False,
                self.slot_mapping, self.kv_cache, "auto", self.k_scale,
            )  # fmt: skip
            return
        if v.split_rope:
            self._rope(q_pe)
            self._rope(k_pe)
        else:
            self._rope(q_pe, k_pe)
        self._cache_write(kv_c, k_pe)

    def _attend(self, v, q_attn):
        s = self.s
        if v.hoist_rows or s.q_len == 1:
            bt, sl = self.bt_rows, self.sl_rows
        else:
            bt, sl = expand_rows(self.block_table, self.seq_lens, s.q_len)
        if v.attn == "proto":
            if isinstance(q_attn, tuple):
                q_nope, q_pe = q_attn
            else:
                q_nope, q_pe = q_attn[..., :KV_LORA], q_attn[..., KV_LORA:]
            o = q_nope.new_empty(s.rows, s.heads, KV_LORA)
            lse = q_nope.new_empty(s.rows, s.heads)
            p, ws, max_splits = self.proto
            p.mla_decode_fused(
                q_nope, q_pe, self.kv_cache, bt, sl, self.scale, o, lse, ws,
                max_splits, launch_pdl=v.pdl,
            )  # fmt: skip
            return o
        if isinstance(q_attn, tuple):
            q_attn = torch.cat(q_attn, dim=-1)
        alloc = torch.zeros if v.memset else torch.empty
        o = alloc(s.rows, s.heads, KV_LORA, dtype=q_attn.dtype, device="cuda")
        lse = alloc(s.rows, s.heads, dtype=q_attn.dtype, device="cuda")
        kv = self.kv_cache.unsqueeze(2)
        args = (q_attn, kv, kv[..., :KV_LORA], o, lse, bt, sl, self.logits)
        if v.pdl:
            decode_attention_fwd_pdl(*args, s.splits, self.scale, s.page, self.k_scale)
        else:
            decode_attention_fwd(
                *args, s.splits, self.scale, s.page, k_scale=self.k_scale,
                v_scale=self.k_scale, is_mla=True,
            )  # fmt: skip
        return o

    def _project(self, o):
        s = self.s
        out = o.new_empty(s.rows, s.heads * s.v_head)
        torch.bmm(
            o.transpose(0, 1),
            self.W_UV,
            out=out.view(s.rows, s.heads, s.v_head).transpose(0, 1),
        )
        return F.linear(out, self.W_o) if s.o_proj else out

    # -- whole layer ---------------------------------------------------------

    def run(self, v: Variant) -> torch.Tensor:
        if v.two_stream:
            q_attn = self._front_two_stream(v)
        else:
            q_attn = self._front_one_stream(v)
        return self._project(self._attend(v, q_attn))

    def _split_a(self, a):
        s = self.s
        kv_raw = a[:, s.q_lora : s.q_lora + KV_LORA]
        k_pe = a[:, s.q_lora + KV_LORA :].unsqueeze(1)
        return a[:, : s.q_lora], kv_raw, k_pe

    def _front_one_stream(self, v):
        s = self.s
        if s.q_lora:
            q_raw, kv_raw, k_pe = self._split_a(F.linear(self.h, self.W_a))
            if v.fused_norms:
                from vllm.models.common.ops import fused_q_kv_rmsnorm

                q_in, kv_c = fused_q_kv_rmsnorm(
                    q_raw, kv_raw, self.w_q_ln, self.w_kv_ln, EPS
                )
            else:
                q_in = self._rms(q_raw, self.w_q_ln)
                kv_c = self._rms(kv_raw, self.w_kv_ln)
        else:
            _, kv_raw, k_pe = self._split_a(F.linear(self.h, self.W_kva))
            kv_c = self._rms(kv_raw, self.w_kv_ln)
            q_in = self.h
        q, q_pe = self._q_gemm(v, q_in)
        self._rope_and_cache(v, q_pe, k_pe, kv_c)
        return self._absorb(v, q, q_pe)

    def _front_two_stream(self, v):
        """Run the q path here and the KV path on ``self.aux``, then join.

        KV-path outputs live in preallocated buffers so nothing is allocated
        on the side stream during capture.
        """
        s = self.s
        main = torch.cuda.current_stream()
        q_in = kv_c = None
        if s.q_lora:
            a = F.linear(self.h, self.W_a)
            q_raw, kv_raw, k_pe = self._split_a(a)
            if v.fused_norms:
                from vllm.models.common.ops import fused_q_kv_rmsnorm

                q_in, kv_c = fused_q_kv_rmsnorm(
                    q_raw, kv_raw, self.w_q_ln, self.w_kv_ln, EPS
                )
        self.aux.wait_stream(main)
        with torch.cuda.stream(self.aux):
            if not s.q_lora:
                torch.mm(self.h, self.W_kva.t(), out=self.kv_lora_buf)
                _, kv_raw, k_pe = self._split_a(self.kv_lora_buf)
            if kv_c is None:
                kv_c = self._rms(kv_raw, self.w_kv_ln, out=self.kv_c_buf)
            self._rope(k_pe)
            self._cache_write(kv_c, k_pe)
        if q_in is None:
            q_in = self._rms(q_raw, self.w_q_ln) if s.q_lora else self.h
        q, q_pe = self._q_gemm(v, q_in)
        self._rope(q_pe)
        q_attn = self._absorb(v, q, q_pe)
        main.wait_stream(self.aux)
        return q_attn


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


def build_variants(s: Shape, pdl_ok: bool, proto_ok: bool) -> list[Variant]:
    out = [
        Variant("base"),
        Variant("no_memset", memset=False),
        Variant("q_copy", q_mode="copy"),
        Variant("absorbed_q", q_mode="absorbed", bitwise=False),
        Variant("rope_cache", rope_cache=True, bitwise=False),
        Variant("split_rope", split_rope=True),
        Variant("two_stream", two_stream=True, split_rope=True),
    ]
    if s.q_lora:
        out.append(Variant("fused_norms", fused_norms=True, bitwise=False))
    if s.q_len > 1:
        out.append(Variant("hoisted_rows", hoist_rows=True))
    if pdl_ok:
        out.append(Variant("pdl", pdl=True))
    if proto_ok:
        out.append(Variant("proto_attn", attn="proto", bitwise=False))
    out.append(
        Variant(
            "combined",
            memset=False,
            q_mode="absorbed",
            fused_norms=bool(s.q_lora),
            hoist_rows=True,
            two_stream=True,
            split_rope=True,
            pdl=pdl_ok,
            bitwise=False,
        )
    )
    return out


def pdl_status() -> tuple[bool, str]:
    if not current_platform.is_arch_support_pdl():
        cap = torch.cuda.get_device_capability()
        return False, f"PDL needs SM90+ (griddepcontrol); this is sm_{cap[0]}{cap[1]}"
    if not hasattr(tl.extra.cuda, "gdc_wait"):
        return False, f"triton {triton.__version__} has no tl.extra.cuda.gdc_wait"
    return True, ""


def load_proto(layer: MLADecodeLayer, max_model_len: int):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import triton_mla_fused_merge as proto

    s = layer.s
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    max_splits = proto.choose_max_splits(
        s.rows, s.heads, 16, sm_count, max_model_len, 256
    )
    ws = proto.MLADecodeWorkspace.create(s.rows, s.heads, max_splits, 16)
    layer.proto = (proto, ws, max_splits)


def max_rel(x, ref) -> float:
    return ((x.float() - ref.float()).abs().max() / ref.float().abs().max()).item()


def compare(out, ref, kv_slots, kv_ref, bitwise):
    """Bitwise variants must match exactly; others within 3e-2 relative."""
    if bitwise and not torch.equal(kv_slots, kv_ref):
        return False, float("nan"), "KV cache write differs"
    if not bitwise and max_rel(kv_slots, kv_ref) > 3e-2:
        return False, float("nan"), "KV cache write differs"
    if bitwise and torch.equal(out, ref):
        return True, 0.0, ""
    rel = max_rel(out, ref)
    if bitwise:
        return rel <= 1e-3, rel, "not bitwise-identical"
    return rel <= 3e-2, rel, ""


def count_kernels(fn) -> tuple[int, list[str]]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.accelerator.synchronize()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    return len(names), names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="glm47flash")
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--q-lens", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--split-policy", choices=["graph", "eager"], default="graph")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--no-o-proj", action="store_true")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument(
        "--count-kernels",
        action="store_true",
        help="After timing, list GPU kernels per variant with torch.profiler.",
    )
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    from flashinfer.testing import bench_gpu_time_with_cupti

    torch.set_default_device("cuda")
    preset = dict(PRESETS[args.preset])
    if args.num_heads:
        preset["heads"] = args.num_heads
    max_model_len = args.max_model_len or preset["max_model_len"]
    layers = args.num_layers or preset["layers"]
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    pdl_ok, pdl_why = pdl_status()
    if not pdl_ok:
        print(f"SKIP pdl variant: {pdl_why}")
    print(
        f"{torch.cuda.get_device_name()} {sm_count} SMs, preset {args.preset} "
        f"{preset}, split policy {args.split_policy}, max_model_len {max_model_len}"
    )

    results, kernel_lists = [], {}
    for batch in args.batch_sizes:
        for kv_len in args.kv_lens:
            for q_len in args.q_lens:
                split_len = max_model_len if args.split_policy == "graph" else kv_len
                s = Shape(
                    heads=preset["heads"],
                    nope=preset["nope"],
                    v_head=preset["v_head"],
                    q_lora=preset["q_lora"],
                    hidden=preset["hidden"],
                    batch=batch,
                    q_len=q_len,
                    kv_len=kv_len,
                    page=args.page_size,
                    splits=_compute_num_kv_splits(split_len, sm_count),
                    o_proj=not args.no_o_proj,
                )
                layer = MLADecodeLayer(s)
                proto_ok = True
                try:
                    load_proto(layer, max_model_len)
                except Exception as e:  # noqa: BLE001
                    proto_ok = False
                    print(f"SKIP proto_attn: {type(e).__name__}: {e}")
                variants = build_variants(s, pdl_ok, proto_ok)
                if args.variants:
                    keep = set(args.variants) | {"base"}
                    variants = [v for v in variants if v.name in keep]

                base = variants[0]
                ref = layer.run(base)
                kv_ref = layer.kv_cache[
                    layer.slot_mapping // s.page, layer.slot_mapping % s.page
                ].clone()
                base_us = None
                for v in variants:
                    row = dict(
                        variant=v.name,
                        batch=batch,
                        q_len=q_len,
                        kv_len=kv_len,
                        heads=s.heads,
                        splits=s.splits,
                    )
                    try:
                        out = layer.run(v)
                        torch.accelerator.synchronize()
                        kv_now = layer.kv_cache[
                            layer.slot_mapping // s.page, layer.slot_mapping % s.page
                        ]
                        ok, rel, note = compare(out, ref, kv_now, kv_ref, v.bitwise)
                        row.update(ok=ok, rel_err=rel, note=note)
                        if not ok:
                            row["note"] = f"{note} -- not timed".strip()
                            results.append(row)
                            print(row)
                            continue
                        fn = lambda v=v, layer=layer: layer.run(v)  # noqa: E731
                        for _ in range(5):
                            fn()
                        torch.accelerator.synchronize()
                        times = bench_gpu_time_with_cupti(fn, use_cuda_graph=True)
                        us = statistics.median(times) * 1e3
                        base_us = us if v is base else base_us
                        row["us"] = us
                        if base_us is not None:
                            row["delta_us"] = us - base_us
                            row["step_delta_ms"] = (us - base_us) * layers / 1e3
                    except Exception as e:  # noqa: BLE001
                        torch.accelerator.synchronize()
                        row.update(ok=False, note=f"{type(e).__name__}: {str(e)[:160]}")
                    results.append(row)
                    print(row)
                if args.count_kernels:
                    for v in variants:
                        try:
                            n, names = count_kernels(
                                lambda v=v, layer=layer: layer.run(v)
                            )
                        except Exception as e:  # noqa: BLE001
                            n, names = -1, [repr(e)]
                        kernel_lists[(batch, q_len, kv_len, v.name)] = (n, names)
                        for r in results:
                            key = (r["batch"], r["q_len"], r["kv_len"], r["variant"])
                            if key == (batch, q_len, kv_len, v.name):
                                r["kernels"] = n
                del layer
                torch.accelerator.empty_cache()

    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(
        f"step_delta_ms = delta_us x {layers} layers; negative is a saving. Shares "
        "of the real decode step need e2e_glm47_flash.sh (nsys)."
    )
    if args.count_kernels:
        for (batch, q_len, kv_len, name), (n, names) in kernel_lists.items():
            if name in ("base", "combined"):
                print(f"\n{name} b={batch} q_len={q_len} kv={kv_len}: {n} GPU ops")
                for k in names:
                    print(f"  {k[:120]}")
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
