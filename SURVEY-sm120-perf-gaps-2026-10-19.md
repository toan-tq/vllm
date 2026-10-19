# Survey: performance gaps for consumer Blackwell (SM120, RTX 5090) in vLLM, 2026-10-19

> Written without a GPU, from `upstream/main` @ `689effee6d` with history deepened to
> 2026-08-10 (659 commits touching csrc, compilation passes, quantization, kernels,
> fused_moe, attention, platform code). Every claim below points at a commit, a PR
> or a file. Nothing has been measured; the measurements are the plan.

**Tóm tắt (vi):** Hai tháng qua làn SM12x được NVIDIA và cộng đồng lấp rất nhanh
(b12x, Humming, CUTLASS blockwise FP8 cho SM120, FlashInfer GDN/NVFP4-KV/XQA trên
SM12x, Triton split-K cho SM120). Kẽ hở còn lại không nằm ở "thiếu kernel" mà ở
**chọn backend mặc định cho consumer Blackwell chưa có số đo trên RTX 5090** và ở
**config Triton MoE chưa tune cho card này**. Các fusion kiểu RMSNorm+quant (nhánh A)
hay GELU+quant (nhánh B) chỉ ở mức 1 đến 3% và không phải chỗ đáng bỏ công nhất.
Hướng được xếp đầu: đo Humming / Marlin / OAI-Triton cho MoE MXFP4 (gpt-oss) và NVFP4
(Qwen3.6-35B-A3B) trên 5090; nếu Humming thắng rõ ở prefill và hòa ở decode, PR là một
dòng (thêm `120` vào `_HUMMING_PREFERRED_CAPABILITIES`) với tiền lệ #56997.

## 1. What landed for SM12x in the last two months (why the lane is crowded)

| Commit | Date | What |
| --- | --- | --- |
| `3c79b1a8bf` #52016, `2740c817ff` #52018, `0d4ad47981` #52017 | Aug 13-Sep 1 | b12x linear, FP4 MoE and paged-attention backends for SM120/121 (opt-in, `b12x==1.5.1` extra) |
| `7d99c2c4fd` #56685, `d1f3d8b870` #56997 | Sep 20, Oct 7 | Humming kernels integrated (default dependency `humming-kernels==0.1.17`); preferred over Marlin **on SM90 only** |
| `4df80187b5` #55180, `2b7fcbf527` #52775 | Aug-Sep | CUTLASS SM12x blockwise FP8 GEMM tuning (raster swizzle, small-M config) |
| `f6326f53bd` #55715 | Sep 8 | FlashInfer GDN prefill kernel enabled on SM12x (`head_k_dim == 128`, CUDA 13) |
| `13cf9e05c1` #55170 | Sep 8 | NVFP4: prefer W4A4 linear kernels over weight-only on SM120/121 |
| `41c4a3ed4e` #57456, `4e0a414c44` #58495 | Sep 19-30 | sm_120 tuned configs for the batch-invariant persistent matmul |
| `edaf3e7b61` #58482, `84f738f187` #56151 | Sep 30 | Triton attention split-K retuned for SM120 |
| `554340f3d3` #46963, `9bd074a996` #60452 | Oct 8-10 | NVFP4 KV cache on SM12x with FlashInfer; XQA decode of NVFP4 KV on SM12x |
| `40c86ae1ce` #58877 | Oct 6 | OAI Triton MXFP4 MoE runs on SM12x; kept behind Marlin (prefill 10% slower on a 5090, 99 KB smem) |
| `2d09fdcdc2` #61274 | Oct 14 | DeepGEMM auto-disabled per model on SM12x (`_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES`: Qwen3.5 dense and MoE on 100/120 for accuracy, Kolibri 1 on 120 for latency) |

Implication: a kernel-porting gap on SM120 is likely already owned by someone
(lukealonso's b12x, the Humming authors, NVIDIA's FlashInfer enablement). What is
not owned is *deciding defaults with 5090 data* and *tuning tables for the 5090*.

## 2. What a single RTX 5090 (32 GB) actually runs

| Model | Weights | MoE kernel on SM120 today | Triton-config dependent? |
| --- | --- | --- | --- |
| `openai/gpt-oss-20b` (MXFP4, expert biases) | ~13 GB | Marlin (W4A16); Humming installed but ranked after Marlin off SM90; OAI Triton opt-in; b12x rejects expert biases | no |
| `nvidia/Qwen3.6-35B-A3B-NVFP4`, Qwen3.5-35B-A3B-NVFP4 | ~19 GB | CUTLASS FP4 experts / FlashInfer (SM120 supported); b12x opt-in; Marlin fallback | no |
| Qwen3-30B-A3B / Qwen3.6-35B-A3B AWQ-4bit | ~17-19 GB | Marlin MoE | no |
| Qwen3.5-27B-NVFP4, Gemma-3-27B NVFP4/AWQ | ~15 GB | dense; CUTLASS SM120 NVFP4 / Marlin | no |
| Qwen3.5-35B-A3B-FP8, Qwen3-30B-A3B-FP8, Qwen3.5-27B-FP8 (block-128) | 27-35 GB | do not fit one 32 GB card with a KV cache; at TP=2 (2x5090) the MoE runs on **Triton `fused_moe`** (DeepGEMM auto-disabled for Qwen3.5 on SM120, CUTLASS FP8 grouped GEMM is SM90-only) | **yes, and no RTX 5090 config exists** |

So the common single-card cases are MXFP4 and NVFP4 with Marlin / CUTLASS /
Humming / b12x, where the open question is *which backend should be the default*;
the Triton FP8 MoE config gap matters for 2x5090 and RTX PRO 6000 Workstation users.

## 3. Ranked directions

### D1. MoE backend default on SM120: measure Humming vs Marlin (vs OAI Triton) on a 5090

- **Mechanism.** Humming on SM12x runs MXFP4 weights with MXFP8 activations through
  the native block-scaled tensor-core path (`quantization/utils/humming/schema.py`:
  `sm_version >= 120 and is_mx_weight` selects group-32 e8m0 dynamic input quant).
  Marlin is W4A16: dequantize to BF16, then BF16 MMA. At decode both are
  bandwidth-bound and should tie; at prefill the FP4 MMA path should win, which is
  the same argument that made Humming the default on SM90 (#56997: "Humming supports
  WGMMA on SM90 and delivers significantly better performance than Marlin").
- **Why it is open.** `_HUMMING_PREFERRED_CAPABILITIES = frozenset({90})`
  (`quantization/utils/humming/priority.py`). In #56997 the only non-SM90 data was
  carmenlok's GB10 (SM121, DGX Spark, ~273 GB/s): Qwen3-30B-A3B MXFP8 and NVFP4,
  "parity within about ±8%, no consistent winner", GSM8K 0.867-0.892 across
  backends; an RTX PRO 6000 rerun was offered and never posted. GB10 is so
  bandwidth-starved that compute differences cannot show; the 5090 (1.79 TB/s,
  170 SMs) is a different regime and has **no posted numbers**.
- **Also in play.** OAI Triton MXFP4 on SM12x: decode equal to Marlin, prefill 10%
  slower (#58877, measured on a 5090). b12x auto-selection (#54669, open): on an
  RTX PRO 6000, B12X decodes 11.8% faster at c=1 but TTFT +55% at 1K input and
  GSM8K 0.945 vs Marlin 0.970 (waynehacking8, Oct 17), so b12x is not the answer
  for the default either; and b12x MoE rejects expert biases (gpt-oss).
- **Expected impact if Humming wins.** Prefill-heavy and mixed workloads of the
  most-run model on 5090s (gpt-oss-20b) and NVFP4 MoE checkpoints: tens of percent
  of TTFT is plausible, decode parity. If it ties (as on GB10), the result is still
  a useful negative that closes the question for maintainers.
- **Deliverable.** One-line PR adding `120` to `_HUMMING_PREFERRED_CAPABILITIES`
  (or a finer rule if only MXFP4 wins), with the benchmark table and GSM8K parity.
  Mergeability is high: #56997 is the exact precedent and mgoin asked for per-arch
  data there.
- **Protocol (5090, same checkpoint, same engine flags, median of 3 runs each):**

  ```bash
  # gpt-oss-20b, MXFP4 with expert biases
  for b in marlin humming triton; do
    vllm serve openai/gpt-oss-20b --moe-backend $b --max-model-len 8192 --port 8000 &
    # wait for /health
    for c in 1 8 32; do
      vllm bench serve --model openai/gpt-oss-20b --dataset-name random \
        --random-input-len 1024 --random-output-len 256 --num-prompts $((c*8)) \
        --max-concurrency $c --save-result --result-dir results/gpt-oss-20b-$b
    done
    # prefill-only view: --random-output-len 1 with --random-input-len 4096
    python tests/evals/gsm8k/gsm8k_eval.py --port 8000 --num-questions 500
    kill %1; wait
  done
  # NVFP4 MoE: repeat with nvidia/Qwen3.6-35B-A3B-NVFP4 (auto vs humming vs marlin)
  ```

  Identity gate: the startup log must print the selected MoE backend
  (`Using ... backend` from the oracle); confirm it per run before reading numbers.
  Report TTFT, TPOT, output tok/s per concurrency, and GSM8K (strict and flexible).
- **Decision gate.** Open the PR only if Humming is within noise at decode for all
  concurrencies, better by >= 10% at prefill/TTFT, and GSM8K is within ±1 point.
  If Marlin wins or ties, post the table on #56997 instead (closes the open
  question; no code).

### D2. Triton fused-MoE tuned configs for RTX 5090 (and RTX PRO 6000 Workstation)

- **Gap.** `vllm/model_executor/layers/fused_moe/configs/` has no
  `device_name=NVIDIA_GeForce_RTX_5090` file at all; the lookup is by exact device
  name (`get_config_file_name`), so the RTX PRO 6000 Server configs do not apply.
  Without a file, `get_default_config` is used, whose block-FP8 branch was tuned on
  Hopper ("the swap-AB kernel keeps it efficient on Hopper") and logs "Using default
  MoE config. Performance might be sub-optimal!".
- **When it matters.** Whenever the Triton kernel is the expert backend: block-FP8
  MoE on SM120 (DeepGEMM auto-disabled for Qwen3.5 MoE, no CUTLASS block-FP8 grouped
  GEMM on SM120), i.e. 2x5090 TP=2 for Qwen3.5/3.6-35B-A3B-FP8 (`E=256,N=256`) and
  Qwen3-30B-A3B-FP8 (`E=128,N=384`); RTX PRO 6000 Workstation (96 GB) at TP=1
  (`E=256,N=512`, `E=128,N=768`), which has no block-FP8 configs either (only the
  Server Edition has three).
- **Precedent and size of gain.** `12579c7f01` #54668 "Tune H20 block-FP8 MoE
  low-batch configs (+21%)", `044b05220e` #53819 "Tune fused_moe FP8 config for
  Qwen3.5 on L40S (+7%)", `7ea4b40954` #52502 GB10 configs for E=256/E=512,
  `f2e2936f91` #55511 A100 E=256,N=512: all merged within days, JSON only.
- **Protocol.**

  ```bash
  # one file per (model, tp); the script derives E, N, block shape from the HF config
  python benchmarks/kernels/benchmark_moe.py --model Qwen/Qwen3.5-35B-A3B-FP8 \
      --dtype fp8_w8a8 --tp-size 2 --tune --trust-remote-code --save-dir tuned/
  python benchmarks/kernels/benchmark_moe.py --model Qwen/Qwen3-30B-A3B-FP8 \
      --dtype fp8_w8a8 --tp-size 2 --tune --save-dir tuned/
  # validate before copying into configs/:
  VLLM_TUNED_CONFIG_FOLDER=$PWD/tuned vllm bench throughput --model ... -tp 2 ...
  ```

  Then copy to `vllm/model_executor/layers/fused_moe/configs/` and report the
  kernel-level table from the tuner plus e2e throughput at batch 1/8/32 against the
  default config. The tuner supports `Qwen3_5MoeForConditionalGeneration`,
  `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`, `NemotronHForCausalLM` and others
  (`get_model_params`).
- **Reach caveat.** Single-card 5090 users rarely hit this path (section 2). It is a
  sure, small, mergeable contribution, not a headline.

### D3. GemmaRMSNorm + quant fusion via `weight_bias` (the redesign of branch A)

Documented in `HANDOFF-feat-gemma-rmsnorm-quant-fusion.md` section 12 on
`feat/gemma-rmsnorm-quant-fusion`. Expected 1-3% TPOT on block-FP8 Qwen3.5 dense
models that need 2 GPUs or are small; only worth doing once an NVFP4 fused norm for
GemmaRMSNorm is wanted on Blackwell. Not first.

### D4. Attention backend default on SM120 (measurement only)

`platforms/cuda.py` orders `FLASH_ATTN` (FA2 on SM120, `fa_utils.py`: FA3 is SM90,
FA4 is SM100) before `FLASHINFER` for every non-SM100 GPU; b12x attention is
opt-in. Nobody has posted a 5090 comparison for long-context Gemma 3 / Qwen3.5
(sliding window, GQA). Worth one afternoon of `vllm bench serve` at 8k-32k input
across the three backends; a default change needs the same kind of table as D1.

### D5. Mamba2 `replayssm` launch table for SM120

`mamba/ops/replayssm_config.py` keys its tuned launch parameters on
`is_device_capability_family(100)`; SM120 silently gets the Hopper row. Relevant to
Mamba2 hybrids that fit a 5090 in NVFP4 (Nemotron-3-Nano class). Gain unknown and
bounded by the SSM kernel's share of a decode step; a retune is a one-row change.

## 4. Checked and set aside

- **Merging the Qwen GDN `in_proj_qkvz` and `in_proj_ba` GEMMs.** Four PRs already
  (#34492 merged the qkv|z and b|a halves; #41457 open since May; #42746 and #58801
  closed); maintainer benchislett prefers multi-stream (#56534). `in_proj_ba`
  cannot be block-FP8 quantized, which is also why the 48 GDN input norms stay
  unfused (two consumers: an FP8 quant and a BF16 GEMM).
- **Deduplicating activation quant in GDN layers.** There is no duplicate: the
  second consumer is a BF16 GEMM.
- **b12x as default** (#54669): mixed data on RTX PRO 6000, accuracy gap, no expert
  biases.
- **OAI Triton MXFP4 as default on SM12x** (#58877): slower prefill on a 5090.
- **Norm / activation fusions** (branches A and B): 1-3% class, see their handoffs.

## 5. Order of work

1. D1 on the 5090 (one day of runs). Outcome either a one-line PR with a strong
   table, or a data comment on #56997.
2. D2 in the background while D1 models are loaded (tuner runs take tens of
   minutes per shape); PR with JSON files and the before/after table.
3. D4 if D1 shows the 5090 behaves differently from GB10 in general.
4. D3/D5 only on demand.

## 6. Reproduction of this survey

```bash
git fetch --shallow-since=2026-08-10 upstream main
git log --since=2026-08-15 --format='%h %ad %s' --date=short upstream/main | grep -i "sm12\|5090\|blackwell"
ls vllm/model_executor/layers/fused_moe/configs | grep -i "5090\|RTX"
sed -n 1,30p vllm/model_executor/layers/quantization/utils/humming/priority.py
sed -n 395,440p vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
sed -n 18,45p vllm/utils/deep_gemm.py
gh pr view 56997 --repo vllm-project/vllm --comments   # GB10 parity table
gh pr view 58877 --repo vllm-project/vllm              # OAI Triton on 5090: -10% prefill
gh pr view 54669 --repo vllm-project/vllm --comments   # b12x vs Marlin on RTX PRO 6000
```
