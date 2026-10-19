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
**Đính chính sau review vòng 4 (19/10, mục 7):** D1 theo mặc định chỉ so Humming W4A16
với Marlin W4A16 (Humming dùng activation BF16 trên SM120, và SM120 không có WGMMA là
lợi thế của #56997 trên SM90), nên kỳ vọng là hòa và sản phẩm là một bảng dữ liệu đăng
lên #56997, không phải PR. D2 trùng #45379 (đúng shape, mở từ 12/6, đo được 1 đến 5%).
Không hướng nào trong D1 đến D5 vượt bar trên 10%; không thuê GPU riêng cho chúng.
**Hướng E (mục 8, tìm sau vòng 4, hạ cấp sau vòng 5):** gpt-oss trên SM120 chạy MoE
bằng Marlin W4A16 trong khi đường FlashInfer CUTLASS MXFP4 × MXFP8 đã có trong cây và
chỉ không được chọn mặc định vì bộ lọc activation BF16. Vòng 5 chỉ ra bộ lọc đó là
chính sách mgoin vừa ghi trong PR 60079 (merge 08/10): "Activations are never
quantized below what the checkpoint declares"; số đo MoE W4A4 thật trên SM120 trong
PR 51158 (mở từ 05/08) chỉ cho TTFT nhanh 1.14 lần ở c=1 và decode chậm hơn; và
PR 47577 (cùng dạng đổi mặc định trên SM120, TTFT giảm 25% ở 8K) bị chính tác giả
đóng ngày 17/10 vì decode chậm 3.5% và không ai review trong ba tháng rưỡi. Kết luận:
không thuê GPU cho E; nếu sẵn 5090, chạy 30 phút arm opt-in để xem đường này có chạy
đúng với expert bias không (nếu hỏng là bug report), không làm PR đổi mặc định.

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

> Corrected after review round 4 (section 7). The first version of this section
> claimed Humming would use MXFP8 activations on SM12x by default; it does not.

- **Mechanism (corrected).** By default Humming runs MXFP4 MoE with **BF16
  activations**: `_backend_activation_key(HUMMING)` returns `None`
  (`fused_moe/oracle/mxfp4.py`), and the Humming branch of that oracle only takes
  the checkpoint's activation scheme when the checkpoint declares one (gpt-oss does
  not) or `VLLM_HUMMING_INPUT_QUANT_CONFIG` is set ("Humming's default BF16
  activations", lines ~1637-1650). So adding `120` to
  `_HUMMING_PREFERRED_CAPABILITIES` compares Humming W4A16 with Marlin W4A16: both
  dequantize to BF16 and run `mma.sync`, because SM120 has no WGMMA, which is the
  SM90 advantage #56997 relied on ("Humming supports WGMMA on SM90"). The GB10
  parity result is most likely explained by this, not only by bandwidth.
- **The only arm with a tensor-core mechanism** is Humming with MXFP8 activations
  forced through `VLLM_HUMMING_INPUT_QUANT_CONFIG` (group-32 e8m0 dynamic input
  quant; `quantization/utils/humming/schema.py`, `sm_version >= 120 and
  is_mx_weight`). That changes numerics (activations quantized to MXFP8), so it is a
  default-input-schema decision with GSM8K evidence, not a priority flip, and the
  Humming author (jinzhen-lin) said in #56997 they would look into the GB10 result
  themselves.
- **Why it is open.** `_HUMMING_PREFERRED_CAPABILITIES = frozenset({90})`
  (`quantization/utils/humming/priority.py`). In #56997 the only non-SM90 data was
  carmenlok's GB10 (SM121, DGX Spark, ~273 GB/s): Qwen3-30B-A3B MXFP8 and NVFP4,
  "parity within about ±8%, no consistent winner", GSM8K 0.867-0.892 across
  backends; an RTX PRO 6000 rerun was offered and never posted. mgoin's only
  comments there were CI triggers; nobody asked for per-arch data. The 5090 has
  **no posted numbers**, but given the mechanism above the expected result is
  parity.
- **Also in play.** OAI Triton MXFP4 on SM12x: decode equal to Marlin, prefill 10%
  slower (#58877, measured on a 5090). b12x auto-selection (#54669, open): on an
  RTX PRO 6000, B12X decodes 11.8% faster at c=1 but TTFT +55% at 1K input and
  GSM8K 0.945 vs Marlin 0.970 (waynehacking8, Oct 17), so b12x is not the answer
  for the default either; and b12x MoE rejects expert biases (gpt-oss).
- **Expected outcome (review round 4 prediction, before any run).** gpt-oss-20b on
  a 5090, Humming W4A16 vs Marlin: decode within ±5%, prefill within ±10%, no
  stable winner (P ~0.7). P(Humming >= 10% faster at prefill and tied at decode)
  ~0.15. The MXFP8-activation arm is the only one that can move the number, at an
  accuracy cost to be measured.
- **Deliverable.** Most likely a data comment on #56997 (three arms, GSM8K), which
  closes an open question for the maintainers. A PR only if the MXFP8 arm wins
  clearly with GSM8K parity, and then it is a default-input-schema change to
  discuss with the Humming author first, not a one-liner.
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
  # third arm: Humming with MXFP8 activations (group-32 e8m0 dynamic); schema fields in
  # vllm/model_executor/layers/quantization/utils/humming/schema.py, env in vllm/envs.py
  VLLM_HUMMING_INPUT_QUANT_CONFIG='{...}' vllm serve openai/gpt-oss-20b --moe-backend humming ...
  # NVFP4 MoE: repeat with nvidia/Qwen3.6-35B-A3B-NVFP4 (auto vs humming vs marlin)
  ```

  Identity gate: the startup log must print the selected MoE backend and, for
  Humming, the resolved input schema (BF16 vs MXFP8); confirm per run before
  reading numbers. Report TTFT, TPOT, output tok/s per concurrency, and GSM8K
  (strict and flexible).
- **Decision gate.** Post the table on #56997 in every case. Consider a PR only if
  an arm is within noise at decode for all concurrencies, better by >= 10% at
  prefill/TTFT, and GSM8K is within ±1 point; for the MXFP8 arm, agree the default
  with the Humming author first. Do not rent a GPU for this; run it when a 5090 is
  in hand for other work.

### D2. Triton fused-MoE tuned configs for RTX 5090 (and RTX PRO 6000 Workstation)

> Corrected after review round 4 (section 7): this is **already claimed by open
> PRs**; the first version of this section missed them (the duplicate check only
> looked at merged commits in `configs/`).

- **Duplicates.** #45379 (shpgy-shpgy, opened 2026-06-12, open, no review, no
  labels): `E=256,N=256,device_name=NVIDIA_GeForce_RTX_5090,dtype=fp8_w8a8,block_shape=[128,128]`,
  the exact Qwen3.5-35B-A3B-FP8 TP=2 shape this section ranked first; its own
  5090 numbers are kernel latency −1.8% to −5.4% for batch 1-512 and ~0 at
  2k-4k. #55360 (CharlesXu-HQ, 2026-09-04): `E=128,N=768` block-FP8 for the 5090.
  The review also lists #51519 (E=128, bf16), #39263 (four bf16 files) and the
  closed #55956; not checked here. The H20 and L40S precedents were merged within
  days, these 5090 PRs have had no maintainer attention for months: the blocker
  is attention, not missing files. The only unclaimed shape is Qwen3-30B-A3B-FP8
  at TP=2 (`E=128,N=384`), too small for a PR of its own.
- **Action instead of a new PR.** If a 2x5090 run happens anyway, post e2e numbers
  (default vs `VLLM_TUNED_CONFIG_FOLDER` with the PR's file) on #45379 and
  @-mention the maintainers who merged the H20/L40S configs (jeejeelee) and the
  requested reviewers (mgoin, pavanimajety, zyongye). Note that this needs two
  cards; a single 5090 does not reach this path (section 2).
- **Gap (as originally stated).** `vllm/model_executor/layers/fused_moe/configs/` has no
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
  `f2e2936f91` #55511 A100 E=256,N=512: all merged within days, JSON only. On the
  5090 itself the measured kernel gain in #45379 is only 1-5%, far from the H20
  figure, so the e2e effect is likely in the low single digits.
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

## 5. Order of work (revised after review round 4)

Nothing in this survey, E included after round 5, clears a "most likely > 10% and
high merge probability" bar; do not rent a GPU for any of it. Cheap, certain
actions when a 5090 is in hand for other reasons:

0. E, reduced to a 30-minute smoke test: start gpt-oss-20b with
   `--moe-backend flashinfer_cutlass_afp8`, check the log names the backend, run
   GSM8K on 100 questions and one `vllm bench serve` at 4K input. A crash or
   garbage output is a bug report (FlashInfer tests the dtype pair on SM120 but
   not the expert-bias path); a clean run is a recipe note, not a PR. Section 8
   has the round-5 reasoning.
1. D1 as a measurement (three arms, GSM8K), posted as data on #56997; the Humming
   MXFP8 arm is shared with the E smoke test.
2. 5090 e2e numbers for #45379's config posted on that PR with a maintainer ping
   (needs 2x5090), instead of a new config PR.
3. D4 only if those runs show the 5090 behaving unlike GB10 in general.
4. D3/D5 only on demand.

## 6. Reproduction of this survey (section 7 lists what later checks corrected)

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
gh pr view 45379 --repo vllm-project/vllm              # existing RTX 5090 E=256,N=256 config PR
grep -n "def _backend_activation_key" -A 20 vllm/model_executor/layers/fused_moe/oracle/mxfp4.py   # HUMMING -> None (BF16)
```

## 7. Review round 4 (2026-10-19): corrections to this survey

- D1 mechanism was wrong: Humming defaults to BF16 activations on SM120, so the
  proposed priority flip compares W4A16 with W4A16; SM120 has no WGMMA. Expected
  result parity; product is a data comment.
- D2 is a duplicate of #45379 (exact shape) and #55360; the blocker is maintainer
  attention. The 5090 gain measured there is 1-5%, not the H20 +21%.
- "mgoin asked for per-arch data" in #56997 was wrong; only CI triggers.
- Everything in sections 1 and 2 (commit list, DeepGEMM exclusion table,
  `replayssm` keying, model-fit table) was confirmed by the review.

## 8. Direction E (found after round 4): gpt-oss on SM120 has an FP4 tensor-core MoE path in the tree that is never selected

**Status after review round 5 (2026-10-19): downgraded from "one GPU day" to a
30-minute smoke test. No default-change PR.** The reviewer's six points, checked
here against `upstream/main` `689effee6d` and the GitHub pages:

1. Confirmed, and decisive. PR 60079 (mgoin, opened Oct 5, merged Oct 8, commit
   `30c7405890`) states the oracle contract in its description: "The
   `activation_key` passed to `select_mxfp4_moe_backend` is the activation
   precision the checkpoint declares (`None` = BF16). Backends that match it are
   preferred. If none supports the deployment, the oracle falls back to
   BF16-activation backends and logs a warning. Activations are never quantized
   below what the checkpoint declares." gpt-oss calls
   `select_mxfp4_moe_backend(moe)` with no key (`quantization/mxfp4.py:158`), so
   the BF16-only filter on SM120 is the policy, not an oversight. A PR preferring
   MXFP8 activations for a checkpoint that declares none asks the owner to
   reverse an eleven-day-old decision. (`select_deepseek_v4_mxfp4_moe_backend` in
   the same file does put TRT-LLM MXFP8 first, but that is a model-specific
   oracle for a model trained with FP8 activations, not a precedent for gpt-oss.)
2. Confirmed. PR 51158 (ayrnb, open since Aug 5, no maintainer review beyond one
   question from Harry-Chen) enables the in-tree CUTLASS MXFP4 W4A4 MoE on SM12x
   and measured it on an RTX PRO 5000 with Qwen3-30B-A3B-Instruct-2507 at 1K
   input: TTFT 1.14x at c=1, 1.15x at c=4, 1.23x at c=8, 1.65x at c=16, 1.08x at
   c=32; end to end at c=1 output tok/s 78.5 vs 87.7 (W4A4 slower by 10%) and TPOT
   12.3 vs 10.8 ms. W4A8 (the FlashInfer path in E) runs on FP8 tensor cores, so
   it has no reason to beat W4A4 on the same GEMMs.
3. Confirmed. `_get_requested_backends` runs only when `moe_backend != "auto"`
   (`oracle/mxfp4.py:675`), so the b12x "prefer W4A8" comment is the policy for an
   explicit `--moe-backend b12x`, not a precedent for auto-selection. The bullet
   below is corrected.
4. Half right. Issue 31085's title and later thread are about NVFP4 MoE; its
   original report is gpt-oss-120b (MXFP4) falling back to Marlin on SM120, which
   mratsim corrected to "gpt-oss is MXFP4, not NVFP4". The 185 vs 140 tok/s
   numbers there are gpt-oss on an RTX 6000 Max-Q (Marlin vs OAI Triton). It is
   not a clean home for gpt-oss MXFP4 data; #42118 or a new issue is.
5. Confirmed in scope: PR 55170 touches only `kernels/linear/__init__.py` (dense
   NVFP4 linear selection). Per-expert M for gpt-oss-20b (32 experts, top-4) at 4K
   input is about 512 rows, so the small-M argument is weaker than for
   Qwen3-30B-A3B (128 experts, top-8, about 64 rows per expert at 1K), but the
   only MoE data on SM120 is point 2, and it says 1.1-1.2x at the concurrencies a
   single-user 5090 runs.
6. Not checkable from this session (the reviewer's own lead list from Oct 15);
   taken as given.

Additional precedent found while checking: PR 47577 (waynehacking8) did exactly
the E-shaped change for NVFP4, auto-selecting `FLASHINFER_B12X` ahead of Marlin on
SM120 only. Opened Jul 3, "ready" label from mgoin, no review comment in three and
a half months, closed by the author on Oct 17 because "on current main the switch
is not a clear win": TTFT at 8K input down 25.2%, decode TPOT up 3.5% at c=1 and
2.3% at c=16. That is the realistic ceiling for E's data and its PR outcome.

Revised odds: P(path runs correctly with expert biases on a 5090) ~0.7; P(TTFT
>= 20% at 4-8K | runs) ~0.25; P(default-change PR merged | both) ~0.1 given points
1 and the 47577 outcome; overall about 0.02-0.05, matching the reviewer. The
sections below are kept as written after round 4 so the reasoning can be audited;
read them with the corrections above.

### The gap

- On SM120 the gpt-oss MoE runs on **Marlin W4A16** (dequantize to BF16, BF16
  `mma.sync`). Prefill is compute-bound there. The b12x MoE rejects expert biases,
  OAI Triton is 10% slower at prefill (#58877), the CUTLASS `mxfp4_experts_quant`
  path is SM10x/11x-only (`mxfp4_experts_quant_sm_supported` in
  `csrc/quantization/fp4/mxfp4_experts_quant.cu`: `100 <= cc < 120`).
- `FlashInferExperts` (`fused_moe/experts/flashinfer_cutlass_moe.py`) accepts
  `(kMxfp4Static, kMxfp8Dynamic)` on `has_device_capability(100)`, which includes
  12.0, lists family 120 as a supported device (with SM110 explicitly excluded),
  supports `SWIGLUOAI`, and passes `fc1_expert_biases`/`fc2_expert_biases` and the
  gpt-oss `swiglu_alpha=1.702, beta=1.0, limit=7.0`. The backend name is
  `FLASHINFER_CUTLASS_MXFP4_MXFP8`, CLI alias `--moe-backend flashinfer_cutlass_afp8`.
- It is never chosen automatically: `_get_priority_backends_for_gpt_oss` lists it
  after TRITON, and `_filter_by_activation` keeps only BF16-activation variants
  when the checkpoint requests no activation format (gpt-oss does not), so on SM120
  the list collapses to MARLIN. (Round 5 correction: the b12x "prefer W4A8" branch
  of `_get_requested_backends` only runs for an explicit `--moe-backend b12x`; it
  is not a precedent for changing auto-selection. The filter itself is the
  contract written in PR 60079, see the status block above.)
- FlashInfer side: the SM120 CUTLASS fused-MoE module is built with
  `-DCOMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS -DENABLE_FP8 -DENABLE_FP4` and
  compiles `moe_gemm_kernels_fp8_fp4.cu` (`flashinfer/jit/fused_moe.py`);
  FlashInfer's `tests/moe/test_trtllm_cutlass_fused_moe.py::test_moe_mxfp8_mxfp4`
  runs on major 10, 11 and 12 ("MXFP8xMXFP4 is only supported on SM100, SM110 and
  SM120/SM121") with the gpt-oss SwiGLU parameters, but without expert biases.
  `cutlass_fused_moe` in `flashinfer/fused_moe/core.py` dispatches "120"/"121" and
  has no SM gate on `use_mxfp8_act_scaling` or `use_w4_group_scaling`. A DGX Spark
  (SM121) user reported the path running under vLLM (NVIDIA forum, "Run VLLM in
  Spark"). vLLM issue #42118 documents it producing garbage **only under expert
  parallelism** (unswizzled MXFP8 scales after all-to-all); single-GPU is not
  affected.
- Demand: issue #31085 (open since 2025-12) opened with gpt-oss-120b MXFP4 falling
  back to Marlin on SM120, but its title and the rest of the thread are NVFP4 MoE
  (round 5 correction); the Marlin ~185 vs OAI Triton ~140 tok/s numbers there are
  gpt-oss on an RTX PRO 6000 Max-Q. Nobody has posted the CUTLASS MXFP4 x MXFP8
  result on a 5090; the nearest MoE data on SM120 is PR 51158 (status block).

### Why the gain can be large (and where it is not)

- Dense precedent on the same architecture: #55170 switched Qwen3.8-27B-NVFP4 on an
  RTX PRO 6000 from W4A16 to W4A4 CUTLASS: TTFT −36%, output +24% at BS1 with a 32K
  cached prefix and 2K input; the author also notes W4A4 "loses BS1 decode by a few
  percent". Expect the same shape here: a large prefill/TTFT win, decode parity or a
  small loss.
- Reference point for the 5090: LMSYS measured gpt-oss-20B MXFP4 under Ollama at
  ~8.5k prompt tok/s and 205 tok/s decode on an RTX 5090; a 1.3-1.5x prefill change
  is well above noise at that scale.
- Not a decode optimization: at M=1 both paths read the same bytes.

### Protocol (one 5090, gpt-oss-20b, median of 3)

```bash
for cfg in "marlin:--moe-backend marlin" "cutlass_afp8:--moe-backend flashinfer_cutlass_afp8" "triton:--moe-backend triton"; do
  name=${cfg%%:*}; flags=${cfg#*:}
  vllm serve openai/gpt-oss-20b $flags --max-model-len 16384 --port 8000 &
  # wait for /health; identity gate: the log must print the selected MoE backend
  for isl in 1024 4096 8192; do
    vllm bench serve --model openai/gpt-oss-20b --dataset-name random \
      --random-input-len $isl --random-output-len 128 --num-prompts 16 --max-concurrency 1 \
      --save-result --result-dir results/E-$name-isl$isl
  done
  vllm bench serve --model openai/gpt-oss-20b --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --num-prompts 64 --max-concurrency 8 \
    --save-result --result-dir results/E-$name-c8
  python tests/evals/gsm8k/gsm8k_eval.py --port 8000 --num-questions 500
  kill %1; wait
done
# fourth arm, shared with D1: Humming with MXFP8 activations
# VLLM_HUMMING_INPUT_QUANT_CONFIG='{...}' vllm serve openai/gpt-oss-20b --moe-backend humming ...
```

Report TTFT per input length, decode TPOT at c=1 and c=8, output tok/s, GSM8K
strict and flexible. If `flashinfer_cutlass_afp8` fails to start or produces
garbage on the 5090, that failure is itself a bug report worth filing (FlashInfer
tests cover the dtype pair on SM120 but not the expert-bias path).

### Decision gate and PR shape

- Gate: TTFT better by >= 20% at 4K-8K input, decode TPOT within ±5%, GSM8K within
  ±1 point of Marlin.
- PR if the gate passes: in `fused_moe/oracle/mxfp4.py`, prefer
  `FLASHINFER_CUTLASS_MXFP4_MXFP8` over MARLIN on `is_device_capability_family(120)`
  when the checkpoint requests no activation format, mirroring the b12x W4A8 policy,
  plus an oracle test and the benchmark table in the description. Size: ~20 lines.
  Expect a policy discussion (MXFP8 activations change numerics; on SM100 the
  default keeps a BF16-activation kernel first), so bring GSM8K and, if possible,
  a second eval. Fallback deliverable if maintainers keep it opt-in: the recipe
  (`docs`/model recipe) and a one-line startup hint on SM12x.
- Odds as written after round 4 (superseded by the status block at the top of
  this section): P(path runs correctly on a single 5090) ~0.7; P(>= 20% TTFT |
  runs) ~0.6; P(decode and GSM8K within gate | runs) ~0.6; roughly 0.3 for a
  default-change PR. Round 5 revised this to 0.02-0.05: the 0.6 for TTFT ignored
  PR 51158's measured 1.14-1.23x, and the PR odds ignored the PR 60079 contract
  and the fate of PR 47577. What survives is the smoke test in section 5 item 0.
