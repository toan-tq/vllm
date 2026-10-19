# Review index: vLLM performance work on the fork `toan-tq/vllm`, 2026-10-18/19

> Entry point for a reviewer or a later session. Everything referenced here is
> pushed to `github.com/toan-tq/vllm`. No GPU was available while any of it was
> written; every number is an estimate and is labelled as such in the documents.

**Tóm tắt (vi):** Hai nhánh code (A: weight fp32 cho kernel RMSNorm+quant, B: GELU-tanh

+ FP8 quant cho Gemma) đã qua ba vòng review bên ngoài; vòng ba chứng minh A gần như
không giảm launch ở đường compile mặc định. Giải pháp đúng cho bài GemmaRMSNorm là
A2 (`weight_bias` trong kernel, idiom upstream), trần 1 đến 3%. Khảo sát hai tháng
commit cho thấy chỗ đáng làm hơn trên RTX 5090 là **quyết định backend MoE mặc định
bằng số đo thật** (Humming vs Marlin, D1) và **config Triton MoE cho 5090** (D2).
Thứ tự đọc: mục 2 của file này, rồi `SURVEY-sm120-perf-gaps-2026-10-19.md`, rồi
handoff của từng nhánh.

## 1. Branches

| Branch | Head | Base | Content |
| --- | --- | --- | --- |
| `survey/sm120-perf-gaps-2026-10-19` | `83c04f47fe` + this file | `upstream/main` `689effee6d` (2026-10-18) | `SURVEY-sm120-perf-gaps-2026-10-19.md` (ranked directions D1-D5 with protocols), `ANALYSIS-index-2026-10-19.md` (this file) |
| `feat/gemma-rmsnorm-quant-fusion` (A) | feature `01488492fe`, handoff `5b3236425d` | `371745ebd` (2026-10-18) | fp32 norm weight in RMSNorm(+quant) kernels, narrowed pass guard, tests, benchmark gate; `HANDOFF-feat-gemma-rmsnorm-quant-fusion.md` sections 1-12 |
| `feat/gelu-tanh-act-quant-fusion` (B) | feature `0cc6e37d4d`, handoff `83597595ac` | `371745ebd` | GELU-tanh+Mul fused with FP8 static and per-block quant, pass patterns, Gemma MLP wiring; `HANDOFF-feat-gelu-tanh-act-quant-fusion.md` sections 1-12 |

Commit hygiene on A and B after review round 2: author and committer
`Toan Tran <toan-tq@outlook.com>`, DCO `Signed-off-by`, one `Co-authored-by` AI
trailer, no session links. The handoff commit sits on top of each feature commit
and must be dropped (`git reset --hard HEAD~1`) before any upstream PR. Upstream PR
heads fetched for comparison in the authoring checkout: `pr-45770`, `pr-59467`,
`pr-36016` (not pushed).

## 2. Reading order for a reviewer

1. This file, section 3 (decision log) and section 4 (A2 design).
2. `SURVEY-sm120-perf-gaps-2026-10-19.md` on this branch: what landed for SM12x in
   two months, what a single 5090 actually runs, directions D1-D5 with commands and
   decision gates.
3. `HANDOFF-feat-gemma-rmsnorm-quant-fusion.md` section 12 (corrected launch
   accounting and upstream precedent), then sections 4-6 for reach and the e2e
   plan, section 10 for the `pr-checklist` notes and a draft PR description.
4. `HANDOFF-feat-gelu-tanh-act-quant-fusion.md` sections 4, 5, 11, 12.

## 3. Decision log

### 3.1 What was built and why

+ Survey of 2026-10-18 found two gaps in the compile-time fusions: GemmaRMSNorm
  graphs (fp32 `1 + w` weight on bf16 activations) were excluded from the
  RMSNorm+quant pass by a dtype guard (#44694 after the #44611 crash), and GeGLU
  (Gemma 2/3/4) had no fused activation+FP8-quant kernels while SwiGLU did.
+ Branch A made the kernels take an fp32 weight (template on `weight_t`) and
  narrowed the guard. Branch B added GELU-tanh variants of the SiLU kernels and
  patterns, plus the manual `maybe_fused_act_quant` producer for the static case.

### 3.2 External review rounds and what changed

+ Round 1 (C1-C8, U1-U23): fixed. Guard narrowed instead of removed, alignment
  made explicit, benchmark identity gate, evaluation order of the GELU polynomial,
  import guards for XPU, defunctionalization handlers, docs corrected.
+ Round 2 (G-1..G-3, A-N1..A-N5, B-N1..B-N7): fixed. Commit history rewritten;
  per-token kernel falls back to its scalar path for a misaligned weight instead
  of raising; pass-level "unsupported dtype mix skips fusion" test; #45770
  description corrected (it covers per-token and per-block by casting the weight
  in the pass); dead per-block manual producer removed from B; kernel renamed
  `act_and_mul_per_block_quant_kernel`; B's overlap with #59467 measured
  (`git merge-tree` clean); `pr-checklist` run on both.
+ Round 3 (prediction document, 2026-10-19): verified in code and accepted. In the
  compiled default path inductor already inlines `weight.float() + 1.0` into the
  norm's Triton kernel; branch A's pattern leaves that expression outside the
  fused op, so it becomes a separate Triton kernel. Launches per norm: baseline
  2 (one Triton norm kernel, one quant kernel; inductor fuses the reduction with
  the row-wise epilogue even past the 1024-element persistent threshold), branch A
  1.5-2. **Net 0 to 1 launch saved; predicted TPOT change in the default config is
  within noise.** The earlier 1-2% estimate in handoff A section 11 was wrong and
  is marked superseded. Branch B's accounting holds (one launch saved per layer)
  but its reach is narrow (block-FP8 Gemma dense on hardware where the fp32-scale
  block pattern matches; not B200 with DeepGEMM UE8M0).

### 3.3 Verdicts

+ **A as designed:** a correctness/enabler change (upstream TODO and stopgap guard
  removed, eager path on `vllm_c`), not a measurable default-config speedup. Do
  not open as a performance PR. Its eager-path numerics change (residual rounded
  to the activation dtype before normalizing) needs parity if it is ever opened.
+ **B:** at most ~1% on a narrow checkpoint set; park unless a fused act+per-token
  FP8 kernel lands (#46273), which would widen the reach to the `-FP8-dynamic`
  checkpoints.
+ **Better design for the GemmaRMSNorm problem:** A2 (section 4), ceiling 1-3%.
+ **Better directions for the 5090:** D1 and D2 in the survey (section 5 below).

## 4. A2: GemmaRMSNorm + quant fusion through `weight_bias` (design, not implemented)

**Idea.** Keep the weight in the model dtype and let the fused norm+quant kernels
apply `float(w) + weight_bias` in fp32, exactly as the upstream all-reduce fusion
already does for Gemma (`allreduce_rms_fusion.py` matches
`vllm.ir.ops.rms_norm(x, weight.float() + 1.0, eps)` and passes the raw weight with
`weight_bias=1.0`; `csrc/fused_qknorm_rope_cache_kernel.cu` takes
`norm_weight_bias`). The `(1 + w)` expression disappears from the graph instead of
becoming a kernel.

**Evidence that this is the accepted idiom.**

+ #29810 (xyang16, Dec 2025 - Apr 2026, closed by the stale bot, never rejected):
  added `weight_bias` to the plain rms_norm kernels; hmellor asked whether it gives
  Gemma access to the quant fusions, answer yes; the author measured ~2 us for the
  custom kernel vs ~6 us for the torch.compile Triton norm; xenova insisted the
  bias be added in fp32.
+ #39014 (ZJY0516, merged Apr 2026): ProExpertProg wrote "I'd want to avoid an
  extra triton kernel to convert to 32bit", which is exactly what branch A
  introduces; the same PR recorded an accuracy failure (GGUF test) when `1 + w`
  was rounded to bf16, which A2 avoids by adding in fp32 inside the kernel.
+ #36016 (jackzhxng, draft since Mar 2026, `needs-rebase`, no maintainer review):
  same idea with six duplicated `gemma_*` ops; heavier than a parameter.
+ RFC #43224 (mgoin): compiler fusions are to be ported to manual fusions
  (RMSNorm+quant is #43500); a kernel-level `weight_bias` serves both the current
  pass and that future, fp32-weight templating serves neither.

**Changes.**

| Layer | Change | Size |
| --- | --- | --- |
| csrc | `float weight_bias = 0` on `rms_norm_static_fp8_quant`, `fused_add_rms_norm_static_fp8_quant`, `rms_norm_dynamic_per_token_quant`, `rms_norm_per_block_quant`; compute `static_cast<scalar_t>(x_norm * (float(w) + bias))` when bias != 0, keep the exact existing expression when bias == 0 (uniform branch, so same-dtype numerics stay bit-identical and no new instantiations) | ~60-80 lines |
| schemas | trailing `float weight_bias=0.0` on the four ops; `fix_functionalization.py` handlers already use the kwargs path for these ops | ~15 lines |
| pass | optional `weight_bias` parameter (float, default none) on the existing pattern classes; the Gemma variant's pattern function is `vllm.ir.ops.rms_norm(x, weight.float() + 1.0, eps)` (and the fused-add form), replacement passes `weight` and `weight_bias=1.0`; strict dtype guard kept for the plain patterns; Gemma guard checks the raw weight dtype equals the activation dtype | ~60 lines |
| tests | pass-level test with `GemmaRMSNorm` expecting `matched_count == 3` for static, per-token and per-block; kernel test vs the native reference with `weight_bias=1.0` | ~60 lines |
| not touched | `GemmaRMSNorm`, the IR op signature, the eager path (a follow-up could add `weight_bias` to `ir.ops.rms_norm` instead) | |

**Expected effect.** Per fused norm: the Triton norm kernel (3-6 us) and the quant
kernel (2-3 us) become one CUDA kernel (2-3 us). Qwen3.5-27B-FP8 has 80 fusable
norms of 129 (the 48 GDN input norms have two consumers: an FP8 quant for
`in_proj_qkvz` and a BF16 GEMM for `in_proj_ba`, which block-FP8 cannot quantize),
so ~0.25-0.5 ms per ~20 ms decode step: **1-2.5% TPOT**; 2-4% on 9B-class models
that fit one 5090. Below a 10% bar. Coordinate with the authors of #36016 and of
the #43500 producer before writing it.

## 5. Two-month survey (2026-08-10 to 2026-10-18), condensed

Full text with commands in `SURVEY-sm120-perf-gaps-2026-10-19.md`.

+ The SM12x lane is actively filled by others: b12x (linear/MoE/attention,
  opt-in), Humming (default dependency, preferred over Marlin on SM90 only since
  #56997), CUTLASS SM120 blockwise FP8, FlashInfer GDN prefill and NVFP4 KV cache
  on SM12x, XQA decode, Triton split-K retunes, OAI Triton MXFP4 on SM12x (behind
  Marlin: -10% prefill on a 5090, #58877), DeepGEMM auto-disabled per model on
  SM12x (#61274).
+ **D1 (first):** measure Humming vs Marlin vs OAI Triton for gpt-oss-20b (MXFP4
  with expert biases) and an NVFP4 MoE checkpoint on the 5090. Only GB10 data
  exists (parity within ±8%, bandwidth-starved device); the 5090 has no numbers.
  If Humming wins prefill and ties decode with GSM8K parity, the PR is one line
  (`_HUMMING_PREFERRED_CAPABILITIES` gains `120`) with #56997 as precedent.
+ **D2:** Triton fused-MoE tuned configs for `NVIDIA_GeForce_RTX_5090` (none exist;
  the RTX PRO 6000 files do not apply by name). Matters when Triton is the expert
  backend: block-FP8 MoE at TP=2 on 2x5090 and on RTX PRO 6000 Workstation.
  Precedents +7% (#53819) and +21% (#54668), merged within days.
+ **D3:** A2 above. **D4:** attention default on SM120 (FA2 vs FlashInfer vs b12x),
  measurement only. **D5:** Mamba2 `replayssm` launch table keyed on SM100 only.
+ Set aside with reasons: merging the GDN projections (four PRs, maintainer prefers
  multi-stream), quant dedup in GDN layers (no duplicate exists), b12x or OAI
  Triton as default (data against), the norm/activation fusions (1-3% class).

## 6. Still unverified

+ No GPU run of any kind: tests, parity, benchmarks, profiles. The prediction
  tables in handoff A section 12 and handoff B section 12 are what to compare
  against first.
+ D1/D2 outcomes depend entirely on 5090 measurements; the survey gives the
  commands, identity gates and decision thresholds.

## 7. Quick reproduction of the key claims

```bash
git fetch upstream main pull/45770/head:pr-45770 pull/59467/head:pr-59467 pull/36016/head:pr-36016
git merge-tree --write-tree pr-59467 origin/feat/gelu-tanh-act-quant-fusion~1          # clean
git merge-tree --write-tree --merge-base=pr-45770^2 pr-45770 origin/feat/gemma-rmsnorm-quant-fusion~1   # 4 conflicts
sed -n 100,135p vllm/compilation/passes/pass_manager.py        # fusion passes -> cleanup -> IR lowering
sed -n 505,600p vllm/compilation/passes/fusion/allreduce_rms_fusion.py   # weight.float() + 1.0 / weight_bias=1.0
sed -n 1,30p vllm/model_executor/layers/quantization/utils/humming/priority.py   # {90}
ls vllm/model_executor/layers/fused_moe/configs | grep -ci 5090                   # 0
```
