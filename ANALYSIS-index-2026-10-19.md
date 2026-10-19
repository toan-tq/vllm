# Review index: vLLM performance work on the fork `toan-tq/vllm`, 2026-10-18/19

> Entry point for a reviewer or a later session. Everything referenced here is
> pushed to `github.com/toan-tq/vllm`. No GPU was available while any of it was
> written; every number is an estimate and is labelled as such in the documents.

**Tóm tắt (vi):** Hai nhánh code (A: weight fp32 cho kernel RMSNorm+quant, B: GELU-tanh

+ FP8 quant cho Gemma) đã qua ba vòng review bên ngoài; vòng ba chứng minh A gần như
không giảm launch ở đường compile mặc định. Giải pháp đúng cho bài GemmaRMSNorm là
A2 (`weight_bias` trong kernel, idiom upstream), trần 1 đến 3%. Khảo sát hai tháng
commit xếp đầu hai hướng D1 (backend MoE mặc định, Humming vs Marlin) và D2 (config
Triton MoE cho 5090), nhưng **review vòng 4 (19/10) bác cả hai**: D1 mặc định chỉ so
W4A16 với W4A16 nên kỳ vọng hòa, D2 trùng #45379. Kết luận hiện tại: không hướng nào
vượt bar trên 10%; việc rẻ và chắc là đăng số đo lên #56997 và #45379 khi sẵn 5090.
Thứ tự đọc: mục 2 của file này, rồi `SURVEY-sm120-perf-gaps-2026-10-19.md` (mục 7
trước), rồi handoff của từng nhánh.

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
+ **Better design for the GemmaRMSNorm problem:** A2 (section 4), ceiling 1-3%,
  and it would be the fifth PR in the same lane (#29810, #36016, #44693, #45770).
+ **Directions for the 5090:** D1 and D2 in the survey were the first ranking;
  review round 4 (section 7) reduced D1 to a measurement and found D2 duplicated.
  No surveyed direction clears a >10% bar.

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
+ #39014 (ZJY0516, merged Apr 2026): ProExpertProg wrote "if `x` is 16bit and
  produced by a custom kernel, I'd want to avoid an extra triton kernel to convert
  to 32bit". That sentence is about converting the **activation** `x` to fp32, not
  the weight (corrected in review round 4); it shows the fusion owner's dislike of
  extra conversion kernels around the norm, which is analogous to branch A's
  `(1 + w)` kernel but not a statement about it. The same PR had a Gemma GGUF
  accuracy test fail; ZJY0516's stated cause is that the test switched from the
  native path to the CUDA kernel ("gemma gguf test uses cuda kernel, but previous
  is native implementation"), with no finer root cause in the thread. That is
  evidence about the risk of moving GemmaRMSNorm onto the `vllm_c` kernel (branch
  A's eager-path hunk, A-N2), not about `1 + w` rounding. A2 adds the bias in fp32
  inside the kernel, which is what xenova asked for in #29810.
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
+ **D1 (corrected in round 4):** Humming defaults to BF16 activations on SM120
  (`_backend_activation_key(HUMMING)` is `None`), and SM120 has no WGMMA, so the
  proposed priority flip compares W4A16 with W4A16; expected parity, as on GB10.
  The only arm with a tensor-core mechanism is Humming with MXFP8 activations via
  `VLLM_HUMMING_INPUT_QUANT_CONFIG`, which changes numerics. Product: a data
  comment on #56997 when a 5090 is in hand, not a PR.
+ **D2 (corrected in round 4):** duplicates #45379 (`E=256,N=256` block-FP8 for
  the 5090, open since 2026-06-12, measured 1-5% at kernel level, no review) and
  #55360 (`E=128,N=768`). The blocker is maintainer attention. Action: post 5090
  e2e numbers on #45379 instead of a new PR.
+ **E (found after round 4, survey section 8):** gpt-oss on SM120 runs its MoE on
  Marlin W4A16 although `FLASHINFER_CUTLASS_MXFP4_MXFP8` (FP4 tensor cores, MXFP8
  activations, expert biases and the gpt-oss SwiGLU parameters wired) is accepted
  on SM120 by vLLM's gate and tested on SM120 by FlashInfer; the BF16-first
  activation filter keeps it out of auto-selection. Dense precedent on the same
  architecture (#55170): TTFT −36%. One GPU day decides it; PR is a ~20-line
  per-architecture policy change in the MXFP4 oracle, with GSM8K.
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

## 7. Review round 4 (2026-10-19) and the bottom line

Checked against `origin/main` `c07c59f821` and GitHub by the reviewer; confirmed
here against the code and the PR pages.

+ Confirmed: the round-3 conclusion on branch A, the A2 idiom (all-reduce fusion),
  the #29810 history, every commit in the survey's SM12x table, the DeepGEMM
  exclusion table, the `replayssm` keying.
+ Corrected: the two #39014 citations (section 4); D1's mechanism (Humming W4A16 by
  default, no WGMMA on SM120); D2 duplicates #45379 and #55360; "mgoin asked for
  per-arch data" was wrong.
+ Reviewer's prediction for D1 on a 5090: decode within ±5%, prefill within ±10%,
  no stable winner (P ~0.7); P(Humming >= 10% prefill win with decode parity)
  ~0.15.
+ **Bottom line after four rounds:** none of the branches or surveyed directions
  reaches "most likely > 10% with a high merge probability". Do not rent a GPU for
  them. Cheap and certain when a 5090 is in hand anyway: post D1 data on #56997
  and 5090 numbers for #45379's config on that PR.
+ **Added after round 4, Direction E** (survey section 8): gpt-oss on SM120 runs
  its MoE through Marlin W4A16 while a FlashInfer CUTLASS MXFP4 x MXFP8 path is
  already in-tree, gated to family 120, and tested on SM120 by FlashInfer; it is
  only excluded from auto-selection by the BF16-activation-first filter. The
  mechanism for a large prefill gain is the dense precedent in PR 55170 (TTFT down
  36% on SM120). Needs one 5090 day (protocol in the survey); PR shape is about
  20 lines in the MXFP4 oracle; odds about 0.3 for a default change; the floor is
  a data comment on #31085.

## 8. Quick reproduction of the key claims

```bash
git fetch upstream main pull/45770/head:pr-45770 pull/59467/head:pr-59467 pull/36016/head:pr-36016
git merge-tree --write-tree pr-59467 origin/feat/gelu-tanh-act-quant-fusion~1          # clean
git merge-tree --write-tree --merge-base=pr-45770^2 pr-45770 origin/feat/gemma-rmsnorm-quant-fusion~1   # 4 conflicts
sed -n 100,135p vllm/compilation/passes/pass_manager.py        # fusion passes -> cleanup -> IR lowering
sed -n 505,600p vllm/compilation/passes/fusion/allreduce_rms_fusion.py   # weight.float() + 1.0 / weight_bias=1.0
sed -n 1,30p vllm/model_executor/layers/quantization/utils/humming/priority.py   # {90}
ls vllm/model_executor/layers/fused_moe/configs | grep -ci 5090                   # 0
```
