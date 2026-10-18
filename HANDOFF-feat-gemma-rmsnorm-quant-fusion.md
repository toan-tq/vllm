# Handoff: `feat/gemma-rmsnorm-quant-fusion`

> Review/handoff notes for this fork branch. **Delete this file before opening an
> upstream PR.** Written so that a reviewer or a later agent session can pick the
> work up without the original conversation. Revision 3 (2026-10-18) incorporates
> both rounds of the external review (`REVIEW_2_NHANH_FUSION_2026-10-18.md` and its
> round-2 follow-up); section 9 lists what each round changed.

**Tóm tắt (vi):** Nhánh này cho phép các kernel RMSNorm và RMSNorm+quant của vLLM
nhận weight fp32 trên activation bf16/fp16, tức là kiểu norm của `GemmaRMSNorm`
(Gemma 1/2/3, Qwen3-Next, Qwen3.5/3.6/3.8 và các MTP head, Step3.5, RNJ1, Alice,
InternS2-Mobius). Mã đã viết xong, lint sạch, các file `.cu` biên dịch được bằng
nvcc 12.0, nhưng **chưa chạy test trên GPU**. Điểm quan trọng nhất sau review:
với checkpoint **block-FP8** (ví dụ `Qwen/Qwen3.8-27B-FP8`, `Qwen3.5-27B-FP8`,
`Qwen3.5-35B-A3B-FP8`) vLLM tự bật custom op `quant_fp8`, nên pass RMSNorm+quant
**đang chạy mặc định** và chỉ bị guard dtype chặn lại với GemmaRMSNorm. Nhánh này
đổi guard đó sang một guard hẹp hơn, nghĩa là **đổi hành vi mặc định** của nhóm model
đó: chúng sẽ được fuse. Cần parity end-to-end (GSM8K) và số TPOT trước khi mở PR.

## 1. Status

| Item | State |
| --- | --- |
| Base | `upstream/main` @ `371745ebd` (2026-10-18) |
| Commits | one feature commit + this handoff commit (`git log upstream/main..`); history rewritten in revision 3: author/committer `Toan Tran <toan-tq@outlook.com>`, DCO `Signed-off-by`, one `Co-authored-by` AI line, no session link. Drop the handoff commit before a PR (`git reset --hard HEAD~1` keeps the feature commit) |
| Lint | `ruff`, `ruff-format`, `clang-format`, `mypy` (oldest py), `typos`, `markdownlint-cli2`: pass |
| CUDA compile | the three modified `.cu` files compile with nvcc 12.0 (sm_90) against the torch 2.14 stable headers; section 7 |
| GPU tests | **not run** (no GPU in the authoring environment) |
| Benchmarks | **not run** |
| Overlap | #45770 solves the same problem differently; `git merge-tree` conflicts in 4 files (section 5). Comment there before opening a PR |

## 2. Problem

`GemmaRMSNorm.forward_native` (`vllm/model_executor/layers/layernorm.py`) computes
`weight = self.weight.float() + 1.0` and calls `ir.ops.rms_norm(x_bf16, weight_fp32, eps)`.
Every fast path refused the mixed dtype:

- `vllm/kernels/vllm_c.py`: `supports_args` required `weight.dtype == x.dtype`, so the
  IR dispatch fell back to the native torch implementation.
- `csrc/layernorm_kernels.cu`, `csrc/layernorm_quant_kernels.cu`,
  `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu`:
  `STD_TORCH_CHECK(weight.scalar_type() == input.scalar_type())`.
  `rms_norm_static_fp8_quant` had no explicit check; the typed accessor
  `const_data_ptr<scalar_t>()` of the stable ABI checks the dtype itself
  (`torch/csrc/stable/tensor_inl.h`), so a mixed call raised there instead of
  reading out of bounds. The branch adds an explicit check with a clearer message.
- `vllm/compilation/passes/fusion/rms_quant_fusion.py`: every RMSNorm+quant pattern
  carried `extra_check=_rms_input_weight_dtype_match`, with the upstream TODO
  "extend rmsnorm quant kernels to support mixed input/weight dtypes, and remove this
  check".

History of that guard (verified on GitHub, 2026-10-18): the guard and the TODO
predate June 2026. Issue #44611 (Qwen3.5-35B-A3B-FP8 on 8x RTX 5090 and on GB200,
default flags) crashed in `rms_norm_per_block_quant` on the kernel's dtype check
because the guard only looked at plain `rms_norm` nodes; **#44694** (merged
2026-06-06) extended it to `fused_add_rms_norm` nodes. #48330 (merged 2026-07-12)
is unrelated to this file: it guards the all-reduce fusions in
`allreduce_rms_fusion.py`.

Models that use `GemmaRMSNorm` on upstream main (`git grep` for the class and its
aliases `Qwen3NextRMSNorm`, `Qwen3_5RMSNorm`): `alice_ai.py`, `gemma.py`,
`gemma2.py`, `gemma3.py`, `gemma3_mm.py`, `interns2_mobius.py`, `qwen3_5.py`,
`qwen3_5_mtp.py`, `qwen3_next.py`, `qwen3_next_mtp.py`, `rnj1.py`, `step3p5.py`,
`step3p5_mtp.py`, and the Transformers backend (`transformers/base.py`,
`transformers/layers.py`, `transformers/fusers/rms_norm.py`). Qwen3.8-27B ships as
`Qwen3_5ForConditionalGeneration`. **Gemma 4 is not affected**: `gemma4.py` uses
plain `RMSNorm` with a bf16 weight.

## 3. Change summary

### C++ / CUDA (shared CUDA + ROCm sources)

- `csrc/layernorm_weight.cuh` (new): `norm_mul_weight<scalar_t>(x_norm, w)` and
  `load_weight_as_float<weight_t, VEC>(ptr, out)`.
- `csrc/dispatch_utils.h`: `VLLM_STABLE_DISPATCH_NORM_WEIGHT_TYPE(INPUT_T, WEIGHT_TYPE, ...)`
  defines `weight_t` as `float` when the weight is fp32, else as the input type.
  Callers validate `weight dtype == input dtype || weight dtype == float32` first.
- `csrc/layernorm_kernels.cu`: `rms_norm_kernel` and both `fused_add_rms_norm_kernel`
  specializations take `typename weight_t`; host functions check and dispatch.
- `csrc/layernorm_quant_kernels.cu`: same for `rms_norm_static_fp8_quant_kernel` and
  both `fused_add_rms_norm_static_fp8_quant_kernel` specializations, plus the
  explicit dtype check on `rms_norm_static_fp8_quant`.
- `csrc/quantization/fused_kernels/layernorm_utils.cuh`: `compute_dynamic_per_token_scales`
  and `norm_and_quant` (scalar + vectorized) gain a trailing `typename weight_t = scalar_t`
  deduced from the weight pointer, so existing call sites are unchanged.
- `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu`: kernels
  take `typename weight_t`; both host entry points dispatch on the weight dtype. The
  vectorized paths load `vec4_t<weight_t>` (16 bytes for fp32, 8 for fp16/bf16).
  The per-token kernel folds the weight alignment into its existing `can_vectorize`
  decision, so a misaligned weight takes the scalar path (review 2, A-N3). The
  per-block kernel has no scalar path, so its host entry rejects a misaligned fp32
  weight (`STD_TORCH_CHECK` in `rms_norm_per_block_quant`) instead of faulting. In the model path
  `weight.float() + 1.0` is a fresh allocation and both conditions hold; inductor's
  own buffers are allocated with at least 16-byte alignment as well, but that was
  not verified on this graph.

### Numerics contract

- Same-dtype weight: **bit-identical to before.** The same expressions are kept
  (`static_cast<scalar_t>(x * rms) * w` for the quant kernels, `x * s_variance * w`
  in fp32 for the plain kernels).
- fp32 weight on fp16/bf16 input: follows the native reference
  (`vllm/ir/ops/layernorm.py`: `x.to(weight.dtype) * weight` then `.to(orig_dtype)`),
  i.e. multiply in fp32 and round to the activation dtype **once**:
  `static_cast<scalar_t>(x_norm * static_cast<float>(w))`. The quant kernels then
  quantize that rounded value, matching the unfused composite.
- Pre-existing, not changed: the CUDA `fused_add_rms_norm` adds the residual in the
  activation dtype before normalizing, while the native op adds in fp32. This
  already applies to every model on the `vllm_c` path; this branch is the first
  time the GemmaRMSNorm models take that path, one more reason for e2e parity.
- Weight loads: `load_weight_as_float` uses one vector load when the group is
  <= 16 bytes (bf16/fp16 x8, fp32 x4, as before) and scalar loads otherwise
  (fp32 weight at an fp16/bf16 vector width of 8). Weights are L1/L2 resident,
  so the extra scalar loads are expected to be noise, but this is unmeasured.
- Path selection: the per-token scalar fallback for a misaligned weight changes the
  reduction order relative to the vectorized path, exactly as a hidden size that is
  not a multiple of 4 already does. Not reachable from the model path.

### Python

- `vllm/kernels/vllm_c.py`: `_weight_dtype_supported` allows an fp32 weight on
  CUDA/ROCm only (`CUDA_ALIKE`); XPU keeps the strict match because its kernels come
  from `vllm-xpu-kernels` and were not changed. This is the eager / `+rms_norm`
  behaviour change described in section 4; the fusion pass does not depend on it.
- `vllm/compilation/passes/fusion/rms_quant_fusion.py`: the six vLLM-kernel patterns
  (static, dynamic per-token, per-block; with/without residual) now use
  `_rms_weight_dtype_supported`: fuse when the dtypes match, or when the platform is
  CUDA/ROCm and the weight is fp32 on fp16/bf16 activations. **Every other mix
  still skips fusion** (previous behaviour) instead of reaching a kernel that would
  raise. That also keeps XPU, which imports this pass, on its old behaviour.
  `_rms_input_weight_dtype_match` (strict) stays for the FlashInfer NVFP4 pattern
  (`add_rmsnorm_fp4quant` documents "weight must have the same dtype as input") and
  for `allreduce_rms_fusion.py`, which imports it.

### Not changed on purpose

- FlashInfer NVFP4 add-RMSNorm pattern (external kernel, same-dtype requirement).
- All-reduce + RMSNorm fusions (`allreduce_rms_fusion.py`): TP>1 FlashInfer/TRT-LLM
  kernels, separate concern; #43953 covers the ROCm side.
- ROCm AITER impls (`vllm/kernels/aiter_ops.py`) keep their own dtype guard.
- XPU kernels.

### Tests, docs, benchmark

- `tests/kernels/core/test_layernorm.py`: `test_rms_norm_fp32_weight`
  (kernel vs native IR impl, with/without residual, strided input, odd hidden
  sizes, opcheck), `test_rms_norm_rejects_other_weight_dtypes` (fp16 input + bf16
  weight must raise), `test_fused_rms_norm_quant_fp32_weight` (static FP8, 1-ULP
  outlier bound).
- `tests/kernels/core/test_fused_quant_layernorm.py`: `test_rms_norm_fp32_weight`
  (dynamic per-token and per-block, FP8 and INT8, residual on/off) against a native
  reference (`_NativeNorm`) so the kernel is not compared with itself.
- `tests/compile/passes/test_fusion.py`: `TestModel(norm_cls=...)`,
  `test_fusion_rmsnorm_quant_fp32_weight` (GemmaRMSNorm, per-tensor / per-token /
  block-128, quant custom op on/off, expects `matched_count == 3`) and
  `test_fusion_rmsnorm_quant_unsupported_weight_dtype_skips` (fp16 `1 + w` weight on
  bf16 activations: `matched_count == 0`, no fused op in the graph, the unfused
  `vllm_ir` norm and quant ops still present, output dtype preserved; review 2, A-N4).
- `docs/design/fusions.md`: one paragraph under "RMSNorm + Quantization".
- `benchmarks/kernels/benchmark_layernorm.py --gemma --ir-impl vllm_c`: pins the
  IR priority, asks the dispatcher which implementation it selected for the exact
  arguments, and aborts if it is not the requested one (identity gate), so a
  reported number cannot silently come from the native path.
- Not covered: the 2D weight path of `rms_norm` (`[num_groups, hidden]`) with
  fp32, ROCm, and any end-to-end run.

## 4. Reach: where this changes execution (reviewer lens)

Verified in `vllm/config/vllm.py` on upstream main:

- `has_blocked_weights()` adds `+quant_fp8` to `custom_ops` for any checkpoint with
  `weight_block_size` set (lines ~1799-1814, both config paths).
- `enable_norm_fusion` is true when the `quant_fp8` custom op is enabled; the default
  optimization level is O2, whose pass config sets `fuse_norm_quant = enable_norm_fusion`.

Therefore:

- **Block-FP8 checkpoints of GemmaRMSNorm models run the RMSNorm+quant pass by
  default today**, and the dtype guard is the only thing that keeps them unfused.
  Issue #44611 is the empirical confirmation (default flags, Qwen3.5-35B-A3B-FP8,
  the per-block fused kernel was reached). With this branch those models get the
  fused `rms_norm_per_block_quant` with an fp32 weight: `Qwen/Qwen3.8-27B-FP8`,
  `Qwen/Qwen3.5-27B-FP8`, `Qwen/Qwen3.5-35B-A3B-FP8`, Qwen3-Next-80B-A3B-FP8 and
  similar. **This is a default-behaviour change for a popular model family and is
  the main reason the branch needs e2e parity before a PR.**
- Non-block FP8 (per-token dynamic, static) in the default config: no change on
  CUDA; `fuse_norm_quant` is off and inductor fuses the native ops itself. The
  branch applies when `+rms_norm` / `+quant_fp8` are set by hand, on ROCm defaults,
  and under `enforce_eager` (IR priority `["vllm_c", "native"]`), where the kernel
  replaces the ~6 unfused torch ops of the native norm. Note the `weight.float() + 1.0`
  of `GemmaRMSNorm` stays outside the kernel in every mode (two small ops per norm).
- **Eager / `+rms_norm` numerics (review 2, A-N2).** `_weight_dtype_supported` makes
  the `vllm_c` impl eligible for GemmaRMSNorm, so under `--enforce-eager` (default
  IR priority `["vllm_c", "native"]` on CUDA, `vllm/platforms/cuda.py`) and with
  `+rms_norm`, Gemma 2/3, Qwen3-Next and Qwen3.5/3.6/3.8 run the CUDA
  `fused_add_rms_norm` kernel for the first time. That kernel rounds `x + residual`
  to the activation dtype before normalizing, the native op normalizes the fp32 sum
  (numerics contract above). Llama-style models have always run this way; for this
  family it is new, so the parity matrix in section 6 includes `--enforce-eager`. If
  eager parity turns out unacceptable, drop the `vllm/kernels/vllm_c.py` hunk: the
  fusion patterns match the `vllm_ir.rms_norm` node in the graph, not an impl, so
  the pass path keeps working without it.
- NVFP4 checkpoints: still excluded (FlashInfer same-dtype kernel). Once #61783
  enables norm fusion by default for NVFP4, Llama-style norms get fused and
  GemmaRMSNorm models do not; a vLLM-owned fused add-RMSNorm+NVFP4-quant kernel that
  accepts an fp32 weight is the natural follow-up and would reuse the dispatch
  added here.

Expected magnitude where it applies: one kernel plus one activation pass per norm
(two norms per decoder layer). For the default block-FP8 case the baseline is
"inductor-fused norm + separate `per_token_group_fp8_quant` kernel", so the gain is
one launch and one read/write of the bf16 activation per norm. **No measurement
exists yet.**

## 5. Related work / duplicate check (2026-10-18)

- #45770 (flutist, opened Jun 2026; on 2026-10-18 OPEN, `CONFLICTING`, last update
  2026-08-19, bot review only, author pinged five times; title says "fixes
  Qwen3.5-FP8"). Its own diff against the main it merged (`git fetch upstream
  pull/45770/head`, then `git diff pr-45770^2 pr-45770`) touches
  `csrc/libtorch_stable/layernorm_quant_kernels.cu`,
  `csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu`,
  `rms_quant_fusion.py` and `allreduce_rms_fusion.py`. What it does: (a) static FP8
  kernels templated on a weight type, mixed dtypes take an element-wise
  (non-vectorized) path; (b) per-token and per-block: the **pass replacement casts
  the weight to the input dtype** (`weight = weight.to(dtype=input.dtype)`), which
  rounds `1 + w` to bf16/fp16 and adds one cast op per norm, and the per-block host
  entry casts again with `torch::stable::to`; the per-token kernel itself is
  unchanged and would still read the weight as the input type (the bot's
  element-size finding), which the cast in the pass hides; (c) the guard accepts
  any floating weight dtype with a one-time warning; (d) the all-reduce fusion
  guard is loosened the same way. Revision 2 of this file said "FP8 only, missing
  per-block"; **that was wrong**, per-block is covered by the cast. How this branch
  differs: every kernel (the plain `rms_norm` / `fused_add_rms_norm`, the static,
  per-token and per-block fused ones) is templated on `weight_t` and keeps its
  vectorized path, no cast op is added, the fp32 multiply matches the native
  reference, the guard is narrowed instead of widened (fp32 weight on fp16/bf16
  only; nothing else changes behaviour), and the eager `vllm_c` path is enabled too.
  `git merge-tree --merge-base=pr-45770^2 pr-45770 <this branch>` conflicts in the
  four files above. Per AGENTS.md this is a materially different approach to the
  same problem: before opening a PR, comment on #45770 (and #44611) explaining the
  difference and ask the #44694 reviewers which direction they want; credit flutist
  if this supersedes it.
- #44694 (merged): the guard extension that this branch replaces with a narrower one.
- #44693 (MrAnayDongre): regression test only; ZJY0516 asked for an e2e Qwen3.5 run.
- #48330 (merged): guards the **all-reduce** RMSNorm fusions; unrelated to this file.
- #43953 (nholmber): ROCm AR+RMS fusion for GemmaRMSNorm (separate pass).
- #61783 (gkswns0531): NVFP4 norm fusion default-on; motivates the follow-up above.

## 6. How to build and test (needs a CUDA GPU)

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt
# csrc changed: full build, not the precompiled wheel
uv pip install -e . --no-build-isolation
uv pip install -r requirements/test/cuda.in

pytest tests/kernels/core/test_layernorm.py -k "fp32_weight or rejects" -v
pytest tests/kernels/core/test_fused_quant_layernorm.py -k fp32_weight -v
pytest tests/compile/passes/test_fusion.py -k fp32_weight -v
# regression: existing suites must stay green
pytest tests/kernels/core/test_layernorm.py tests/kernels/core/test_fused_quant_layernorm.py
pytest tests/compile/passes/test_fusion.py -k "rmsnorm_quant and not aiter"

python benchmarks/kernels/benchmark_layernorm.py --gemma --ir-impl vllm_c \
    --dtype bfloat16 --num-tokens 1 --hidden-size 5120 --add-residual
python benchmarks/kernels/benchmark_layernorm.py --gemma --ir-impl native \
    --dtype bfloat16 --num-tokens 1 --hidden-size 5120 --add-residual
```

End-to-end, in priority order (RTX 5090, 32 GB):

| Model | Config | Why |
| --- | --- | --- |
| `Qwen/Qwen3.8-27B-FP8` (block-128) | default | **default config changes** (pass path); tight on 32 GB, use a short `--max-model-len` |
| `Qwen/Qwen3.5-35B-A3B-FP8` or `Qwen3.5-27B-FP8` | default | same path, the issue-#44611 model family |
| the same Qwen3.5/3.8 checkpoint | `--enforce-eager` | eager path now on the `vllm_c` kernel (A-N2); parity only |
| `google/gemma-3-12b-it` (bf16) | `--enforce-eager` | eager path for the unquantized Gemma family; parity only |
| `RedHatAI/Qwen3.5-9B-FP8-dynamic` | `+rms_norm,+quant_fp8` | per-token pattern (custom-op configs only) |
| any Gemma 4 checkpoint | default | control: must be unchanged |

Run `main` vs this branch interleaved, each with its own `VLLM_CACHE_ROOT`. Parity:
GSM8K (at least 200 samples, same seed, greedy) or NLL on a fixed text set, for
every row; a difference beyond run-to-run noise blocks the PR. Before reading any
performance number confirm the identity gate: the compile log
(`VLLM_LOGGING_LEVEL=DEBUG`) must show `rmsnorm_quant_fusion_pass` replacing
patterns, and an nsys trace must show `rms_norm_per_block_quant_kernel` instead of
`per_token_group_fp8_quant`; in eager mode the trace must show
`fused_add_rms_norm_kernel` instead of the native elementwise kernels. Then TPOT at
batch 1/8/32 for the default-config rows.

## 7. Verification log

- Lint hooks: pass (see Status).
- CUDA compile check (no GPU; object compile only, 2026-10-18, re-run after
  revision 3's alignment fallback):
    - Toolchain: Ubuntu `nvidia-cuda-toolkit` nvcc 12.0.140, host `g++-12`, torch 2.14.0
    headers (`torch/csrc/stable`, `torch/headeronly`), CUB from the same toolkit.
    - Flags mirrored from the extension build: `-std=c++20 --expt-relaxed-constexpr
    --expt-extended-lambda -DTORCH_TARGET_VERSION=0x020B000000000000ULL -DUSE_CUDA
    -DENABLE_FP8 -gencode arch=compute_90,code=sm_90 -Icsrc -I<torch include>`.
    - Result: `csrc/layernorm_kernels.cu`, `csrc/layernorm_quant_kernels.cu`,
    `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu`
    compile cleanly on this branch; the same files compile on unmodified
    `upstream/main` with the same command (harness validated). Because the host
    dispatch instantiates every (input, weight) dtype pair, the fp32-weight
    instantiations were compiled, not just the same-dtype ones.
    - Not covered: ROCm/HIP, sm_120 codegen, runtime behaviour.
- Nothing has run on a GPU.

## 8. Review checklist / known risks

- [ ] e2e parity for block-FP8 Qwen3.5/3.8 (the default-config change).
- [ ] e2e parity under `--enforce-eager` for one Gemma and one Qwen3.5 checkpoint
      (A-N2); fall back to dropping the `vllm_c.py` hunk if it fails.
- [ ] Comment on #45770 / #44611 before opening a PR (A-N1).
- [ ] Alignment of the `1 + w` buffer under inductor (the per-block entry raises if
      it is not 16-byte aligned; per-token falls back to the scalar path).
- [ ] fp32 weight + fp16/bf16 input adds instantiations (per-block kernel: roughly
      +67%, from 48 to 80 combinations). Build time and binary size unmeasured.
- [ ] ROCm: `norm_mul_weight` relies on `c10::Half`/`c10::BFloat16` `operator*` on
      device, as the existing code did. Not compiled for HIP here.
- [ ] 2D weight (`rms_norm` with `[num_groups, hidden]`) works with fp32 since the
      pointer arithmetic is in `weight_t` units; not covered by a dedicated test.
- [ ] Tolerances of the new tests are copied from the same-dtype tests; whether
      they hold for the fp32-weight path is only known after a GPU run.
- [ ] The FlashInfer NVFP4 pattern still silently skips GemmaRMSNorm graphs.

## 9. Revision history

### Revision 3 (review round 2)

- A-N1: the #45770 description in revision 2 ("FP8 only, missing per-block") was
  wrong; it covers per-token and per-block by casting the weight in the pass. The
  overlap is now measured (`git merge-tree`: 4 conflicting files) and the
  recommendation is to comment on #45770 first.
- A-N2: the eager / `+rms_norm` numerics change for the GemmaRMSNorm family is
  documented with a parity matrix and a fallback (drop the `vllm_c.py` hunk).
- A-N3: the per-token kernel falls back to its scalar path for a misaligned weight
  instead of raising; the per-block entry keeps the check because that kernel has
  no scalar path.
- A-N4: pass-level test for the "unsupported dtype mix skips fusion" branch.
- G-1: commit history rewritten (author, DCO sign-off, one AI trailer, no session
  link). G-2/G-3 (GPU tests, measurements) remain open.

### Revision 2 (review round 1)

- Guard origin: previously attributed to #48330; it is #44694 (fused-add
  extension) on top of an older guard. #48330 is the all-reduce guard.
- Reach: previously "no change in the default CUDA config". Wrong for block-FP8
  checkpoints, where `+quant_fp8` is auto-enabled and the pass runs by default.
- `rms_norm_static_fp8_quant`: previously described as an out-of-bounds read risk;
  the stable typed accessor already raised on a dtype mismatch.
- Model list extended with the alias users (MTP heads, Alice, InternS2-Mobius).
- "~600 fewer launches" in eager mode overstated; the `weight.float() + 1.0`
  ops remain outside the kernel.
- #45770 description corrected (it does touch the per-block path, by casting).
- Guard narrowed instead of removed (unsupported mixes keep skipping fusion).
- fp32 weight alignment made an explicit check; benchmark got an identity gate;
  negative dtype test added.

## 10. `pr-checklist` notes (2026-10-18, revision 3)

**Verdict: not ready to open.** Blocked on GPU validation (nothing has run) and on
the #45770 conversation. The code-quality items of the checklist are addressed.

### Design fit

- 1.1 Core components: none touched (csrc kernels, `vllm/kernels/vllm_c.py`, one
  compiler pass, tests, one benchmark, one docs paragraph).
- 1.2 Replication: `norm_mul_weight` / `load_weight_as_float` are shared by the
  three kernel files; the per-token/per-block utilities reuse `vec4_t<weight_t>`;
  the only new abstraction is the `VLLM_STABLE_DISPATCH_NORM_WEIGHT_TYPE` macro.
- 1.3 Complexity: 351 added / 184 deleted non-test, non-doc lines in 9 files
  (plus 46 in the benchmark): upper end of "medium". Most of it is mechanical
  (`scalar_t` -> `weight_t` in signatures and dispatch). Justification: removes an
  upstream TODO and closes the GemmaRMSNorm gap of a pass that is already on by
  default for block-FP8 Qwen3.5/3.6/3.8.
- 1.4 Correctness: same-dtype instantiations are the same code; the stable typed
  accessors still reject any other dtype; the fp32 path is only reachable with an
  fp32 weight. ROCm builds the same sources (not compiled here). XPU, the
  all-reduce fusions and the FlashInfer NVFP4 pattern are unchanged.
- 1.5 Tradeoffs: more template instantiations (per-block kernel roughly 48 -> 80
  combinations; build time and binary size unmeasured); eager numerics change for
  the GemmaRMSNorm family (section 4); no CPU/GPU synchronization added (the
  alignment checks read pointers only).
- 1.6 MRV1/MRV2: not applicable.

### Testing and validation

- 2.1 Kernel tests compare against the native reference, not the kernel against
  itself; pass-level positive (`matched_count == 3`) and negative
  (`matched_count == 0`) tests; negative dtype test at kernel level. The only mock
  is the existing `TestFP8Layer` helper.
- 2.2 Flakiness: tolerances are copied from the same-dtype tests; the 1-ULP outlier
  bound in `test_fused_rms_norm_quant_fp32_weight` is the riskiest assertion.
- 2.3 CI: the tests live in files already wired into Buildkite
  (`check-test-tethering` hook passes); CI would cover H100-class GPUs, not ROCm or
  SM120. **Nothing has run on any GPU yet; the human submitter must run section 6
  before publishing.**

### Code quality

- 3.1 Comments kept to the alignment and dtype decisions.
- 3.2 `docs/design/fusions.md` updated.
- 3.3 `check_fp32_weight_alignment` had a single call site after revision 3 and
  was inlined.
- 3.4 No unrelated formatting changes (`git diff --stat` reviewed).

### Pull request contents (draft, template `.github/PULL_REQUEST_TEMPLATE.md`)

```markdown
## Overview

Let the RMSNorm and RMSNorm+quant kernels take an fp32 weight on fp16/bf16
activations, so the RMSNorm+quant fusion (and the vllm_c kernels in eager mode)
cover GemmaRMSNorm models (Gemma 2/3, Qwen3-Next, Qwen3.5/3.6/3.8). Removes the
dtype guard from #44694 (#44611). Related: #45770 (different approach, see Details).

## Claims

- RMSNorm+quant fusion fires for GemmaRMSNorm graphs (FP8 static, per-token,
  per-block 128/64) instead of being skipped; other dtype mixes still skip.
- Same-dtype kernels are unchanged; the fp32-weight path follows the native
  reference rounding.
- TODO: TPOT on <model> at batch 1/8/32, main vs PR (table), nsys identity gate.

## Validation

- TODO: pytest commands from section 6 with results.
- TODO: GSM8K parity table (default config and --enforce-eager), main vs PR.
- Compile check without GPU: nvcc 12.0 (sm_90) on the three .cu files.

## Details

- weight_t template in every kernel instead of casting the weight (no extra op,
  fp32 multiply kept); guard narrowed to "same dtype or fp32 on fp16/bf16".
- Differs from #45770, which casts the weight to the activation dtype in the pass
  replacement and widens the guard to any floating weight.
- Tradeoffs: more instantiations; eager path of this model family now uses the
  vllm_c fused_add_rms_norm kernel (parity run above).
- AI assistance was used (Claude); every changed line was reviewed by the
  submitter. Not a duplicate of #45770 for the reasons above (commented there).
```

## 11. Value assessment and go/no-go (2026-10-18, before any measurement)

> **Superseded by section 12 (2026-10-19).** The launch accounting below assumes
> the baseline runs the norm and the quant as two separate launches and this
> branch as one. In the compiled default path that is wrong: inductor already
> inlines `weight.float() + 1.0` into the norm kernel, and this branch forces it
> out into its own kernel. Read section 12 before planning any measurement.

**Expected magnitude.** Each fused norm removes one kernel and one read/write of
the bf16 activation. With CUDA graphs the saving is mostly the inter-kernel gap,
about 2-3 us per norm. For a 27B FP8 model on an RTX 5090 at batch 1: 2 norms x
64 layers x ~2.5 us is ~0.3 ms on a ~20 ms step, i.e. **1-2% TPOT**. The saved
bandwidth at large batch is negligible next to the weight reads, so the number
does not grow with batch. `docs/design/fusions.md` lists 1-4% for this fusion
class, which upstream accepted for Llama-style models. A ">10%" bar is not
reachable with this change.

**Why it is still worth a PR.** This is not a new optimization; it closes a gap in
a fusion that is already on by default for block-FP8 checkpoints (official
Qwen3.5/3.6/3.8 FP8), where only the dtype guard keeps GemmaRMSNorm unfused.
Upstream left a TODO asking for exactly this, the guard is a stopgap after the
crash in #44611, and #45770 shows independent demand. The main risk is
maintainer bandwidth (#45770 has waited since June), not the technique.

**Go/no-go.**

1. Measure first (section 6) on a Qwen3.5 FP8 block checkpoint that fits 32 GB,
   with GSM8K parity and the identity gate.
2. TPOT gain >= ~1.5% with clean parity: open the PR after the #45770 / #44611
   conversation (offer the templating approach there if the maintainers prefer
   flutist's PR).
3. Gain < 1% with clean parity: still mergeable on "removes the TODO and the
   stopgap guard" grounds, but expect low priority; say the number honestly.
4. Any parity regression: do not open; first try dropping the `vllm_c.py` hunk
   (eager path) and re-measure.

**Longer-term value.** The kernels now accept an fp32 weight, which is the
prerequisite for a vLLM-owned fused add-RMSNorm + NVFP4 quant for the GemmaRMSNorm
family (#61783 makes NVFP4 norm fusion default-on). On Blackwell, NVFP4 is the
format that matters more than FP8 for this GPU; that follow-up has clearer value
than either FP8 branch.

## 12. Review round 3 (prediction document, 2026-10-19): corrected accounting

The third review round was a prediction document written before any run. Its
central claim was checked against the code on 2026-10-19 and holds.

### What was verified

- Pass order (`vllm/compilation/passes/pass_manager.py`): the fusion passes run
  first, then `PostCleanupPass`, then `VllmIRLoweringPass`, which traces the
  selected implementation (native under inductor) into the graph. Inductor
  therefore sees the native norm as aten ops.
- Baseline compiled path for a block-FP8 GemmaRMSNorm model: `weight.float() +
  1.0` is a single-use pointwise op, so inductor inlines it into the norm's
  Triton kernel; the quant is the opaque `per_token_group_fp8_quant` custom op.
  Per norm: most likely one Triton kernel plus one CUDA quant kernel = **2
  launches**. Inductor fuses the variance reduction with the row-wise apply even
  when the row is too long for a persistent reduction (threshold 1024 for inner
  reductions, `torch/_inductor/choices.py`): `SIMDScheduling.can_fuse` accepts a
  pointwise over `numel * rnumel` after a reduction, and `generate_node_schedule`
  emits it as a second loop in the same kernel. Two Triton kernels (3 launches)
  is the less likely case; the profile decides.
- This branch: the pattern replaces only `vllm_ir.(fused_add_)rms_norm` plus the
  quant. `weight.float() + 1.0` stays outside the pattern and becomes an input of
  the fused custom op, so it must be materialized: **one extra Triton kernel per
  norm** (fp32, `hidden` elements), unless inductor's combo-kernel pass
  (`combo_kernels=True` by default, `vllm/config/compilation.py`) merges the
  independent `(1 + w)` kernels of a graph partition into one launch. Per norm:
  1 (or ~0.5) Triton plus 1 fused CUDA kernel: **1.5-2 launches**.
- Net effect: **0 to 1 launch saved per norm**, not 1 as section 11 assumed. At
  M=1 the activation traffic saved (~10 KB at hidden 5120) is smaller than the
  fp32 weight the branch now writes and reads back (~20 KB each way). Predicted
  TPOT change in the default config: **within noise (−0.5% to +1%)**; prefill
  below 1%. The eager path (`--enforce-eager`, not the default) should gain a
  few percent, since ~8 torch kernels per norm become 3.

### Upstream precedent (verified on `upstream/main` 689effee6d)

- `allreduce_rms_fusion.py` has Gemma-specific patterns that match
  `vllm.ir.ops.rms_norm(x, weight.float() + 1.0, eps)` and pass the **raw weight
  with `weight_bias=1.0`** to the FlashInfer kernel; the manual path
  `fused_allreduce_gemma_rms_norm.py` does the same, and vLLM's own
  `csrc/fused_qknorm_rope_cache_kernel.cu` takes a `norm_weight_bias`. The vLLM
  idiom for GemmaRMSNorm fusion is "model-dtype weight plus a bias applied in
  the kernel", not an fp32 weight.
- #36016 (jackzhxng, draft since 2026-03-04, `needs-rebase` since May, no
  maintainer review, e2e test "blocked by GPU memory availability"): adds
  `gemma_*` copies of all six kernels and ops with an `is_gemma` template
  computing `1 + w` in-kernel, Gemma matchers, and makes
  `GemmaRMSNorm.forward_cuda` call the custom op. Same idea as `weight_bias`,
  heavier (six new op schemas). Fetched as `pr-36016` in this checkout.
- RFC #43224 (mgoin, open): compiler fusion passes are to be ported to manual
  fusions in model code; RMSNorm+quant is sub-issue #43500 and a contributor's
  producer PR is referenced there. Under that design GemmaRMSNorm calls the fused
  kernel directly with its raw weight, so a kernel-level `weight_bias` serves both
  the current pass and the future manual path. fp32-weight templating serves
  neither well.

### Consequence for this branch

- As designed, this is a correctness and enabler change (TODO and stopgap guard
  removed, eager path on `vllm_c`), **not a measurable default-config speedup**.
  Do not open it as a performance PR, and do not spend GPU time expecting one.
- The design that can show a gain ("A2"): keep the weight in the model dtype, add
  `weight_bias` (float, default 0) to the fused norm+quant kernels, give the eager
  path a Gemma entry (a `weight_bias` argument on `ir.ops.rms_norm` /
  `fused_add_rms_norm`, or a Gemma op as in #36016), and register Gemma patterns
  that match `weight.float() + 1.0` like the all-reduce pass. That removes the
  `(1 + w)` kernel entirely: baseline 2-3 launches per norm become 1. Expected
  gain is still only **1-3% TPOT** (80-130 fused norms x 2-5 us on a ~20 ms
  step), below any 10% bar. Size comparable to this branch; most of this branch's
  kernel templating would be replaced, not extended. Coordinate with the #36016
  author and the #43500 owner first; this is their lane.
- Recommendation: park this branch unless a GemmaRMSNorm fusion is needed for the
  NVFP4 direction; in that case build A2 directly with `weight_bias` in the
  NVFP4 fused kernel instead of extending this branch.

### Predictions to compare against if a run happens

| # | Item | Prediction |
| --- | --- | --- |
| A1 | Full csrc build (sm_120 + sm_90) first try | passes |
| A2 | New kernel tests (`-k "fp32_weight or rejects"`) first try | pass; if not, the 1-ULP static FP8 bound or the per-block tolerance |
| A3 | Existing kernel/pass suites | green (same-dtype path is unchanged code) |
| A4 | `test_fusion_rmsnorm_quant_fp32_weight` | `matched_count == 3` |
| A5 | Qwen3.x-27B-FP8 compile log | ~80 of 129 norms fused (16 full-attention input norms + 64 post-attention); 0 of the 48 GDN input norms |
| A6 | nsys, decode | ≤ 0.5 launch saved per norm; a separate Triton `(1 + w)` kernel appears |
| A7 | TPOT decode bs1/8, default config | −0.5% to +1%, most likely ~0% |
| A8 | Prefill / TTFT at 8k tokens | < 1% |
| A9/A10 | GSM8K parity, default and `--enforce-eager` | within noise |
| A11 | Eager TPOT (not default) | a few percent better |
| A12 | Qwen3.x-27B-FP8 on one 32 GB RTX 5090 with usable KV cache | does not fit |
