# Handoff: `feat/gelu-tanh-act-quant-fusion`

> Review/handoff notes for this fork branch. **Delete this file before opening an
> upstream PR.** Written so that a reviewer or a later agent session can pick the
> work up without the original conversation. Revision 3 (2026-10-18) incorporates
> both rounds of the external review (`REVIEW_2_NHANH_FUSION_2026-10-18.md` and its
> round-2 follow-up); section 9 lists what each round changed.

**Tóm tắt (vi):** Nhánh này thêm biến thể GELU-tanh (GeGLU, dùng trong Gemma 2/3/4)
cho hai kernel fused activation+FP8-quant hiện có của SiLU: static per-tensor và
per-block (128/64). Pattern trong `ActivationQuantFusionPass`, registry fusion thủ
công `maybe_fused_act_quant`, handler defunctionalization, và MLP của Gemma 2/3/4
được nối tương ứng. Mã đã viết xong, lint sạch, các kernel biên dịch được bằng
nvcc 12.0, **chưa chạy test trên GPU**. Phạm vi ở cấu hình mặc định: (1) với
checkpoint Gemma **block-FP8** (ví dụ `RedHatAI/gemma-4-31B-it-FP8-block`) pass
act-fusion chạy mặc định vì `+quant_fp8` được tự bật, nên pattern GELU mới sẽ fire;
(2) với checkpoint FP8 **static per-tensor**, đường thủ công fire; (3) với
`*-FP8-dynamic` (per-token), là loại phổ biến nhất của RedHatAI, **không đổi gì**.

## 1. Status

| Item | State |
| --- | --- |
| Base | `upstream/main` @ `371745ebd` (2026-10-18) |
| Commits | one feature commit + this handoff commit (`git log upstream/main..`); history rewritten in revision 3: author/committer `Toan Tran <toan-tq@outlook.com>`, DCO `Signed-off-by`, one `Co-authored-by` AI line, no session link. Drop the handoff commit before a PR (`git reset --hard HEAD~1` keeps the feature commit) |
| Lint | `ruff`, `ruff-format`, `clang-format`, `mypy` (oldest py), `typos`, `markdownlint-cli2`: pass |
| CUDA compile | modified `.cu` files compile with nvcc 12.0 (sm_90) against the torch 2.14 stable headers; section 7 |
| GPU tests | **not run** (no GPU in the authoring environment) |
| Benchmarks | **not run** |
| Overlap | #59467 (same author) touches 8 of the same files, but `git merge-tree` of the two heads merges **cleanly** (identical Gemma MLP hunks, disjoint edits elsewhere); revision 2 wrongly predicted a conflict. Decision in section 5: separate PR, not stacked |

## 2. Problem

`vllm/compilation/passes/fusion/act_quant_fusion.py` and
`vllm/model_executor/layers/fusion/fused_act_quant.py` only know `silu_and_mul`
(`MatcherSiluAndMul`, `(SiluAndMul, key)` registry rows). Gemma 2/3/4 use
`GeluAndMul(approximate="tanh")` (`gelu_pytorch_tanh`), so their MLP activation was
never fused with the following quantization, while Llama-style models got
`silu_and_mul_quant` / `silu_and_mul_per_block_quant` and the manual
`maybe_fused_act_quant` path.

## 3. Change summary

### C++ / CUDA

- `csrc/quantization/activation_kernels.cu`: `gelu_tanh_kernel<T>` with the same
  formula **and the same evaluation order** as the unfused kernel in
  `csrc/activation_kernels.cu` (`x_cube = f*f*f; inner = BETA*(f + KAPPA*x_cube)`),
  and the host entry `gelu_tanh_and_mul_quant`, which reuses
  `act_and_mul_quant_kernel` through `LAUNCH_ACTIVATION_GATE_KERNEL`.
- `csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`: the kernel is
  renamed `act_and_mul_per_block_quant_kernel` (review 2, B-N4: a profile must not
  show "silu" for the GELU instantiation) and gets a `bool kGeluTanh = false`
  template parameter and an fp32 `act_and_mul<kGeluTanh>` helper (same evaluation
  order); the host body becomes `act_and_mul_per_block_quant<kGeluTanh>` with two
  thin entry points (`silu_and_mul_per_block_quant` unchanged in behaviour,
  `gelu_tanh_and_mul_per_block_quant` new). `#include <cmath>` added for `M_SQRT2`.
- `csrc/ops.h`, `csrc/torch_bindings.cpp`: declarations, schemas and impls for
  `gelu_tanh_and_mul_quant(Tensor! result, Tensor input, Tensor scale)` and
  `gelu_tanh_and_mul_per_block_quant(...)` (same schema as the SiLU op).

### Python

- `vllm/_custom_ops.py`: `gelu_tanh_and_mul_per_block_quant` wrapper; the SiLU wrapper
  and it share `_act_and_mul_per_block_quant(op, ...)`.
- `vllm/compilation/passes/fusion/matcher_utils.py`: `GELU_TANH_MUL_OP` and
  `MatcherGeluTanhAndMul`. Its native path calls a `GeluAndMul(approximate="tanh")`
  instance's `forward_native`, so the traced pattern is identical to the model graph
  (including ROCm's exact-GELU fallback).
- `vllm/compilation/passes/fusion/act_quant_fusion.py`: `ActivationQuantPattern` takes
  an `act_mul_matcher` and a `fused_ops` table; `GELU_TANH_FUSED_OPS` holds the GELU
  ops and is only populated when `torch.ops._C.gelu_tanh_and_mul_quant` exists
  (`gelu_tanh_and_mul_quant_supported`), so builds that ship other kernels (XPU
  imports this pass) keep importing. The pass registers static and per-block
  patterns for both activations; on ROCm the GELU patterns are registered only when
  the `gelu_and_mul` custom op is enabled, because the native ROCm GeluAndMul
  computes the exact GELU and must not be replaced by a tanh kernel.
- `vllm/compilation/passes/utility/fix_functionalization.py`: defunctionalization
  handlers for `gelu_tanh_and_mul` (`out`) and `gelu_tanh_and_mul_quant`
  (`result`), mirroring the SiLU handlers. No handler for the per-block ops, same
  as `silu_and_mul_per_block_quant` today. The `gelu_tanh_and_mul` handler also
  runs in graphs with `+gelu_and_mul` where nothing is fused (review 2, B-N5); it
  is kept because the SiLU precedent has the same pair and nothing else depends on
  it, so it can be dropped if a reviewer prefers. The argument order was checked
  against the schemas (`out, input` / `result, input, scale`); only a GPU run of
  `test_silu_mul_quant_fusion.py` proves it.
- `vllm/model_executor/layers/fusion/fused_act_quant.py`: producer
  `_gelu_tanh_and_mul_fp8_static` and the registry row
  `(GeluAndMul, kFp8StaticTensorSym)` with a support predicate requiring
  `approximate == "tanh"`, registered only when the op exists. **No block row**: no
  block-FP8 GEMM kernel exposes an input quant key (`cutlass.py`, `flashinfer.py`,
  `pytorch.py` return `kFp8StaticTensorSym` only, AITER returns per-token), so a
  per-block manual producer could never be selected; revision 2 shipped one and it
  was dead code (review 2, B-N3). The per-block case is served by the compiler
  pattern. `_silu_and_mul_fp8_dynamic_block` is as on main.
- `vllm/model_executor/models/gemma2.py`, `gemma3.py`, `gemma4.py`: the dense MLPs call
  `maybe_fused_act_quant(self.act_fn, gate_up, self.down_proj)` like `LlamaMLP`.
  Gemma 4 MoE experts are untouched (they go through the fused-MoE kernels).

### Tests and docs

- `tests/kernels/test_fused_quant_activation.py`: static kernel test and the static
  `maybe_fused_act_quant` test parametrized over `{"silu", "gelu_tanh"}`; the
  per-block manual test stays SiLU-only (there is no GELU block producer).
- `tests/kernels/core/test_fused_silu_mul_block_quant.py`: main per-block test
  parametrized over the activation (FP8 and INT8, group 64/128, transposed scales).
- `tests/compile/passes/test_silu_mul_quant_fusion.py`: `TestGeluTanhMulFp8QuantModel`
  and `TestGeluTanhMulBlockQuantModel` added to the matrix; `+gelu_and_mul` is passed
  alongside `+silu_and_mul`; ROCm skips the native-GELU case.
- Not covered: a defunctionalization test for the new handlers
  (`tests/compile/passes/test_functionalization.py` only models SiLU).
- `docs/design/fusions.md`: one paragraph under "SiLU+Mul + Quantization".

## 4. Reach (reviewer lens)

Verified in `vllm/config/vllm.py` and the linear kernels on upstream main:

- **Pattern-matcher path.** `enable_act_fusion` is true when the `quant_fp8`
  custom op is enabled, and `has_blocked_weights()` auto-enables it for every
  checkpoint with `weight_block_size`. So for a **block-FP8 Gemma checkpoint** the
  pass runs in the default config and the new GELU per-block pattern fires.
  `RedHatAI/gemma-4-31B-it-FP8-block` exists (128x128 weight blocks, per-group-128
  activations, LLM Compressor `FP8_BLOCK`); its popularity relative to the
  `-FP8-dynamic` variant is unknown. For any other checkpoint the pass needs
  `+silu_and_mul`/`+gelu_and_mul` or `+quant_fp8` set by hand.
- **Where the block pattern actually fires (review 2, B-N1).** The pattern matches
  the fp32-scale `per_token_group_fp8_quant` op. With DeepGEMM UE8M0
  (`VLLM_USE_DEEP_GEMM_E8M0`, B200-class) the activation quant lowers to the
  packed-int32-scale op and neither the SiLU nor the GELU block pattern matches
  (TODO in `tests/compile/passes/test_fusion.py`, skip in
  `test_silu_mul_quant_fusion.py`). On SM90 with DeepGEMM and fp32 scales the
  pattern should match; unverified. On SM120 (RTX 5090) the consumer is the Triton
  or CUTLASS block GEMM with fp32 scales, so the pattern is expected to match;
  confirm `matched_count > 0` in the compile log before measuring anything.
  `RedHatAI/gemma-4-31B-it-FP8-block` (~31 GB of weights) does not fit one 32 GB
  RTX 5090 together with a KV cache: measure with TP=2, or quantize a smaller
  Gemma (Gemma 4 12B or Gemma 3 12B) with llm-compressor's `FP8_BLOCK` recipe.
- **Manual path (`maybe_fused_act_quant`)** runs in the default config only when the
  `down_proj` kernel advertises an input quant key. On CUDA that is the FP8
  **static per-tensor** scaled-mm kernels (`cutlass.py`, `flashinfer.py`,
  `pytorch.py` all return `kFp8StaticTensorSym` only) and NVFP4 (FlashInfer,
  SM100). No block-FP8 GEMM kernel exposes a key, so the `kFp8Dynamic128Sym`
  registry rows (SiLU and GELU alike) are dormant outside AITER and the mock-based
  tests. No widely used static-FP8 Gemma checkpoint was found on 2026-10-18
  (ModelOpt FP8 Gemma would qualify if one exists).
- **Per-token dynamic FP8** (`RedHatAI/gemma-3-27b-it-FP8-dynamic`,
  `RedHatAI/gemma-4-31B-it-FP8-dynamic`, `strategy: token`): **no change.** There is
  no fused act+per-token-quant kernel for any activation; #46273 (CUDA) and #44018
  (Helion) are pending. Covering that case means building on #46273 and having the
  CUTLASS kernel expose `kFp8DynamicTokenSym`.
- Whether the manual fused kernel beats inductor's own fusion for the static-FP8 case
  has not been measured for GELU; Llama adopted the same path upstream (#51415 and
  follow-ups), so the expectation is parity or a small gain.
- ROCm note: the manual path always uses the tanh kernel, while the unfused native
  ROCm `GeluAndMul` computes the exact GELU. That mirrors what the `gelu_and_mul`
  custom op already does on ROCm, but it is a behaviour difference worth a ROCm
  maintainer's opinion.

## 5. Related work / duplicate check (2026-10-18)

- #59467 (toan-tq; on 2026-10-18 OPEN, `MERGEABLE`, `REVIEW_REQUIRED`, last update
  2026-10-14): GELU-tanh + NVFP4, manual registry row, the same Gemma MLP wiring,
  and a defunctionalization handler for its NVFP4 op. `git fetch upstream
  pull/59467/head` and `git merge-tree pr-59467 <this branch>` merge cleanly: the
  three Gemma MLP hunks are byte-identical and the `fused_act_quant.py`,
  `fix_functionalization.py` and docs edits are in disjoint regions. **Decision
  (review 2, B-N2): separate PR, neither stacked on #59467 nor merged into it.**
  Stacking saves no rebase work (no conflicts either way) and would show the NVFP4
  kernel in this PR's diff until #59467 lands; merging would grow a PR that is
  already waiting for review. Let #59467 land first if it is close; otherwise open
  this one and say in the description that the Gemma MLP hunks are identical to
  #59467 and disappear on rebase.
- #46273 (archibate): fused SiLU+per-token FP8 CUDA kernel, open since June,
  unreviewed; #44018 (Helion variant). Prerequisite for the per-token case.
- #55251, #59153, #59610: ROCm AITER MoE / MXFP4 variants, not CUDA dense.
- No open CUDA PR adds GELU-tanh FP8 fused activation kernels.

## 6. How to build and test (needs a CUDA GPU)

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt
uv pip install -e . --no-build-isolation      # csrc changed
uv pip install -r requirements/test/cuda.in

pytest tests/kernels/test_fused_quant_activation.py -v
pytest tests/kernels/core/test_fused_silu_mul_block_quant.py -k "gelu_tanh" -v
pytest tests/compile/passes/test_silu_mul_quant_fusion.py -k "GeluTanh" -v
# regression
pytest tests/compile/passes/test_silu_mul_quant_fusion.py \
       tests/compile/passes/test_silu_mul_quant_manual_fusion.py \
       tests/compile/passes/test_functionalization.py
```

End-to-end (RTX 5090):

| Model | Path exercised |
| --- | --- |
| `RedHatAI/gemma-4-31B-it-FP8-block` | pattern path in the **default config** (block-FP8 auto-enables `+quant_fp8`); does not fit one 32 GB card with a KV cache: TP=2, or a self-quantized smaller block-FP8 Gemma (section 4) |
| a Gemma 3 / Gemma 4 dense checkpoint with **static** FP8, if one is found | manual path in the default config |
| `RedHatAI/gemma-4-31B-it-FP8-dynamic`, `RedHatAI/gemma-3-27b-it-FP8-dynamic` | control: per-token, expect no change |
| any Gemma + `--compilation-config '{"custom_ops":["+gelu_and_mul","+quant_fp8"]}'` | pattern path with the custom op |

Identity gate before reading numbers: the compile log must show
`activation_quant_fusion_pass` replacing patterns (`matched_count > 0`; see the
DeepGEMM caveat in section 4) and an nsys trace must show
`act_and_mul_per_block_quant_kernel<..., kGeluTanh=true>` / `gelu_tanh_and_mul_quant`
instead of the separate activation and quant kernels. Measure `vllm bench latency` (batch 1/8/32)
and `vllm bench serve` TPOT/TTFT, plus a GSM8K subset for parity.

## 7. Verification log

- Lint hooks: pass.
- CUDA compile check (no GPU; object compile only, 2026-10-18, re-run after
  revision 3's kernel rename):
    - Toolchain: Ubuntu `nvidia-cuda-toolkit` nvcc 12.0.140, host `g++-12`, torch 2.14.0
    headers (`torch/csrc/stable`, `torch/headeronly`), CUB from the same toolkit.
    - Flags mirrored from the extension build: `-std=c++20 --expt-relaxed-constexpr
    --expt-extended-lambda -DTORCH_TARGET_VERSION=0x020B000000000000ULL -DUSE_CUDA
    -DENABLE_FP8 -gencode arch=compute_90,code=sm_90 -Icsrc -I<torch include>`.
    - `csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`: compiles cleanly
    (both the SiLU and the GELU-tanh instantiations are reached from the host code).
    - `csrc/quantization/activation_kernels.cu`: nvcc 12.0 rejects the pre-existing
    DeepGEMM persistent kernel in this file (`static constexpr __nv_bfloat16`, needs a
    newer nvcc) on unmodified `upstream/main` as well, so that kernel and its host
    entry were masked with `#if 0` in a scratch copy; the remaining file, including
    `act_and_mul_quant_kernel`, `gelu_tanh_kernel` and `gelu_tanh_and_mul_quant`,
    compiles cleanly on this branch and on baseline.
    - `csrc/torch_bindings.cpp`: `g++-12 -std=c++20 -fsyntax-only` with the same
    defines passes on this branch and on baseline (it does not prove the schema
    strings are valid TorchScript signatures; only loading the extension checks that).
    - Not covered: ROCm/HIP, sm_120 codegen, runtime behaviour.
- Nothing has run on a GPU.

## 8. Review checklist / known risks

- [ ] `MatcherGeluTanhAndMul` instantiates a `GeluAndMul` module at pass-construction
      time (needs the vLLM config context, which the pass has). Confirm no warnings
      about custom-op enablement during pass init.
- [ ] Pattern count doubles (static 1 -> 2, block 16 -> 32 on CUDA); pass
      construction time unmeasured.
- [ ] Whether inductor's decomposition of `F.gelu(approximate="tanh")` matches the
      traced native pattern is only proven by `test_silu_mul_quant_fusion.py -k GeluTanh`
      on a GPU.
- [ ] Hardware on which the block pattern fires at all (B-N1): not with DeepGEMM
      UE8M0; SM90 and SM120 to be confirmed through `matched_count`.
- [ ] The static kernel test compares FP8 outputs exactly (`torch.allclose`
      defaults); a single-element GPU mismatch would need a tolerance (B-N7).
- [ ] Scope of the `gelu_tanh_and_mul` defunctionalization handler (B-N5).
- [ ] `Gemma4MLP` is also used for the dense layers of the MoE variants; the
      experts' activation is not touched here.

## 9. Revision history

### Revision 3 (review round 2)

- B-N2: overlap with #59467 measured with `git merge-tree` (clean); decision:
  separate PR, not stacked. Revision 2's "will conflict" was wrong.
- B-N3: dead per-block manual producer and registry row removed; the block case
  belongs to the compiler pattern.
- B-N4: kernel renamed `act_and_mul_per_block_quant_kernel`.
- B-N1: DeepGEMM UE8M0 / SM100 caveat and the 31B-on-32-GB problem recorded.
- B-N5: defunctionalization handler scope noted.
- G-1: commit history rewritten (author, DCO sign-off, one AI trailer, no session
  link). G-2/G-3 (GPU tests, measurements) remain open.

### Revision 2 (review round 1)

- Evaluation order of the GELU-tanh polynomial now matches the unfused kernel
  (previously `KAPPA * f * f * f`, which can differ by a few float32 ULPs; a CPU
  probe in the review showed no FP8 output differences, but exact parity with the
  unfused kernel is free).
- `torch.ops._C.gelu_tanh_and_mul_quant` is no longer accessed unguarded at import
  time (XPU imports this pass with `vllm-xpu-kernels`).
- Defunctionalization handlers added for the two new ops.
- Docs paragraph no longer claims the manual fusion covers per-group consumers.
- Reach section now records the default-config block-FP8 case and the absence of a
  known static-FP8 Gemma checkpoint.

## 10. `pr-checklist` notes (2026-10-18, revision 3)

**Verdict: not ready to open.** Blocked on GPU validation (nothing has run), on
finding hardware where the block pattern fires (section 4), and on the ordering
with #59467 (section 5). The code-quality items of the checklist are addressed.

### Design fit

- 1.1 Core components: none touched (csrc kernels and bindings, one compiler pass,
  the matcher utilities, the manual-fusion registry, one line in each Gemma MLP).
- 1.2 Replication: the static kernel reuses `act_and_mul_quant_kernel` through
  `LAUNCH_ACTIVATION_GATE_KERNEL`; the per-block kernel is parametrized on the
  activation instead of copied; `_act_and_mul_fp8_static` and
  `_act_and_mul_per_block_quant` are shared by the SiLU and GELU wrappers.
  `MatcherGeluTanhAndMul` mirrors `MatcherSiluAndMul`; a reviewer may ask for one
  matcher parametrized on the activation module instead of two classes.
- 1.3 Complexity: 310 added / 52 deleted non-test, non-doc lines in 12 files:
  "high" by the rule of thumb. Breakdown: ~90 csrc (kernel + host entries +
  bindings), ~90 in the pass (patterns parametrized on the activation), ~35
  matcher, ~40 registry and `_custom_ops`, 9 in the three model files.
  Justification: GeGLU models get the fusion SiLU models already have, and it is on
  by default for block-FP8 Gemma checkpoints. **The gain is unmeasured; expect a
  maintainer to ask for the e2e number before reviewing.**
- 1.4 Correctness: on ROCm the native `GeluAndMul` is the exact GELU, so the GELU
  patterns register there only with the custom op enabled; XPU imports the pass
  and the registry, both guard on the op existing; the CPU build has
  `gelu_tanh_and_mul` but not the quant ops, also covered by the guards.
- 1.5 Tradeoffs: pattern count doubles (static 1 -> 2, block 16 -> 32 on CUDA;
  pass construction time unmeasured); the ROCm manual path uses the tanh kernel
  where the unfused native op is the exact GELU (same as the existing custom op);
  no CPU/GPU synchronization added.
- 1.6 MRV1/MRV2: not applicable.

### Testing and validation

- 2.1 Kernel tests are parametrized over the activation and compare against
  unfused torch references; the pass test adds two model classes to the existing
  matrix; the manual-fusion test covers the static GELU row only (the only live
  one). Mocks: the existing `MockLinear*` helpers of that file.
- 2.2 Flakiness: the static kernel test compares FP8 outputs exactly, as the SiLU
  test does; the tanh polynomial uses the same evaluation order as the unfused
  kernel, but a one-element GPU mismatch would need a tolerance (B-N7).
- 2.3 CI: the tests live in files already wired into Buildkite
  (`check-test-tethering` hook passes); CI would cover H100/B200-class GPUs (B200
  with DeepGEMM UE8M0 skips the block pattern), not ROCm or SM120. **Nothing has
  run on any GPU yet; the human submitter must run section 6 before publishing.**

### Code quality

- 3.1 Comments trimmed to the guards and the ROCm exception.
- 3.2 `docs/design/fusions.md` updated.
- 3.3 Helpers with one call site: `act_and_mul<kGeluTanh>` (device helper, kept so
  the kernel body stays readable) and `_gelu_tanh_supported` (registry predicate).
- 3.4 No unrelated formatting changes (`git diff --stat` reviewed).

### Pull request contents (draft, template `.github/PULL_REQUEST_TEMPLATE.md`)

```markdown
## Overview

Add GELU-tanh+Mul counterparts of the fused SiLU+Mul FP8 activation kernels
(static per-tensor and per-block 128/64), match them in ActivationQuantFusionPass,
and wire the Gemma 2/3/4 dense MLPs through maybe_fused_act_quant. Related: #59467
(GELU-tanh + NVFP4, same Gemma MLP wiring; merges cleanly).

## Claims

- Gemma 2/3/4 graphs get the activation+FP8-quant fusion that SiLU models have; for
  block-FP8 Gemma checkpoints it is on by default.
- Same evaluation order as the unfused gelu_tanh_and_mul kernel.
- TODO: TPOT on <block-FP8 Gemma checkpoint> at batch 1/8/32, main vs PR, with the
  compile log showing matched_count > 0 and nsys showing the fused kernel.

## Validation

- TODO: pytest commands from section 6 with results.
- TODO: GSM8K parity, main vs PR.
- Compile check without GPU: nvcc 12.0 (sm_90) on the changed kernels.

## Details

- Per-block kernel parametrized on the activation (renamed
  act_and_mul_per_block_quant_kernel); static kernel reuses the existing
  activation+quant template.
- No manual per-block producer: no block-FP8 GEMM kernel exposes an input quant
  key. On ROCm the GELU patterns need the gelu_and_mul custom op (native ROCm
  GeluAndMul is the exact GELU). Not matched with DeepGEMM UE8M0 (packed scales).
- AI assistance was used (Claude); every changed line was reviewed by the
  submitter. Overlap with #59467 is limited to identical Gemma MLP hunks.
```

## 11. Value assessment and go/no-go (2026-10-18, before any measurement)

**Expected magnitude.** One fused activation per layer removes one kernel and one
read/write of the bf16 gate-up activation. With CUDA graphs the saving is mostly
the inter-kernel gap, about 2-3 us per layer: on a ~20 ms decode step that is
**about 1% TPOT or less**, and the saved bandwidth at large batch is negligible
next to the weight reads. Same class as the SiLU fusion upstream rates at 1-4%.

**Why this branch is weaker than `feat/gemma-rmsnorm-quant-fusion`.**

- Default-config reach is narrow: Gemma dense MLPs with block-FP8 checkpoints,
  on hardware where the fp32-scale block pattern matches (not B200 with DeepGEMM
  UE8M0; SM120 still to be confirmed). Per-token FP8-dynamic Gemma checkpoints,
  the most common ones, get nothing until a fused act+per-token kernel exists
  (#46273).
- The manual path (static FP8) has no known Gemma checkpoint using it, and the
  Gemma MLP wiring duplicates #59467.
- 310 non-test lines and a doubled pattern count for one model family; a reviewer
  will ask for the e2e number before reading, and ~1% is hard to defend.

**Go/no-go.**

1. Do not measure B before the measurement workflow of branch A works (same
   identity-gate discipline).
2. If measured, measure the **trimmed** version: kernels + pass patterns only;
   drop the manual registry row and the Gemma MLP wiring (leave those to #59467).
   Hardware: TP=2 for `RedHatAI/gemma-4-31B-it-FP8-block`, or a self-quantized
   smaller block-FP8 Gemma.
3. TPOT gain < ~1.5%: do not open a PR. Keep the branch as a reference for when a
   fused act+per-token-FP8 kernel lands, which would widen the reach to the
   `-FP8-dynamic` checkpoints.
4. Gain >= ~1.5% with clean parity and `matched_count > 0` on SM120: open the
   trimmed PR after #59467 lands.

## 12. Review round 3 (prediction document, 2026-10-19)

The third review round predicted outcomes before any run. For this branch the
launch accounting holds: `gelu_tanh_and_mul` plus the quant become one kernel,
so exactly one launch per layer is saved (unlike the RMSNorm branch, where the
`weight.float() + 1.0` of GemmaRMSNorm leaves the pattern and costs a kernel; see
`HANDOFF-feat-gemma-rmsnorm-quant-fusion.md` section 12). No code change follows
from that round. Compare a GPU run against these rows:

| # | Item | Prediction |
| --- | --- | --- |
| B1 | Build and first load of the extension (new schemas) | passes |
| B2 | GELU kernel tests (static exact compare, per-block) first try | pass; risk is the exact FP8 compare of the static test |
| B3 | `test_silu_mul_quant_fusion.py -k GeluTanh` | matches on SM120; skipped on H100/B200 with DeepGEMM |
| B4 | `RedHatAI/gemma-4-31B-it-FP8-block`, TP=2 on RTX 5090 | `matched_count > 0` |
| B5 | TPOT decode bs1 | +0.3% to +1% |
| B6 | GSM8K parity | within noise |

Conclusion of the round, unchanged by the check above: at most ~1% on a narrow
set of checkpoints; below any 10% bar.
