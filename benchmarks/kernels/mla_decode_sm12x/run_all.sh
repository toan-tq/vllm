#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Run the MLA decode kernel benchmarks on the current GPU. Writes one CSV per
# benchmark, a log per step and env.txt into $OUT. A failing step is recorded
# in $OUT/steps.txt and the remaining steps still run.
#
# Usage (from the repo root, after the README setup):
#   bash benchmarks/kernels/mla_decode_sm12x/run_all.sh
#   QUICK=1 bash benchmarks/kernels/mla_decode_sm12x/run_all.sh   # small sweep
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HERE="$REPO_ROOT/benchmarks/kernels/mla_decode_sm12x"
PY=${PY:-$REPO_ROOT/.venv/bin/python}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-202752}
QUICK=${QUICK:-0}
OUT=${OUT:-$REPO_ROOT/mla_kernels_$(hostname)_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
[[ -x "$PY" ]] || { echo "missing $PY (see README: uv venv + install)"; exit 1; }

{
    echo "date: $(date -u +%FT%TZ)"
    echo "git: $(git -C "$REPO_ROOT" rev-parse HEAD) ($(git -C "$REPO_ROOT" status --porcelain | wc -l) changed files)"
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm,clocks.max.mem \
        --format=csv || echo "nvidia-smi failed"
    nvcc --version | tail -n 2 || echo "nvcc: not on PATH (FlashInfer JIT needs it)"
    echo "CUDA_HOME=${CUDA_HOME:-unset} FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-unset}"
    "$PY" - <<'EOF'
import importlib.metadata as md
import torch
p = torch.cuda.get_device_properties(0)
print(f"device: {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count} "
      f"L2={getattr(p, 'L2_cache_size', 'n/a')} "
      f"smem/block(optin)={getattr(p, 'shared_memory_per_block_optin', 'n/a')} "
      f"smem/SM={getattr(p, 'shared_memory_per_multiprocessor', 'n/a')}")
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
for dist in ("triton", "flashinfer-python", "cupti-python", "vllm"):
    try:
        print(f"{dist} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"{dist} not installed")
EOF
} >"$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

: >"$OUT/steps.txt"
run_step() {
    local name=$1
    shift
    echo "=== $name: $*"
    if "$@" 2>&1 | tee "$OUT/$name.log"; then
        echo "$name OK" >>"$OUT/steps.txt"
    else
        echo "$name FAILED (see $name.log)" >>"$OUT/steps.txt"
    fi
}

if [[ "$QUICK" == 1 ]]; then
    HEADS=(20); BATCHES=(1 8); KV_LENS=(8192); PROTO_BATCHES=(1)
else
    HEADS=(16 20 64 128); BATCHES=(1 8 32); KV_LENS=(8192 32768); PROTO_BATCHES=(1 4 16)
fi

run_step proto_test "$PY" "$HERE/triton_mla_fused_merge.py" --test
BACKENDS=(triton fa2)
if grep -q "^proto_test OK" "$OUT/steps.txt"; then
    BACKENDS+=(triton_fused)
fi

run_step kernels_bf16 "$PY" "$HERE/bench_mla_decode_kernels.py" \
    --num-heads "${HEADS[@]}" --batch-sizes "${BATCHES[@]}" --kv-lens "${KV_LENS[@]}" \
    --q-lens 1 --backends "${BACKENDS[@]}" --max-model-len "$MAX_MODEL_LEN" \
    --csv "$OUT/kernels_bf16.csv"
run_step kernels_bf16_mtp "$PY" "$HERE/bench_mla_decode_kernels.py" \
    --num-heads 16 20 --batch-sizes 1 8 --kv-lens "${KV_LENS[@]}" --q-lens 2 4 \
    --backends "${BACKENDS[@]}" --max-model-len "$MAX_MODEL_LEN" \
    --csv "$OUT/kernels_bf16_mtp.csv"
# Expected to record errors on SM89/SM12x: TRITON_MLA FP8 needs 102400 B smem at
# 2 stages (vllm#60172); FlashInfer 0.7.2 FA2 FP8 KV is SM90-only.
run_step kernels_fp8 "$PY" "$HERE/bench_mla_decode_kernels.py" \
    --num-heads 20 --batch-sizes 1 8 --kv-lens 8192 --q-lens 1 --kv-dtype fp8 \
    --backends triton fa2 --max-model-len "$MAX_MODEL_LEN" --csv "$OUT/kernels_fp8.csv"

run_step chain_glm_graph "$PY" "$HERE/bench_mla_decode_layer_chain.py" \
    --preset glm47flash --batch-sizes "${BATCHES[@]:0:2}" --kv-lens "${KV_LENS[@]}" \
    --q-lens 1 2 --split-policy graph --max-model-len "$MAX_MODEL_LEN" \
    --count-kernels --csv "$OUT/chain_glm_graph.csv"
run_step chain_glm_eager "$PY" "$HERE/bench_mla_decode_layer_chain.py" \
    --preset glm47flash --batch-sizes "${BATCHES[@]:0:2}" --kv-lens "${KV_LENS[@]}" \
    --q-lens 1 --split-policy eager --variants no_memset two_stream combined \
    --csv "$OUT/chain_glm_eager.csv"
run_step chain_dsv2lite "$PY" "$HERE/bench_mla_decode_layer_chain.py" \
    --preset dsv2lite --batch-sizes "${BATCHES[@]:0:2}" --kv-lens 8192 --q-lens 1 \
    --csv "$OUT/chain_dsv2lite.csv"

if grep -q "^proto_test OK" "$OUT/steps.txt"; then
    run_step proto_bench "$PY" "$HERE/triton_mla_fused_merge.py" --bench \
        --batch-sizes "${PROTO_BATCHES[@]}" --kv-lens "${KV_LENS[@]}" \
        --max-model-len "$MAX_MODEL_LEN" --csv "$OUT/proto_bench.csv"
else
    echo "proto_bench SKIPPED (prototype tests did not pass)" >>"$OUT/steps.txt"
fi

echo
cat "$OUT/steps.txt"
echo "results in $OUT"
