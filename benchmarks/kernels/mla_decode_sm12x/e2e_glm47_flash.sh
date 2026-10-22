#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end decode measurements for GLM-4.7-Flash on the TRITON_MLA backend.
#
# Primary target: DGX Spark / GB10 (BF16 weights, ~60 GB estimated, fit in
# 128 GB). RTX 5090 (32 GB) / RTX 4090 (24 GB): BF16 weights do not fit. Set
# MODEL to a 4-bit GLM-4.7-Flash checkpoint you have validated yourself, or run
# only the "forward" step, which uses dummy weights and EXTRA_ARGS such as
#   EXTRA_ARGS='--hf-overrides {"num_hidden_layers":8}'
# (the config may also need its per-layer lists shortened; check the log).
#
# Steps (default: latency serve nsys):
#   latency  Split-count experiment: decode TPOT vs --max-model-len. A FULL
#            decode CUDA graph replays num_kv_splits computed from
#            max_model_len (gpu_model_runner.py / model_states/default.py).
#            TPOT = (t(out=N+1) - t(out=1)) / N from `vllm bench latency`.
#   serve    `vllm serve` + `vllm bench serve`, random dataset, input 8k/32k,
#            output 256, concurrency 1 and 8.
#   nsys     Per-kernel decode time (nsys --cuda-graph-trace=node), decode-only
#            by differencing out=1 and out=N+1 runs; prints category shares.
#   forward  `vllm bench forward` per-kernel profile (no torch.compile, eager
#            split heuristic: NOT the serving split count; see README).
#
# Usage (from the repo root):
#   bash benchmarks/kernels/mla_decode_sm12x/e2e_glm47_flash.sh latency nsys
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
VLLM=${VLLM:-$REPO_ROOT/.venv/bin/vllm}
PY=${PY:-$REPO_ROOT/.venv/bin/python}
MODEL=${MODEL:-zai-org/GLM-4.7-Flash}
OUT=${OUT:-$REPO_ROOT/mla_e2e_$(date +%Y%m%d_%H%M%S)}
PORT=${PORT:-8000}
BACKEND=${BACKEND:-TRITON_MLA}
GPU_UTIL=${GPU_UTIL:-0.85}
INPUT_LEN=${INPUT_LEN:-8192}
DECODE_TOKENS=${DECODE_TOKENS:-256}
BATCHES=${BATCHES:-"1 8"}
# 9216 >= INPUT_LEN + DECODE_TOKENS + 1; 202752 is the config default.
MAX_MODEL_LENS=${MAX_MODEL_LENS:-"9216 65536 202752"}
SERVE_MAX_MODEL_LEN=${SERVE_MAX_MODEL_LEN:-65536}
NSYS_MAX_MODEL_LEN=${NSYS_MAX_MODEL_LEN:-65536}
NSYS_BATCH=${NSYS_BATCH:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
STEPS=${*:-latency serve nsys}

mkdir -p "$OUT"
for tool in "$VLLM" "$PY"; do
    [[ -x "$tool" ]] || { echo "missing $tool (see README: uv venv + install)"; exit 1; }
done
echo "results -> $OUT"
"$PY" - <<EOF | tee "$OUT/splits.txt"
import torch
from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
sms = torch.cuda.get_device_properties(0).multi_processor_count
for mml in "$MAX_MODEL_LENS $SERVE_MAX_MODEL_LEN $NSYS_MAX_MODEL_LEN".split():
    print(f"max_model_len={mml}: FULL-graph TRITON_MLA num_kv_splits="
          f"{_compute_num_kv_splits(int(mml), sms)} ({sms} SMs)")
EOF

read -r -a EXTRA <<<"$EXTRA_ARGS"
ENGINE=(--attention-backend "$BACKEND" --gpu-memory-utilization "$GPU_UTIL"
    ${EXTRA[@]+"${EXTRA[@]}"})
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

run_latency() {
    for mml in $MAX_MODEL_LENS; do
        for bs in $BATCHES; do
            for out_len in 1 $((DECODE_TOKENS + 1)); do
                tag="latency_mml${mml}_b${bs}_out${out_len}"
                "$VLLM" bench latency --model "$MODEL" "${ENGINE[@]}" \
                    --max-model-len "$mml" \
                    --input-len "$INPUT_LEN" --output-len "$out_len" \
                    --batch-size "$bs" --num-iters-warmup 2 --num-iters 5 \
                    --output-json "$OUT/$tag.json" 2>&1 | tee "$OUT/$tag.log"
            done
        done
    done
    "$PY" - "$OUT" "$DECODE_TOKENS" <<'EOF' | tee "$OUT/latency_tpot.txt"
import json, pathlib, re, sys
out, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
runs = {}
for p in out.glob("latency_mml*_b*_out*.json"):
    m = re.match(r"latency_mml(\d+)_b(\d+)_out(\d+)\.json", p.name)
    runs[tuple(map(int, m.groups()))] = json.loads(p.read_text())["avg_latency"]
print("max_model_len batch decode_TPOT_ms")
for (mml, bs, o), t in sorted(runs.items()):
    if o == 1 and (mml, bs, n + 1) in runs:
        print(mml, bs, f"{(runs[(mml, bs, n + 1)] - t) / n * 1e3:.3f}")
EOF
}

run_serve() {
    "$VLLM" serve "$MODEL" "${ENGINE[@]}" \
        --max-model-len "$SERVE_MAX_MODEL_LEN" --port "$PORT" \
        >"$OUT/serve.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 360); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died, see $OUT/serve.log"; exit 1; }
        sleep 5
    done
    for in_len in 8192 32768; do
        for conc in 1 8; do
            "$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" \
                --dataset-name random --random-input-len "$in_len" \
                --random-output-len 256 --max-concurrency "$conc" \
                --num-prompts $((conc < 2 ? 8 : conc * 4)) --ignore-eos \
                --percentile-metrics ttft,tpot,itl,e2el --save-result \
                --result-dir "$OUT" --result-filename "serve_in${in_len}_c${conc}.json" \
                2>&1 | tee "$OUT/serve_in${in_len}_c${conc}.log"
        done
    done
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

run_nsys() {
    command -v nsys >/dev/null || { echo "nsys not found (Nsight Systems CLI)"; exit 1; }
    for out_len in 1 $((DECODE_TOKENS + 1)); do
        rm -f "$OUT"/nsys_out"${out_len}"*.nsys-rep
        VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile --trace=cuda,nvtx \
            --trace-fork-before-exec=true --cuda-graph-trace=node \
            --capture-range=cudaProfilerApi --capture-range-end=repeat \
            --force-overwrite=true -o "$OUT/nsys_out${out_len}" \
            "$VLLM" bench latency --model "$MODEL" "${ENGINE[@]}" \
            --max-model-len "$NSYS_MAX_MODEL_LEN" --input-len "$INPUT_LEN" \
            --output-len "$out_len" --batch-size "$NSYS_BATCH" \
            --num-iters-warmup 1 --profile --profiler-config.profiler cuda \
            2>&1 | tee "$OUT/nsys_out${out_len}.log"
        for rep in "$OUT"/nsys_out"${out_len}"*.nsys-rep; do
            nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
                --output "${rep%.nsys-rep}" "$rep" >/dev/null
        done
    done
    "$PY" - "$OUT" "$DECODE_TOKENS" <<'EOF' | tee "$OUT/nsys_decode_share.txt"
import csv, pathlib, re, sys
from collections import defaultdict

out, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
CATS = [
    ("mla_attention", r"_fwd_grouped_kernel_stage1|_fwd_kernel_stage2|_mla_split_kernel|_mla_merge_kernel|BatchMLAPagedAttention"),
    ("mla_cache_rope", r"concat_and_cache_mla|rotary_embedding"),
    ("cat_fill", r"CatArrayBatchedCopy|FillFunctor|fill_kernel"),
    ("rmsnorm", r"rms_norm|rmsnorm|RMSNorm"),
    ("moe", r"moe|MoE|expert|topk|grouped_gemm"),
    ("gemm", r"gemm|Gemm|GEMM|cutlass|xmma|cublas|sm\d+_|ampere|gemv|splitK|Kernel2"),
    ("inductor", r"^triton_"),
]

def totals(prefix):
    acc = defaultdict(float)
    for f in out.glob(f"{prefix}*cuda_gpu_kern_sum*.csv"):
        for row in csv.DictReader(f.open()):
            name, ns = row["Name"], float(row["Total Time (ns)"])
            cat = next((c for c, rx in CATS if re.search(rx, name)), "other")
            acc[cat] += ns
    return acc

short, long = totals("nsys_out1"), totals(f"nsys_out{n + 1}")
if not short or not long:
    sys.exit("no nsys CSVs found; check the nsys logs")
per_step = {c: (long[c] - short.get(c, 0.0)) / n for c in long}
total = sum(per_step.values())
print(f"decode-only kernel time per step: {total / 1e3:.1f} us (summed, overlap ignored)")
for c, ns in sorted(per_step.items(), key=lambda kv: -kv[1]):
    print(f"{c:15s} {ns / 1e3:10.1f} us  {100 * ns / total:5.1f}%")
EOF
}

run_forward() {
    "$VLLM" bench forward --model "$MODEL" "${ENGINE[@]}" --load-format dummy \
        --max-model-len $((INPUT_LEN + 1024)) --kv-len-jitter 0 \
        --case "decode_b1=1*1@$INPUT_LEN" --case "decode_b8=8*1@$INPUT_LEN" \
        --profile-dir "$OUT/forward_prof" --output-json "$OUT/forward.json" \
        2>&1 | tee "$OUT/forward.log"
    for f in "$OUT"/forward_prof/*.rank0.kernels.json; do
        "$PY" -m vllm.benchmarks.forward.profile "$f" | tee "${f%.json}.txt"
    done
}

for step in $STEPS; do
    case "$step" in
        latency) run_latency ;;
        serve) run_serve ;;
        nsys) run_nsys ;;
        forward) run_forward ;;
        *) echo "unknown step $step"; exit 1 ;;
    esac
done
echo "done: $OUT"
