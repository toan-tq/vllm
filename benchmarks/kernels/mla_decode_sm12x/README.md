# Dense MLA decode trên GPU ~100 KB smem (SM86/89/120/121): fusion + concurrent launch

Thư mục này là gói tái lập (reproduction kit) cho hướng đóng góp "dense MLA decode"
(latent cache 512 + 64 = 576) trên RTX 4090 (SM89), RTX 5090 (SM120) và DGX Spark/GB10
(SM121), với hai trục tối ưu: **kernel fusion** và **launch đồng thời/chồng lấn**
(multi-stream trong CUDA graph, PDL, merge split-KV trong cùng kernel).

**Trạng thái: chưa chạy trên GPU nào.** Mọi phát biểu về code đã đối chiếu với source
(vLLM `87b804f0`, FlashInfer wheel `0.7.2` từ PyPI, Triton `v3.8.0`). Kernel Triton được
compile offline bằng Triton 3.8.0 (ptxas đi kèm wheel) cho `sm_89/sm_120/sm_121`, không có GPU.

Nhãn dùng cho mọi con số:

- **[V]** đã kiểm chứng tĩnh (đọc code / compile offline), kèm `file:line`.
- **[M]** đo bởi người khác, kèm link.
- **[E]** ước lượng, kèm công thức và giả định.
- **[U]** chưa kiểm chứng được (spec phần cứng, hành vi runtime).

## 0. Tóm tắt

1. **Số split của TRITON_MLA bị "đóng băng" theo `max_model_len` trong FULL CUDA graph** [V].
   Khi capture, runner đặt `max_seq_len = max_model_len`
   (`vllm/v1/worker/gpu_model_runner.py:2276-2280`,
   `vllm/v1/worker/gpu/model_states/default.py:195-197`); `forward_mqa` tính
   `num_kv_splits` từ đó (`vllm/v1/attention/backends/mla/triton_mla.py:268-273`, công thức ở
   `:44-50`); grid và constexpr `NUM_KV_SPLITS` được ghi vào graph. GLM-4.7-Flash mặc định
   `max_position_embeddings=202752` → **340 split trên 5090** (170 SM), 256 trên 4090 [V: công
   thức]. Ở context 8K mỗi split chỉ ~25 token (< 1 tile `BLOCK_N=32`). Partial state
   (fp32, 513/head) ghi + đọc = **27.9 MB/layer ở B=1** so với 9.4 MB KV [E: `2·B·20·S·513·4`],
   ≤ 0.73 ms/step nếu đi DRAM trên 5090 [E], và stage2 lặp tuần tự qua cả 340 split cho mỗi
   (row, head) (`vllm/v1/attention/ops/triton_decode_attention.py:606`). Không thấy PR trùng
   (xem §5). Đây là ứng viên lớn nhất và rẻ nhất — **đo trước bằng `e2e_glm47_flash.sh latency`**
   (không cần sửa code). Hệ quả thứ hai là bộ nhớ: workspace logits được giữ sẵn theo
   `max_num_seqs · q_len · H · S · 513 · 4` B (`triton_mla.py:53-70,238-244`) → với
   `max_num_seqs=256`, H=20, S=340 là ≈ 3.6 GB trên 5090 (≈ 2.7 GB trên 4090, S=256) [E], lấy
   bớt chỗ của KV cache. Split động trên device với `max_splits` chặn theo occupancy giảm cả
   lưu lượng partial lẫn workspace này.
2. **Stage1 của TRITON_MLA chạy 1 CTA (4 warp)/SM và không pipeline tile CKV 32 KB** [V, compile
   offline Triton 3.8.0 với divisibility hints như JIT]: smem BF16 = 59,392 B (1 stage) /
   63,488 B (2 stage) → 2 CTA không vừa 102,400 B/SM; TTGIR chỉ có `async_copy` cho tile KPE
   64×32, tile CKV 512×32 vẫn `ld.global` đồng bộ. FP8 2-stage = **102,400 B > 101,376 B** — tái
   hiện đúng lỗi vllm#60172 trên cả sm_89/120/121 (số khớp với
   [#61068](https://github.com/vllm-project/vllm/pull/61068) và
   [#54013](https://github.com/vllm-project/vllm/pull/54013), đo trên Triton 3.7.x).
3. **Mỗi layer phát ~14 kernel trong khối attention** (GLM-4.7-Flash, BF16, `q_len=1`), +4 khi
   MTP verify [V, §4]. **6 kernel bỏ được không cần kernel mới** [V]: 2 fill → `torch.empty`;
   q_a/kv_a norm → `fused_q_kv_rmsnorm`; RoPE + cache-write → `concat_and_cache_mla_rope_fused`;
   cat + W_UK bmm → hấp thụ W_UK vào `q_b_proj` (F4, có đánh đổi). Với giả định 2–4 µs/kernel
   nhỏ trong graph [E, giả định] → 0.56–1.13 ms/step ở 47 layer. Chi phí thật đo bằng `bench_mla_decode_layer_chain.py`.
4. **Không có multi-stream và không có PDL trong đường MLA của DeepSeek-V2/V3/GLM** [V]. Helper
   có sẵn (`vllm/utils/multi_stream_utils.py:16-63`, dùng ở
   `vllm/models/deepseek_v41/attention.py:889-895`). PDL dùng được trên SM120/121 (Triton 3.8
   `launch_pdl` + `tl.extra.cuda.gdc_wait`; wrapper của chúng tôi compile được) và **không thể
   trên SM89** (ptxas: "Instruction 'griddepcontrol' requires .target sm_90 or higher") [V].
5. **FlashInfer FA2 MLA trên ~100 KB smem**: Q64/KV16/1 stage, QK tính trùng ở 2 warpgroup, warp
   chỉ chứa hàng padding vẫn chạy MMA → với 20 head làm **4.9×** công MMA có ích; cường độ
   executed 185 FLOP/B > ridge 5090 ≈ 117 FLOP/B [E; spec U]. **vLLM hiện không có backend
   dense nào dùng `BatchMLAPagedAttentionWrapper`** (chỉ sparse SM90:
   `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py`), nên cải tiến kernel
   FlashInfer cần thêm một backend vLLM mới thì mới đến được E2E.
6. **Thứ tự đề xuất**: (1) đo (Exp. A + nsys); (2) split count an toàn với graph cho
   TRITON_MLA; (3) gói cắt launch (fill, cat, norm, hoist row expansion) + two-stream + PDL;
   (4) kernel FA2 BF16 small-Q cho ~100 KB + backend vLLM. Mục tiêu >8% E2E chỉ đạt được khi
   attention chiếm ≥8%/(1−1/s) thời gian decode (Amdahl; s = speedup attention) — ở 8K/B=1
   phần KV lý tưởng chỉ 5.8% [E], ở 32K/B=1 là 19.9% [E] (Phụ lục).

## 1. Cài đặt môi trường (theo `AGENTS.md`)

```bash
git clone <fork-của-bạn>/vllm.git && cd vllm
git checkout <nhánh-chứa-thư-mục-này>
curl -LsSf https://astral.sh/uv/install.sh | sh        # nếu chưa có uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt && pre-commit install
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pandas "cupti-python>=13"               # bench + CUPTI timing
```

- Không dùng `python3`/`pip` hệ thống; mọi lệnh qua `.venv/bin/python` hoặc `uv`.
- vLLM pin `torch==2.14.0`, `flashinfer-python==0.7.2` (`requirements/cuda.txt`). Theo metadata
  PyPI, torch 2.14.0 kéo `triton~=3.8.0` và `cuda-toolkit==13.0.3` [V] → driver phải hỗ trợ
  CUDA 13 [U: thường là R580+].
- **FlashInfer JIT cần `nvcc`** (CUDA toolkit 13.x khớp torch cu13) trong `PATH` và `CUDA_HOME`;
  lần `plan()` đầu sẽ JIT-compile module `batch_mla`. Kiến trúc tự dò; có thể ép:
  `export FLASHINFER_CUDA_ARCH_LIST="8.9"` (4090), `"12.0"` (5090, thành `12.0f` khi CUDA ≥12.9),
  `"12.1"` (GB10, thành `12.1a`) (`flashinfer/compilation_context.py:81-125` trong wheel 0.7.2).
- **CUPTI**: `bench_gpu_time_with_cupti` cần `cupti-python>=13`; thiếu thì FlashInfer cảnh báo
  "CUPTI is not installed ..." và rơi về CUDA event/graph timing
  (`flashinfer/testing/utils.py:1042-1094`).
- GB10 là aarch64: nếu `VLLM_USE_PRECOMPILED=1` không có wheel phù hợp thì build từ source theo
  `docs/contributing/incremental_build.md`; test deps dùng `requirements/test/cuda.in`.
- Nsight Systems CLI (`nsys`) cho bước `nsys` của `e2e_glm47_flash.sh`.

## 2. Các script

| File | Đo gì | Ghi chú |
| --- | --- | --- |
| `bench_mla_decode_kernels.py` | Kernel attention thuần: Triton (`decode_attention_fwd`) với split theo `graph` (từ `--max-model-len`) và `eager` (từ KV thật), FlashInfer FA2, prototype | Bản sửa lỗi của `benchmarks/kernels/benchmark_mla_decode_backends.py` (file cũ, đã xoá) |
| `bench_mla_decode_layer_chain.py` | Cả khối attention một layer (A-proj → norm → q_b → RoPE → cache write → W_UK bmm → cat → fill → stage1/2 → W_UV bmm → o_proj) trong **một CUDA graph**, cold L2, CUPTI; các ablation §5/§6 | Mỗi biến thể được so số học với `base` trước khi đo |
| `triton_mla_fused_merge.py` | Prototype: 1 launch, split động theo `seq_lens` trên device, q hai con trỏ, ghi đủ o/lse, tuỳ chọn PDL. `--test`, `--bench` | **UNTESTED ON GPU** |
| `e2e_glm47_flash.sh` | `latency` (TPOT theo `--max-model-len`), `serve` (8k/32k × c1/c8), `nsys` (tỷ trọng kernel decode), `forward` | GB10 chạy được BF16; 5090/4090 xem §2.1 |
| `run_all.sh` | Chạy tất cả bench kernel, ghi CSV + `env.txt` + `steps.txt` | `QUICK=1` cho sweep nhỏ |

### 2.1 Lệnh theo GPU

```bash
# Chung (từ repo root, venv đã activate)
bash benchmarks/kernels/mla_decode_sm12x/run_all.sh            # sweep đầy đủ
QUICK=1 bash benchmarks/kernels/mla_decode_sm12x/run_all.sh    # sweep nhỏ để thử trước

# RTX 4090 (SM89): PDL tự bỏ qua; FP8 Triton và FA2 FP8 dự kiến lỗi (§2.2)
FLASHINFER_CUDA_ARCH_LIST="8.9" bash benchmarks/kernels/mla_decode_sm12x/run_all.sh

# RTX 5090 (SM120)
FLASHINFER_CUDA_ARCH_LIST="12.0" bash benchmarks/kernels/mla_decode_sm12x/run_all.sh

# GB10 (SM121) + E2E BF16 (trọng số ~60 GB [E: ~29.9B param × 2 B, từ config mặc định] vừa 128 GB)
FLASHINFER_CUDA_ARCH_LIST="12.1" bash benchmarks/kernels/mla_decode_sm12x/run_all.sh
bash benchmarks/kernels/mla_decode_sm12x/e2e_glm47_flash.sh latency nsys serve
```

E2E trên 5090 (32 GB) / 4090 (24 GB): BF16 không vừa. Hai lựa chọn:
`MODEL=<checkpoint 4-bit GLM-4.7-Flash bạn tự chọn và đã kiểm tra> bash .../e2e_glm47_flash.sh latency nsys`,
hoặc chỉ đo tỷ trọng kernel với trọng số giả:
`EXTRA_ARGS='--hf-overrides {"num_hidden_layers":8}' bash .../e2e_glm47_flash.sh forward`
(có thể phải rút ngắn thêm các list theo layer trong config; `vllm bench forward` không dùng
torch.compile và dùng split theo KV thật, nên **không** tái hiện hiện tượng (1) ở §0).

Thử nghiệm A (không sửa code, quan trọng nhất): `e2e_glm47_flash.sh latency` chạy
`vllm bench latency` với `--max-model-len 9216 / 65536 / 202752` (split 32 / 128 / 340 trên
5090 [V: công thức]) và in `latency_tpot.txt` (TPOT = (t(out=257) − t(out=1))/256).
Trên 5090/4090, `max_model_len=202752` có thể không khởi động được (KV cho một chuỗi 202752
token ≈ 11 GB + workspace ở trên [E]); khi đó dùng
`MAX_MODEL_LENS="9216 65536 131072"` — so sánh vẫn hợp lệ vì chỉ có số split là khác nhau.

### 2.2 Lỗi dự kiến (không phải bug của script)

| Lỗi | Ở đâu | Lý do |
| --- | --- | --- |
| `out of resource: shared memory, Required: 102400, Hardware limit: 101376` | Triton FP8 (`--kv-dtype fp8`) trên SM89/120/121 | vllm#60172; compile offline Triton 3.8.0 tái hiện 102,400 B [V] |
| `FP8 kv_data_type for MLA requires an SM90 (Hopper) device` | FA2 FP8 | flashinfer 0.7.2 `_fa_common.py:153-157`; PR #6987 (open) chỉ thêm SM89 single-token |
| `ninja`/`nvcc` not found khi `plan()` | FA2 | JIT cần CUDA toolkit |
| Cảnh báo "CUPTI is not installed" | mọi bench | thiếu `cupti-python>=13` → timing kém chính xác |
| `SKIP pdl variant: PDL needs SM90+` | 4090 | đúng thiết kế (griddepcontrol cần sm_90) |
| Prototype `FAIL ...` | `triton_mla_fused_merge.py --test` | prototype chưa chạy GPU; `run_all.sh` khi đó bỏ qua bench prototype |
| `kernels = -1` | `--count-kernels` | torch.profiler (kineto) có thể xung đột CUPTI sau khi FlashInfer đã dùng; chạy riêng nếu cần |
| OOM ở H=128, B=32, split `graph` | kernel bench | buffer logits `B·H·S·513·4` ≈ 2.9 GB ở S=340 [E] |

### 2.3 Đọc kết quả

- `kernels_*.csv`: `arm` = `triton:graph` / `triton:eager` / `fa2` / `triton_fused`; `splits`;
  `us` (median CUPTI); `graph` (đo trong CUDA graph hay không); `gbps` = byte có ích (KV đọc một
  lần + q + o) / thời gian; `pct_copy_bw` so với copy D2D đo cùng lúc; `tflops` = FLOP có ích
  `2·rows·H·kv·(576+512)`; `rel_err`/`ok` so với reference FP32 (ngưỡng 2e-2). **`triton:graph` vs
  `triton:eager` = giá của split đóng băng; `fa2` vs `triton:eager` = chất lượng kernel.**
- `chain_*.csv`: `us` = cả khối một layer; `delta_us` so với `base`; `step_delta_ms` =
  `delta_us × số layer` (âm là tiết kiệm); `ok`/`rel_err` (biến thể bitwise phải bằng hệt);
  `kernels` (khi `--count-kernels`: số GPU op của một lần chạy eager dưới torch.profiler, gồm cả
  memset) và danh sách op của `base`/`combined` cuối log.
- `proto_bench.csv`: `vllm forward_mqa graph|eager` (cat + 2 fill + stage1 + stage2) vs prototype
  `fused` / `split+merge` (± `pdl`), `kv_gbps` = KV byte / thời gian.
- E2E: `latency_tpot.txt`; `serve_in*_c*.json` (`median_tpot_ms`, `median_itl_ms`);
  `nsys_decode_share.txt` (thời gian kernel mỗi bước decode theo nhóm, sai phân out=257 vs out=1).

### 2.4 Tiêu chí quyết định / dừng

- **G1 (tuần 1, Exp. A)**: TPOT ở 8K, B∈{1,8} chênh < 3% giữa `max_model_len` 9216 và 202752 trên
  cả 5090 và GB10 → bác giả thuyết (1), bỏ PR split.
- **G2 (tuần 1, nsys)**: `mla_attention + mla_cache_rope + cat_fill` < 10% bước decode ở 32K/B=1
  và B=8 → >8% E2E không đạt được chỉ bằng attention (Amdahl); chuyển trọng tâm.
- **G3 (tuần 3, layer chain)**: biến thể `combined` tiết kiệm < 1.5% bước decode đo được → không
  gửi PR fusion lẻ (quy tắc "no busywork" của `AGENTS.md`), chỉ gộp vào PR lớn.
- **G4 (tuần 5)**: kernel FA2 small-Q phải nhanh hơn Triton (sau PR split) ≥1.3× ở H=20,
  B∈{1,8}, KV∈{8K,32K} trên 5090 và GB10, nếu không thì dừng nhánh FlashInfer.
- **Mục tiêu**: TPOT giảm >8% ở (32K, B=1) và (8K, B=8) trên ≥2/3 GPU; GSM8K lệch ≤ nhiễu.

## 3. Kiểm chứng các claim trước đó

| # | Claim | Kết luận | Bằng chứng / sửa |
| --- | --- | --- | --- |
| 1 | SM12x chỉ có `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]` → dense luôn TRITON_MLA | **ĐÚNG** | `vllm/platforms/cuda.py:134-138`. SM8x (nhánh `else`, `:139-155`): FLASH_ATTN_MLA chỉ `major==9` (`flashattn_mla.py:69-70`), FLASHMLA `9,10` (`flashmla.py:73-74`), FLASHINFER_MLA `10` → thực tế cũng TRITON_MLA |
| 2 | FLASHINFER_MLA chỉ `major==10` | **ĐÚNG** | `flashinfer_mla.py:220-221`; thêm: backend này gọi `trtllm_batch_decode_with_kv_cache_mla` (`:8`), không phải FA2 wrapper |
| 3 | BLOCK=32, BLOCK_H=16, stages=2, warps=4, grid như mô tả; 20 head → 2 head-block cùng stream KV; FP8 vượt smem | **ĐÚNG** | `triton_decode_attention.py:505,512,514-518,521,565`; head-block 1 chỉ 4/16 hàng hợp lệ (`:319-322`). FP8 2-stage 102,400 B > 101,376 B [V compile] trên sm_89/120/121 (không chỉ sm_120); BF16 63,488 B |
| 4 | `repeat_interleave` cho MTP; `torch.cat`; `o`/`lse` = `torch.zeros` mỗi lần; có thể `empty` | **ĐÚNG** | `triton_mla.py:307-324`, `:256-257`, `:262-265`. Stage2 ghi **mọi** `o[b,h,0:512]` và `lse[b,h]` với mọi (row, head) trong grid (`triton_decode_attention.py:626-638`; `BLOCK_DV=Lv=512`) → zeros bị ghi đè, `empty` tương đương. Hàng padding `seq_len=0` cho 0/0 = **NaN** dù zeros hay empty (`:599-626`); PR [#57094](https://github.com/vllm-project/vllm/pull/57094) đổi thành 0. Với DCP, `lse=-inf` × NaN vẫn ra NaN — vấn đề có sẵn |
| 5 | `DISPATCH_SMEM_CONFIG` [92672,147968) → stages=1, KV16, QK_SHARD=false; Q=64 cố định; QK trùng; warp padding vẫn MMA; 1 stage tuần tự | **ĐÚNG** | `mla.cuh:1216-1237`, `:1266`; `scheduler.cuh:1655` (`cta_tile_q=64`); QK non-shard: cả 2 warpgroup lặp mọi `NUM_MMA_KV` (`mla.cuh:419-435`, chú thích `:83-85`); `load_q` chỉ zero-fill (`:187-198`), không warp nào bỏ MMA; vòng lặp: `wait_group<0>` → compute → mới `load_kv` tile kế (`:1098-1132`). Thêm: KV16 chỉ warpgroup 0 nạp (`:223-262`). SharedStorage BF16 Q64/KV16/1 stage = **92,704 B** (2 stage 111,136 B) [V: biên dịch struct bằng g++]; khớp `_generated_fa_shared_bytes` (wheel `_fa_common.py:219-252`). FP8 bị ép KV32 (`mla.cuh:1262`) → 129,536 B |
| 6 | Roofline | **MỘT PHẦN** | Số học đúng: QK 2·576=1152, PV 2·512=1024 FLOP/(row·token); 1152 B/token BF16; 1.89 FLOP/B/row; H=20: 37.8; FA2 executed 64·(2·1152+1024)=212,992 FLOP/token = 184.9 FLOP/B (gấp 4.9× có ích). Spec **[U]**: 5090 209.5 TFLOPS FP16/BF16 FP32-acc dense chỉ từ nguồn thứ cấp và mơ hồ ([diễn đàn NVIDIA](https://forums.developer.nvidia.com/t/rtx-5090-specs-fp16-tensor-tflops-is-ambiguous/351063)); 1.792 TB/s = 28 Gbps·512 bit/8; ridge ≈ 117. 4090: ≈165 TFLOPS (số của brief, chưa kiểm) / 1.008 TB/s (21 Gbps·384 bit/8) → ridge ≈ 164 < 185 → cũng compute-bound theo công executed. GB10: 273 GB/s [U], TFLOPS không rõ → gần như chắc chắn memory-bound |
| 7 | Auto SM12x: fa2 → xqa → cutile; XQA MLA cần đúng 128 head | **ĐÚNG** | wheel 0.7.2 `_auto_policy.py:247-249`; `xqa_backend.py:117-120` (+ q/kv cùng dtype, packed, non-causal, page ∈ {16,32,64,128}) |
| 8 | PR #6987: SM89 + FP8 + single-token; SM89_FP8_Q32/COMPACT; 1 stage; prefetch FP8 thô; BF16 & SM12x không đụng | **ĐÚNG (với lưu ý)** | `git diff 595bc57 pr6987`: dispatch riêng khi `major==8 && minor==9` + FP8 + 512/64; Python từ chối `q_len>1` ("SM89 FP8 MLA currently supports single-token decode only"); cờ `SM89_FP8_Q32`, `SM89_FP8_COMPACT`, `SM89_FP8_ADAPTED`, `PREFETCH_RAW_KV`, `OMIT_WG_STATS`; COMPACT bỏ QK trùng (chỉ warpgroup 0 tính, trao điểm qua smem KPE đã dùng xong), Q32 dùng 4 output group. **Lưu ý**: đường BF16 không đổi về dispatch nhưng code chung bị refactor (`NUM_MMA_D_O`, `WARPS_PER_GROUP` trong `load_q/load_kv/compute_mla_pv/write_o`, ghi LSE trong `DevicePersistentMergeStates`, chữ ký `MLAPlan(..., tile_q, chunk_quantum)`) → xung đột với mọi việc BF16. #4906: SM90 swap-AB + merge song song dùng chung với FA2 + `MLAPlan(min_cta_tile_q, graph_num_blks_x, graph_cta_tile_q)`, `MLAPlanInfo.cta_tile_q`, `MLASelectCtaTileQ` → xung đột trực tiếp với #6987 ở `MLAPlan` và `DevicePersistentMergeStates` |
| 9 | #6308: FA2 BF16 B16/KV32K/H20 1254 µs trên RTX PRO 5000, ~36% của 1.34 TB/s | **MỘT PHẦN** | [M] commit `30a74f3` của [#6308](https://github.com/flashinfer-ai/flashinfer/pull/6308): "B16/KV32K decode falling from 1254.26 us BF16 to 415.74 us native FP8", RTX PRO 5000 **72GB**, FlashInfer 0.6.15.post1, shape GLM-4.7-Flash; **không ghi số head và không nêu tên backend**. Byte: 16·32768·1152 = 604 MB ✓ → 482 GB/s = 35.8% của 1,344 GB/s ([trang NVIDIA](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000), qua kết quả tìm kiếm) |
| 10 | Script benchmark cũ | **CÓ LỖI** | xem dưới |

**Lỗi trong `benchmark_mla_decode_backends.py` (đã sửa trong `bench_mla_decode_kernels.py`):**

1. Split của Triton tính từ KV thật → không giống FULL graph khi serving (từ `max_model_len`), bỏ
   sót hiện tượng lớn nhất. Nay có `--split-policies graph eager` + `--max-model-len`.
2. `--page-size 64` mặc định, trong khi vLLM mặc định block 16 (`vllm/config/cache.py:182`).
3. FA2 FP8: không truyền `ckv_scale`/`kpe_scale` (bắt buộc khi plan chọn `kv-per-tensor`,
   `_wrapper.py` plan + `_fa_common.py` run) và FP8 FA2 chỉ chạy SM90 ở 0.7.2; nhánh "retry với
   ckv/kpe contiguous" chẩn đoán sai (lỗi là ở `plan`, không phải layout) và nhân đôi bộ nhớ.
4. Dùng API đã deprecated (plan phẳng + positional, `run(q_nope=..., ckv_cache=...)`) và copy
   `.contiguous()`; nay dùng `metadata=MLAPlanMetadata.csr(...)` + `run(query=q, kv_cache=cache)`
   packed → FA2 tự tách view không copy (wheel `mla/_batch_mla/_contracts.py:388-437`).
5. Fallback graph bằng `try/except` quanh `bench_gpu_time_with_cupti(use_cuda_graph=True)` có thể
   để stream ở trạng thái capture hỏng; nay probe capture riêng trước.
6. Không có ngưỡng đúng/sai (`rel_err` in ra nhưng vẫn coi timing hợp lệ); nay có `ok`.
7. Reference FP8 index trực tiếp tensor `float8` (`cache[block_table[b]]`); nay qua view `uint8`.
8. Scale giả định qk_head 192; GLM-4.7-Flash là 256 (`--qk-head-dim`, chỉ ảnh hưởng số học).
9. Ví dụ dùng `python` trần (trái `AGENTS.md`); không in version môi trường; `--backends` không
   validate.

Ghi chú về FA2 trong CUDA graph mà không `use_cuda_graph=True`: `run()` chỉ đọc `plan_info` và
workspace đã ghi sẵn, không đồng bộ host, không cấp phát khi có `out` → capture được nếu không
`plan()` lại [V: `_fa_common.py:666` `_run_generated_fa`]; kernel dùng `cudaLaunchCooperativeKernel`
(`mla.cuh:1274`) — capture cooperative launch là [U] cho tới khi chạy thật (script probe trước).
`bench_gpu_time_with_cupti`: mỗi lần lặp lấy `max(end) − min(start)` của mọi kernel/memset được
khởi chạy trong cửa sổ CPU của lần đó (`testing/utils.py:1287-1316`) → với graph nhiều kernel nó
đo cả khoảng trống giữa kernel (đúng thứ ta cần); flush L2 nằm ngoài cửa sổ (`:1246-1251`).

## 4. (A) Chuỗi kernel decode mỗi layer — GLM-4.7-Flash, TRITON_MLA, BF16

Chiều [V]: transformers 5.20.2 `Glm4MoeLiteConfig` mặc định: 20 head, `qk_nope=192`, `rope=64`,
`v_head=256`, `kv_lora=512`, `q_lora=768`, `hidden=2048`, 47 layer, `max_position_embeddings=202752`
(`config.json` thật trên HF chưa đọc được [U]). Model dùng lại `DeepseekV2MLAAttention`
(`vllm/model_executor/models/glm4_moe_lite.py:59-61,104`). Trên CUDA, attention là custom op mờ
(`vllm/platforms/cuda.py:595-596` → `vllm/model_executor/layers/attention/mla_attention.py:878-896`).
Viết tắt trong bảng: `mla.py` = `vllm/model_executor/layers/mla.py`, `triton_mla.py` =
`vllm/v1/attention/backends/mla/triton_mla.py`, `triton_decode_attention.py` =
`vllm/v1/attention/ops/triton_decode_attention.py`.

| # | Kernel | Phát từ | Đã fuse? | Aux stream / PDL |
| --- | --- | --- | --- | --- |
| 1 | `fused_qkv_a_proj` GEMM [T,2048]→[T,1344] | `layers/mla.py:183` | q_a+kv_a gộp. GEMM min-latency `dsv3_fused_a_gemm` (có PDL) chỉ cho 2112×7168 trên SM90/100 (`deepseek_v2.py:972-982`) | không / không |
| 2 | `q_a_layernorm` RMSNorm [T,768] | `mla.py:190` | Không; `fused_q_kv_rmsnorm` có sẵn nhưng opt-in (`mla.py:110-112,204-214`), chỉ GLM5-Next bật (`models/glm5next/common/attention.py:568`) | không / không |
| 3 | `kv_a_layernorm` RMSNorm [T,512] | `mla.py:217` | như trên | không / không |
| 4 | `q_b_proj` GEMM [T,768]→[T,5120] | `mla.py:221` | không | không / không |
| 5 | RoPE in-place (q_pe, k_pe) | `mla.py:228` → `rotary_embedding/base.py:221-251` | Pass `fuse_rope_kvcache_cat_mla` (→ `concat_and_cache_mla_rope_fused`, `csrc/cache_kernels_fused.cu:217`) **tắt mặc định với BF16** vì `unified_mla_kv_cache_update` là splitting op (`config/vllm.py:221-226`, `config/compilation.py:1184-1213,1285-1305`) | không / không |
| 6 | `concat_and_cache_mla` | `mla_attention.py:879,1436-1462` → `v1/attention/backend.py:1121-1128` | như #5 | không / không |
| 7 | W_UK `bmm` (N,T,192)×(N,192,512) ghi thẳng (T,N,512) | `mla_attention.py:1097-1103` | đã tránh transpose copy | không / không |
| 8 | `torch.cat(ql_nope, q_pe)` | `triton_mla.py:256-257` | không (PR #51861 mở) | không / không |
| 9–10 | `torch.zeros(o)`, `torch.zeros(lse)` | `triton_mla.py:262-265` | không | không / không |
| (11–14) | chỉ `q_len>1`: `repeat_interleave`, `arange`, `add`, `clamp` | `triton_mla.py:307-321` | lặp lại **mỗi layer** | không / không |
| 15 | stage1 grid `(T, 2, S)` | `triton_decode_attention.py:514-571` | — | không / không |
| 16 | stage2 grid `(T, 20)` | `:662-680` | không (#52019 chỉ khi 1 split) | không / không |
| 17 | W_UV `bmm` vào output | `mla_attention.py:1163,1401-1433` | không | không / không |
| 18 | `o_proj` GEMM [T,5120]→[T,2048] | `mla.py:255` | không | không / không |

- **Đếm**: 14 kernel/layer (`q_len=1`), 18 khi MTP verify → 658 / 846 kernel mỗi bước cho 47 layer
  [V: đếm theo code]. DeepSeek-V2-Lite (không q_lora): `kv_a_proj_with_mqa` + `q_proj` (2 GEMM
  cùng input, `mla.py:199-221`), không có q_a norm/q_b → 13. DeepSeek-V3/Kimi trên SM12x: như
  GLM (GEMM min-latency không bật ngoài SM90/100). Dưới torch.compile mặc định, RMSNorm/RoPE có
  thể thành kernel Inductor (tên `triton_*`, có thể gộp) → số đếm thực tế xác nhận bằng
  `--count-kernels` / nsys.
- **Giả định chi phí cố định**: 2–4 µs cho mỗi kernel nhỏ phụ thuộc nối tiếp trong graph
  (launch, ramp, tail) [E, giả định]; `bench_mla_decode_layer_chain.py` đo trực tiếp qua `delta_us`.

## 5. (B) Cơ hội fusion

Ước lượng dùng: 47 layer, 2–4 µs/kernel nhỏ [giả định], 5090 1.792 TB/s, GB10 273 GB/s [U].
Duplicate-check: tìm tiêu đề PR vLLM "triton mla" (3 trang), "num_kv_splits",
"fused_q_kv_rmsnorm", "mla pdl", "kv_a_proj_with_mqa", "flashinfer mla sm120", issue
"TRITON_MLA slow decode", "triton mla cudagraph max_model_len"; FlashInfer "mla sm120", "mla fa2"
(2 trang), "mla pdl" — ngày 2026-10-22.

| ID | Fusion | Vị trí | Ước lượng tiết kiệm | Rủi ro | Trùng lặp |
| --- | --- | --- | --- | --- | --- |
| F1 | **Split count an toàn với graph**: `n_split = clamp(cdiv(seq_len, MIN_TOK), 1, max_splits)` tính trên device; `max_splits` chặn theo occupancy (`B` cố định cho mỗi graph); stage2 chỉ đọc split hiệu lực | `triton_mla.py:268-273`, `triton_decode_attention.py:352-354,606-609` | ≤ 0.73 ms/step (5090, 8K, B=1) nếu partial đi DRAM; 0.98 ms trên 4090 (S=256) [E: `2·B·20·S·513·4·47/BW`] + bỏ vòng stage2 340 lần | Grid vẫn = max (CTA thừa thoát sớm); số học giữ nguyên | Không trùng. Gần nhất: [#40750](https://github.com/vllm-project/vllm/pull/40750) (closed, không merge; "batch-aware splits" + bảng tuning sm120/FP8 cho Kimi trên RTX PRO 6000; tác giả khuyên gửi PR nhỏ, tập trung) và [#40614](https://github.com/vllm-project/vllm/pull/40614) (closed draft). Liên quan: [#50629](https://github.com/vllm-project/vllm/pull/50629) (draft, ROCm, cap occupancy; [M] +21.7% throughput GLM-4.7-Flash TP4 W7900 8k/1k), [#42885](https://github.com/vllm-project/vllm/pull/42885) (bật FULL graph, "capture uses the worst-case max_seq_len"), [#33529](https://github.com/vllm-project/vllm/pull/33529). Hệ quả suy ra: fast path của [#52019](https://github.com/vllm-project/vllm/pull/52019) chỉ kích hoạt trong FULL graph khi `max_model_len < 1024` [suy luận, cần xác nhận] |
| F2 | `o`/`lse` → `torch.empty` | `triton_mla.py:262-265` | 2 kernel/layer → 0.19–0.38 ms/step [E] | Giữ bất biến "stage2 ghi mọi phần tử"; phối hợp ngữ nghĩa hàng rỗng với #57094/#52019 | Không thấy PR bỏ zeros |
| F3 | Bỏ `cat`: stage1 nhận `q_pe` qua con trỏ riêng | `triton_mla.py:256`, `triton_decode_attention.py:331-350` | 1 kernel/layer; [M] #51861: `CatArrayBatchedCopy` 10.6 ms→0 trên 128 bước, E2E +0.46% (DS-V2-Lite, B=64, RTX PRO 6000) | thấp | **TRÙNG** [#51861](https://github.com/vllm-project/vllm/pull/51861) (draft, needs-rebase) → review/hỗ trợ, không làm lại |
| F4 | Hấp thụ W_UK vào trọng số q (1 GEMM ra query 576) | `mla_attention.py:1083-1103,1343-1347` | −2 kernel/layer (bmm + cat) nhưng đọc thêm 5.9 MB/layer = +0.15 ms/step (5090), **+1.02 ms/step (GB10)** [E: `768·20·(576−256)·2 − 20·192·512·2`]; +277 MB VRAM | Số học khác (bf16 làm tròn tích W); có thể **lỗ trên GB10** | Không thấy PR. Tiền lệ: Kimi-K3 specialized decode gộp RMSNorm+Q_B+W_UK+KV insert có PDL (`models/kimi_k3/nvidia/specialized_decode/cutedsl/mla/query.py:1-4`, chỉ SM100/103) |
| F5 | Gộp q_a/kv_a RMSNorm (`fused_q_kv_rmsnorm`, Triton, có PDL) cho `DeepseekV2MLAAttention` | `deepseek_v2.py:1215-1238` (không truyền `fuse_qkv_rmsnorm`) | 1 kernel/layer → 0.09–0.19 ms/step [E] | Inductor có thể đã gộp; đo trước | ROCm có ([#39242](https://github.com/vllm-project/vllm/pull/39242), [#52494](https://github.com/vllm-project/vllm/pull/52494)); CUDA/DeepSeek-V2: không thấy |
| F6 | RoPE + cache write (`concat_and_cache_mla_rope_fused`) | pass có sẵn, tắt mặc định (xem §4 #5) | 1 kernel/layer [E] | RoPE trong kernel gộp tính bằng dtype của q (`csrc/cache_kernels_fused.cu:59-80`) → có thể lệch bit so với op RoPE riêng; bật cần `use_inductor_graph_partition` | Tính năng có sẵn — thử `-cc.use_inductor_graph_partition=true` ở E2E |
| F7 | Hoist row expansion (`q_len>1`) lên metadata builder | `triton_mla.py:307-324` | 4 kernel/layer → 0.38–0.75 ms mỗi bước verify [E] | Cần buffer cố định địa chỉ cho graph | Đã được gợi ý trong review [#51065](https://github.com/vllm-project/vllm/pull/51065) (bị hoãn); không có PR |
| F8 | Gộp stage2 vào stage1 (CTA cuối merge, counter tự reset) | prototype `triton_mla_fused_merge.py` | 1 kernel/layer; chỉ lời khi `n_split` nhỏ vì 1 SM đọc `n_split·16·513·4` B [E] | Thứ tự bộ nhớ (fence + atomic acq_rel; PTX đã kiểm) | Trùng một phần #52019 (chỉ 1 split) |
| F9 | 1 CTA cho cả 20 head (`BLOCK_H=32`, 8 warp) | `triton_decode_attention.py:512-518` | bỏ CTA head-block 2 (KV đọc lại, 12/16 hàng padding) | smem 77,824 B, 187 reg, không spill [V compile] | Không thấy PR |
| F10 | MTP: gói `q_len×H` hàng vào 1 CTA, không nhân bản hàng | `triton_mla.py:307-324` | KV đọc `q_len` lần → 1 lần (L2 có thể che một phần) [E] | mask nhân quả theo hàng | #51065 (merged) chọn flatten; [#52218](https://github.com/vllm-project/vllm/pull/52218) closed; không có PR gói |
| F11 | Partial state BF16 (hiện là FP32) | `triton_decode_attention.py:442-466` | −50% lưu lượng partial [E] | sai số (FlashInfer đã lưu partial bằng `DTypeO`: `mla.cuh:769-770,861-863`) | Không thấy PR |
| F12 | W_UV vào epilogue attention/merge | `mla_attention.py:1401-1433` | 1 kernel/layer; merge phải đọc 5.2 MB W_UV/layer [E] | phức tạp, chỉ đáng khi merge phân tán | Tiền lệ SM100: Kimi-K3 `mla/value.py` |

## 6. (C) Launch đồng thời / chồng lấn

| ID | Cơ chế | Vị trí | Ước lượng | Rủi ro | Trùng lặp |
| --- | --- | --- | --- | --- | --- |
| C1 | **Two-stream**: KV path (kv_a norm → k RoPE → cache write) song song q path (q norm → q_b → q RoPE → W_UK bmm), join trước attention, trong cùng CUDA graph | `layers/mla.py:183-250` với `maybe_execute_in_parallel` (`utils/multi_stream_utils.py:16-63`), `aux_stream()` (`utils/torch_utils.py:860-869`) | che ~2–3 kernel/layer → 0.19–0.56 ms/step [E]; RoPE phải tách q/k (+1 kernel) | `forward` của wrapper nằm trong vùng torch.compile → phần stream nên nằm trong custom op mờ [U]; helper tự tắt khi `BreakableCUDAGraphCapture` active; thứ tự với `kv_cache_dummy_dep` | Có ở DeepSeek-V4.1 (`models/deepseek_v41/attention.py:889-895`), ROCm Kimi-K3 [#60248](https://github.com/vllm-project/vllm/pull/60248) (open); shared experts đã ở aux stream (`fused_moe/runner/shared_experts.py:130-157`); DeepSeek-V2/GLM trên CUDA: không có |
| C2 | **PDL stage1→stage2** (Triton `launch_pdl=True`, `gdc_wait` đầu kernel, `gdc_launch_dependents` ở stage1) | `triton_decode_attention.py:533-571,663-680` | 1 khoảng trống/layer ≈ 1–2 µs → 0.05–0.1 ms/step [E] | SM90+ (vLLM: `platforms/cuda.py:738-744`); SM89 không thể [V ptxas]; kernel nào bật PDL đều phải `gdc_wait` trước khi đọc dữ liệu kernel trước | Không thấy PR (chỉ [#53526](https://github.com/vllm-project/vllm/pull/53526) Kimi-K3, closed) |
| C3 | Chuỗi PDL dài hơn: `fused_q_kv_rmsnorm` đã hỗ trợ PDL (`models/common/ops/fused_qk_rmsnorm.py:188`); GEMM FlashInfer BF16 nhận `pdl` (`model_executor/layers/utils.py:707-711`, chỉ backend `flashinfer_cutedsl`); cuBLAS (`torch.bmm`, `F.linear`) không điều khiển được | nhiều nơi | phụ thuộc số cạnh PDL liên tiếp [E] | một kernel không PDL làm đứt chuỗi | — |
| C4 | FA2 MLA: merge split-KV đã nằm trong cùng kernel cooperative (`grid.sync` + `DevicePersistentMergeStates`, `mla.cuh:1205-1213`) → so với Triton đã bớt 1 launch; thêm PDL cần `cudaLaunchKernelEx` với Cooperative + ProgrammaticStreamSerialization | `mla.cuh:1274`; FA2 `supports_enable_pdl=False` (wheel `_capabilities.py:17,124-125`, `fa2_backend.py:21-28`) | che launch vào attention; [E] ~1–2 µs/layer | Kết hợp Cooperative+PDL **[U]**: launcher Triton 3.8 dựng đúng cặp thuộc tính này không chặn (wheel `triton/backends/nvidia/driver.c:1004-1017`), nhưng driver có nhận hay không chưa kiểm | Không thấy PR |

Triton/vLLM đã hỗ trợ PDL [V]: Triton 3.8.0 wheel `CUDAOptions.launch_pdl`
(`triton/backends/nvidia/compiler.py:124`), `gdc_wait`/`gdc_launch_dependents`
(`triton/language/extra/cuda/gdc.py:13-37`, "also safe to execute when programmatic dependent
launch is disabled"); mẫu trong vLLM: `model_executor/layers/mamba/ops/scatter_states.py:28-37,104`. Kwarg
`launch_pdl` vừa bind vào kernel param cùng tên (nếu có) vừa thành option launch
(`triton/runtime/jit.py:702-716`).
Wrapper PDL trong `bench_mla_decode_layer_chain.py` gọi chính thân kernel của vLLM, compile ra
cùng 63,488 B smem / 190 reg trên sm_120 [V].

## 7. (D) Thiết kế kernel FlashInfer FA2 MLA BF16 cho ~100 KB smem

Mục tiêu: bỏ QK trùng và hàng padding, pipeline nhiều stage, giữ merge trong cùng kernel, thêm PDL.

**Ngân sách smem** [V: `sizeof` của struct mô phỏng, biên dịch bằng g++; tile có `alignas(128)`;
`o_stage` dùng union với vùng Q+KV sau vòng lặp; giới hạn 101,376 B/block]:

| Tile (Q / KV / stage) | Score exchange | Byte | Dùng cho |
| --- | --- | --- | --- |
| Q16 / KV32 / 2 | 1 buffer (QK chia theo cột KV) | 94,336 | `q_len·H ≤ 16` (DS-V2-Lite H=16) |
| Q16 / KV16 / 4 | 1 buffer | 93,312 | như trên, nhiều byte đang bay hơn |
| Q32 / KV16 / 3 | 1 buffer / 2 buffer (chia theo K) | 94,464 / 96,512 | `q_len·H ≤ 32` (GLM H=20) |
| Q32 / KV16 / 2 | 2 buffer | 78,080 | dự phòng |
| Q64 / KV16 / 1 | 1 / 2 buffer | 96,768 / 100,864 | `q_len·H ≤ 64`; chỉ bỏ QK trùng |
| Q32 / KV32 / 2 | 2 buffer | 119,040 ✗ | không vừa |
| hiện tại Q64 / KV16 / 1 (FlashInfer) | m/d 512 B | 92,704 | |

Struct dùng để đo (rút gọn):
`union { struct { bf16 q_nope[TQ*512]; bf16 q_pe[TQ*64]; bf16 ckv[S][TKV*512]; bf16 kpe[S][TKV*64]; float scores[G][TQ*TKV]; float m[TQ], d[TQ]; }; bf16 o_stage[TQ*512]; }`.

**Phân công warp (8 warp = 256 thread, Q32/KV16/3)**: QK là M32×N16×K576 = 4 tile m16n8 →
8 warp = 4 tile × 2 nửa K (288) → 2 buffer điểm FP32 (2·32·16·4 = 4 KB) rồi cộng; mọi warp đọc
đủ 32×16 điểm để tự tính max/exp (ý tưởng "4 output group, mọi group đọc điểm" của #6987 Q32,
tổng quát hoá khỏi FP8). PV là M32×N512×K16 → mỗi warp 64 cột đầu ra (2 m16 × 8 n8 = 16 MMA/tile),
O accumulator 32·64/32 = 64 reg FP32/thread. Hàng padding: Q32 cho H=20 còn 12 hàng thừa so với
44 ở Q64; công executed = 32·2176 = 69,632 FLOP/token = 60 FLOP/B (< mọi ridge) [E].

**Pipeline**: `cp.async` S stage: đầu vòng `wait_group<S−2>`, phát nạp tile `t+S−1` trước khi tính
tile `t` (Q32/KV16/3: 2 tile = 36 KB đang bay/SM). Little: 36 KB / ~0.7 µs [giả định độ trễ] ≈
51 GB/s/SM → 170 SM ≈ 8.8 TB/s ≥ 1.79 TB/s (5090); GB10 chỉ cần ≈ 273/51 ≈ 6 SM bận [E].

**Merge**: giữ cooperative `grid.sync` + `DevicePersistentMergeStates` (một launch), lấy bản merge
song song của #4906 (dùng chung cho FA2). Partial vẫn ở `DTypeO` (BF16).

**PDL**: launch bằng `cudaLaunchKernelEx` + {Cooperative, ProgrammaticStreamSerialization}
(nếu driver từ chối: bỏ PDL hoặc đổi `grid.sync` sang merge của CTA cuối). Trước
`griddepcontrol.wait`: đọc `work_indptr` của CTA, metadata work item đầu (`q_indptr`,
`kv_indptr`, `kv_start/end` — ghi bởi `plan()` trước khi replay graph), tính `kv_tile_idx`, nạp
`kv_indices` của S tile đầu vào thanh ghi, dựng con trỏ smem. Sau `wait`: `load_q` (q do kernel
ngay trước tạo) và `cp.async` KV (token mới do `concat_and_cache_mla` ghi trước đó trong layer;
lưu ý vòng lặp FA2 đi từ tile **cuối** (`mla.cuh:1065-1099`, `--kv_tile_idx`) nên tile nạp đầu
tiên chứa token mới → không prefetch KV trước `wait`). Sau `grid.sync`: `griddepcontrol.launch_dependents`, rồi merge. Kernel kế tiếp trong vLLM
là W_UV `bmm` (cuBLAS, không PDL) nên lợi ích chủ yếu ở cạnh vào attention.

**Planner**: dùng API của #4906 (`MLASelectCtaTileQ(max_packed_qo_len, min_cta_tile_q,
cluster_size)`, `plan_info.cta_tile_q` giữ nguyên khi replan dưới graph): trên ~100 KB chọn
Q16 (≤16), Q32 (≤32), Q64 (≤64), cluster 1. Đổi hàm làm tròn `f()` (`scheduler.cuh:1670-1683`:
mọi `total_kv/num_sm ≤ 64` đều thành 192 token) bằng lượng tử `TKV·S` (#6987 thêm
`chunk_quantum`): B=1, 8K, 170 SM → hiện ~43 CTA có việc (8192/192) [E] so với 128 nếu chunk 64.

**Xung đột & phối hợp**: #4906 (mở từ 2026-09-02) và #6987 (mở 2026-10-22) cùng sửa
`MLAPlan` và `DevicePersistentMergeStates`; #6987 tổng quát hoá `KernelTraits`
(`OUTPUT_GROUPS`, `NUM_MMA_D_O`). [#6308](https://github.com/flashinfer-ai/flashinfer/pull/6308)
(mở, FP8 SM120 cho shape GLM-4.7-Flash, BF16 giữ nguyên) là việc song song. Đề xuất: (1) comment
trên #4906 và #6987 đề nghị dùng chung API planner của #4906 (`tile_q` của #6987 → `min_cta_tile_q`,
thêm `chunk_quantum`); (2) phát triển kernel BF16 small-Q như một specialization riêng
(`SMALL_Q_NONDUP`) dựa trên `OUTPUT_GROUPS` của #6987 để diff vào code chung nhỏ nhất, rebase sau
khi một trong hai PR merge; (3) PDL + launch cooperative là PR riêng; (4) phía vLLM cần backend dense
mới bọc `BatchMLAPagedAttentionWrapper(backend="fa2")` cho SM8x/SM12x, theo mẫu plan-ngoài-graph
của `flashinfer_mla_sparse_sm90.py` (không thấy PR trùng khi tìm "flashinfer mla sm120").

## 8. Kế hoạch theo tuần (10–20 h/tuần)

| Tuần | Việc | Đầu ra / cổng |
| --- | --- | --- |
| 1 (22/10–28/10) | Cài đặt 3 máy; `run_all.sh`; `e2e_glm47_flash.sh latency nsys` trên GB10 + 5090 (4-bit hoặc `forward`); đọc lại số với §0 | G1, G2; comment số đo lên #50629/#52019 nếu liên quan |
| 2 (29/10–4/11) | PR vLLM A: split an toàn với graph cho TRITON_MLA (device `n_split`, cap occupancy, stage2 đọc split hiệu lực; có thể partial BF16); test `tests/kernels/attention/test_triton_decode_attention.py` thêm replay graph với `seq_lens` khác nhau; E2E + GSM8K | PR A (sau skill `pr-checklist`) |
| 3 (5/11–11/11) | PR vLLM B (gộp, không lẻ): `empty` o/lse, hoist row expansion, `fuse_qkv_rmsnorm` cho `DeepseekV2MLAAttention`, two-stream KV path (sau cờ), PDL stage1→2; hỗ trợ #51861, không làm lại cat | G3 |
| 4–5 (12/11–25/11) | FlashInfer: specialization BF16 small-Q (Q16/Q32), QK không trùng, 3–4 stage, planner #4906; PDL tuỳ chọn; test `tests/attention/test_deepseek_mla.py`, `test_mla_wrapper.py` trên SM89/120/121 | G4 |
| 6 (26/11–2/12) | Backend vLLM dense FA2 cho SM8x/SM12x (nếu G4 đạt); eval GSM8K/`tests/evals`; PR có tuyên bố hỗ trợ AI, lệnh test, kết quả | PR C/D |
| đệm (3/12–9/12) | Sửa theo review | |

## 9. Chưa kiểm chứng được

- Mọi thời gian chạy trên GPU; prototype đúng hay sai; capture CUDA graph của launch cooperative
  FA2; Cooperative + PDL cùng lúc; `torch.profiler` sau CUPTI.
- Spec: 5090 209.5 TFLOPS (nguồn thứ cấp mơ hồ), 4090 ≈165 TFLOPS, TFLOPS/SM/L2 của GB10, L2 5090;
  RTX PRO 5000 1,344 GB/s chỉ qua kết quả tìm kiếm trích trang NVIDIA.
- `config.json` thật của GLM-4.7-Flash (huggingface.co bị chặn; dùng mặc định transformers) và các
  chiều DeepSeek-V2-Lite (theo model card).
- Số head và backend trong phép đo 1254.26 µs của #6308.
- `torch.bmm(out=view)` với stride 576: suy từ đường `bmm_out_cuda` của ATen (kết quả có
  `stride[2]==1`, `stride[1] ≥ size[2]` → không copy) — kiểm bằng `--count-kernels`.
- Đặt tên file khi `nsys --capture-range-end=repeat` (script glob `nsys_out*.nsys-rep`).
- Inductor có gộp RMSNorm/RoPE dưới compile mặc định hay không (ảnh hưởng số kernel thật).

## Phụ lục: công thức

- Byte mỗi bước decode GLM-4.7-Flash BF16, B=1 [E, từ config mặc định]: attention
  `2048·1344 + 768·5120 + 512·20·448 + 5120·2048 = 21.76M` param/layer; MoE thưa
  `(4+1)·3·2048·1536` + router; layer 0 dày `3·2048·10240`; lm_head `154880·2048` → **7.16 GB**.
  KV/bước = `47·L·1152` B = 0.444 GB (8K), 1.774 GB (32K). Bước lý tưởng: 5090 4.24/4.98 ms,
  4090 7.54/8.86 ms, GB10 27.9/32.7 ms (8K/32K) [E]; phần KV = 5.8% / 19.9%.
- Amdahl: giảm thời gian bước = `f·(1 − 1/s)`; cần `f ≥ 0.08/(1 − 1/s)` (s=2 → f ≥ 16%).
- Split đóng băng: `S = min(next_pow2(max_model_len//512), 2·SM)`; partial ghi+đọc/layer =
  `2·rows·20·S·513·4` B (27.9 MB ở B=1, S=340; 223 MB ở B=8).
- FA2 hiện tại: `64·(2·1152+1024)/1152 = 184.9` FLOP/B; B16/KV32K trên 5090: sàn compute
  `16·32768·212992/209.5e12 = 533 µs` > sàn bộ nhớ `604 MB/1.792 TB/s = 337 µs` [E, spec U].
