# DreamZero-DROID Inference Bottleneck Analysis

Model: `GEAR-Dreams/DreamZero-DROID` (14B params, Wan2.1 backbone, CFG-parallel
across 2 GPUs). Hardware: 2x NVIDIA H100 80GB HBM3 (`h100sxm` SLURM partition).
Workload: 1 initial frame + 6 streaming 4-frame chunks, `--enable-dit-cache` on,
`TORCHDYNAMO_DISABLE=1` unless noted.

Jobs referenced below (all under `benchmarks/results/` on the cluster):
- `baseline_5886564` — plain run, model's own per-chunk timing (uploaded to
  `MPI/dreamzero_profiling/`)
- `profile_5887121` — deep profile: `nsys` with GPU-metrics sampling +
  custom NVTX ranges, PyTorch profiler on both CFG ranks, all 7 calls
- `ncu_5887174` — Nsight Compute kernel-level occupancy/throughput, scoped
  to the `diffusion_loop` NVTX range
- `compile_5887123` — same workload with `torch.compile` re-enabled, for
  comparison against the eager-mode baseline

## 1. Where the time goes (steady-state chunk, ~1.6-2.2s)

The model has its own built-in CUDA-event timer
(`wan_flow_matching_action_tf.py:~1326`, prints `"Time taken: ..."`) that
was already being logged but not surfaced in the first round of profiling:

| Phase | Typical cost | Share of total |
|---|---|---|
| **Diffusion (DiT denoising loop)** | 1.3-1.6s | **~85%** |
| KV cache creation | 0.17-0.5s | ~10% |
| Text/Image encoder | ~0 usually; spikes to 0.3-1.6s periodically | variable |
| VAE | ~0.05-0.08s | ~4% |
| Transform/Untransform (pre/post-processing) | ~0.02s | <2% |

**The diffusion loop dominates by a wide margin.** Text/image encoder costs
are near-zero most chunks because the model only re-encodes when the
language prompt changes or a new anchor frame is needed (periodic, not
every chunk) — but when it does trigger, it's a real 0.3-1.6s spike.

## 2. Inside the diffusion loop: the DiT-cache skip pattern

Custom NVTX ranges were added per denoising step
(`diffusion_step_0` .. `diffusion_step_15`) and around the whole loop
(`diffusion_loop`) in `wan_flow_matching_action_tf.py`, matching the
existing CUDA-event boundaries 1:1. `nsys stats --report nvtx_sum` on
`profile_5887121/nsys_report.nsys-rep` shows a strong bimodal pattern
across the 16 scheduler steps:

- Steps that **actually run the 40-layer DiT**: ~193-528ms each (confirmed
  by the log's `DIT Compute Steps: 8` — only 8 of 16 steps compute)
- Steps that **reuse a cached extrapolated prediction**
  (`cache_predict_order1`, the `--enable-dit-cache` skip path): ~0.08-0.12ms
  each — essentially free

This is the DiT-cache optimization working as designed, already halving
real compute. One extra finding: the *first two* real-compute steps
(step 0, step 1) run ~2.7x slower (514-528ms) than later real steps
(~195ms) — **a warmup tax that recurs inside every chunk**, not just the
first chunk overall. Likely cuDNN/kernel algorithm re-selection.

## 3. GPU utilization during the dominant phase: not saturated

`nsys`'s GPU-metrics sampling (`--gpu-metrics-set=gh100`, 10kHz), queried
from the exported sqlite and restricted to timestamps *inside* the 14
`diffusion_loop` NVTX windows (not the ~10min of idle model-loading time
that dominates the raw trace):

| Metric | Average (in diffusion_loop) | Peak |
|---|---|---|
| SMs Active | 52.0% | 100% |
| **Tensor Active** | **23.5%** | 100% |
| DRAM Read Bandwidth | 14.1% | 45% |
| DRAM Write Bandwidth | 7.8% | 41% |

Neither compute-saturated (tensor cores idle ~76% of the time on average)
nor memory-bandwidth-bound. There is real headroom during the one phase
that matters most.

## 4. Why: kernel composition

`nsys stats --report cuda_gpu_kern_sum` (whole run): GEMM kernels
(`nvjet_tst_*`, the DiT's Linear-layer matmuls) are the single largest
consumer at ~24%+ of all GPU time combined; flash-attention is ~9.3%.
Expected for a 40-layer, 5120-dim transformer.

`ncu_5887174/ncu_report.ncu-rep` (Nsight Compute, `--set basic`, scoped to
`diffusion_loop` via NVTX, `--launch-count 200`) landed in the loop's
**scheduler/noise-math prologue** (elementwise `log`/`expm1`/`neg`/add/mul
ops — the flow-matching noise/sigma computations that run before each
DiT forward). These hit **single-digit-percent achieved occupancy**
(6-10% vs. 75-100% theoretical), and `ncu`'s own rule engine flags them
directly:
> *"This kernel grid is too small to fill the available resources on this
> device, resulting in only 0.1 full waves across all SMs."*
> *"The grid for this launch is configured to execute only 110 blocks,
> which is less than the GPU's 132 multiprocessors."*

This is consistent with finding #3: a lot of GPU time goes to launching
many small, under-occupied kernels rather than keeping tensor cores busy.

**Caveat**: the `--launch-count 200` cap meant this specific `ncu` capture
never reached the GEMM/flash-attention kernels inside the DiT layers
themselves — only the prologue. A follow-up `ncu` run with
`--launch-skip` to jump past the prologue would get occupancy numbers for
the actual GEMM/attention kernels, if needed.

## 5. torch.compile: tested, no measurable benefit

Re-enabling `torch.compile` (`compile_5887123`, dynamo left on) cost
**~5 minutes** of one-time compilation overhead across the first two
calls (189s + 104s). Steady-state `Diffusion` time afterward
(chunks 1-5, avg 1.43s) was **statistically identical** to eager mode's
`Diffusion` time (`baseline_5886564`, avg 1.42s).

Plausible explanation given finding #4: if the bottleneck is many small,
separately-dispatched kernels, compile *should* help by fusing them —
but this model's tensor shapes change across denoising steps and chunks
(hence `torch._dynamo.config.recompile_limit = 800` set in
`socket_test_optimized_AR.py`), which likely prevents fusion from ever
stabilizing across calls. Not a clear win as currently configured.

## Bottom line

The single real lever is the **diffusion loop's DiT compute**, already
half-mitigated by the DiT-cache skip. Two more specific, actionable
findings on top of that:

1. A repeating ~500ms "first-2-real-steps" warmup tax inside *every*
   chunk (not just the first chunk ever) — worth investigating whether
   this is cuDNN algorithm caching that could be pre-warmed once at
   server startup instead of repeating per chunk.
2. A scheduler-math prologue made of many tiny, poorly-occupied kernels
   that `torch.compile` isn't fusing in practice — a hand-written fused
   kernel (or a `torch.compile`-friendly restructuring that avoids the
   shape-dependent recompilation) is the most concrete remaining lever.

## 6. Deeper NVTX instrumentation and a targeted `ncu` follow-up

Closes the limitation flagged in §4: `ncu_5887174` was scoped only to the
whole `diffusion_loop` NVTX range and, capped at `--launch-count 200`,
only ever reached the small scheduler-math prologue kernels — never the
actual GEMM/flash-attention kernels inside the DiT blocks.

### New NVTX ranges

Added via `torch.cuda.nvtx.range_push/pop` (the same push/pop convention
already used by `wan_flow_matching_action_tf.py`'s existing
`text_encoder`/`image_encoder`/`vae`/`kv_cache_creation`/`diffusion_loop`/
`diffusion_step_N` ranges), in two files:

- `wan_video_dit_action_casual_chunk.py`'s `_forward_blocks` loop: one
  `dit_block_{i}` range per DiT layer (40 layers), wrapping each
  `block(...)` call.
- `CausalWanAttentionBlock.forward` itself: `self_attn`, `cross_attn`,
  `ffn` sub-ranges around each of the block's three compute phases.
- `wan_video_vae.py`'s `WanVideoVAE.decode()`: a single `vae_decode` range
  (covers all 4 call sites in `socket_test_optimized_AR.py` that share this
  method).

**Granularity choice**: block-level + sub-module, not block-level only —
so a follow-up `ncu` pass can isolate GEMM-category kernels (self-attn's
flash-attention, the FFN's two Linears) from each other, not just from the
model as a whole. Overhead: `nvtx.range_push/pop` is a non-blocking CPU
marker write (~100ns-1µs, no CUDA sync) — negligible next to the
multi-second per-chunk runtime.

Resulting nesting, confirmed via `nsys stats --report nvtx_sum` on
`profile_5892595/nsys_report.nsys-rep` (a plain re-run of `job_profile.sbatch`,
unmodified, now carrying the new ranges):
```
diffusion_loop > diffusion_step_{index} > dit_block_{0..39} > self_attn / cross_attn / ffn
```
- All 40 `dit_block_0` through `dit_block_39` present, 126 occurrences
  each (one per real-compute step across the run's ranks/chunks).
- `self_attn`/`cross_attn`/`ffn` each show **5,040** occurrences — exactly
  40 blocks × 126 calls, confirming every block call triggers all three
  sub-ranges exactly once, correctly nested.
- `vae_decode`: exactly **1** occurrence (813ms) — correctly rare, fires
  only at episode-reset/video-save time, not per chunk.
- `diffusion_step_{N}` timings reconfirm the real-compute vs. skip-step
  split (8 real-compute steps ~190-200ms each, skip-steps ~70-150µs) and
  the "first two real-compute steps run slower" warmup tax (steps 0/1:
  480-490ms mean, pulled up by outliers as large as 3.1s — consistent
  with the ~2.7x warmup tax already documented in §2, now visible at the
  system level too).

### Targeted `ncu` pass: `diffusion_step_10/dit_block_20/`

New job `job_ncu_dit_block.sbatch` (A100x2 — see note on partition choice
below), scoped via `--nvtx-include "diffusion_step_10/dit_block_20/"`
(trailing `/` for descendant-inclusive matching — **not** the trailing `]`
the original `job_ncu.sbatch` uses for `diffusion_loop]`; this session hit
a real bug this pass where a `]`-suffixed filter matched zero kernels, so
`/` was used deliberately here), **no `--launch-count` cap** (one block's
kernel population is small and naturally bounded, unlike the whole
`diffusion_loop`). `diffusion_step_10` chosen from the `nvtx_sum` data
above: a real-compute step, past the first-two-steps warmup tax.
`dit_block_20` is an arbitrary representative mid-network layer (all 40
share identical shapes/config).

**Result: this pass lands squarely on real compute, unlike `ncu_5887174`.**
Captured kernels include multiple real GEMM variants and a flash-attention
kernel, each with genuine occupancy/throughput metrics — not the
elementwise `log`/`expm1`/`neg` prologue ops the original pass was stuck
on:

| Kernel | Invocations | Compute (SM) Throughput | Achieved Occupancy |
|---|---|---|---|
| `ampere_bf16_s16816gemm_bf16_256x128...` (grid 20×14) | 7 | 78.6-82.0% (avg 79.1%) | 12.41-12.47% |
| `ampere_bf16_s16816gemm_bf16_256x128...` (grid 54×14, FFN-in) | 1 | 91.24% | 12.47% |
| `ampere_bf16_s16816gemm_bf16_256x64...` (grid 20×5) | 2 | 72.1-73.8% | 6.13-6.14% |
| `flash::flash_fwd_kernel<...>` (self-attn) | 3 (×2 ranks) | 43.6-57.4% (avg ~49.8%) | 10.97-11.38% |

Per-kernel NVTX attribution (visible directly in the raw `ncu` export's
NVTX-range column) correctly identifies which sub-module each kernel
belongs to — e.g. the 256×128 GEMM at grid (20,14,1) appears under both
`self_attn` (Q/K/V/O projections) and `cross_attn` (Q projection), while a
distinct grid (54,14,1) instance of the *same* kernel shape appears only
under `ffn` (the `dim→ffn_dim` expansion Linear, larger grid matching
`ffn_dim=13824`).

**New finding this pass**: the single slowest individual kernel inside the
block is the FFN's *second* Linear (the `ffn_dim→dim` contraction,
`13824→5120`) at **1.32ms** — longer than any attention-projection GEMM
(~0.51ms) or the flash-attention kernel itself. It is genuinely
compute-saturated (82.0% Compute Throughput, `ncu`: *"utilizing greater
than 80% of the available compute or memory performance of the device"*)
despite only 12.47% achieved occupancy, which `ncu` attributes to **both**
register pressure and shared-memory allocation (*"theoretical occupancy
(12.5%) is limited by the number of required registers, and the required
amount of shared memory"*, estimated 87.5% local speedup if resolved) —
unlike the GR00T N1.5/Motus/Fast-WAM dominant kernels profiled elsewhere
this session (shared-memory-limited only), this one is doubly constrained.

### GPU-metrics, restricted to `dit_block_20` windows only

Using the `profile_5892595` nsys trace (GPU-metrics sampling, `gh100`
preset, already captured for the whole run) and an NVTX-bounded SQL query
restricted to just the 252 `dit_block_20` occurrences (both ranks, all
real-compute steps across the 7-chunk run):

| Metric | Average | Peak |
|---|---|---|
| SMs Active | **74.2%** | 100% |
| Tensor Active | **34.7%** | 100% |
| DRAM Read Bandwidth | 20.6% | 45% |
| DRAM Write Bandwidth | 11.3% | 42% |

**This is the highest steady-state GPU utilization measured for any window,
on any model, this entire session** — well above the original
`diffusion_loop`-wide figures (52.0% SMs Active, 23.5% Tensor Active) that
are diluted by the scheduler-math prologue and the two slower cache-skip
and kv-cache-creation phases folded into that coarser window. Restricting
to just one block's actual GEMM+attention+FFN compute shows the DiT layers
themselves are, in isolation, reasonably well-utilized — the system-level
23.5% Tensor Active figure says more about how much *non-block* overhead
surrounds the DiT loop than about the blocks' own efficiency. This
reframes the earlier finding: the biggest lever is not "the DiT blocks are
inefficient," it's "there's a lot of per-step overhead (scheduler math,
cache bookkeeping) *outside* the blocks diluting the aggregate" — consistent
with Pattern B (small-kernel tail) in `DEEP_OPTIMIZATION_ANALYSIS.md`,
now with a precise before/after number to back it.

### GPU partition note

This follow-up ran on `--partition=a100 --gres=gpu:a100:2`, not DreamZero's
usual `h100sxm` x2. The torchrun/CFG-parallel setup needs 2 GPUs regardless
of type; A40 was ruled out (`job_baseline.sbatch`'s own comment: *"The
A40's 44GB is tight for this 14B model's growing autoregressive
KV-cache"*), and A100 80GB was chosen over H100 to avoid the more
contended tier while still clearing that VRAM margin comfortably.

### Caveat
The `ncu_probe.py` process used for this pass runs exactly one chunk (one
`wrapper_policy.infer(obs)` call), not the 7-chunk streaming workload
`job_profile.sbatch` uses — `diffusion_step_10` here is the 10th step of
that single chunk's 16-step denoising loop, not a cross-chunk aggregate.
Since steps 0/1's warmup tax is paid once per chunk (confirmed recurring
in §2), and step 10 comes well after it within this same chunk, the
captured kernels should still represent genuine steady-state behavior, not
warmup-contaminated kernels — but this is inferred from the within-chunk
step ordering, not independently re-verified against a second chunk.
