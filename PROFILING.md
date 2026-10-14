# Profiling DreamZero-DROID

This commit layers GPU profiling infrastructure on top of the pristine
DreamZero codebase from the previous commit. It covers the PyTorch
profiler + Nsight Systems (`nsys`) + Nsight Compute (`ncu`) passes used to
answer "where does the time go, and why." The fine-grained NVTX
instrumentation that lets a later `ncu` pass scope directly onto a single
DiT block is a separate, follow-up commit — see `NVTX_PROFILING.md`.

## What changed

- **`socket_test_optimized_AR.py`**: three small, additive changes to the
  real inference server (no behavior change when unused):
  1. A per-rank `TRITON_CACHE_DIR` fix — a shared cache directory makes
     both CFG ranks' `torch.compile` workers race each other over the
     same compiled-kernel files on disk; each rank now gets its own
     subdirectory.
  2. An import-order fix (`import deepspeed` before the module touches
     anything from `transformers.modeling_utils`) to avoid a circular
     import deepspeed triggers.
  3. An optional PyTorch profiler hook, gated by `TORCH_PROFILE=1`. When
     enabled, it wraps the first `TORCH_PROFILE_STEPS` forward passes per
     rank in `torch.profiler.profile(...)` and writes a Chrome trace to
     `TORCH_PROFILE_DIR/rank{rank}_step{i}.json`. When disabled (the
     default), this code path is a no-op (`contextlib.nullcontext()`).
- **`benchmarks/`** (new directory): the sbatch jobs and a standalone
  `ncu` probe script, described below.
- **`activate_env.sh`**: the one environment-setup script every sbatch
  job sources, so cache directories and `TORCHDYNAMO_DISABLE` stay
  consistent across runs.

## The jobs, in the order we ran them

| Job | What it measures | GPUs |
|---|---|---|
| `job_baseline.sbatch` | Plain run: starts the real inference server (`torchrun`, 2-GPU CFG parallelism), drives it with the repo's own test client, collects the model's own per-chunk `Time taken:` timing + GPU memory snapshots. No profiler attached — this is "how fast is it, unobserved." | 2x H100 |
| `job_profile.sbatch` | Same server+client, but the whole process tree runs under `nsys profile` (CUDA/NVTX/cuDNN/cuBLAS trace + GPU-metrics sampling: SM/Tensor-core/DRAM utilization over time) with `TORCH_PROFILE=1` also set, so every one of the run's calls gets both an `nsys` system-level trace *and* a PyTorch-profiler Chrome trace. | 2x H100 |
| `job_compile.sbatch` | Identical workload with `torch.compile` re-enabled (every other job disables it via `TORCHDYNAMO_DISABLE=1` — dynamo/Triton otherwise hit cross-rank cache races and transient NFS I/O errors under concurrent compile workers, see the file's own comments), to directly compare eager-mode vs. compiled steady-state latency. | 2x H100 |
| `job_ncu.sbatch` | Nsight Compute kernel-level occupancy/throughput — answers *why* a kernel is slow (achieved occupancy %, compute vs. memory-bound), not just how long it ran. `ncu`'s per-kernel instrumentation overhead is large (10-100x), so this pass is scoped to only the `diffusion_loop` NVTX range (already ~98% of per-chunk wall time) via `--nvtx-include "diffusion_loop]"`, capped at `--launch-count 200` to keep the capture finishing in reasonable time. Uses `benchmarks/ncu_probe.py` instead of the real server (see below for why). | 2x H100 |

## Why `ncu_probe.py` exists instead of profiling the real server directly

`ncu` only flushes its `.ncu-rep` report when the profiled process tree
exits **on its own**. The real server (`socket_test_optimized_AR.py`) runs
an event loop (`serve_forever()`-equivalent) that never exits by itself —
killing it externally, even gracefully, truncated the report both times
this was tried. `ncu_probe.py` reuses the exact same model-construction
and observation-conversion code the real server uses, but skips the
websocket layer: each rank runs exactly one forward pass, then exits
normally, so `ncu` gets a clean process-exit to flush against.

## What `job_ncu.sbatch`'s pass actually found (the gap this fixes)

Scoped to the whole `diffusion_loop` range and capped at 200 kernels, this
pass never got past the loop's **scheduler-math prologue** — the small
elementwise `log`/`expm1`/`neg`/add/mul ops that compute the flow-matching
noise schedule before each DiT forward. `ncu`'s own rule engine flagged
these directly:

> *"This kernel grid is too small to fill the available resources on this
> device, resulting in only 0.1 full waves across all SMs."*

It never reached the actual GEMM/flash-attention kernels inside the DiT
blocks — those are what `NVTX_PROFILING.md` and `job_ncu_dit_block.sbatch`
(the next commit) are for.

## Full numeric results

See `benchmarks/results/INSIGHTS.md` (added in the next commit, since it
also documents the NVTX follow-up) for the actual measured numbers: kernel
breakdowns, GPU utilization percentages, the `torch.compile` comparison
result, and caveats.
