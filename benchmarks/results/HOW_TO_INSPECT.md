# How to inspect these profiling artifacts

## `profile_5887121/nsys_report.nsys-rep` (301MB) — Nsight Systems
System-wide timeline across both GPUs/ranks: CUDA kernels, NCCL
collectives, GPU occupancy/bandwidth sampling, and the custom NVTX ranges
(`text_encoder`, `image_encoder`, `vae`, `kv_cache_creation`,
`diffusion_loop`, `diffusion_step_0..15`).

- **GUI** (recommended for exploring): install [Nsight Systems](
  https://developer.nvidia.com/nsight-systems) (free, Win/Mac/Linux),
  File > Open, pick the `.nsys-rep`. Useful views:
  - Timeline: expand each process/rank, look at the CUDA HW row for
    kernel activity, the NVTX row for the custom range labels.
  - "GPU Metrics" row: SM/Tensor-core/DRAM utilization over time — this
    is where you see the 23.5% Tensor Active average directly as a
    graph instead of a single number.
  - Right-click any kernel/range > "Show in Events View" for exact
    durations.
- **CLI, no GUI needed** — text summaries via `nsys stats`:
  ```
  nsys stats --report cuda_gpu_kern_sum nsys_report.nsys-rep   # top kernels by time
  nsys stats --report nvtx_sum nsys_report.nsys-rep            # custom phase breakdown
  ```
  (first run generates a `.sqlite` sidecar; reuse it for faster re-queries)

## `profile_5887121/torch_profile/rank{0,1}_step{0..6}.json` (~220-242MB each, ~3.1GB total) — PyTorch Profiler
Per-rank, per-call Chrome traces with Python call stacks
(`with_stack=True`) and module names (`with_modules=True`).
- `rank0_*` = the CFG **conditional** pass (server's main handler)
- `rank1_*` = the CFG **unconditional** pass (worker loop)
- `step0` = the initial 1-frame call (includes warmup); `step1`-`step6` =
  chunks 0-5 (steady-state)

**Open with**: go to `chrome://tracing` in Chrome, or
[ui.perfetto.dev](https://ui.perfetto.dev) (works in any browser), and
load/drag the JSON file. Use the search box to find a module name (e.g.
"CausalWanModel" or "WanVideoVAE"), click a block to see its self-time
and full Python call stack in the side panel.

## `ncu_5887174/ncu_report.ncu-rep` (82MB) — Nsight Compute
Kernel-level occupancy/throughput for ~200 kernels sampled from inside
the `diffusion_loop` NVTX range (the loop's scheduler-math prologue —
see INSIGHTS.md caveat, this didn't reach the GEMM/attention kernels).

- **GUI**: install [Nsight Compute](
  https://developer.nvidia.com/nsight-compute) (free), File > Open the
  `.ncu-rep`. The "Details" page per kernel shows the SpeedOfLight
  roofline chart (compute % vs memory % at a glance) and the
  auto-generated "Rule" recommendations (the occupancy/grid-size
  warnings quoted in INSIGHTS.md come from here).
- **CLI**:
  ```
  ncu -i ncu_report.ncu-rep --page details | less      # full per-kernel detail
  ncu -i ncu_report.ncu-rep --page summary              # one-line-per-kernel overview
  ```

## `compile_5887123/` — torch.compile comparison
Plain text, no special viewer needed:
- `summary.txt` — per-chunk `Time taken:`/`Inference Time:` lines plus
  peak GPU memory
- `client.log` — client-side round-trip times (note the ~189s + ~104s
  first two calls — that's compilation, not a bug)

Compare its steady-state `Diffusion` numbers directly against
`baseline_5886564/summary.txt` (in the sibling `MPI/dreamzero_profiling/`
Drive folder) for the eager-vs-compiled numbers cited in INSIGHTS.md.

## Quick orientation if you only have 5 minutes
Read `INSIGHTS.md` first — it has the conclusions. Then, if you want to
see the "too many small kernels" finding visually, open
`nsys_report.nsys-rep` in the GUI, find any `diffusion_step_N` NVTX range
on the timeline, and zoom in: you'll see a dense cluster of short
elementwise-kernel bars rather than a few long GEMM bars.
