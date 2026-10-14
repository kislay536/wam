# Profiling X-WAM

This commit adds a deep-profiling probe + sbatch jobs on top of the
pristine X-WAM codebase (previous commit) — PyTorch profiler, Nsight
Systems (`nsys`), then Nsight Compute (`ncu`), in that order.

## No single-GPU inference script exists upstream

The repo only ships a ZMQ broker/server/client microservice architecture
for benchmark evaluation (`evaluation/{policy_broker,policy_server,robocasa_client}.py`).
`benchmarks/run_deep_profile.py` mirrors `policy_server.py`'s exact
model-instantiation and inference-call logic directly, bypassing the
sockets entirely — same pattern used for the other models profiled in
this project:

```python
from runners.xwam_runner import XWAMRunner
config = OmegaConf.load(exp_path / "config.yaml")
config.wan_checkpoint_dir = <path to a local Wan2.2-TI2V-5B checkpoint>
config.action_num = config.dataset.frame_skip // config.dataset.action_skip  # see bug #1 below
model = XWAMRunner(config=config).cuda().bfloat16()
model.load_state_dict(torch.load(ckpt_path)["module"])
model.eval()
model.model = torch.compile(model.model)
pred_videos, xt_actions, xt_proprios, xt_depth_latents = model.generate(
    rgb=..., proprio=..., prompt=[...], seeds=[i],
    early_stop=True, cfg=0.0, run_depth=False)
```

## Two real bugs this surfaced (both instructive, not X-WAM-specific)

1. **`config.action_num` isn't in the checkpoint's own `config.yaml`.**
   `XWAMRunner.__init__` reads it directly, but it's actually derived at
   runtime by `evaluation/policy_server.py`
   (`config.action_num = config.dataset.frame_skip // config.dataset.action_skip`)
   before ever touching `XWAMRunner` — missed on the first attempt
   (`ConfigAttributeError: Missing key action_num`), fixed by mirroring
   that exact derivation in the probe.
2. **Missing `torch.no_grad()` caused two real OOMs, including on a true
   80GB GPU.** `model.eval()` only disables dropout/batchnorm — it does
   nothing about gradient tracking. Without `torch.no_grad()`, PyTorch
   retains the full autograd graph across this 5B+-param model's internal
   denoising steps. This OOM'd on a 44GB GPU, then OOM'd *again* on a
   genuine 80GB GPU (`76.46 GiB allocated ... 79.24 GiB memory in use`),
   which is what actually surfaced the bug rather than it looking like a
   true capacity problem. Fixed by decorating the probe's `main()` with
   `@torch.no_grad()` — peak memory dropped to a sane 40.89GB.

## The two passes

| Job | What it measures |
|---|---|
| `job_xwam_deep_nsys.sbatch` | `nsys profile --capture-range=cudaProfilerApi`, scoped to only the steady-state calls (model load + the one-time `torch.compile` warmup run outside `cudaProfilerStart/Stop`, so they're excluded from the capture). Each steady call also gets wrapped in `torch.profiler.profile()`, exporting a per-call Chrome trace. |
| `job_xwam_deep_ncu.sbatch` | Nsight Compute, `--set basic --launch-count 300 --profile-from-start off`, scoped to one steady-state call (the 300-kernel cap keeps a `--set basic` replay practical on a model this dense). |

Both use `torch.cuda.nvtx.range(...)` markers (`warmup_step0`,
`steady_step_{i}`) purely to bound the `cudaProfilerApi` capture window
and to show up as named regions in the `nsys` timeline — a lighter use of
NVTX than the dreamzero branch's per-DiT-block instrumentation, just
enough to cleanly separate "warmup" from "steady state" in the trace.

## GPU note

Needed a genuine 80GB A100 — not just "the `a100` partition" (this
cluster's `a100` partition mixes 3×40GB nodes and 5×80GB nodes under the
same partition name; `sinfo -p a100 -N -o "%N %G %m"` distinguishes
them), and not an A40 (44GB OOM'd during `torch.compile` warmup even
before the `no_grad()` bug was found). See `benchmarks/results/INSIGHTS.md`
for the exact sequence of three attempts.

## Full numeric results

See `benchmarks/results/INSIGHTS.md` for the measured timings, kernel/API
breakdown, and `ncu` occupancy table — including a kernel-occupancy
finding (a GEMM kernel limited by *both* register pressure and shared
memory simultaneously) that matches a finding on the dreamzero branch's
deepest-profiled kernel, in this unrelated codebase built on the same
Wan2.2-TI2V-5B backbone family.
