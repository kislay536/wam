# X-WAM Profiling Results

Model: `sharinka0715/X-WAM` ("Unified 4D World Action Modeling from Video
Priors with Asynchronous Denoising", arXiv:2604.26694). Architecture:
Wan2.2-TI2V-5B base DiT + a lightweight "interleaved depth branch" (extra
replicated blocks for 3D spatial reconstruction, 10 `extra_blocks` per the
config's `num_extra_layers: 10`) + "Asynchronous Noise Sampling" for fast
action decoding from the same backbone that jointly generates video.
Checkpoint: `robocasa_sft` (RoboCasa fine-tune, 79.2% avg success rate per
the paper's own benchmark table). Hardware: 1x NVIDIA A100 80GB (`a100`
partition — see GPU note below for why not A40).

## Architecture note -- same Wan2.2-TI2V-5B family as Motus/Fast-WAM

X-WAM, Motus, and Fast-WAM all build on the identical `Wan-AI/Wan2.2-TI2V-5B`
base weights. This let the Wan2.2 base checkpoint already downloaded for
Motus (`Motus/pretrained_models/Wan2.2-TI2V-5B/`, ~34GB: DiT safetensors
shards, `Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`) be reused
directly via `config.wan_checkpoint_dir` override -- no redundant download.
Only the `robocasa_sft` fine-tune checkpoint (38.9GB, downloaded via
`hf download sharinka0715/X-WAM-checkpoints --include "robocasa_sft/*"`,
avoiding the full 117GB repo) was new.

## No single-GPU inference script in the repo -- bypassed the broker/client

The repo only ships a ZMQ broker/server/client microservice architecture
for benchmark evaluation (`evaluation/{policy_broker,policy_server,robocasa_client}.py`).
Per this session's established pattern (RynnVLA-002/Fast-WAM/AVDC/GR00T
N1.5 all did this), `benchmarks/run_deep_profile.py` mirrors
`policy_server.py`'s exact model-instantiation and inference-call logic
directly, bypassing the sockets:
```python
from runners.xwam_runner import XWAMRunner
config = OmegaConf.load(exp_path / "config.yaml")
config.wan_checkpoint_dir = <Motus's local Wan2.2-TI2V-5B path>
config.action_num = config.dataset.frame_skip // config.dataset.action_skip  # derived at runtime by policy_server.py, not stored in config.yaml
model = XWAMRunner(config=config).cuda().bfloat16()
model.load_state_dict(torch.load(ckpt_path)["module"])
model.eval()
model.model = torch.compile(model.model)
pred_videos, xt_actions, xt_proprios, xt_depth_latents = model.generate(
    rgb=..., proprio=..., prompt=["pick up the object"], seeds=[i],
    early_stop=True, cfg=0.0, run_depth=False)
```

## Input data (DROID substitution, as requested)

Same 3 DROID clips used for every other model this session, mapped onto
X-WAM's expected camera names (`robot0_agentview_left/right`,
`robot0_eye_in_hand`), resized to `config.dataset.video_size = [256, 320]`,
normalized to `[-1, 1]`. Proprio zeroed (`config.proprio_dim = 16`).
`frame_num = 9`, `action_dim = 14`, `action_num = 4` (derived) ->
`xt_actions` shape `(1, 32, 14)` (`(frame_num-1) * action_num = 32`
timesteps), matching the model's own docstring exactly.

## Bugs found and fixed

1. **`config.action_num` missing from the checkpoint's own `config.yaml`.**
   `XWAMRunner.__init__` reads `config.action_num` directly, but the
   downloaded checkpoint's `config.yaml` doesn't contain that field --
   `evaluation/policy_server.py` derives it at runtime
   (`config.action_num = config.dataset.frame_skip // config.dataset.action_skip`)
   before ever touching `XWAMRunner`. Missed on the first attempt (hit
   `ConfigAttributeError: Missing key action_num`), fixed by mirroring the
   exact same derivation in the probe script.
2. **Missing `torch.no_grad()` -- not a config issue, caused two real OOMs.**
   Neither `XWAMRunner.generate()`/`forward()` nor the original probe
   script wrapped inference in `torch.no_grad()` (`model.eval()` only
   disables dropout/batchnorm training behavior -- a different thing,
   easy to conflate). Without it, PyTorch retains the full autograd graph
   across this 5B+-param model's internal denoising steps, which **OOM'd
   twice**: first on an A40 (44GB, `Tried to allocate 22.00 MiB ... 44.41
   GiB memory in use`), then even on a genuine 80GB A100
   (`sws-8a100-01..05`; `76.46 GiB allocated ... 79.24 GiB memory in use`).
   Fixed by adding `@torch.no_grad()` to the probe's `main()` -- peak
   memory dropped to a sane **40.89GB**, fitting comfortably on a single
   80GB A100 (would likely now also fit on an A40 with little margin, not
   re-tested given the A100 was already working).
3. **`huggingface-hub` version conflict.** Installing `huggingface_hub[cli]`
   pulled `2.2.0`, but `transformers==4.51.3` (from `requirements.txt`)
   requires `huggingface-hub>=0.30.0,<1.0` -- `import transformers` failed
   with `ImportError` before any model code ran. Fixed by pinning
   `huggingface_hub>=0.30.0,<1.0` after the CLI install (the modern `hf`
   command, not the removed `huggingface-cli` alias, still works at this
   version).
4. **flash-attn has no prebuilt wheel for torch 2.8.0+cu129** (the versions
   `requirements.txt`/the README's install instructions specify), and
   building from source isn't viable on this cluster's allocation nodes
   (no `nvcc`, 2 CPUs -- same constraint hit for GR00T N1.5). Fixed by
   reusing GR00T's exact already-proven-working combo instead: downgraded
   to `torch==2.5.1+cu124` (loose `torch>=2.4.0` requirement in
   `requirements.txt` permits this) + the prebuilt
   `flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
   from the project's GitHub releases.

## Result: inference succeeded, real outputs produced

5 calls to `model.generate(..., early_stop=True)`, each producing a valid
`(1, 32, 14)` action-chunk tensor:
- warmup (model load + `torch.compile` warmup): **207.81s** -- `torch.compile`
  is unconditional here (`model.model = torch.compile(model.model)` in
  `policy_server.py` itself, not optional), same one-time-compile-tax
  pattern Fast-WAM showed, bracketed outside `cudaProfilerStart/Stop` from
  the start this time so the steady-state capture below is clean on the
  first successful attempt.
- steady calls: **1.24-1.63s** each (1.633s, 1.506s, 1.239s, 1.242s, 1.524s)
  -- a tighter, more compute-bound-shaped latency than most other models'
  per-chunk numbers this session, consistent with the kernel breakdown
  below.

Model load (checkpoint deserialization + Wan2.2 base + depth-branch extra
blocks): **113.88s** (standalone, from the `ncu` run's log, excluding
`torch.compile`).

## Profiling artifacts

- `benchmarks/results/run_deep_nsys_5892611/nsys_report.nsys-rep` -- full
  system CUDA/NVTX trace, `--capture-range=cudaProfilerApi` scoped to the
  5 steady-state calls only (warmup + `torch.compile` excluded by
  construction). `torch_profile/step{0-4}.json` -- per-call PyTorch
  profiler Chrome traces.
- `benchmarks/results/run_deep_ncu_5892612/ncu_report.ncu-rep` -- Nsight
  Compute, `--set basic --launch-count 300 --profile-from-start off`,
  scoped to one steady-state call.

## Kernel breakdown (`nsys stats`, clean steady-state-only capture)

`cuda_api_sum`: **`cudaStreamSynchronize` dominates at 65.9%** of CPU API
time (890 calls) -- not `cudaMemcpyAsync` (only 1.9%, 2,240 calls). X-WAM
joins GR00T N1.5 and AVDC as a third model this session that breaks the
memcpy-dominance pattern seen in DreamZero/Motus/Fast-WAM(original)/
RynnVLA-002 -- consistent with the `DEEP_OPTIMIZATION_ANALYSIS.md`
observation that this is architecture-family-specific, not universal.
`cudaLaunchKernel` is 23.1% (42,645 calls).

`cuda_gpu_kern_sum` -- **the most GEMM/conv-dominated kernel mix of any
model profiled this session**:
- `ampere_bf16_s16816gemm_bf16_256x128...`: **45.2%** alone, 11,340 instances
- `triton_*` (custom fused kernels from the `torch.compile`'d graph): 8.6%, 16,400 instances
- `ampere_sgemm_128x64_tn`: 8.2%, 150 instances
- `sm80_xmma_fprop_implicit_gemm...` (depth-branch conv layers): 7.3% + 1.5% + 1.3%
- `flash::flash_fwd_kernel<...>`: 4.4%, 3,000 instances
- two more `ampere_bf16_s16816gemm...` variants: 3.6% + 2.0% + 2.0%
- Collectively, explicitly GEMM/conv/attention-shaped kernels account for
  **~74%** of total GPU kernel time -- higher than Fast-WAM's 57.7% (the
  previous highest this session), consistent with X-WAM jointly computing
  video + depth + action every call (more total DiT-equivalent compute
  per call than an action-only model).

## `ncu` occupancy, aggregated by kernel category (300-kernel capped window, one steady call)

| Category | Count | % of window time | Avg SM throughput | Avg achieved occupancy |
|---|---|---|---|---|
| elementwise/copy/reduce | 251 | 46.2% | 23.0% | 72.3% |
| GEMM/conv/triton | 39 | **42.5%** | **63.9%** | **12.3%** |
| other | 10 | 11.3% | 59.6% | 51.0% |

The dominant kernel captured (`ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn`,
274.5µs) shows the now-familiar cross-model pattern: compute-saturated
(67.0% Compute Throughput, `ncu`: *"Compute is more heavily utilized than
Memory"*) despite only 12.45% achieved occupancy -- `ncu` attributes this
to **both** register pressure and shared memory (*"theoretical occupancy
(12.5%) is limited by the number of required registers, and the required
amount of shared memory"*, 87.5% estimated local speedup), the identical
dual-constraint signature found on DreamZero's slowest FFN kernel this
session (see `dreamzero/benchmarks/results/INSIGHTS.md` §6) -- now observed
in two independent Wan2.2-based DiT codebases.

## GPU note: A40 and even one 40GB A100 node both OOM'd; needed real 80GB A100

Three attempts before success: (1) A40 44GB OOM'd during `torch.compile`
warmup (43.56GB in use, failed allocating 22MB more); (2) `sws-3a100grid-01`
turned out to be a **40GB** A100 node (this cluster's `a100` partition has
a mix of 3x40GB nodes and 5x80GB nodes under the same partition name,
`sinfo -p a100 -N -o "%N %G %m"` distinguishes them) -- OOM'd identically;
(3) a genuine `sws-8a100-*` 80GB node OOM'd too, which is what actually
surfaced the real `torch.no_grad()` bug (bug #2 above) rather than it being
a true capacity problem -- after that fix, peak memory dropped to 40.89GB
and the job succeeded on the first retry on the same 80GB node. Future
X-WAM runs should explicitly `--exclude=sws-3a100grid-01` (or any other
sub-80GB `a100` node discovered later) when requesting this partition.

## Caveats

- Checkpoint is RoboCasa-trained; DROID frames + zeroed proprio are
  out-of-distribution -- output actions are not semantically meaningful,
  profiling-only, same caveat as every other model this session.
- `run_depth=False` was passed to `generate()`, but `config.use_depth=True`
  is baked into the model's architecture (`num_modalities = 2 if
  config.use_depth else 1`, fixed at construction) -- the depth branch's
  conv layers still execute every call (visible in the kernel breakdown
  above), `run_depth` only controls whether the depth *output* is returned,
  not whether it's computed. The measured timings/kernel mix reflect the
  full joint video+depth+action compute, not an action-only fast path.
- `early_stop=True` skips the VAE decode entirely (`generate()` returns
  `(None, xt_actions, xt_proprios, None)` without ever calling
  `self.vae.decode`) -- so this profile does not include video-decode
  cost, only the diffusion/action-generation forward pass.
- Only 5 steady-state calls with identical input (same frame, same
  zeroed proprio, only the `seeds` argument varies) -- not a large sample
  or varied-observation test.
- No GPU-metrics sampling (SM/Tensor Active %) was captured for this
  model -- time-boxed to the `nsys` system trace + `ncu` kernel-occupancy
  pass only, matching the depth given to most other models this session
  (not DreamZero's GPU-metrics-sampled deep profile).
