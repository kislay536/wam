# NVTX instrumentation: what it is, and what's marked here

This commit adds fine-grained NVTX region markers to the DiT forward pass
and the VAE decoder, closing a real gap: the previous `ncu` pass
(`job_ncu.sbatch`, scoped to the whole `diffusion_loop` range) never
reached the actual GEMM/attention kernels — it only ever landed on the
scheduler-math prologue. The new, narrower ranges let a follow-up `ncu`
pass (`job_ncu_dit_block.sbatch`) scope directly onto one DiT block and
finally measure the real compute kernels.

## What NVTX actually is

**NVTX (NVIDIA Tools Extension)** is not a profiler — it's an annotation
API. You call it from Python to drop named markers into the program's
execution timeline; `nsys` and `ncu` pick these markers up and use them as
time-boundaries, with essentially zero runtime cost (a marker write is a
CPU-side operation, no GPU synchronization, no kernel launch — on the
order of 100ns-1µs regardless of how much work happens inside the marked
region).

Two equivalent styles exist; this codebase uses the explicit push/pop
style throughout (already established in
`wan_flow_matching_action_tf.py` before this commit — see below):

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("self_attn")
y = self.self_attn(x, ...)
nvtx.range_pop()

# equivalent, context-manager form (not used here, for comparison):
with nvtx.range("self_attn"):
    y = self.self_attn(x, ...)
```

Ranges **nest automatically by call order** — `range_push` while another
range is still open makes it a child, purely because it's a stack. No
explicit parent reference is needed; that's why wrapping new ranges
inside an existing `diffusion_step_{index}` loop just works without
touching that outer code.

## What was already marked (previous commit, not new here)

`wan_flow_matching_action_tf.py` already had **phase-level** ranges marking
each stage of a single chunk's forward pass:

```
text_encoder          -- WanTextEncoder.encode_prompt (only runs when the language prompt changes)
image_encoder         -- CLIP feature extraction (only runs when a new anchor frame is needed)
vae                   -- VAE encode of the latent video / previous frames
kv_cache_creation     -- warm-up calls into the DiT to build/update the causal KV cache
diffusion_loop         -- the main denoising loop over all 16 scheduler steps
  diffusion_step_0 .. diffusion_step_15   -- one range per scheduler step
```

This was already enough to see *that* the DiT denoising loop dominates
per-chunk time (~85%) and *which* scheduler steps are real DiT-compute vs.
cheap cache-skips. It was **not** enough to see *inside* a single DiT
call — `diffusion_step_N` just wraps a call into a 40-layer transformer as
one opaque block.

## What's new in this commit

Two files get NVTX added, both wrapping the actual per-layer compute:

**`wan_video_dit_action_casual_chunk.py`** — the loop that runs all 40 DiT
layers now tags each one:

```python
for block_index, block in enumerate(self.blocks):
    nvtx.range_push(f"dit_block_{block_index}")
    x, updated_kv_cache = block(x=x, e=e0, freqs=freqs, ...)
    nvtx.range_pop()
```

And inside `CausalWanAttentionBlock.forward` itself — each block has 3
natural sub-phases — self-attention, then a `cross_attn_ffn` closure doing
cross-attention and the FFN:

```python
nvtx.range_push("self_attn")
y, updated_kv_cache = self.self_attn(x=..., kv_cache=kv_cache, ...)
nvtx.range_pop()
x = x + (y * e[2].squeeze(2))

def cross_attn_ffn(x, context, e):
    nvtx.range_push("cross_attn")
    x = x + self.cross_attn(self.norm3(x), context)
    nvtx.range_pop()
    nvtx.range_push("ffn")
    y = self.ffn((self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2)))
    x = x + (y * e[5].squeeze(2))
    nvtx.range_pop()
    return x
```

**`wan_video_vae.py`** — the `decode()` method (called from 4 sites, but
only at episode-reset/video-save time, not every chunk) gets a single
`vae_decode` range covering both its tiled and non-tiled paths.

### The resulting nesting

```
diffusion_loop
  diffusion_step_{0..15}
    dit_block_{0..39}
      self_attn
      cross_attn
      ffn
```
(`vae_decode` is a sibling range elsewhere in the call graph, not nested
under `diffusion_loop`.)

### Why block-level *and* sub-module, not just block-level

The point is letting a follow-up `ncu` pass isolate GEMM-category kernels
(self-attn's flash-attention call, the FFN's two Linear layers) from each
other, not just from the model as a whole — a block-level-only range
would still lump all three sub-phases' kernels together in one capture.

Overhead: 40 blocks × 3 sub-ranges × ~8 real-compute steps per chunk (the
cache-skip steps bypass the whole DiT call, so this is already the upper
bound, not amplified by them) — a few thousand push/pop pairs per chunk,
each effectively free. Negligible next to the ~1.3-1.6s/chunk cost the
loop already measured.

## Verifying the ranges actually show up

```bash
nsys stats --report nvtx_sum nsys_report.nsys-rep
```
Confirmed on our own capture: all of `dit_block_0` through `dit_block_39`
appear (not fewer — a gap would mean the loop edit didn't fire for every
block), `self_attn`/`cross_attn`/`ffn` each show exactly 40×(number of
real-compute-step calls) occurrences — confirming every block call
triggers all three sub-ranges exactly once, correctly nested — and
`vae_decode` shows up rarely (once per saved episode, not per chunk).

## What region is actually profiled by `job_ncu_dit_block.sbatch`

```bash
ncu --nvtx --nvtx-include "diffusion_step_10/dit_block_20/" \
    --set basic --target-processes all \
    -o ncu_report torchrun --standalone --nproc_per_node=2 benchmarks/ncu_probe.py
```

- `diffusion_step_10` — a real-DiT-compute scheduler step (confirmed from
  the `nvtx_sum` output above: ~190-200ms, not a ~70-150µs skip-step),
  past the first-two-real-steps warmup tax the model shows every chunk.
- `dit_block_20` — an arbitrary representative mid-network layer; all 40
  share identical shapes/config, so any index works.
- The trailing **`/`** (not `]`) means "this range and everything nested
  inside it" — i.e. `self_attn`, `cross_attn`, and `ffn` are all included
  in the capture, not just `dit_block_20` itself.
- **No `--launch-count` cap** is needed here (unlike the original
  `job_ncu.sbatch`'s `--launch-count 200` on the much coarser
  `diffusion_loop` range) — one block's kernel population (a handful of
  GEMMs + one flash-attention kernel + a few layernorm/elementwise ops
  across all three sub-phases) is small and naturally bounded.

### Result: this time it lands on real compute

Captured kernels this time include multiple GEMM variants (66-91% Compute
Throughput) and a flash-attention kernel (~50% throughput), each with
genuine per-kernel occupancy numbers — not the elementwise
`log`/`expm1`/`neg` prologue the original `job_ncu.sbatch` pass was stuck
on. See `benchmarks/results/INSIGHTS.md` §6 for the full kernel table and
occupancy numbers, including a new finding: the FFN's second Linear
(the 13824→5120 contraction) is the single slowest kernel in the block,
compute-saturated but occupancy-limited by both register pressure and
shared memory simultaneously.
