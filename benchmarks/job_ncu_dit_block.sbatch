#!/bin/bash
#SBATCH --job-name=dz_ncu_dit_block
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sws0/user/karya/wam/dreamzero/benchmarks/logs/ncu_dit_block_%j.out
#SBATCH --error=/scratch/sws0/user/karya/wam/dreamzero/benchmarks/logs/ncu_dit_block_%j.err

# Follow-up to job_ncu.sbatch (ncu_5887174): that pass was scoped to the
# whole "diffusion_loop" NVTX range with --launch-count 200, which only
# ever reached the small scheduler-math prologue kernels, never the real
# GEMM/flash-attention kernels inside the DiT blocks. New per-block/
# per-sub-module NVTX ranges (dit_block_{i} > self_attn/cross_attn/ffn,
# added this pass) let us scope directly onto ONE block instance instead --
# small enough kernel population that no --launch-count cap is needed.
#
# diffusion_step_10 chosen from profile_5892595's nvtx_sum export: a real-
# DiT-compute step (~191ms mean, low stddev), past the first-two-real-steps
# warmup tax (steps 0/1, which run ~2.7x slower). dit_block_20 is an
# arbitrary representative mid-network layer (all 40 share identical
# shapes/config).
#
# Uses a trailing "/" (descendant-inclusive match), NOT the "]" the
# original job_ncu.sbatch uses for "diffusion_loop]" -- this session hit a
# real bug where a "]"-suffixed ncu --nvtx-include filter matched zero
# kernels (AVDC's first attempt); "/" is the confirmed-working syntax used
# successfully for RynnVLA-002/GR00T N1.5's deep-profiling passes.
#
# On a100 x2 (not h100sxm, DreamZero's usual partition): resolved with the
# user as a middle ground between the A40 VRAM-tightness precedent
# documented in job_baseline.sbatch and the contended H100 tier.

set -uo pipefail
cd /scratch/sws0/user/karya/wam/dreamzero
source activate_env.sh
export TORCHDYNAMO_DISABLE=1

NCU=/usr/lib/cuda-12.8/bin/ncu
MODEL_PATH=./checkpoints/DreamZero-DROID
RUN_DIR=benchmarks/results/ncu_dit_block_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
export NCU_PROBE_MODEL_PATH="$MODEL_PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv > "$RUN_DIR/gpu_before.csv"

"$NCU" \
    --nvtx \
    --nvtx-include "diffusion_step_10/dit_block_20/" \
    --set basic \
    --target-processes all \
    -f -o "$RUN_DIR/ncu_report" \
    torchrun --standalone --nproc_per_node=2 benchmarks/ncu_probe.py \
    > "$RUN_DIR/server.log" 2>&1
NCU_EXIT=$?
echo "ncu/torchrun exited with code $NCU_EXIT"

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv > "$RUN_DIR/gpu_after.csv"

echo "=== ncu report ==="
ls -la "$RUN_DIR"/ncu_report* 2>&1
echo "Full results in $RUN_DIR"
