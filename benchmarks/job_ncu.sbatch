#!/bin/bash
#SBATCH --job-name=dz_ncu
#SBATCH --partition=h100sxm
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sws0/user/karya/wam/dreamzero/benchmarks/logs/ncu_%j.out
#SBATCH --error=/scratch/sws0/user/karya/wam/dreamzero/benchmarks/logs/ncu_%j.err

# Kernel-level bottleneck pass: Nsight Compute answers the question nsys
# and the PyTorch profiler can't -- WHY a kernel is slow (achieved
# occupancy, memory bandwidth %, compute throughput %, i.e. compute-bound
# vs memory-bound), not just how long it ran. ncu's per-kernel
# instrumentation overhead is huge (10-100x), so this is scoped to the
# "diffusion_loop" NVTX range added in wan_flow_matching_action_tf.py --
# that's ~98% of per-chunk wall time per the model's own "Time taken:"
# breakdown, so it's the only region worth this cost.
#
# Uses benchmarks/ncu_probe.py instead of the real websocket server: ncu
# only flushes its .ncu-rep report when the profiled process tree exits
# on its own (confirmed: killing the serve_forever() server, even
# gracefully, truncated the report both times it was tried). The probe
# runs exactly one forward pass per rank then exits normally.

set -uo pipefail
cd /scratch/sws0/user/karya/wam/dreamzero
source activate_env.sh
export TORCHDYNAMO_DISABLE=1

NCU=/usr/lib/cuda-12.8/bin/ncu
MODEL_PATH=./checkpoints/DreamZero-DROID
RUN_DIR=benchmarks/results/ncu_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
export NCU_PROBE_MODEL_PATH="$MODEL_PATH"
# ncu_probe.py lives in benchmarks/, so torchrun's implicit sys.path[0]
# (the script's own directory) won't include the repo root that
# eval_utils/groot/socket_test_optimized_AR/test_client_AR need.
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv > "$RUN_DIR/gpu_before.csv"

"$NCU" \
    --nvtx \
    --nvtx-include "diffusion_loop]" \
    --set basic \
    --target-processes all \
    --launch-count 200 \
    -f -o "$RUN_DIR/ncu_report" \
    torchrun --standalone --nproc_per_node=2 benchmarks/ncu_probe.py \
    > "$RUN_DIR/server.log" 2>&1
NCU_EXIT=$?
echo "ncu/torchrun exited with code $NCU_EXIT"

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv > "$RUN_DIR/gpu_after.csv"

echo "=== ncu report ==="
ls -la "$RUN_DIR"/ncu_report* 2>&1
echo "Full results in $RUN_DIR"
echo "Quick text summary: ncu -i $RUN_DIR/ncu_report.ncu-rep --page details | less"
echo "Or open the .ncu-rep in the Nsight Compute GUI for the roofline/occupancy charts."
