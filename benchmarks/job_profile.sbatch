#!/bin/bash
#SBATCH --job-name=dz_profile
#SBATCH --partition=h100sxm
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sws0/user/karya/wam/dreamzero/benchmarks/logs/profile_%j.out
#SBATCH --error=/scratch/sws0/user/karya/wam/dreamzero/benchmarks/logs/profile_%j.err

# Deep profiled run: same server+client as job_baseline.sbatch, but the
# whole torchrun process tree runs under `nsys profile` with GPU metrics
# sampling (SM occupancy / memory bandwidth / tensor-core utilization,
# gh100 metric set) on top of the CUDA/NVTX/cuDNN/cuBLAS trace across both
# ranks. Custom NVTX ranges (text_encoder/image_encoder/vae/
# kv_cache_creation/diffusion_loop/diffusion_step_N) were added in
# wan_flow_matching_action_tf.py so the nsys timeline lines up 1:1 with the
# model's own "Time taken: ..." breakdown. TORCH_PROFILE=1 turns on the
# PyTorch profiler hook (now with with_stack/with_modules, and covering
# both ranks, not just rank 1), which dumps Chrome traces per chunk.

set -uo pipefail
cd /scratch/sws0/user/karya/wam/dreamzero
source activate_env.sh
# See job_baseline.sbatch: dynamo/Triton hit cross-rank cache races and a
# transient NFS I/O error under concurrent compile workers (worse under
# nsys instrumentation). Disable for a reliable, if slower, profiled run.
export TORCHDYNAMO_DISABLE=1

NSYS=/usr/lib/cuda-12.8/nsight-systems-2024.6.2/bin/nsys
PORT=$(( 20000 + (SLURM_JOB_ID % 10000) ))
MODEL_PATH=./checkpoints/DreamZero-DROID
RUN_DIR=benchmarks/results/profile_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

export TORCH_PROFILE=1
export TORCH_PROFILE_DIR="$RUN_DIR/torch_profile"
export TORCH_PROFILE_STEPS=7
mkdir -p "$TORCH_PROFILE_DIR"

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Port: $PORT"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv > "$RUN_DIR/gpu_before.csv"

"$NSYS" profile \
    -o "$RUN_DIR/nsys_report" \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --trace-fork-before-exec=true \
    --force-overwrite=true \
    --gpu-metrics-devices=cuda-visible \
    --gpu-metrics-set=gh100 \
    --gpu-metrics-frequency=10000 \
    torchrun --standalone --nproc_per_node=2 socket_test_optimized_AR.py \
        --port "$PORT" --model-path "$MODEL_PATH" --enable-dit-cache \
    > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for server port to open (profiled startup is slower)..."
# See job_baseline.sbatch: the real rank-0 server has no HTTP /healthz
# route, so use a raw TCP connect test instead of curl.
READY=0
for i in $(seq 1 240); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
        exec 3>&- 3<&- 2>/dev/null
        echo "Server port open after ~$((i*10))s"
        READY=1
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died while warming up; see server.log"
        break
    fi
    sleep 10
done

if [ "$READY" != "1" ]; then
    echo "Server port never opened; dumping server.log tail:"
    tail -100 "$RUN_DIR/server.log"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# TORCH_PROFILE_STEPS=7 covers the initial call + all 6 chunks below, so
# warmup vs. steady-state is directly comparable in the traces.
python test_client_AR.py --port "$PORT" --num-chunks 6 2>&1 | tee "$RUN_DIR/client.log"

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv > "$RUN_DIR/gpu_after.csv"

echo "=== nsys report ==="
ls -la "$RUN_DIR"/nsys_report* 2>&1
echo "=== torch profiler traces ==="
ls -la "$TORCH_PROFILE_DIR" 2>&1
echo "Full results in $RUN_DIR"
echo "View nsys report with: nsys stats $RUN_DIR/nsys_report.nsys-rep   (or open the .nsys-rep in the Nsight Systems GUI)"
