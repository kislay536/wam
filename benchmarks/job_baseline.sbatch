#!/bin/bash
#SBATCH --job-name=dz_baseline
#SBATCH --partition=h100sxm
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sws0/user/karya/wam/dreamzero/benchmarks/logs/baseline_%j.out
#SBATCH --error=/scratch/sws0/user/karya/wam/dreamzero/benchmarks/logs/baseline_%j.err

# Baseline run: start the DreamZero-DROID inference server (torchrun, 2 GPUs
# for CFG parallelism as in the paper), drive it with the repo's own test
# client, and collect whatever timing/memory numbers the stack already
# reports (per-chunk Forward/Inference/Wait time server-side, round-trip
# latency client-side, GPU memory before/during/after).

set -uo pipefail
cd /scratch/sws0/user/karya/wam/dreamzero
source activate_env.sh
# The A40's 44GB is tight for this 14B model's growing autoregressive
# KV-cache; torch.compile/Triton also hit cross-rank cache races and
# transient NFS I/O errors under concurrent compile workers. Disable
# dynamo for these diagnostic runs: eager mode is slower but avoids both
# failure classes so we can get complete, reliable timing numbers.
export TORCHDYNAMO_DISABLE=1

PORT=$(( 20000 + (SLURM_JOB_ID % 10000) ))
MODEL_PATH=./checkpoints/DreamZero-DROID
RUN_DIR=benchmarks/results/baseline_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Port: $PORT"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv > "$RUN_DIR/gpu_before.csv"

torchrun --standalone --nproc_per_node=2 socket_test_optimized_AR.py \
    --port "$PORT" --model-path "$MODEL_PATH" --enable-dit-cache \
    > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for server port to open..."
# The real rank-0 server (eval_utils/policy_server.py) is a bare websocket
# server with no HTTP /healthz route, so plain curl gets a 426. A raw TCP
# connect test is the accurate readiness signal: the port only opens once
# model construction/loading (the slow part) has finished.
READY=0
for i in $(seq 1 180); do
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

# Background GPU sampler while the client drives inference.
( while kill -0 $SERVER_PID 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> "$RUN_DIR/gpu_during.csv"
    sleep 2
  done ) &
SAMPLER_PID=$!

python test_client_AR.py --port "$PORT" --num-chunks 6 2>&1 | tee "$RUN_DIR/client.log"

kill $SAMPLER_PID 2>/dev/null
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv > "$RUN_DIR/gpu_after.csv"

{
    echo "=== Server timings (Wait/Forward/Inference per chunk) ==="
    grep -E "Forward Time|Inference Time|Wait Time" "$RUN_DIR/server.log"
    echo
    echo "=== Client round-trip per chunk ==="
    grep "Action shape" "$RUN_DIR/client.log"
    echo
    echo "=== Peak GPU memory during run (MiB) ==="
    awk -F',' 'NR>0{gsub(/ MiB/,"",$2); print $2}' "$RUN_DIR/gpu_during.csv" 2>/dev/null | sort -n | tail -1
} > "$RUN_DIR/summary.txt"

cat "$RUN_DIR/summary.txt"
echo "Full results in $RUN_DIR"
