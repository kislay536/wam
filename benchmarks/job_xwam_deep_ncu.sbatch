#!/bin/bash
#SBATCH --job-name=xwam_deep_ncu
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sws-3a100grid-01
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/sws0/user/karya/wam/X-WAM/benchmarks/results/slurm_deep_ncu_%j.log

set -euo pipefail
cd /scratch/sws0/user/karya/wam/X-WAM
source activate_env.sh
export PROFILE_MODE=ncu
export XWAM_STEPS_CKPT="${XWAM_STEPS_CKPT:?set XWAM_STEPS_CKPT}"

NCU=/usr/lib/cuda-12.8/bin/ncu
OUTDIR="benchmarks/results/run_deep_ncu_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
export DEEP_OUTDIR="$(pwd)/$OUTDIR"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$OUTDIR/gpu_before.csv"

"$NCU" --set basic --target-processes all \
  --profile-from-start off \
  --launch-count 300 \
  -o "$OUTDIR/ncu_report" --force-overwrite \
  python3 benchmarks/run_deep_profile.py > "$OUTDIR/run.log" 2>&1

echo "exit code: $?" >> "$OUTDIR/run.log"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$OUTDIR/gpu_after.csv"
tail -60 "$OUTDIR/run.log"
