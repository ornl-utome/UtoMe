#!/bin/bash
#SBATCH --job-name=utome
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=02:00:00
#SBATCH --partition=batch

# Activate your ROCm PyTorch environment before submitting this Frontier job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
utome_config=${1:?Usage: sbatch --account=<allocation> scripts/train_amd.sbatch <config> [--ckpt ...]}
shift
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="${TMPDIR:-/tmp}/utome-miopen-${SLURM_JOB_ID}"
mkdir -p "$MIOPEN_USER_DB_PATH"
export NCCL_SOCKET_IFNAME=hsn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
ulimit -c 0
srun --gpu-bind=none python scripts/train_global_forecast.py --config "$utome_config" "$@"
