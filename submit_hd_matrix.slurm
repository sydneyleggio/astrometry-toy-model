#!/bin/bash
#============================================================
# submit_hd_matrix.slurm
#
# Submits ONE run of hd_full_matrix_snr.py on USC CARC Discovery.
#
# This is a single job, not a job array — it runs whatever
# N_STARS / FIELD_SIZE_DEG you set in section 1 below. To run a
# different N, edit those two values and resubmit. Submitting one
# job at a time like this is fine cluster etiquette as long as you
# don't have many of these queued/running simultaneously.
#
# Submit with:
#     sbatch submit_hd_matrix.slurm
#
# Check status with:
#     squeue -u $USER
#============================================================

#SBATCH --job-name=hd_matrix
#SBATCH --account=kmpardo_1034
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/hd_matrix_%j.out
#SBATCH --error=logs/hd_matrix_%j.err

set -euo pipefail

# ------------------------------------------------------------
# 1. PARAMETERS — edit these two values, then resubmit for each run
# ------------------------------------------------------------
export SLURM_N_STARS=100
export SLURM_FIELD_SIZE_DEG=10

echo "=========================================="
echo "Running with N_STARS=${SLURM_N_STARS}, FIELD_SIZE_DEG=${SLURM_FIELD_SIZE_DEG}"
echo "=========================================="

# ------------------------------------------------------------
# 2. ENVIRONMENT
# ------------------------------------------------------------
module purge
module load gcc/13.3.0
module load python/3.11.9
# If you use a conda env instead, comment out the two module load
# lines above and uncomment these two:
# module load conda
# conda activate YOUR_ENV_NAME

mkdir -p logs

# ------------------------------------------------------------
# 3. RUN
# ------------------------------------------------------------
# Run from the repo directory containing main.py and hd_full_matrix_snr.py.
cd "${SLURM_SUBMIT_DIR}"

python3 hd_full_matrix_snr.py

echo "Finished (N_STARS=${SLURM_N_STARS}, FOV=${SLURM_FIELD_SIZE_DEG})."
