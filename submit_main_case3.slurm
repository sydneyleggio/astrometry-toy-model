#!/bin/bash
#SBATCH --job-name=main_case3
#SBATCH --account=kmpardo_1034
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/output.out
#SBATCH --error=logs/error.err

set -euo pipefail

export SLURM_N_STARS=1250
export SLURM_FIELD_SIZE_DEG=125

echo "=========================================="
echo "Running with N_STARS=${SLURM_N_STARS}, FIELD_SIZE_DEG=${SLURM_FIELD_SIZE_DEG}"
echo "=========================================="

PYTHON=/home1/sleggio/.conda/envs/astrosnr/bin/python3

echo "Using interpreter: ${PYTHON}"
"${PYTHON}" -c "import matplotlib; print('matplotlib OK:', matplotlib.__file__)"

mkdir -p logs

cd "${SLURM_SUBMIT_DIR}"
echo "Running from: $(pwd)"

"${PYTHON}" main.py

echo "Finished (N_STARS=${SLURM_N_STARS}, FOV=${SLURM_FIELD_SIZE_DEG})."
