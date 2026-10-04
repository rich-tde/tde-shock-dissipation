#!/usr/bin/env bash
#SBATCH --partition=cpu-zen4
#SBATCH --account=strw
#SBATCH --job-name=cell-histories
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Usage: sbatch jobs/submit-cell-histories.sh scan|extract|census|nozzle [analysis options]
# The archive-only nozzle selection needs one CPU (--cpus-per-task=1).
# The inexpensive select stage can run directly in the analysis environment.
set -euo pipefail
cd /home/hey4/rich_tde
export MPLCONFIGDIR=/home/hey4/rich_tde/.cache/matplotlib
export NUMBA_CACHE_DIR=/home/hey4/rich_tde/.cache/numba
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

/home/hey4/.conda/envs/richanalysis/bin/python -u \
    works/shock-tde/trace-cell-histories.py "$@"
