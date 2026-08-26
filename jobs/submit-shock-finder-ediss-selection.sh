#!/usr/bin/env bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=shockfinder-ediss
#SBATCH --array=0-12%4

#SBATCH --time=3-00:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=ALL

set -euo pipefail

export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd /home/hey4/rich_tde
/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/shock-tde/shock-finder-ediss-selection.py \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --workers "${SLURM_CPUS_PER_TASK}"
