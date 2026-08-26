#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --account=strw
#SBATCH --job-name=shock-location-slices
#SBATCH --array=0-12%2

#SBATCH --time=4:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=ALL

set -euo pipefail

export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1

cd /home/hey4/rich_tde
extra_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    extra_args+=(--overwrite)
fi
if [[ "${RERENDER:-0}" == "1" ]]; then
    extra_args+=(--rerender)
fi
/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/shock-tde/shock-location-slices.py \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    "${extra_args[@]}"
