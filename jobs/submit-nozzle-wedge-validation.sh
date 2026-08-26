#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --account=strw
#SBATCH --job-name=nozzle-wedge
#SBATCH --array=1-3

#SBATCH --time=1:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=ALL

set -euo pipefail

export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1

cd /home/hey4/rich_tde
/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/cooling-checks/nozzle-wedge-validation.py \
    --mode "${SLURM_ARRAY_TASK_ID}" \
    --overwrite
