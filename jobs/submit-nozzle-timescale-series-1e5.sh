#!/usr/bin/env bash
#SBATCH --partition=cpu-zen4
#SBATCH --account=strw
#SBATCH --job-name=nozzle-ts-1e5
#SBATCH --array=0-161%24
#SBATCH --time=8:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=FAIL,END

set -euo pipefail

export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1

cd /home/hey4/rich_tde
/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/cooling-checks/nozzle-timescale-series.py \
    --action worker \
    --mode 2 \
    --snapshot-index "${SLURM_ARRAY_TASK_ID}" \
    --resolution 256 \
    --resolution-z 512 \
    --workers 8
