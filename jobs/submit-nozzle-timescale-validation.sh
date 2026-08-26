#!/bin/bash
#SBATCH --partition=cpu-zen4
#SBATCH --account=strw
#SBATCH --job-name=nozzle-ts-val
#SBATCH --array=1-3

#SBATCH --time=8:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%A_%a_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

cd /home/hey4/rich_tde || exit 1
/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/cooling-checks/nozzle-timescale-validation.py \
    --mode "${SLURM_ARRAY_TASK_ID}" \
    --resolution 256 \
    --resolution 384 \
    --workers 8
