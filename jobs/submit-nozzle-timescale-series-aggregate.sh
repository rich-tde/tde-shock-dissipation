#!/usr/bin/env bash
#SBATCH --partition=cpu-zen4
#SBATCH --account=strw
#SBATCH --job-name=nozzle-ts-aggregate
#SBATCH --time=2:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=FAIL,END

set -euo pipefail

export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1

cd /home/hey4/rich_tde
/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/cooling-checks/nozzle-timescale-series.py \
    --action aggregate \
    --resolution 256 \
    --resolution-z 512
