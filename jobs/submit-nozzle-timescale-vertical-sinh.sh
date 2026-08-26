#!/usr/bin/env bash
#SBATCH --partition=cpu-zen4
#SBATCH --account=strw
#SBATCH --job-name=nozzle-ts-sinh
#SBATCH --array=1-8

#SBATCH --time=8:00:00
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

modes=(1 2 3 3 3 3 3 3)
snapshots=(108 142 513 626 688 793 850 961)
index=$((${SLURM_ARRAY_TASK_ID} - 1))

cd /home/hey4/rich_tde
/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/cooling-checks/nozzle-timescale-validation.py \
    --mode "${modes[${index}]}" \
    --snapshot-number "${snapshots[${index}]}" \
    --resolution-xy 256 \
    --resolution-z 512 \
    --z-spacing sinh \
    --sinh-scale-rp 0.1 \
    --workers 8 \
    --output-root /home/hey4/rich_tde/data/processed/CoolingChecks/nozzle-timescale-series/stage2-vertical-sinh
