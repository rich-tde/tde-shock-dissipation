#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --account=strw
#SBATCH --job-name=shock-zoom-caches
#SBATCH --array=0,2,3%3

#SBATCH --time=4:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=ALL

set -euo pipefail

export OMP_NUM_THREADS=1

cd /home/hey4/rich_tde
/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/shock-tde/shock-zoom-caches.py \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --workers "${SLURM_CPUS_PER_TASK}"
