#!/usr/bin/env bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=E-t
#SBATCH --array=1-3

#SBATCH --time=2-00:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=ALL

set -euo pipefail

/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/shock-tde/E-t.py \
    --mode "${SLURM_ARRAY_TASK_ID}"
