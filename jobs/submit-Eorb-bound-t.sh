#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --job-name=Eorb-bound-t
#SBATCH --array=0-5
#SBATCH --time=04:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%A_%a_%x.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=ALL

set -euo pipefail
runs=(1e4 1e5 1e6 ss24 1e4midres 1e4lowres)
/home/hey4/.conda/envs/richanalysis/bin/python /home/hey4/rich_tde/works/shock-tde/Eorb-bound-t.py \
    --run "${runs[SLURM_ARRAY_TASK_ID]}"
