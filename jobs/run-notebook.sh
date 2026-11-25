#!/usr/bin/env bash
#SBATCH --partition=cpu-zen4
#SBATCH --account=strw
#SBATCH --job-name=run-notebook
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
# Usage: sbatch jobs/run-notebook.sh works/<path>.ipynb [more notebooks]
set -euo pipefail
cd /home/hey4/rich_tde
export MPLCONFIGDIR=/home/hey4/rich_tde/.cache/matplotlib
export IPYTHONDIR="/home/hey4/rich_tde/.cache/ipython/${SLURM_JOB_ID}"
for notebook in "$@"; do
    /home/hey4/.conda/envs/richanalysis/bin/jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 "$notebook"
done
