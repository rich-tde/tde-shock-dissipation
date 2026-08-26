#!/usr/bin/env bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=plot-regional-circularization-t

#SBATCH --time=00:15:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

/home/hey4/.conda/envs/richanalysis/bin/jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    /home/hey4/rich_tde/works/shock-tde/0.7-plot-regional-circularization-t.ipynb
