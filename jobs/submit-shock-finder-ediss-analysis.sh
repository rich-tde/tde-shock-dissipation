#!/usr/bin/env bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=shockfinder-ediss-analysis

#SBATCH --time=1-00:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=ALL

set -euo pipefail

export MPLCONFIGDIR="/home/hey4/rich_tde/.cache/matplotlib"
export IPYTHONDIR="/home/hey4/rich_tde/.cache/ipython/${SLURM_JOB_ID}"

cd /home/hey4/rich_tde
/home/hey4/.conda/envs/richanalysis/bin/jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    /home/hey4/rich_tde/works/shock-tde/0.5-shock-finder-ediss-analysis.ipynb
