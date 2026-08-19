#!/bin/bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=nozzle-dissipation-analysis

#SBATCH --time=01:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${SLURM_JOB_ID}
export IPYTHONDIR=/tmp/ipython-${USER}-${SLURM_JOB_ID}

cd /home/hey4/rich_tde || exit 1
/home/hey4/.conda/envs/richanalysis/bin/jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    /home/hey4/rich_tde/works/shock-tde/1.3-nozzle-dissipation-all-analysis.ipynb
