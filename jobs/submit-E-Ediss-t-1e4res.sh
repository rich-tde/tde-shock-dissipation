#!/usr/bin/env bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=E-Ediss-1e4res
#SBATCH --array=0-3
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%A_%a_%x.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=ALL

set -euo pipefail

data=/data1/projects/pi-rossiem/TDE_data
out=/home/hey4/rich_tde/data/processed/SimpleTimeseries
runs=(midres lowres)
dirs=("$data/NewSnellius/R0.47M0.5BH10000beta1S60Compton" "$data/R0.47M0.5BH10000beta1S60n1.5ComptonLowResNewAMR")
run=${runs[SLURM_ARRAY_TASK_ID % 2]}
dir=${dirs[SLURM_ARRAY_TASK_ID % 2]}

if (( SLURM_ARRAY_TASK_ID < 2 )); then
    script=Ediss-t.py; output=$out/Ediss-t-1e4$run-final.txt
else
    script=E-t.py; output=$out/E-t-1e4$run.txt
fi

/home/hey4/.conda/envs/richanalysis/bin/python /home/hey4/rich_tde/works/shock-tde/$script \
    --mode 1 --data-dir "$dir" --output "$output"
