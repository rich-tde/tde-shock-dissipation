#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --job-name=stream-convergence
#SBATCH --array=0-5
#SBATCH --time=04:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%A_%a_%x.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl
#SBATCH --mail-type=ALL

set -euo pipefail

python=/home/hey4/.conda/envs/richanalysis/bin/python
works=/home/hey4/rich_tde/works/shock-tde
out=/home/hey4/rich_tde/data/processed/SimpleTimeseries
data=/data1/projects/pi-rossiem/TDE_data/StreamConvergence
runs=(1e4streamhires 1e4streamhires2)
dirs=("$data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResStream" "$data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResStream2")
run=${runs[SLURM_ARRAY_TASK_ID % 2]}
dir=${dirs[SLURM_ARRAY_TASK_ID % 2]}

case $((SLURM_ARRAY_TASK_ID / 2)) in
    0) $python $works/Ediss-t.py --mode 1 --data-dir "$dir" --output "$out/Ediss-t-$run-final.txt" ;;
    1) $python $works/E-t.py --mode 1 --data-dir "$dir" --output "$out/E-t-$run.txt" ;;
    2) $python $works/Eorb-bound-t.py --run "$run" ;;
esac
