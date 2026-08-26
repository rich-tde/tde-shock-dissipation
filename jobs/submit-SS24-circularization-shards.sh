#!/usr/bin/env bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=SS24-circ-shard
#SBATCH --array=0-3

#SBATCH --time=1-00:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

set -euo pipefail

starts=(923 944 965 986)
ends=(943 964 985 1006)
start=${starts[${SLURM_ARRAY_TASK_ID}]}
end=${ends[${SLURM_ARRAY_TASK_ID}]}
output=/home/hey4/rich_tde/data/processed/SS24-circularization-t/shard-${start}-${end}.txt

/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/shock-tde/SS24-circularization-t.py \
    --start-snapshot "${start}" \
    --end-snapshot "${end}" \
    --timeseries-file "${output}" \
    --skip-fallback
