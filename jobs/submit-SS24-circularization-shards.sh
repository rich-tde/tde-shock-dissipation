#!/bin/bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=SS24-circ-shard
#SBATCH --array=0-3

#SBATCH --time=1-00:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%A_%a_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

STARTS=(923 944 965 986)
ENDS=(943 964 985 1006)
START=${STARTS[$SLURM_ARRAY_TASK_ID]}
END=${ENDS[$SLURM_ARRAY_TASK_ID]}
OUT=/home/hey4/rich_tde/data/processed/SS24-circularization-t/shard-${START}-${END}.txt

/home/hey4/.conda/envs/richanalysis/bin/python \
    /home/hey4/rich_tde/works/shock-tde/SS24-circularization-t.py \
    --start-snapshot "$START" \
    --end-snapshot "$END" \
    --timeseries-file "$OUT" \
    --skip-fallback
