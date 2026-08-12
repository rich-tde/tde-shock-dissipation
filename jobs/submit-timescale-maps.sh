#!/bin/bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=timescale-maps
#SBATCH --array=1-3

#SBATCH --time=3-00:00:00 # max 7 days
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out   # %x: Job name, %j: Job id
#SBATCH --nodes=1   # max 2 nodes for gpu_strw
#SBATCH --ntasks=1  # max 24 per node for gpu_strw
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G   # max 61G per node for gpu_strw
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

/home/hey4/.conda/envs/richanalysis/bin/python /home/hey4/rich_tde/works/cooling-checks/timescale-maps.py --mode "$SLURM_ARRAY_TASK_ID"
