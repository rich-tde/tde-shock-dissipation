#!/bin/bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=job
#SBATCH --time=0-03:00:00 # max 7 days
#SBATCH --output=/home/hey4/RICH/jobs/logs/%j_%x.out   # %x: Job name, %j: Job id
#SBATCH --nodes=1
#SBATCH --ntasks=48	# max 48 per node for gpu_strw
#SBATCH --cpus-per-task=1
#SBATCH --mem=61G	# max 61G per node for gpu_strw (this is per node requrest)
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

echo "====== Date: $(date) ======"
echo "====== Batch script: ======"
cat "$0"
echo "=========================="
echo ""

# Load environment
module purge                        # start with a clean environment
module restore new_rich_build          # load my saved configuration for rich

# Run
srun /home/hey4/RICH-fwrk/build/rich
