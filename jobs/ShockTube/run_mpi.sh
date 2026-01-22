#!/bin/bash -l
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=shocktubes
#SBATCH --time=0-01:00:00 # max 7 days
#SBATCH --output=/home/hey4/RICH/jobs/logs/%j_%x.out   # %x: Job name, %j: Job id
#SBATCH --nodes=1
#SBATCH --ntasks=12	# max 48 per node for gpu_strw
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G	    # max 61G per node for gpu_strw
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
echo "Date: $(date)"
echo "Batch script:"
cat "$0"
echo "==============================================="
module purge && module restore new_rich_build			
sleep $(shuf -i 0-20 -n 1) # pause a random time between 0-20 seconds to avoid the hdf5 writing error
srun /home/hey4/RICH-fwrk/build/rich
