#!/bin/bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=ApptainerRICH
#SBATCH --time=0-10:00:00 # max 7 days
#SBATCH --output=/home/hey4/RICH/jobs/logs/%j_%x.out   # %x: Job name, %j: Job id
#SBATCH --nodes=1
#SBATCH --ntasks=24	# max 24 per node for gpu_strw
#SBATCH --cpus-per-task=1
#SBATCH --mem=61G	# max 61G per node for gpu_strw
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

echo "====== Date: $(date) ======"
echo "====== Batch script: ======"
cat "$0"
echo "=========================="
echo ""

# ALICE Instruction: https://pubappslu.atlassian.net/wiki/spaces/HPCWIKI/pages/103809025/Apptainer+Singularity+containers
export APPTAINER_TMPDIR=$SCRATCH/.apptainer-tmp
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer-cache
mkdir -p $APPTAINER_TMPDIR
mkdir -p $APPTAINER_CACHEDIR

# Load MPI 4.1.6 to match version of the image
module purge
module load ALICE/default OpenMPI/5.0.8-GCC-14.3.0

# The Hybrid approach: https://apptainer.org/docs/user/1.0/mpi.html#hybrid-model
srun apptainer exec --cleanenv \
       /home/hey4/rich_env.sif /home/hey4/RICH-fwrk/build/rich

# --cleanenv option is important otherwise MPI bugs out. It cleans the
# environment variable in the container such that it doesn't mix up MPI from
# HPC and MPI inside the container.
