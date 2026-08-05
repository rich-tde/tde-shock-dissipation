#!/bin/bash
#SBATCH --job-name=vr_proj
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/vr_proj_%j.out
#SBATCH --error=jobs/vr_proj_%j.err

# Projection (column-density) movie of a RICH TDE snapshot.
# WHICH (wide|disk|both) is passed via --export=WHICH=...; defaults to both.

set -euo pipefail
cd /zfsstore/user/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

WHICH="${WHICH:-both}"
SNAP=/zfsstore/user/hey4/rich_tde/data/external/R0.47M0.5BH10000beta1S60ComptonHiRes/snap_full_151.h5

echo "Host: $(hostname)  CPUs: $(nproc)  WHICH=$WHICH  Started: $(date)"

$PY scripts/render_tde_movies.py "$SNAP" \
    --field density \
    --mode projection \
    --res 256 \
    --resolution 1024 \
    --nframes 120 \
    --fps 24 \
    --n-jobs 24 \
    --disk-radius 250 \
    --which "$WHICH" \
    --outdir reports/gifs

echo "Finished: $(date)"
