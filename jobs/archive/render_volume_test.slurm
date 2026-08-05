#!/bin/bash
#SBATCH --job-name=vr_render
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=jobs/vr_render_%j.out
#SBATCH --error=jobs/vr_render_%j.err

# Scalability + quality test for richio.render on a ~70M-cell TDE snapshot.
# Single-node multiprocessing: yt's renderer is single-threaded, so frames are
# split across `--n-jobs` forked workers (the grid is built once, shared COW).

set -euo pipefail
cd /zfsstore/user/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
# Keep scipy's KDTree query from oversubscribing the allocation.
export OMP_NUM_THREADS=1

SNAP=/zfsstore/user/hey4/rich_tde/data/external/R0.47M0.5BH10000beta1S60ComptonHiRes/snap_full_151.h5
OUT=reports/gifs/density_spin_70M.mp4

echo "Host: $(hostname)  CPUs: $(nproc)  Started: $(date)"

$PY scripts/render_volume_movie.py "$SNAP" \
    --field density \
    --res 256 \
    --resolution 1024 \
    --nframes 120 \
    --elevation 26 \
    --box auto \
    --tf-mode map \
    --cmap twilight \
    --zoom 1.1 \
    --colorbar \
    --sigma-clip 4.0 \
    --fps 24 \
    --n-jobs 16 \
    --frames-dir /tmp/vr_frames_${SLURM_JOB_ID} \
    --out "$OUT"

echo "Finished: $(date)  ->  $OUT"
