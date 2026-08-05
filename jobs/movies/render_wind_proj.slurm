#!/bin/bash
#SBATCH --job-name=vr_windproj
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/vr_windproj_%j.out
#SBATCH --error=jobs/vr_windproj_%j.err

# Wind-projection evolution movie: per-snapshot 2x3 panels (density top,
# dissipation bottom; xy/xz/yz columns) of the unbound wind (B>0 & v_r>0),
# reproducing works/windstudy/0.1-tde-wind.ipynb for every post-switch
# snapshot.  Matplotlib column projections via snap.plots.projection.
#
# Parametrise via --export:
#   BOXHALF = symmetric cube half-width in code length units (200 wide, 30 narrow)
#   BOX     = explicit 'x0,y0,z0,x1,y1,z1' box (overrides BOXHALF; for zoomed views)
#   VMIN    = per-field log10 vmin overrides 'density,dissipation' (blank = auto)
#   TAG     = label used in the output filename (wide | narrow | zoom | ...)
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

BOXHALF="${BOXHALF:-200}"
TAG="${TAG:-wide}"
RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR

# Explicit BOX wins over the symmetric BOXHALF cube.
if [ -n "${BOX:-}" ]; then
  boxflag=(--box "$BOX")
else
  boxflag=(--box-half "$BOXHALF")
fi
vminflag=()
if [ -n "${VMIN:-}" ]; then vminflag=(--vmin "$VMIN"); fi

OUTDIR=reports/movies/wind_proj
mkdir -p "$OUTDIR"
OUT="${OUTDIR}/wind_proj_${TAG}.mp4"

echo "Host: $(hostname)  BOXHALF=$BOXHALF BOX=${BOX:-} VMIN=${VMIN:-} TAG=$TAG -> $OUT  Started: $(date)"

$PY scripts/render_wind_proj_evolution.py "$RUN" \
    --start 21 --end 151 \
    --coords CMx,CMy,CMz \
    --res 512 \
    "${boxflag[@]}" \
    "${vminflag[@]}" \
    --fields density,dissipation \
    --cmaps twilight,viridis \
    --n-jobs 6 \
    --fps 24 \
    --frames-dir "/tmp/windproj_${TAG}_${SLURM_JOB_ID}" \
    --out "$OUT"

echo "Finished: $(date)"
