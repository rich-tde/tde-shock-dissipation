#!/bin/bash
#SBATCH --job-name=vr_rosseland
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=61G
#SBATCH --time=7-00:00:00
#SBATCH --output=jobs/logs/vr_rosseland_%j.out
#SBATCH --error=jobs/logs/vr_rosseland_%j.err

# Rosseland optical-depth movie: tau = int alpha_ross dr (scripts/render_rosseland_movie.py).
# Single derived field (not stored on disk, not registered in richio) -- lighter
# than the 4-field weighted conference-movie set (no weight cube), so the same
# gpu_strw defaults as render_multi.slurm have headroom, not less.
#
# Parametrise via --export=ALL,VAR=...:
#   CAMERA   = faceon | side   (default faceon; az/el presets in render_rosseland_movie.py)
#   BOX      = A | B | C   (C = pericentre close-up, face-on only)
#   OUTDIR   = output folder
#   TAG      = output stem: <outdir>/<tag>_<camera>_<box>.mp4
#   VMIN,VMAX = fixed colorbar bounds (tau, log); blank = auto
# Optional:
#   RES        = interpolation grid res (default 1024 = 1024^3 box)
#   RESOLUTION = output image px        (default 1024)
#   NJOBS      = frame-parallel worker processes (default 2)
#   WORKERS    = KDTree query threads per build  (default 24)
#   START,END  = snapshot range (default 21..151)
#   STEP       = use every Nth snapshot (default 1)
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

CAMERA="${CAMERA:-faceon}"
BOX="${BOX:-A}"
OUTDIR="${OUTDIR:-reports/movies/rosseland}"
TAG="${TAG:-rosseland}"
VMIN="${VMIN:-}"
VMAX="${VMAX:-}"
RES="${RES:-1024}"
RESOLUTION="${RESOLUTION:-1024}"
# cpu-zen4 (128c/384G): NJOBS=5 (~250-290G, memory-safe -- NJOBS=6 caused a few
# oom_kill events) with WORKERS=25 (NJOBS*WORKERS<=128).
NJOBS="${NJOBS:-5}"
WORKERS="${WORKERS:-25}"
START="${START:-21}"
END="${END:-151}"
STEP="${STEP:-1}"
# Persist frames on the SHARED filesystem (not node-local /tmp) and keep them, so
# a walltime/OOM kill doesn't discard a nearly-finished render -- a re-submit with
# the same FRAMES_ROOT resumes (render_rosseland_movie.py skips existing frames).
FRAMES_ROOT="${FRAMES_ROOT:-}"
KEEP_FRAMES="${KEEP_FRAMES:-1}"
RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR

mkdir -p "$OUTDIR"

EXTRA=()
[[ -n "$FRAMES_ROOT" ]] && { mkdir -p "$FRAMES_ROOT"; EXTRA+=(--frames-root "$FRAMES_ROOT"); }
[[ "$KEEP_FRAMES" == 1 ]] && EXTRA+=(--keep-frames)

echo "Host: $(hostname)  CAMERA=$CAMERA BOX=$BOX RES=$RES RESOLUTION=$RESOLUTION"\
" NJOBS=$NJOBS WORKERS=$WORKERS FRAMES_ROOT=${FRAMES_ROOT:-/tmp} KEEP=$KEEP_FRAMES"\
" -> $OUTDIR/${TAG}_${CAMERA}_${BOX}.mp4  Started: $(date)"

$PY scripts/render_rosseland_movie.py "$RUN" \
    --camera "$CAMERA" \
    --box "$BOX" \
    --scalebar \
    --bh-frame \
    --flip-x \
    --coords CMx,CMy,CMz \
    --start "$START" \
    --end "$END" \
    --step "$STEP" \
    --res "$RES" \
    --resolution "$RESOLUTION" \
    ${VMIN:+--vmin "$VMIN"} \
    ${VMAX:+--vmax "$VMAX"} \
    --workers "$WORKERS" \
    --n-jobs "$NJOBS" \
    --fps 24 \
    --outdir "$OUTDIR" \
    --tag "$TAG" \
    "${EXTRA[@]}"

echo "Finished: $(date)"
