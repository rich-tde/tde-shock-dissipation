#!/bin/bash
#SBATCH --job-name=vr_evo2
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=61G
#SBATCH --time=20:00:00
#SBATCH --output=jobs/vr_evo2_%j.out
#SBATCH --error=jobs/vr_evo2_%j.err

# General evolution-movie driver: fixed boxes A/B, r_t scale bar, BH-frame, x
# flipped, single trailing rotation (NO multi-spin).  Per-field colours match
# the existing evo_* movies.  Parametrise via --export:
#   FIELD  = density | dissipation | temperature | bernoulli
#   BOX    = A | B
#   OUTDIR = output folder
#   TAG    = output filename stem (TAG.mp4)
# Optional:
#   SELECT     = wind            (unbound disc-plane wind; default none)
#   PLANE      = xy | xz | yz    (exact axis-aligned projection; forces no rotation)
#   AZ, EL     = camera azimuth/elevation (default 45/26)
#   SPINFRAMES = trailing rotation frames (default 90; 0 = none)
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

FIELD="${FIELD:-density}"
BOX="${BOX:-A}"
OUTDIR="${OUTDIR:-reports/movies/canonical}"
TAG="${TAG:-${FIELD}_${BOX}}"
RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR

WEIGHT=""; VMIN=""; LINTHRESH=""
case "$FIELD" in
  density)     CMAP=twilight; NORM=log;    VMIN=1e13                      ;;
  dissipation) CMAP=viridis;  NORM=log                                    ;;
  temperature) CMAP=inferno;  NORM=log;    WEIGHT=density                 ;;
  bernoulli)   CMAP=RdBu_r;   NORM=symlog; WEIGHT=density; LINTHRESH=1     ;;
  *) echo "unknown FIELD=$FIELD" >&2; exit 1 ;;
esac
# Caller overrides: VMIN_OVR (set to "" for the field's own auto colorbar),
# grid sampling RES and output RESOLUTION.
VMIN="${VMIN_OVR-$VMIN}"
RES="${RES:-224}"
RESOLUTION="${RESOLUTION:-1024}"

mkdir -p "$OUTDIR"
OUT="${OUTDIR}/${TAG}.mp4"

echo "Host: $(hostname)  FIELD=$FIELD BOX=$BOX SELECT=${SELECT:-none} PLANE=${PLANE:-none} "\
"AZ=${AZ:-45} EL=${EL:-26} SPIN=${SPINFRAMES:-90} -> $OUT  Started: $(date)"

$PY scripts/render_evolution.py "$RUN" \
    --field "$FIELD" \
    --mode projection \
    --box "$BOX" \
    ${SELECT:+--select "$SELECT"} \
    ${PLANE:+--plane "$PLANE"} \
    --scalebar \
    --bh-frame \
    --flip-x \
    --cmap "$CMAP" \
    --norm "$NORM" \
    ${WEIGHT:+--weight "$WEIGHT"} \
    ${VMIN:+--vmin "$VMIN"} \
    ${LINTHRESH:+--linthresh "$LINTHRESH"} \
    ${AZ:+--azimuth "$AZ"} \
    ${EL:+--elevation "$EL"} \
    --spin-frames "${SPINFRAMES:-90}" \
    --spin-angle 360 \
    --coords CMx,CMy,CMz \
    --res "$RES" \
    --resolution "$RESOLUTION" \
    --workers "${WORKERS:-8}" \
    --n-jobs "${NJOBS:-6}" \
    --fps 24 \
    --frames-dir "/tmp/${TAG}_${SLURM_JOB_ID}" \
    --out "$OUT"

echo "Finished: $(date)"
