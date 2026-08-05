#!/bin/bash
#SBATCH --job-name=vr_evosel
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=06:00:00
#SBATCH --output=jobs/vr_evosel_%j.out
#SBATCH --error=jobs/vr_evosel_%j.err

# Unbound-outflow selection movies: fixed boxes A/B, map-style r_t scale bar,
# multi-spin camera (pause+spin 360 at 0.5/1.0/1.5 t_fb, 2x360 at the end), all
# frame-corrected to the BH frame.  Per-field colour settings match the existing
# evo_* movies so the two are directly comparable.
#
# Parametrise via --export:
#   FIELD  = density | dissipation | temperature | bernoulli
#   BOX    = A | B
#   SELECT = unbound_out | unbound_out_xpos | unbound_out_xneg | none
#   OUTDIR = output folder (e.g. reports/movies/sel_xpos)
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

FIELD="${FIELD:-density}"
BOX="${BOX:-A}"
SELECT="${SELECT:-unbound_out_xpos}"
OUTDIR="${OUTDIR:-reports/movies/sel_xpos}"
RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR

# Per-field colour settings (reuse the existing evo_* recipe).
WEIGHT=""; VMIN=""; LINTHRESH=""
case "$FIELD" in
  density)     CMAP=twilight; NORM=log;    VMIN=1e13                      ;;
  dissipation) CMAP=viridis;  NORM=log                                    ;;
  temperature) CMAP=inferno;  NORM=log;    WEIGHT=density                 ;;
  bernoulli)   CMAP=RdBu_r;   NORM=symlog; WEIGHT=density; LINTHRESH=1     ;;
  *) echo "unknown FIELD=$FIELD" >&2; exit 1 ;;
esac

mkdir -p "$OUTDIR"
OUT="${OUTDIR}/sel_${BOX}_${FIELD}.mp4"

echo "Host: $(hostname)  FIELD=$FIELD BOX=$BOX SELECT=$SELECT -> $OUT  Started: $(date)"

$PY scripts/render_evolution.py "$RUN" \
    --field "$FIELD" \
    --mode projection \
    --box "$BOX" \
    --select "$SELECT" \
    --scalebar \
    --bh-frame \
    --flip-x \
    --cmap "$CMAP" \
    --norm "$NORM" \
    ${WEIGHT:+--weight "$WEIGHT"} \
    ${VMIN:+--vmin "$VMIN"} \
    ${LINTHRESH:+--linthresh "$LINTHRESH"} \
    --coords CMx,CMy,CMz \
    --res 224 \
    --resolution 1024 \
    --spin-tfb 0.5,1.0,1.5 \
    --final-turns 2 \
    --spin-frames 90 \
    --n-jobs 6 \
    --fps 24 \
    --frames-dir "/tmp/sel_${SELECT}_${BOX}_${FIELD}_${SLURM_JOB_ID}" \
    --out "$OUT"

echo "Finished: $(date)"
