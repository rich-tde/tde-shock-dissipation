#!/bin/bash
#SBATCH --job-name=vr_evosp
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=02:30:00
#SBATCH --output=jobs/vr_evosp_%j.out
#SBATCH --error=jobs/vr_evosp_%j.err

# Evolution (fixed camera) then spin the final state, projection only.
# Combo via --export: FIELD (density|dissipation), BOX (wide|disk), CMAP.
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

FIELD="${FIELD:-density}"
BOX="${BOX:-wide}"
CMAP="${CMAP:-twilight}"
RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR

name="evo_${BOX}_${FIELD}"
OUT="reports/movies/${name}.mp4"

echo "Host: $(hostname)  FIELD=$FIELD BOX=$BOX CMAP=$CMAP -> $OUT  Started: $(date)"

$PY works/movies/render_evolution.py "$RUN" \
    --field "$FIELD" \
    --mode projection \
    --box "$BOX" \
    ${BOXFIELD:+--box-field "$BOXFIELD"} \
    ${WEIGHT:+--weight "$WEIGHT"} \
    ${FLIPX:+--flip-x} \
    ${BHFRAME:+--bh-frame} \
    ${VMIN:+--vmin "$VMIN"} \
    ${NORM:+--norm "$NORM"} \
    ${LINTHRESH:+--linthresh "$LINTHRESH"} \
    ${VMAX:+--vmax "$VMAX"} \
    --cmap "$CMAP" \
    --coords CMx,CMy,CMz \
    --res 224 \
    --resolution 1024 \
    --spin-frames 90 \
    --spin-angle 360 \
    --n-jobs 6 \
    --fps 24 \
    --frames-dir "/tmp/${name}_${SLURM_JOB_ID}" \
    --out "$OUT"

echo "Finished: $(date)"
