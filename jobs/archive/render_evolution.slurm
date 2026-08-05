#!/bin/bash
#SBATCH --job-name=vr_evo
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=02:30:00
#SBATCH --output=jobs/vr_evo_%j.out
#SBATCH --error=jobs/vr_evo_%j.err

# Time-evolution movie. Combo passed via --export:
#   MODE = projection|volume   BOX = wide|disk   ROT = 0|1
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

MODE="${MODE:-projection}"
BOX="${BOX:-wide}"
ROT="${ROT:-0}"
RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR

name="evo_${BOX}_${MODE}"
rotflag=""
if [ "$ROT" = "1" ]; then rotflag="--rotate"; name="${name}_rot"; fi
OUT="reports/gifs/${name}.mp4"

echo "Host: $(hostname)  MODE=$MODE BOX=$BOX ROT=$ROT  -> $OUT  Started: $(date)"

# 63M-cell NPY snapshots are memory-heavy: keep workers modest.
$PY scripts/render_evolution.py "$RUN" \
    --field density \
    --mode "$MODE" \
    --box "$BOX" \
    $rotflag \
    --res 224 \
    --resolution 1024 \
    --coords CMx,CMy,CMz \
    --n-jobs 6 \
    --fps 24 \
    --frames-dir "/tmp/${name}_${SLURM_JOB_ID}" \
    --out "$OUT"

echo "Finished: $(date)"
