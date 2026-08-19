#!/bin/bash
#SBATCH --job-name=faceon_density
#SBATCH --partition=cpu-zen4
#SBATCH --account=strw
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out

set -euo pipefail
cd /home/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
QUALITY="${QUALITY:-preview}"
NJOBS="${NJOBS:-8}"
WORKERS="${WORKERS:-8}"
FRAME_START="${FRAME_START:-0}"
FRAME_STOP="${FRAME_STOP:--1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/hey4/rich_tde/reports/movies/crete}"
VMAX="${VMAX:-3162277.6601683795}"
AMBIENT_FACTOR="${AMBIENT_FACTOR:-1e-8}"

EXTRA=()
[[ "${OVERWRITE:-0}" == 1 ]] && EXTRA+=(--overwrite)
[[ "${NO_ENCODE:-0}" == 1 ]] && EXTRA+=(--no-encode)
[[ "${ENCODE_ONLY:-0}" == 1 ]] && EXTRA+=(--encode-only)
[[ "${NO_EXAMPLES:-0}" == 1 ]] && EXTRA+=(--no-examples)

echo "Host=$(hostname) mode=$SLURM_ARRAY_TASK_ID quality=$QUALITY n_jobs=$NJOBS workers=$WORKERS vmax=$VMAX ambient_factor=$AMBIENT_FACTOR frames=[$FRAME_START,$FRAME_STOP)"
"$PY" works/movies/faceon_density.py \
    --mode "$SLURM_ARRAY_TASK_ID" \
    --quality "$QUALITY" \
    --output-root "$OUTPUT_ROOT" \
    --frame-start "$FRAME_START" \
    --frame-stop "$FRAME_STOP" \
    --n-jobs "$NJOBS" \
    --workers "$WORKERS" \
    --vmax "$VMAX" \
    --ambient-factor "$AMBIENT_FACTOR" \
    "${EXTRA[@]}"
