#!/bin/bash
#SBATCH --job-name=scan_rosseland
#SBATCH --partition=cpu-short
#SBATCH --account=strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=1:30:00
#SBATCH --output=jobs/logs/%x_%j.out
#SBATCH --error=jobs/logs/%x_%j.err

# Rosseland optical-depth colorbar-range study (scripts/scan_rosseland_range.py).
# Pools tau = int alpha_ross dr over sampled snapshots x (box A/B) x (faceon/side)
# and prints recommended fixed VMIN/VMAX per (box, camera).
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR
SNAPS="${SNAPS:-21,40,60,80,100,120,140,151}"
RES="${RES:-512}"
RESOLUTION="${RESOLUTION:-512}"
WORKERS="${WORKERS:-24}"

echo "Host: $(hostname)  SNAPS=$SNAPS RES=$RES  Started: $(date)"

$PY scripts/scan_rosseland_range.py "$RUN" \
    --boxes A,B \
    --cameras faceon,side \
    --snaps "$SNAPS" \
    --res "$RES" \
    --resolution "$RESOLUTION" \
    --workers "$WORKERS" \
    --out reports/movies/rosseland/tau_range.json

echo "Finished: $(date)"
