#!/bin/bash
#SBATCH --job-name=cr_scan
#SBATCH --partition=cpu-short
#SBATCH --account=strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=0:40:00
#SBATCH --output=jobs/logs/cr_scan_%j.out
#SBATCH --error=jobs/logs/cr_scan_%j.err

# Scan first+last snapshots for fixed projection colour limits, per (box, camera).
# Serial builds (one index at a time) -> all cores on the KDTree query.  Scans at
# RES (default 512): colour bounds are percentiles of a column-integral projection
# and are ~independent of interpolation res, while keeping memory small.
# Writes a JSON of per-config vmins/vmaxs consumed by jobs/submit_movies.sh.
#
# Parametrise via --export=ALL,VAR=...:
#   CAMERAS = comma list of cameras (canonical,side,top,faceon)   default: all four
#   BOXES   = comma list of boxes                                 default: A,B
#   SNAPS   = first,last snapshot indices                         default: 21,151
#   RES     = scan interpolation res                              default: 512
#   OUT     = output JSON                          default: reports/movies/color_ranges.json
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde
PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR

CAMERAS="${CAMERAS:-canonical,side,top,faceon}"
BOXES="${BOXES:-A,B}"
SNAPS="${SNAPS:-21,151}"
RES="${RES:-512}"
OUT="${OUT:-reports/movies/color_ranges.json}"

echo "Host: $(hostname)  CAMERAS=$CAMERAS BOXES=$BOXES SNAPS=$SNAPS RES=$RES -> $OUT  Started: $(date)"
$PY works/movies/scan_color_range.py "$RUN" \
    --boxes "$BOXES" --snaps "$SNAPS" --cameras "$CAMERAS" \
    --res "$RES" --resolution "$RES" --workers 48 --out "$OUT"
echo "Finished: $(date)"
