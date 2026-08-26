#!/usr/bin/env bash
# Submit the wind-projection evolution movies: wide (+-200) and narrow (+-30).
# Each frame is a 2x3 panel (density/dissipation x xy/xz/yz) of the unbound wind.
# Usage: bash jobs/movies/submit_wind_proj.sh
set -euo pipefail
cd /home/hey4/rich_tde

mkdir -p reports/movies/wind_proj

for spec in "200:wide" "30:narrow"; do
  BOXHALF="${spec%%:*}"
  TAG="${spec##*:}"
  job_id=$(sbatch --parsable \
    --export=ALL,BOXHALF="${BOXHALF}",TAG="${TAG}" \
    jobs/movies/render_wind_proj.slurm)
  echo "submitted ${job_id}  BOXHALF=${BOXHALF} TAG=${TAG} -> reports/movies/wind_proj/wind_proj_${TAG}.mp4"
done
