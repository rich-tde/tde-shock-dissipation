#!/bin/bash
set -euo pipefail
cd /home/hey4/rich_tde

case "${1:-}" in
  preview)
    QUALITY=preview
    DEFAULT_NJOBS=4
    DEFAULT_WORKERS=8
    SCHEDULER=(--partition=cpu-short --account=strw --time=4:00:00 --mem=120G --cpus-per-task=32)
    ;;
  production)
    QUALITY=production
    DEFAULT_NJOBS=8
    DEFAULT_WORKERS=8
    SCHEDULER=(--partition=cpu-zen4 --account=strw --time=7-00:00:00 --mem=240G --cpus-per-task=64)
    ;;
  *)
    echo "usage: bash jobs/movies/submit_faceon_density.sh preview|production [density|dissipation]" >&2
    exit 2
    ;;
esac

FIELD="${2:-${FIELD:-density}}"
if [[ "$FIELD" != density && "$FIELD" != dissipation ]]; then
  echo "field must be density or dissipation" >&2
  exit 2
fi

# Production is intentionally not invoked by the preview workflow. Submit it
# only after the preview contact sheets and movies have been approved.
sbatch --parsable \
  "${SCHEDULER[@]}" \
  --export=ALL,QUALITY="$QUALITY",FIELD="$FIELD",NJOBS="${NJOBS:-$DEFAULT_NJOBS}",WORKERS="${WORKERS:-$DEFAULT_WORKERS}" \
  jobs/movies/render_faceon_density.slurm
