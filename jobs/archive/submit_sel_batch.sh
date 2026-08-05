#!/bin/bash
# Submit one selection batch = 4 fields x 2 boxes = 8 movies.
# Usage:  bash jobs/submit_sel_batch.sh <SELECT> <OUTDIR>
#   SELECT = unbound_out_xpos | unbound_out_xneg | unbound_out | none
#   OUTDIR = e.g. reports/movies/sel_xpos
# Example (Batch P, pericenter side, render first):
#   bash jobs/submit_sel_batch.sh unbound_out_xpos reports/movies/sel_xpos
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde

SELECT="${1:?need SELECT}"
OUTDIR="${2:?need OUTDIR}"
mkdir -p "$OUTDIR"

for FIELD in density dissipation temperature bernoulli; do
  for BOX in A B; do
    jid=$(sbatch --parsable \
      --export=ALL,FIELD="$FIELD",BOX="$BOX",SELECT="$SELECT",OUTDIR="$OUTDIR" \
      jobs/render_evolution_select.slurm)
    echo "submitted $jid  FIELD=$FIELD BOX=$BOX SELECT=$SELECT -> $OUTDIR/sel_${BOX}_${FIELD}.mp4"
  done
done
