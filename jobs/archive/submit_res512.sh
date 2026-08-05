#!/bin/bash
# FAST set: g2/g3 conference movies at interpolation RES=512 (output RESOLUTION=1024),
# whole movie per job (render+encode, MODE=full) on cpu-short.  At res 512 a worker
# peaks ~10-15 GB and the 512^3 KDTree query is ~8x cheaper than 1024^3, so a full
# 221-frame movie fits comfortably in one <4 h job at high n_jobs.  Colour limits come
# from reports/movies/color_ranges.json (which was itself scanned at res 512 -> exact).
#
# Usage:
#   bash jobs/submit_res512.sh g2     # canonical A,B           (2 jobs)
#   bash jobs/submit_res512.sh g3     # side & top x A,B        (4 jobs)
#   bash jobs/submit_res512.sh all    # everything             (6 jobs)
# Tunables (env): RES RESOLUTION NJOBS WORKERS MEM TIME CPUS
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde
S=jobs/render_multi.slurm
PY=/home/hey4/.conda/envs/richanalysis/bin/python
CR_JSON="${CR_JSON:-reports/movies/color_ranges.json}"
FIELDS_SEMI="density;dissipation;temperature;bernoulli"

RES="${RES:-512}"
RESOLUTION="${RESOLUTION:-1024}"
NJOBS="${NJOBS:-8}"
WORKERS="${WORKERS:-8}"     # NJOBS*WORKERS <= 64 (cpu-short MaxCPUsPerNode)
MEM="${MEM:-180G}"
TIME="${TIME:-4:00:00}"
CPUS="${CPUS:-64}"
PART=(--partition=cpu-short --account=strw)

ranges() {
  local label="$1"; [[ -f "$CR_JSON" ]] || { echo ""; return; }
  "$PY" - "$CR_JSON" "$label" <<'PYEOF'
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))["configs"].get(sys.argv[2])
    if cfg:
        print(f"VMINS={cfg['vmins'].replace(',',';')},VMAXS={cfg['vmaxs'].replace(',',';')}")
except Exception:
    pass
PYEOF
}

# submit_movie OUTDIR NAME_TMPL BOX AZ EL RANGES_LABEL TAG
submit_movie() {
  local OUTDIR="$1" NAME_TMPL="$2" BOX="$3" AZ="$4" EL="$5" RL="$6" TAG="$7"
  local cr; cr="$(ranges "$RL")"
  # Frames to a per-movie network dir (not /tmp): MODE=full encodes then cleans it up,
  # but if the encode step ever fails the frames survive for a cheap MODE=encode rerun.
  local FR="reports/movies/_frames/r512_$RL"; mkdir -p "$FR"
  local exp="FIELDS=$FIELDS_SEMI,BOX=$BOX,AZ=$AZ,EL=$EL,RES=$RES,RESOLUTION=$RESOLUTION,NJOBS=$NJOBS,WORKERS=$WORKERS,OUTDIR=$OUTDIR,TAG=$TAG,NAME_TMPL=$NAME_TMPL,FRAMES_ROOT=$FR${cr:+,$cr}"
  sbatch --parsable "${PART[@]}" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
    --job-name="r512_$RL" --output=jobs/r512_%j.out --error=jobs/r512_%j.err \
    --export=ALL,"$exp" "$S" | xargs -I{} echo "  {}  $RL ($NAME_TMPL) res=$RES"
}

g2() {
  echo "== res-512 group 2: canonical A,B =="
  for BOX in A B; do
    submit_movie reports/movies/canonical "g2_{field}_${BOX}" "$BOX" 45 26 "g2_${BOX}" g2
  done
}

g3() {
  echo "== res-512 group 3: side & top x A,B =="
  declare -A AZ=( [side]=0  [top]=20 )
  declare -A EL=( [side]=15 [top]=72 )
  for ANGLE in side top; do for BOX in A B; do
    submit_movie reports/movies/angles "g3_${ANGLE}_{field}_${BOX}" "$BOX" \
      "${AZ[$ANGLE]}" "${EL[$ANGLE]}" "g3_${ANGLE}_${BOX}" "g3_${ANGLE}"
  done; done
}

case "${1:?need g2|g3|all}" in
  g2) g2 ;;
  g3) g3 ;;
  all) g2; g3 ;;
  *) echo "unknown group $1" >&2; exit 1 ;;
esac
