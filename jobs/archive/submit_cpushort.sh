#!/bin/bash
# Submit the g2/g3 conference movies on the cpu-short partition (4 h walltime cap),
# chunking each movie's frames across several short render jobs + one dependent
# encode job.  cpu-short has big RAM (~370 GB/node) so res-1024 fits at n_jobs>=4.
#
# Per movie (box[/angle]):
#   * total frames = N_EVO (snaps 21..151) + SPIN
#   * render jobs cover disjoint global frame windows [s, s+CHUNK) into a SHARED
#     frames-root on zfsstore (NOT /tmp: chunks land on different nodes)
#   * an --encode-only job (afterok on all chunks) stitches the movie per field
#
# Usage:
#   bash jobs/submit_cpushort.sh g2    # canonical A,B            (2 movies)
#   bash jobs/submit_cpushort.sh g3    # side & top x A,B         (4 movies)
#   bash jobs/submit_cpushort.sh all   # everything              (6 movies)
# Tunables (env): CHUNK NJOBS WORKERS MEM TIME CPUS
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde
S=jobs/render_multi.slurm
PY=/home/hey4/.conda/envs/richanalysis/bin/python
RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR
CR_JSON="${CR_JSON:-reports/movies/color_ranges.json}"
FIELDS_SEMI="density;dissipation;temperature;bernoulli"

SPIN="${SPIN:-90}"
CHUNK="${CHUNK:-75}"          # frames per render job (<<4 h at NJOBS>=4)
NJOBS="${NJOBS:-4}"
WORKERS="${WORKERS:-16}"      # NJOBS*WORKERS <= 64 (cpu-short MaxCPUsPerNode)
MEM="${MEM:-330G}"           # NJOBS * ~72 GB/worker (res 1024) < MEM
TIME="${TIME:-4:00:00}"
CPUS="${CPUS:-64}"
PART=(--partition=cpu-short --account=strw)
FRAMEDIR_BASE=reports/movies/_frames

# Number of evolution frames (snaps 21..151).
N_EVO=$("$PY" -c "import sys; sys.path.insert(0,'works/movies'); import render_evolution as r; print(len(r.find_snapshots('$RUN',21,151)))")
TOTAL=$((N_EVO + SPIN))
echo "N_EVO=$N_EVO SPIN=$SPIN TOTAL=$TOTAL  CHUNK=$CHUNK NJOBS=$NJOBS WORKERS=$WORKERS MEM=$MEM"

# ranges LABEL -> "VMINS=...;...,VMAXS=...;..." (semicolon lists), or empty.
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

# submit_movie OUTDIR NAME_TMPL BOX AZ EL RANGES_LABEL FRAMES_SUBDIR TAG
submit_movie() {
  local OUTDIR="$1" NAME_TMPL="$2" BOX="$3" AZ="$4" EL="$5" RL="$6" SUB="$7" TAG="$8"
  local FR="$FRAMEDIR_BASE/$SUB"
  local cr; cr="$(ranges "$RL")"
  local common="FIELDS=$FIELDS_SEMI,BOX=$BOX,AZ=$AZ,EL=$EL,OUTDIR=$OUTDIR,TAG=$TAG,NAME_TMPL=$NAME_TMPL,SPINFRAMES=$SPIN,NJOBS=$NJOBS,WORKERS=$WORKERS,FRAMES_ROOT=$FR${cr:+,$cr}"
  echo "-- movie $SUB ($RL): frames-root $FR"
  mkdir -p "$FR"
  local deps=() s e jid
  for ((s=0; s<TOTAL; s+=CHUNK)); do
    e=$((s + CHUNK)); (( e > TOTAL )) && e=$TOTAL
    jid=$(sbatch --parsable "${PART[@]}" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
      --job-name="cs_${SUB}_${s}" \
      --export=ALL,"$common,MODE=render,FRAME_START=$s,FRAME_STOP=$e" "$S")
    echo "   render [$s,$e) -> $jid"
    deps+=("$jid")
  done
  local depstr; depstr=$(IFS=:; echo "${deps[*]}")
  jid=$(sbatch --parsable "${PART[@]}" --cpus-per-task=4 --mem=8G --time=0:30:00 \
    --job-name="cs_${SUB}_enc" --dependency=afterok:"$depstr" \
    --export=ALL,"$common,MODE=encode" "$S")
  echo "   encode (afterok:$depstr) -> $jid"
}

g2() {
  echo "== group 2 on cpu-short: canonical A,B =="
  for BOX in A B; do
    submit_movie reports/movies/canonical "g2_{field}_${BOX}" "$BOX" 45 26 "g2_${BOX}" "g2_${BOX}" g2
  done
}

g3() {
  echo "== group 3 on cpu-short: side & top x A,B =="
  declare -A AZ=( [side]=0  [top]=20 )
  declare -A EL=( [side]=15 [top]=72 )
  for ANGLE in side top; do for BOX in A B; do
    submit_movie reports/movies/angles "g3_${ANGLE}_{field}_${BOX}" "$BOX" \
      "${AZ[$ANGLE]}" "${EL[$ANGLE]}" "g3_${ANGLE}_${BOX}" "g3_${ANGLE}_${BOX}" "g3_${ANGLE}"
  done; done
}

case "${1:?need g2|g3|all}" in
  g2) g2 ;;
  g3) g3 ;;
  all) g2; g3 ;;
  *) echo "unknown group $1" >&2; exit 1 ;;
esac
