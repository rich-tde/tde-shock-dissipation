#!/bin/bash
# ============================================================================
# Single entry point to (re)produce the conference TDE movie sets.
#
#   bash jobs/submit_movies.sh scan      # colour-range scan (all cameras) -> color_ranges.json
#   bash jobs/submit_movies.sh g2        # canonical 3/4 view      (boxes A,B; 2 jobs x4 fields)
#   bash jobs/submit_movies.sh g3        # side + top angles       (boxes A,B; 4 jobs x4 fields)
#   bash jobs/submit_movies.sh faceon    # face-on xy projection, no rotation, frames KEPT
#                                        # boxes A, B (wide) and C (+-2.5 r_p close-up)
#   bash jobs/submit_movies.sh all       # g2 + g3 + faceon
#
# Each render job uses the shared-index multi-field driver (jobs/render_multi.slurm
# -> works/movies/render_evolution_multi.py): ONE KDTree build per snapshot renders all 4
# fields (density, dissipation, temperature, bernoulli).  Colour limits are read from
# reports/movies/color_ranges.json (produced by the `scan` case); if a label is
# missing the field auto-scales.  Output movies:
#   g2     -> reports/movies/canonical/g2_<field>_<A|B>.mp4
#   g3     -> reports/movies/angles/g3_<side|top>_<field>_<A|B>.mp4
#   faceon -> reports/movies/faceon/faceon_<field>_<A|B|C>.mp4 (+ PNGs under faceon/frames/)
#
# Defaults render at interpolation RES=512 (output RESOLUTION=1024) as whole movies on
# cpu-short (big RAM, fits res 512 in <4 h).  Tunables (env):
#   RES RESOLUTION NJOBS WORKERS MEM TIME CPUS PARTITION ACCOUNT SPIN
# For a res-1024 set the whole-movie wall time can exceed cpu-short's 4 h cap; render
# in frame windows instead (render_multi.slurm supports MODE=render/encode +
# FRAME_START/FRAME_STOP/FRAMES_ROOT -- see jobs/archive/submit_cpushort.sh).
# ============================================================================
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde
RENDER=jobs/render_multi.slurm
SCAN=jobs/scan_color.slurm
PY=/home/hey4/.conda/envs/richanalysis/bin/python
CR_JSON="${CR_JSON:-reports/movies/color_ranges.json}"
FIELDS_SEMI="${FIELDS:-density;dissipation;temperature;bernoulli}"  # ';' (sbatch --export splits on ',')

RES="${RES:-512}"; RESOLUTION="${RESOLUTION:-1024}"
NJOBS="${NJOBS:-8}"; WORKERS="${WORKERS:-8}"       # NJOBS*WORKERS <= CPUS
MEM="${MEM:-180G}"; TIME="${TIME:-3:00:00}"; CPUS="${CPUS:-64}"
PART=(--partition="${PARTITION:-cpu-short}" --account="${ACCOUNT:-strw}")

# Boxes rendered by the faceon case.  C is the pericentre close-up (+-2.5 r_p
# across, see render_evolution.BOX_PRESETS): a pencil beam keeping the full
# line-of-sight depth, so it is a true magnification of the wide frames -- same
# column values, so it reuses box A's colour ranges and needs no new scan.
FACEON_BOXES="${FACEON_BOXES:-A B C}"
LOG=(--output=jobs/logs/%x_%j.out --error=jobs/logs/%x_%j.err)

# "VMINS=..;..,VMAXS=..;.." (semicolon lists, aligned with FIELDS_SEMI's order --
# may be a subset of the full field list, e.g. FIELDS=dissipation for a
# single-field re-render) for a scan label, or empty.
ranges() {
  local label="$1"; local fields_csv="${FIELDS_SEMI//;/,}"
  [[ -f "$CR_JSON" ]] || { echo ""; return; }
  "$PY" - "$CR_JSON" "$label" "$fields_csv" <<'PYEOF'
import json, sys
FULL = ["density", "dissipation", "temperature", "bernoulli"]
try:
    c = json.load(open(sys.argv[1]))["configs"].get(sys.argv[2])
    if c:
        fields = sys.argv[3].split(",")
        vmins_full = c["vmins"].split(",")
        vmaxs_full = c["vmaxs"].split(",")
        idx = [FULL.index(f) for f in fields]
        vmins = ";".join(vmins_full[i] for i in idx)
        vmaxs = ";".join(vmaxs_full[i] for i in idx)
        print(f"VMINS={vmins},VMAXS={vmaxs}")
except Exception: pass
PYEOF
}

# emit_movie OUTDIR NAME_TMPL RANGES_LABEL BOX AZ EL SPIN KEEP TAG FRAMES_ROOT
emit_movie() {
  local OUTDIR="$1" NAME_TMPL="$2" RL="$3" BOX="$4" AZ="$5" EL="$6" SPIN="$7" KEEP="$8" TAG="$9" FR="${10}"
  local cr; cr="$(ranges "$RL")"
  [[ -n "$FR" ]] && mkdir -p "$FR"
  local exp="FIELDS=$FIELDS_SEMI,BOX=$BOX,AZ=$AZ,EL=$EL,RES=$RES,RESOLUTION=$RESOLUTION,SPINFRAMES=${SPIN_OVR:-$SPIN},NJOBS=$NJOBS,WORKERS=$WORKERS,MODE=full,OUTDIR=$OUTDIR,TAG=$TAG,NAME_TMPL=$NAME_TMPL${KEEP:+,KEEP=1}${FR:+,FRAMES_ROOT=$FR}${cr:+,$cr}"
  sbatch --parsable "${PART[@]}" "${LOG[@]}" --job-name="${TAG}_${BOX}" \
    --cpus-per-task="$CPUS" --mem="$MEM" --time="${TIME_OVR:-$TIME}" \
    --export=ALL,"$exp" "$RENDER" | xargs -I{} echo "  {}  $NAME_TMPL (box $BOX, az$AZ el$EL spin${SPIN_OVR:-$SPIN})"
}

g2() {  # canonical 3/4 view + trailing rotation
  echo "== g2: canonical 3/4 view =="
  for B in A B; do emit_movie reports/movies/canonical "g2_{field}_$B" "g2_$B" "$B" 45 26 90 "" g2 ""; done
}
g3() {  # side (along y) & top (along z) angles + trailing rotation
  echo "== g3: side & top angles =="
  declare -A AZ=( [side]=0 [top]=20 ) EL=( [side]=15 [top]=72 )
  for V in ${ANGLES:-side top}; do for B in A B; do
    emit_movie reports/movies/angles "g3_${V}_{field}_$B" "g3_${V}_$B" "$B" "${AZ[$V]}" "${EL[$V]}" 90 "" "g3_${V}" ""
  done; done
}
faceon() {  # exact xy-plane projection, NO rotation, every frame PNG kept under movies/
  # One job per box in FACEON_BOXES (A, B wide; C the pericentre close-up).
  # Run on a 7-day partition: cpu-short's 4 h cap has bitten these before, and
  # overshooting the request costs nothing.
  echo "== faceon: xy-plane, no rotation, frames kept, boxes: $FACEON_BOXES =="
  local PART_SAVE=("${PART[@]}")
  PART=(--partition="${FACEON_PARTITION:-cpu-zen4}" --account="${ACCOUNT:-strw}")
  TIME_OVR="${FACEON_TIME:-7-00:00:00}"
  for B in $FACEON_BOXES; do
    # C is spatially inside A and carries the same column, so it reuses A's ranges.
    local RL="faceon_$B"; [[ "$B" == C ]] && RL="faceon_A"
    emit_movie reports/movies/faceon "faceon_{field}_$B" "$RL" "$B" 0 90 0 1 faceon "reports/movies/faceon/frames/$B"
  done
  PART=("${PART_SAVE[@]}"); unset TIME_OVR
}
scan() {  # colour-range scan over all cameras -> CR_JSON
  echo "== scan: colour ranges (all cameras) =="
  sbatch --parsable "${PART[@]}" "${LOG[@]}" --job-name=cr_scan "$SCAN" | xargs -I{} echo "  {}  -> $CR_JSON"
}

case "${1:?need scan|g2|g3|faceon|all}" in
  scan) scan ;;
  g2) g2 ;;
  g3) g3 ;;
  faceon) faceon ;;
  all) g2; g3; faceon ;;
  *) echo "unknown case $1" >&2; exit 1 ;;
esac
