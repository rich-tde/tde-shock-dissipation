#!/bin/bash
# Submit the redirected movie matrix (groups 1-3).  Usage:
#   bash jobs/submit_v2_groups.sh g1   # wind exact-plane maps (3, single-field; unchanged)
#   bash jobs/submit_v2_groups.sh g2   # canonical 3/4 view, multi-field 1024^3 (2 jobs x4 fields)
#   bash jobs/submit_v2_groups.sh g3   # side & top angles, multi-field 1024^3 (4 jobs x4 fields)
#   bash jobs/submit_v2_groups.sh all  # everything
#
# g2/g3 now use the shared-index multi-field driver (jobs/render_multi.slurm): one job
# per box[/angle] renders all 4 fields from ONE KDTree index build per snapshot at
# RES=1024 (was 4 separate single-field jobs at RES=224).  Colour limits are read from
# CR_JSON (works/movies/scan_color_range.py output); if absent, each field auto-scales.
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde
S=jobs/render_evolution_v2.slurm        # legacy single-field driver (g1 still uses it)
S2=jobs/render_multi.slurm              # shared-index multi-field driver (g2/g3)
FIELDS="density dissipation temperature bernoulli"
# sbatch --export splits items on commas, so list-valued vars (FIELDS/VMINS/VMAXS)
# use ';' as their delimiter; jobs/render_multi.slurm converts ';' -> ',' for the driver.
FIELDS_SEMI="density;dissipation;temperature;bernoulli"
CR_JSON="${CR_JSON:-reports/movies/color_ranges.json}"  # written by works/movies/scan_color_range.py
PY=/home/hey4/.conda/envs/richanalysis/bin/python

sub() { sbatch --parsable --export=ALL,"$1" "$S" | xargs -I{} echo "  {}  $1"; }

# Multi-field submit: one job renders all 4 fields from one index/snapshot.
sub2() { sbatch --parsable --export=ALL,"$1" "$S2" | xargs -I{} echo "  {}  $1"; }

# Echo "VMINS=...;...,VMAXS=...;..." (semicolon-delimited, comma-separated KEY=VAL)
# for a scan label from CR_JSON, or empty if the JSON/label is missing (-> the driver
# falls back to per-field auto colour bounds).
ranges() {
  local label="$1"
  [[ -f "$CR_JSON" ]] || { echo ""; return; }
  "$PY" - "$CR_JSON" "$label" <<'PYEOF'
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))["configs"].get(sys.argv[2])
    if cfg:
        vmins = cfg["vmins"].replace(",", ";")
        vmaxs = cfg["vmaxs"].replace(",", ";")
        print(f"VMINS={vmins},VMAXS={vmaxs}")
except Exception:
    pass
PYEOF
}

g1() {  # wind density, exact xy/xz/yz planes, no rotation
  # Box C (+-200), grid res 1024^3 (the physics resolution), own auto colorbar.
  # 1024^3 ~ 50 GB/build -> one build at a time (NJOBS=1) with all cores on the
  # query (WORKERS=16); ~minutes/frame, fine under the 7-day partition limit.
  echo "== group 1: wind exact-plane maps (box C, res 1024^3, auto colorbar) =="
  for PLANE in xy xz yz; do
    sbatch --parsable --cpus-per-task=48 \
      --export=ALL,FIELD=density,BOX=C,SELECT=wind,PLANE=$PLANE,SPINFRAMES=0,VMIN_OVR=,RES=1024,RESOLUTION=2048,NJOBS=2,WORKERS=24,OUTDIR=reports/movies/wind_planes,TAG=g1_${PLANE}_C_density \
      "$S" | xargs -I{} echo "  {}  PLANE=$PLANE res=1024^3 (48 cores, njobs=2 workers=24)"
  done
}

g2() {  # canonical 3/4 view, all 4 fields per box from one shared index (1024^3)
  echo "== group 2: canonical 3/4 view (multi-field, res 1024^3) =="
  for BOX in A B; do
    local cr; cr="$(ranges g2_${BOX})"
    sub2 "FIELDS=$FIELDS_SEMI,BOX=$BOX,OUTDIR=reports/movies/canonical,TAG=g2,NAME_TMPL=g2_{field}_${BOX}${cr:+,$cr}"
  done
}

g3() {  # side (along y) & top (along z) angles, all 4 fields per box (1024^3)
  echo "== group 3: canonical side & top angles (multi-field, res 1024^3) =="
  declare -A AZ=( [side]=0  [top]=20 )
  declare -A EL=( [side]=15 [top]=72 )
  for ANGLE in side top; do for BOX in A B; do
    local cr; cr="$(ranges g3_${ANGLE}_${BOX})"
    sub2 "FIELDS=$FIELDS_SEMI,BOX=$BOX,AZ=${AZ[$ANGLE]},EL=${EL[$ANGLE]},OUTDIR=reports/movies/angles,TAG=g3_${ANGLE},NAME_TMPL=g3_${ANGLE}_{field}_${BOX}${cr:+,$cr}"
  done; done
}

case "${1:?need g1|g2|g3|all}" in
  g1) g1 ;;
  g2) g2 ;;
  g3) g3 ;;
  all) g1; g2; g3 ;;
  *) echo "unknown group $1" >&2; exit 1 ;;
esac
