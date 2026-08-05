#!/bin/bash
# ============================================================================
# Submit Rosseland optical-depth movies (tau = int alpha_ross dr): face-on and
# roughly side-on cameras, box presets A, B and C (mirrors submit_movies.sh's
# g3()/faceon() pattern).
#
#   bash jobs/submit_rosseland.sh faceon   # face-on xy projection      (3 jobs: A,B,C)
#   bash jobs/submit_rosseland.sh side     # roughly side-on (az0 el15) (2 jobs: A,B)
#   bash jobs/submit_rosseland.sh both     # faceon + side              (5 jobs)
#
# Each job runs jobs/render_rosseland.slurm -> scripts/render_rosseland_movie.py.
# Output: reports/movies/rosseland/rosseland_<faceon|side>_<A|B|C>.mp4
#
# Tunables (env): RES RESOLUTION NJOBS WORKERS MEM TIME CPUS PARTITION ACCOUNT
# ============================================================================
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde
RENDER=jobs/render_rosseland.slurm
PY=/home/hey4/.conda/envs/richanalysis/bin/python

# RESOLUTION IS NON-COMPROMISE: keep interpolation RES=1024 (output
# RESOLUTION=1024) always. Trade compute/memory/partition knobs, never resolution.
#
# Each 1024^3 frame needs at least: int64 index map 8.6G + float64 cube 8.6G
# + yt's own copy of the cube in to_yt + KDTree/coords + snapshot fields). That
# does NOT fit gpu_strw's 62G nodes for NJOBS>1. Run on cpu-zen4 (128c/384G/7-DAY
# cap).
#
# NJOBS=3. The evidence is the OOM kills, NOT a per-frame memory measurement:
#   * NJOBS=5 (4574247/4574248) was OOM-killed, one worker each. Worse, a dead
#     worker leaves mp.Pool waiting forever on its result, so both jobs sat DEAD
#     for 31 h looking exactly like slow progress. NEVER trust "still RUNNING".
#   * NJOBS=4 (4596952 side_A) completed 131 frames.
#   * 3 is one step back from the last setting that completed. Extra wall time
#     comes out of a 7-day budget, which is nothing.
#
# DO NOT size NJOBS from `sacct MaxRSS` here -- it is useless for this workload.
# The cgroup high-water mark counts reclaimable PAGE CACHE, and these jobs stream
# ~57 M-cell NPY arrays every frame, so it just fills whatever limit you give it:
# jobs 4574244/4574246/4596943 all reported *exactly* 100.0% of a 180G limit, and
# side_A 96.7% of 370G. Real per-frame anonymous usage is unknown; the old
# "~50 GiB/frame" figure in these notes was never verified either. If you need a
# real number, measure RSS inside a worker, not the cgroup peak.
# WORKERS=42 keeps NJOBS*WORKERS<=128.
#
# WALLTIME: ASK FOR THE 7-DAY CAP, not a tight estimate. A whole 1024 movie takes
# ~24-26 h on these (variable) nodes; TIME=1-day was killed at frame 129/131 with
# nothing encoded, and TIME=2-day only just fit. Overshooting the request costs
# nothing (the job exits when it finishes), undershooting costs a whole render.
# Frames are KEPT on the SHARED filesystem (per-job FRAMES_ROOT under
# OUTDIR/frames/) so a kill never discards a nearly-finished render: re-running
# the same command RESUMES (render_rosseland_movie.py skips frames already on disk)
# and just encodes. Fallbacks (same 384G): cpu-skylake (48c, 7-day); cpu-short
# (48c, 4h cap -- too short).
RES="${RES:-1024}"; RESOLUTION="${RESOLUTION:-1024}"
NJOBS="${NJOBS:-3}"; WORKERS="${WORKERS:-42}"      # NJOBS*WORKERS<=CPUS; see the OOM note above
MEM="${MEM:-370G}"; TIME="${TIME:-7-00:00:00}"; CPUS="${CPUS:-128}"

# Boxes to render. C is the pericentre close-up (+-2.5 r_p across): a pencil beam
# keeping the full line-of-sight depth, so tau stays a real optical depth and the
# same VMIN/VMAX apply. Face-on only -- viewed from the side a beam is a sliver.
FACEON_BOXES="${FACEON_BOXES:-A B C}"
SIDE_BOXES="${SIDE_BOXES:-A B}"
PART=(--partition="${PARTITION:-cpu-zen4}" --account="${ACCOUNT:-strw}")
LOG=(--output=jobs/logs/%x_%j.out --error=jobs/logs/%x_%j.err)
OUTDIR="reports/movies/rosseland"

# Fixed colorbar bounds on tau = int alpha_ross dr: log10 tau from -1 to 2, i.e.
# one decade below and two above the tau=1 photosphere, which is where all the
# structure that means anything physically lives. Everything deeper than tau~100
# is equally opaque, and everything thinner than 0.1 equally transparent, so
# widening the range only wastes colour resolution on the interesting decade.
# Same range for every box, so A, B and C are directly comparable.
declare -A VMIN_BOX=( [A]=1e-1 [B]=1e-1 [C]=1e-1 )
declare -A VMAX_BOX=( [A]=1e2  [B]=1e2  [C]=1e2  )

# emit_movie CAMERA BOX
emit_movie() {
  local CAMERA="$1" BOX="$2"
  local VMIN="${VMIN_BOX[$BOX]}" VMAX="${VMAX_BOX[$BOX]}"
  # Persistent per-job frames dir on the shared FS -> survives a kill, enables resume.
  local FR="$OUTDIR/frames/${CAMERA}_${BOX}"
  local exp="CAMERA=$CAMERA,BOX=$BOX,RES=$RES,RESOLUTION=$RESOLUTION,NJOBS=$NJOBS,WORKERS=$WORKERS,OUTDIR=$OUTDIR,TAG=rosseland,VMIN=$VMIN,VMAX=$VMAX,FRAMES_ROOT=$FR,KEEP_FRAMES=1"
  sbatch --parsable "${PART[@]}" "${LOG[@]}" --job-name="rosseland_${CAMERA}_${BOX}" \
    --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
    --export=ALL,"$exp" "$RENDER" \
    | xargs -I{} echo "  {}  rosseland_${CAMERA}_${BOX}.mp4 (camera=$CAMERA box=$BOX vmin=$VMIN vmax=$VMAX)"
}

# BOXES env restricts which boxes to submit (default both) -- e.g. BOXES=A to
# resubmit just one box without disturbing a still-running job for the other.
faceon() { echo "== faceon =="; for B in ${BOXES:-$FACEON_BOXES}; do emit_movie faceon "$B"; done; }
side()   { echo "== side ==";   for B in ${BOXES:-$SIDE_BOXES};   do emit_movie side "$B";   done; }

case "${1:?need faceon|side|both}" in
  faceon) faceon ;;
  side) side ;;
  both) faceon; side ;;
  *) echo "unknown case $1" >&2; exit 1 ;;
esac
