#!/usr/bin/env bash
#SBATCH --job-name=vr_multi
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=61G
#SBATCH --time=20:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out

# Multi-field, shared-index evolution-movie driver (works/movies/render_evolution_multi.py).
# Builds the geometry-only KDTree index ONCE per snapshot and projects every field
# from it -> ~Nfield fewer tree builds and ~Nfield less HDF5 I/O than launching one
# single-field job per field (the old render_evolution_v2.slurm path).  Fixed boxes
# A/B, r_t scale bar, BH-frame, x flipped, single trailing rotation.
#
# Parametrise via --export=ALL,VAR=...:
#   FIELDS    = comma-sep field list   (default: density,dissipation,temperature,bernoulli)
#   BOX       = A | B | C   (C = pericentre close-up, face-on only)
#   AZ, EL    = camera azimuth/elevation   (default 45/26 = canonical 3/4 view)
#   OUTDIR    = output folder
#   TAG       = output stem base used by NAME_TMPL
#   NAME_TMPL = output name template, '{tag}'/'{field}' substituted
#               (e.g. 'g2_{field}_A' -> g2_density_A.mp4)
#   VMINS     = per-field vmin list aligned with FIELDS (comma-sep; blank = auto)
#   VMAXS     = per-field vmax list aligned with FIELDS
# Optional:
#   RES        = interpolation grid res (default 1024 = 1024^3 box)
#   RESOLUTION = output image px        (default 1024)
#   SPINFRAMES = trailing rotation frames (default 90; 0 = none)
#   NJOBS      = frame-parallel worker processes (default 2)
#   WORKERS    = KDTree query threads per build  (default 24)
#   START,END  = snapshot range (default 21..151)
#   STEP       = use every Nth snapshot (default 1)
#   RICHIO_FONT_SCALE = on-frame text multiplier (default 1.0 = unchanged)
#
# ----------------------------------------------------------------------------
# MEMORY NOTES  (why these defaults; how to run at 1024^3)
# ----------------------------------------------------------------------------
# Per worker, peak RAM for ONE grid build at RES=1024 (1024^3):
#   index map  int64  1024^3  = 8.6 GB   (geometry; built once/snapshot, reused/field)
#   field cube float64 1024^3 = 8.6 GB   (one at a time)
#   weight cube (temperature/bernoulli only, weight=density) = +8.6 GB
#   KDTree + particle coords (~70 M cells) = ~3-5 GB
#   => ~17 GB unweighted, ~26-30 GB weighted, per worker.
# Peak job RAM ~= NJOBS * per-worker-peak.  The single-threaded tree BUILD
# (~30-60 s/frame) does not scale with cores; the 1024^3 KDTree QUERY is huge
# (~1e9 points) and IS threaded, so set WORKERS high to fill cores during query.
#
# gpu_strw nodes = 48c / 61 GB  ->  NJOBS=2, WORKERS=24 (=48 query threads).
#   2 * ~30 GB (weighted) can brush 61 GB: watch `sacct -j <id> --format=MaxRSS`.
#   If OOM/tight: enable the int32 index downcast in richio/data.py, OR run the
#   weighted fields with NJOBS=1.  Rule of thumb: NJOBS = floor((mem-8GB)/peak),
#   WORKERS = floor(cpus/NJOBS).
#
# cpu-short BURST fallback (only when gpu_strw is busy; account=strw, 4 h cap):
#   sbatch --partition=cpu-short --account=strw --time=4:00:00 \
#          --cpus-per-task=64 --mem=240G --export=ALL,NJOBS=8,WORKERS=8,... jobs/movies/render_multi.slurm
# ----------------------------------------------------------------------------
set -euo pipefail
cd /home/hey4/rich_tde

PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export RICHIO_FONT_SCALE="${RICHIO_FONT_SCALE:-1.0}"

# List-valued vars arrive ';'-delimited (sbatch --export splits on commas); the
# driver wants commas.  Accept either delimiter.
FIELDS="${FIELDS:-density,dissipation,temperature,bernoulli}"; FIELDS="${FIELDS//;/,}"
VMINS="${VMINS:-}"; VMINS="${VMINS//;/,}"
VMAXS="${VMAXS:-}"; VMAXS="${VMAXS//;/,}"
BOX="${BOX:-A}"
AZ="${AZ:-45}"
EL="${EL:-26}"
OUTDIR="${OUTDIR:-reports/movies/canonical}"
TAG="${TAG:-g2}"
# NB: do NOT inline a brace-containing default here -- bash closes ${VAR:-...} at the
# first bare '}', which would append a stray '_{field}}' to the value.  Set separately.
NAME_TMPL="${NAME_TMPL:-}"; [[ -z "$NAME_TMPL" ]] && NAME_TMPL='{tag}_{field}'
RES="${RES:-1024}"
RESOLUTION="${RESOLUTION:-1024}"
SPINFRAMES="${SPINFRAMES:-90}"
NJOBS="${NJOBS:-2}"
WORKERS="${WORKERS:-24}"
START="${START:-21}"
END="${END:-151}"
STEP="${STEP:-1}"
# Chunked rendering (for short-walltime partitions like cpu-short, 4 h cap):
#   MODE        = full | render | encode   (full: render+encode in one job)
#   FRAME_START = first global frame index  (evolution [0,n), spin [n,n+spin))
#   FRAME_STOP  = stop global frame index (exclusive; -1 = all)
#   FRAMES_ROOT = shared frames dir on a NETWORK FS (NOT /tmp: chunks land on
#                 different nodes).  Required for MODE=render/encode.
MODE="${MODE:-full}"
FRAME_START="${FRAME_START:-0}"
FRAME_STOP="${FRAME_STOP:--1}"
FRAMES_ROOT="${FRAMES_ROOT:-}"
RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR

mkdir -p "$OUTDIR"

# Assemble mode/window/frames-root flags.
EXTRA=(--frame-start "$FRAME_START" --frame-stop "$FRAME_STOP")
case "$MODE" in
  render) EXTRA+=(--no-encode --keep-frames) ;;
  encode) EXTRA+=(--encode-only) ;;
  full)   : ;;
  *) echo "unknown MODE=$MODE" >&2; exit 1 ;;
esac
# KEEP=1 keeps the per-frame PNGs after encoding (MODE=full); use a persistent
# FRAMES_ROOT (e.g. under reports/movies/) to archive every frame.
[[ "${KEEP:-0}" == 1 ]] && EXTRA+=(--keep-frames)
[[ -n "$FRAMES_ROOT" ]] && EXTRA+=(--frames-root "$FRAMES_ROOT")

echo "Host: $(hostname)  FIELDS=$FIELDS BOX=$BOX AZ=$AZ EL=$EL RES=$RES RESOLUTION=$RESOLUTION"\
" NJOBS=$NJOBS WORKERS=$WORKERS SPIN=$SPINFRAMES FONT=$RICHIO_FONT_SCALE MODE=$MODE"\
" frames[$FRAME_START,$FRAME_STOP) -> $OUTDIR/$NAME_TMPL  Started: $(date)"

"${PY}" works/movies/render_evolution_multi.py "${RUN}" \
    --fields "$FIELDS" \
    --box "$BOX" \
    --azimuth "$AZ" \
    --elevation "$EL" \
    --scalebar \
    --bh-frame \
    --flip-x \
    --coords CMx,CMy,CMz \
    --start "$START" \
    --end "$END" \
    --step "$STEP" \
    --res "$RES" \
    --resolution "$RESOLUTION" \
    --spin-frames "$SPINFRAMES" \
    --spin-angle 360 \
    ${VMINS:+--vmins "$VMINS"} \
    ${VMAXS:+--vmaxs "$VMAXS"} \
    --workers "$WORKERS" \
    --n-jobs "$NJOBS" \
    --fps 24 \
    --outdir "$OUTDIR" \
    --tag "$TAG" \
    --name-template "$NAME_TMPL" \
    "${EXTRA[@]}"

echo "Finished: $(date)"
