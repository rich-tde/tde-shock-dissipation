#!/bin/bash
#SBATCH --job-name=vr_test
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=61G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/vr_test_%j.out
#SBATCH --error=jobs/vr_test_%j.err

# Conference-movie TEST: render snaps 21/86/151 (g2 box A canonical view) at the
# production RES=1024 with the locked g2_A colour limits, keep the PNGs for review,
# and measure per-worker peak RAM (n_jobs=1) to decide the production n_jobs.
set -euo pipefail
cd /zfsstore/user/hey4/rich_tde
PY=/home/hey4/.conda/envs/richanalysis/bin/python
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export RICHIO_FONT_SCALE=1.0
RUN=/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR
FR=reports/movies/_tests/g2_A_frames
mkdir -p "$FR"

# Locked g2_A ranges from reports/movies/color_ranges.json (field order
# density,dissipation,temperature,bernoulli). Direct call -> commas are fine.
VMINS="0.00024817,0.0969962,65066.4,-5"
VMAXS="2.58966e+06,3.91509e+17,1.16836e+08,5"

echo "Host: $(hostname)  Started: $(date)"
/usr/bin/time -v $PY works/movies/render_evolution_multi.py "$RUN" \
    --fields density,dissipation,temperature,bernoulli \
    --box A --azimuth 45 --elevation 26 \
    --scalebar --bh-frame --flip-x --coords CMx,CMy,CMz \
    --start 21 --end 151 --step 65 \
    --res 1024 --resolution 1024 \
    --spin-frames 0 \
    --vmins "$VMINS" --vmaxs "$VMAXS" \
    --workers "${WORKERS:-48}" --n-jobs "${NJOBS:-1}" --fps 24 \
    --outdir reports/movies/_tests --tag g2test \
    --name-template "g2test_{field}_A" \
    --frames-root "$FR" --keep-frames
echo "Finished: $(date)"
