#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --account=strw
#SBATCH --job-name=fallback-rate
#SBATCH --time=04:00:00
#SBATCH --output=/home/hey4/rich_tde/jobs/logs/%j_%x.out
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

cd /home/hey4/rich_tde
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/home/hey4/rich_tde/.cache/matplotlib
export MPLBACKEND=Agg
export PYTHONPATH="/home/hey4/rich_tde/dev${PYTHONPATH:+:$PYTHONPATH}"

# One worker per node keeps the native-snapshot reads on separate network links.
srun --kill-on-bad-exit=1 bash -c '
    mode=$((SLURM_PROCID / 3 + 1))
    part=$((SLURM_PROCID % 3))
    runs=(1e4 1e5 1e6)
    run=${runs[$((mode - 1))]}
    base=data/processed/FallbackRate
    output=$base/parts/$run-$part
    mkdir -p "$output"
    for kind in flux energy; do
        if [[ ! -e "$output/$kind-$run.txt" && -e "$base/$kind-$run.txt" ]]; then
            cp "$base/$kind-$run.txt" "$output/$kind-$run.txt"
        fi
    done
    /home/hey4/.conda/envs/richanalysis/bin/python -u \
        works/dynamics/fallback-rate.py --mode "$mode" \
        --part "$part" --parts 3 --output-dir "$output"
'

# Publish complete tables only after every worker succeeds.
/home/hey4/.conda/envs/richanalysis/bin/python - <<'PY'
import os
import shutil

import numpy as np

import dev

base = "data/processed/FallbackRate"
for run in ["1e4", "1e5", "1e6"]:
    rows = {}
    for part in range(3):
        path = f"{base}/parts/{run}-{part}/flux-{run}.txt"
        with open(path) as handle:
            header = "".join(
                line[2:] for line in handle if line.startswith("# ")
            ).rstrip()
        rows.update({int(row[0]): row for row in np.loadtxt(path, ndmin=2)})
    expected = {
        int(os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[1])
        for path in dev.DATAPATHS(run)
    }
    assert set(rows) == expected, f"Incomplete {run} snapshot coverage"
    output = f"{base}/flux-{run}.txt"
    np.savetxt(
        output + ".partial",
        sorted(rows.values(), key=lambda row: row[1]),
        header=header,
        fmt=["%d", "%.16e", "%.16e", "%.16e", "%.16e", "%d", "%.16e"],
    )
    os.replace(output + ".partial", output)
    energy = f"{base}/energy-{run}.txt"
    shutil.copyfile(f"{base}/parts/{run}-0/energy-{run}.txt", energy + ".partial")
    os.replace(energy + ".partial", energy)
    print(f"Merged {run}: {len(rows)} snapshots -> {output}", flush=True)
PY
