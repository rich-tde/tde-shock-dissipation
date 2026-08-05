# jobs/ — conference TDE movie pipeline

One entry point drives everything: **`jobs/submit_movies.sh`**.

```bash
bash jobs/submit_movies.sh scan      # colour-range scan (all cameras) -> reports/movies/color_ranges.json
bash jobs/submit_movies.sh g2        # canonical 3/4 view      (boxes A,B)
bash jobs/submit_movies.sh g3        # side + top angles       (boxes A,B)
bash jobs/submit_movies.sh faceon    # face-on xy projection, no rotation, frames kept
bash jobs/submit_movies.sh all       # g2 + g3 + faceon
```

Each render job uses the shared-index multi-field driver: **one** KDTree build per
snapshot renders all four fields (density, dissipation, temperature, bernoulli).
Colour limits come from `reports/movies/color_ranges.json` (run `scan` first).

## Layout

| Path | Role |
|------|------|
| `submit_movies.sh`   | **main entry** — submit the scan / g2 / g3 / faceon / all cases |
| `render_multi.slurm` | render worker → `scripts/render_evolution_multi.py` (MODE=full/render/encode) |
| `scan_color.slurm`   | colour-scan worker → `scripts/scan_color_range.py` |
| `logs/`              | all Slurm `*.out` / `*.err` |
| `archive/`           | superseded / pre-conference scripts (kept for reference) |
| `render_wind_proj.slurm`, `submit_wind_proj.sh` | separate wind-projection workflow (not the conference movies) |

## Outputs

```
reports/movies/canonical/g2_<field>_<A|B>.mp4
reports/movies/angles/g3_<side|top>_<field>_<A|B>.mp4
reports/movies/faceon/faceon_<field>_<A|B>.mp4   + per-frame PNGs in faceon/frames/<box>/<field>/
reports/movies/color_ranges.json                 # locked vmin/vmax per (field,box,camera)
```

Fields/boxes/cameras: density(twilight,log), dissipation(viridis,log),
temperature(inferno,log,weight=density), bernoulli(RdBu_r,symlog,weight=density);
box A = ±400 R⊙ cube, box B = wide; cameras canonical(az45/el26), side(az0/el15),
top(az20/el72), faceon(az0/el90, no rotation). Snapshots 21–151, BH-frame, x-flipped,
scale bar = n·r_t, time label + xyz triad on every frame.

## Tunables (env on `submit_movies.sh`)

`RES` (default 512 interpolation) · `RESOLUTION` (1024 output px) · `NJOBS` `WORKERS`
(NJOBS·WORKERS ≤ `CPUS`=64) · `MEM` (180G) · `TIME` (3:00:00) · `PARTITION` (cpu-short)
· `ACCOUNT` (strw).

**Partition note:** renders are CPU-only (scipy KDTree + yt off-axis projection).
gpu_strw nodes are 48c/61 GB — too little RAM for res-512 whole movies at NJOBS=8;
cpu-short gives ~370 GB so a whole 221-frame movie fits one <4 h job. A res-**1024**
set needs ~70 GB/worker and the whole-movie wall time can exceed cpu-short's 4 h cap —
render it in frame windows (`render_multi.slurm` MODE=render/encode +
FRAME_START/FRAME_STOP/FRAMES_ROOT; pattern in `archive/submit_cpushort.sh`).
