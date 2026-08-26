# Face-on projection movies for the presentation

The current presentation workflow renders face-on density or dissipation movies
for each of the `1e4`, `1e5`, and `1e6` black-hole-mass runs. It uses a fixed box in
units of each run's `r_amin` and corrects only datasets that are still in the
orbiting reference frame. Every frame and mass uses the fixed range
`10^0.5`--`10^6.5 g cm^-2`: exactly six decades, with the ceiling rounded above
the legacy `faceon_A` scan maximum. This removes colour-scale flashing.
Dissipation uses viridis and `10^14`--`10^18 erg s^-1 cm^-2`: four fixed
decades with the ceiling rounded above the established face-on scan maximum.
Negative or non-finite dissipation cells are zeroed before projection; non-positive
projected pixels are drawn at the viridis colorbar floor.

```bash
# Required review run: 256 x 180 image plane, 128 sinh-spaced z intervals.
bash jobs/movies/submit_faceon_density.sh preview

# Dissipation preview; inspect and approve before production.
bash jobs/movies/submit_faceon_density.sh preview dissipation

# Do not run until the preview has been approved. This uses a 2048 x 1434
# image plane, 256 sinh-spaced z intervals, and 300-DPI PNG output.
bash jobs/movies/submit_faceon_density.sh production
```

Preview submissions use 32 CPUs and 120 GB on the four-hour `cpu-short`
partition; production uses 64 CPUs and 240 GB on the seven-day `cpu-zen4`
partition.

The Slurm array maps modes 1, 2, and 3 to `1e4`, `1e5`, and `1e6`. Frames are
restartable and are never removed automatically. Outputs are written to:

```
reports/movies/crete/<preview|production>/<mass>/
├── frames/frame_*.png
├── faceon_density_<mass>.mp4
├── dissipation/
│   ├── frames/frame_*.png
│   └── faceon_dissipation_<mass>.mp4
├── timeline.png                  # preview only
├── window-comparison.png         # preview only
└── window-examples/*.png         # preview only
```

The `1e4` window is `x/r_amin = [-2, 0.5]` and
`y/r_amin = [-0.875, 0.875]`, preserving the standard image-plane aspect
ratio. The `1e5` and `1e6` windows use `x/r_amin = [-1.5, 0.5]` and
`y/r_amin = [-0.7, 0.7]`. All three use `z/r_amin = [-0.7, 0.7]`. The
projection is not horizontally flipped. Frames
reuse the established `richio.render` presentation layer: black canvas,
linear-value log colorbar, time annotation, lower-left x/y/z orientation triad,
and a `0.5 r_amin` scale bar. The black-hole mass is intentionally omitted for
addition in post. Preview contact sheets also
compare taller and wider alternatives. Playback averages 8 fps for `1e4`,
16 fps for `1e5` (about 10 seconds total), and 24 fps for `1e6`.

A small black point marks the BH at `(x,y)=(0,0)`. For density only, before `0.3 t_fb`, cells
with `tracers/Star < 0.99` are suppressed for the movie projection. The factor
is `1e-8 * V_box/V_initial` (capped at one), compensating the simulation's
inverse-box-volume ambient-density scaling so its projected color stays fixed
below the displayed six-decade range.
Set `AMBIENT_FACTOR` in the Slurm environment to change the factor at the
initial `10^3` code-volume box.

Movie encoding uses the physical `unyt` snapshot times. Each rendered snapshot
is assigned a variable H.264 presentation duration proportional to the interval
before the next snapshot, normalized to retain the intended total movie length.
Thus sparse intervals freeze the preceding image longer while every available
snapshot remains present; the encoder repeats the final image only to close its
last presentation interval.

Useful restart overrides can be passed through `sbatch --export`, including
`FRAME_START`, `FRAME_STOP`, `NO_ENCODE=1`, `ENCODE_ONLY=1`, `NJOBS`, `WORKERS`,
`VMAX`, and `OVERWRITE=1`. `VMAX` changes the shared ceiling while the lower
limit remains exactly six decades below it. Encoding is attempted only when the
requested window covers the complete movie, and it refuses to proceed if any
frame is absent or empty.

## Legacy multi-field conference pipeline

The commands below describe the older fixed-box, four-field workflow. They are
kept for reproducing its existing products, but they are not used for the three
new presentation movies.

---

# jobs/ — legacy conference TDE movie pipeline

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
| `render_multi.slurm` | render worker → `works/movies/render_evolution_multi.py` (MODE=full/render/encode) |
| `scan_color.slurm`   | colour-scan worker → `works/movies/scan_color_range.py` |
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
