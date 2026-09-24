# Research notebooks and tools

We use this directory to explore the TDE simulations, check our analysis choices,
and develop results for papers. The notebooks keep the scientific reasoning close
to the figures; the Python tools handle calculations that we need to repeat across
snapshots or runs.

We use notebook prefixes to track maturity. We start each new notebook at `0.1`
and reserve `1.0` for results that we consider finished, or close enough to use in
a paper. We choose when to change that version ourselves. A short exploration can
answer its question and remain at `0.1`; it does not need to become a larger study.

For a quick calculation, roughly less than ten minutes, we usually work in an
existing exploratory notebook. For expensive or repeated work, we extend the
closest suitable command-line tool before adding another script. We keep the
calculations and physical assumptions visible, and save large figure collections
outside notebooks so that we can still open them comfortably over a remote
connection.

We run the tools from the repository root, using our analysis environment
(`richanalysis` on our cluster). For example:

```bash
python works/shock-tde/shock-zoom-caches.py --help
```

Each script opens with a guide to its scientific purpose, input files, output
files, usage and loading examples. We document array shapes, column or key
meanings, units and plotting orientation there, so that we can use a saved result
without reconstructing the calculation from its source. We follow the same
Sphinx/reStructuredText convention as `richio`; helper modules explain how to
call their Python interfaces.

For the loading examples in an interactive Python session at the repository
root, we also put the local packages on the import path:

```bash
PYTHONPATH="$PWD/dev:$PWD/richio" python
```

This lets Python find the `dev` package that applies our plotting style, rather
than the outer source directory of the same name.

We save analysis data and figures under `data/processed/<StudyName>`, retain movie
frames in explicit output directories, and keep plotting configuration under
`.cache/matplotlib`. Our default simulation paths refer to our cluster; the script
guides describe their inputs and available path overrides. When we change a
selection or an input dataset, we choose a new output directory or explicitly
recompute the affected products. Each guide explains which results the tool
reuses when we run it again.

## Shock and dissipation studies

We use these tools to locate dissipation and compare energy budgets across the
three black-hole masses. We keep spatially selected power, shock-finder estimates
and bound-orbital-energy changes distinct because they answer different questions.

| Tool in `shock-tde/` | Question or product |
|---|---|
| [E-t.py](shock-tde/E-t.py) | All-cell orbital, kinetic, gravitational, gas and radiation energy budgets. |
| [Ediss-t.py](shock-tde/Ediss-t.py) | Regional dissipation powers; `--regions standard` or `--regions nozzle-split`. |
| [Rdiss-t.py](shock-tde/Rdiss-t.py) | Dissipation-weighted radius and mean radial direction. |
| [epsilondiss-t.py](shock-tde/epsilondiss-t.py) | Regional power per unit selected mass. |
| [plot-Ediss-t-four-regions.py](shock-tde/plot-Ediss-t-four-regions.py) | Rates and cumulative energy fractions from nozzle-split tables. |
| [SS24-circularization-t.py](shock-tde/SS24-circularization-t.py) | Bound orbital-energy and fallback tables; `--merge-input` combines shards. |
| [regional-circularization-t.py](shock-tde/regional-circularization-t.py) | Regional dissipation divided by fallback circularization power. |
| [nozzle-dissipation-all.py](shock-tde/nozzle-dissipation-all.py) | Internal/kinetic energy and heating-rate proxies within a configurable nozzle wedge. |
| [nozzle-zoom-slices.py](shock-tde/nozzle-zoom-slices.py) | Cached XY physical-field slices and figures. |
| [nozzle-yz-slices.py](shock-tde/nozzle-yz-slices.py) | Cached vertical physical-field and diagnostic slices and figures. |
| [shock-finder-ediss-selection.py](shock-tde/shock-finder-ediss-selection.py) | Shock detector on chosen snapshots; writes surface-cell results. |
| [shock-location-slices.py](shock-tde/shock-location-slices.py) | Histograms and locations of detected shock surfaces. |
| [shock-zoom-caches.py](shock-tde/shock-zoom-caches.py) | Small physical/Mach/geometry NPZ products for presentation notebooks; produces no figures. |
| [pw-orbit-nozzle-slice-test.py](shock-tde/pw-orbit-nozzle-slice-test.py) | Ballistic orbit overlays on existing cached nozzle slices. |

We choose between the standard and nozzle-split regions through `Ediss-t.py`,
and combine SS24 time-series shards through the same tool that produces them:

```bash
# Sample the nozzle-split dissipation in ten snapshots.
python works/shock-tde/Ediss-t.py --mode 1 --regions nozzle-split --npoints 10

# Combine existing shards; replace these example filenames with your own.
python works/shock-tde/SS24-circularization-t.py \
    --merge-input data/processed/SS24-circularization-t/shard-a.txt \
    --merge-input data/processed/SS24-circularization-t/shard-b.txt \
    --timeseries-file data/processed/SS24-circularization-t/combined.txt \
    --require-contiguous
```

We can merge any number of shards that share the SS24 column schema. The tool
checks for conflicting snapshot duplicates and requires `--overwrite` before
replacing an existing destination. To include an earlier checkpoint, we pass it
as another `--merge-input`.

For shock zooms, we first run `shock-finder-ediss-selection.py` to identify the
surface cells. We then use the shock-finder dissipation-analysis notebook to
estimate their individual powers. `shock-zoom-caches.py` brings these results
together with small field grids that we can load in the presentation notebooks.
Its opening guide lists every output key and shows how to load and plot the
arrays; `--list-only` shows the four cases we currently use.

## Cooling and resolution

We check the nozzle selection and grid resolution before building a time series.
We compare emission, photon escape and vertical-flow times separately, so that a
short emission time alone does not stand in for efficient net cooling.

| Tool | Purpose |
|---|---|
| [timescale-maps.py](cooling-checks/timescale-maps.py) | Legacy whole-plane projected timescale maps; definitions differ from the nozzle tools. |
| [nozzle-selection-validation.py](cooling-checks/nozzle-selection-validation.py) | Compare candidate nozzle selections. |
| [nozzle-wedge-validation.py](cooling-checks/nozzle-wedge-validation.py) | Check the angular wedge and cache its direction. |
| [nozzle-timescale-validation.py](cooling-checks/nozzle-timescale-validation.py) | Compare grid resolution and vertical spacing. |
| [nozzle-timescale-series.py](cooling-checks/nozzle-timescale-series.py) | Per-snapshot production caches and time-series aggregation. |
| [nozzle_timescales.py](cooling-checks/nozzle_timescales.py) | Shared physical calculations and cache/statistics API; import this module. |
| [check_nozzle_resolution.py](nozzle-resolution/check_nozzle_resolution.py) | Native-cell volume, size and mass diagnostics with JSONL output. |

## Movies and rendering checks

We retain separate rendering tools where the projection, camera treatment or
physical quantity differs. Their opening guides explain those choices and how
to read the resulting images, movies or numerical data.

| Tool in `movies/` | Purpose |
|---|---|
| [make_gif.py](movies/make_gif.py) | Simple configurable snapshot projection/slice PNGs and GIFs. |
| [render_evolution.py](movies/render_evolution.py) | One movie over a sequence of snapshots. |
| [render_evolution_multi.py](movies/render_evolution_multi.py) | Multiple fields with frame-window resume. |
| [render_wind_proj_evolution.py](movies/render_wind_proj_evolution.py) | Wind projections with gas selections. |
| [render_tde_movies.py](movies/render_tde_movies.py) | Established TDE projection/slice movie recipes. |
| [render_volume_movie.py](movies/render_volume_movie.py) | Volume-rendered movies. |
| [render_movie_mpi.py](movies/render_movie_mpi.py) | MPI-distributed snapshot rendering. |
| [render_rosseland_movie.py](movies/render_rosseland_movie.py) | Rosseland optical-depth views. |
| [scan_color_range.py](movies/scan_color_range.py) | Fixed colour-range scans across snapshots. |
| [scan_rosseland_range.py](movies/scan_rosseland_range.py) | Optical-depth colour-range scans. |
| [faceon_density.py](movies/faceon_density.py) | Shared face-on density API. |
| [movie_zoom.py](movies/movie_zoom.py), [tde_frame.py](movies/tde_frame.py) | Shared camera and reference-frame helpers. |

We use [benchmark_gridding.py](richio-development/benchmark_gridding.py) to measure
gridding performance, [test_gridding.py](richio-development/test_gridding.py) to
check nearest-neighbour results, and
[sinh-projection-convergence.py](richio-development/sinh-projection-convergence.py)
to examine how the projection changes with grid spacing and resolution.
