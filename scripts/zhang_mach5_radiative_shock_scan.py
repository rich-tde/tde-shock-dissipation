#!/usr/bin/env python3
"""Scan a RadHydroZhangMach5 (grey-diffusion, M0=5) run and compare every
snapshot against the analytic NLTE radiative-shock solution (Lowrie & Edwards
2008), using the Zhang et al. opacity/density/temperature scale at the paper's
actual M0=5 (unlike RadHydroMach2, which uses the same opacity scale but is
tuned to M0~2 for a stationary shock).

Same approach as mach2_radiative_shock_scan.py:
  * load every ZhangM5_*.h5 snapshot directly with richio (raw cgs values --
    no unit conversion, see that script's docstring for why),
  * evaluate the analytic profile at each snapshot's own simulation time
    (shifted back into the analytic lab frame: this run sits in the
    shock-stationary frame, v_sim = v_analytic - v_shock, so x - v_shock*t
    in the lab frame maps to x in the sim frame at the same t),
  * save a per-snapshot comparison figure and a summary L1-error-vs-time table.

Physical parameters must match runs/RadHydroZhangMach5/test.cpp.

Run inside the richanalysis env:  python scripts/zhang_mach5_radiative_shock_scan.py
"""

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import richio
from nlte_radiative_shock import NLTERadiativeShock
from radiative_shock import Units

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_DIR = "/home/hey4/RICH/alice_data/raw/RadHydroZhangMach5"
SNAP_PATTERN = "ZhangM5_*.h5"

OUTPUT_BASE = "/zfsstore/user/hey4/rich_tde/data/processed/mach2_radiative_shock"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, os.path.basename(INPUT_DIR))
TABLE_NAME = "zhang_mach5_radiative_shock.txt"

# Left (unshocked, downstream) state and opacities -- must match test.cpp
GAMMA = 5.0 / 3.0
MU = 1.67e-24  # g, mean particle mass
K_BOLTZ = 1.380649e-16  # erg/K
CV = K_BOLTZ / (MU * (GAMMA - 1.0))

RHO_LEFT = 5.45969e-13  # g/cc
V_LEFT = 5.88588e5  # cm/s (analytic lab frame, M0=5)
T_LEFT = 100.0  # K
SIGMA_ROSS = 0.848903  # 1/cm
SIGMA_ABS = 3.92664e-5  # 1/cm

CS_LEFT = np.sqrt(GAMMA * (GAMMA - 1.0) * CV * T_LEFT)
M0 = V_LEFT / CS_LEFT
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.join(OUTPUT_DIR, "profiles"), exist_ok=True)

solver = NLTERadiativeShock(
    M0=M0,
    gamma=GAMMA,
    sigma_ross=lambda T, rho: SIGMA_ROSS,
    sigma_abs=lambda T, rho: SIGMA_ABS,
    cv=CV,
    rho_left=RHO_LEFT,
    v_left=V_LEFT,
    T_left=T_LEFT,
    eps_nlte_solver=0.5e-4,
)
V_SHOCK = solver.v_shock  # sim runs in the frame where the shock sits at x=0


def rel_l1_error(numeric, analytic):
    """Per-cell relative L1 error: mean(|num - ana| / |num|) over the shocked region."""
    mask = np.abs(numeric) > 0.01 * np.max(np.abs(numeric))
    if np.sum(mask) < 2:
        mask = np.ones(len(numeric), dtype=bool)
    return float(np.mean(np.abs(numeric[mask] - analytic[mask]) / np.abs(numeric[mask])))


files = glob.glob(os.path.join(INPUT_DIR, SNAP_PATTERN))
print(f"{len(files)} snapshots in {INPUT_DIR}")
print(f"v_shock = {V_SHOCK:.6e} cm/s (sim frame offset)")

rows = []  # (snapnum, time, name, density_l1, tgas_l1, trad_l1)

for path in files:
    name = os.path.splitext(os.path.basename(path))[0]
    snap = richio.load(path)

    # Raw cgs values -- no unit conversion (see module docstring)
    x_num = np.asarray(snap.x)
    rho_num = np.asarray(snap.density)
    T_num = np.asarray(snap.temperature)
    Erad_num = np.asarray(snap.Erad)  # specific radiation energy [erg/g]
    Trad_num = (Erad_num * rho_num / Units.arad) ** 0.25
    t = float(np.asarray(snap.time))

    order = np.argsort(x_num)
    x_num, rho_num, T_num, Trad_num = (
        x_num[order],
        rho_num[order],
        T_num[order],
        Trad_num[order],
    )

    if t <= 0:
        print(f"{name}: t = {t}, no shock yet -- skipped")
        continue

    # The sim is run in the shock-stationary frame (x_sim = x_shock=0 fixed);
    # the analytic solver expects the lab frame where the shock moves at
    # v_shock, i.e. x_lab = x_sim + v_shock * t.
    x_lab = x_num + V_SHOCK * t
    solution = solver.solve_profiles(time=t, x=x_lab)
    rho_ana = solution["density"]
    T_ana = solution["temperature"]
    Trad_ana = solution["radiation_temperature"]
    v_ana = solution["velocity"] - V_SHOCK  # back into sim frame for plotting

    density_l1 = rel_l1_error(rho_num, rho_ana)
    tgas_l1 = rel_l1_error(T_num, T_ana)
    trad_l1 = rel_l1_error(Trad_num, Trad_ana)

    cycle_match = re.search(r"_(\d+)$", name)
    cycle = int(cycle_match.group(1)) if cycle_match else -1  # -1 for ZhangM5_final
    rows.append((cycle, t, name, density_l1, tgas_l1, trad_l1))
    print(
        f"{name}: t={t:.4e}s  density_L1={density_l1:.3e}  "
        f"Tgas_L1={tgas_l1:.3e}  Trad_L1={trad_l1:.3e}"
    )

    # Per-snapshot comparison figure
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axes[0].plot(x_num, rho_num, ".", ms=2, label="RICH")
    axes[0].plot(x_num, rho_ana, "-", label="analytic")
    axes[0].set_ylabel(r"$\rho$ [g/cc]")
    axes[0].legend()

    axes[1].plot(x_num, T_num, ".", ms=2, label="RICH $T_{gas}$")
    axes[1].plot(x_num, T_ana, "-", label="analytic $T_{gas}$")
    axes[1].plot(x_num, Trad_num, ".", ms=2, label="RICH $T_{rad}$")
    axes[1].plot(x_num, Trad_ana, "-", label="analytic $T_{rad}$")
    axes[1].set_ylabel("T [K]")
    axes[1].legend()

    axes[2].plot(x_num, v_ana, "-", label="analytic velocity")
    axes[2].set_ylabel("v [cm/s] (sim frame)")
    axes[2].set_xlabel("x [cm] (sim frame)")
    axes[2].legend()

    fig.suptitle(f"{name}  t={t:.4e}s")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "profiles", f"{name}.png"), dpi=130)
    plt.close(fig)

rows.sort(key=lambda r: r[1])  # sort by time

table_path = os.path.join(OUTPUT_DIR, TABLE_NAME)
with open(table_path, "w") as f:
    f.write("# snapnum  time[s]  name  density_rel_L1  Tgas_rel_L1  Trad_rel_L1\n")
    for snapnum, t, name, d_l1, t_l1, tr_l1 in rows:
        f.write(f"{snapnum:6d}  {t:.6e}  {name:>16s}  {d_l1:.6e}  {t_l1:.6e}  {tr_l1:.6e}\n")

print(f"Wrote {len(rows)} rows to {table_path}")
print(f"Profile figures in {os.path.join(OUTPUT_DIR, 'profiles')}")
