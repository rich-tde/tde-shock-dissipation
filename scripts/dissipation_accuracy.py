import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import unyt as u
import pandas as pd

from rs import RiemannSolver
import richio
richio.plots.use_nice_style()
import warnings

def _get_at_x(snap,
           quantity, 
           x, 
           xeps=0.05,
           qeps=1e-2):
    """
    Get quantity at x=indices, with coordinate error within xeps and quantity
    error within qeps.
    """
    try:
        indices = np.abs(snap.x - x * snap.x.units) <= xeps * snap.x.units
    except:
        indices = np.abs(snap.x - x) <= xeps * snap.x.units
    qxs = quantity[indices]

    qerr = (np.max(qxs) - np.min(qxs)) / np.min(qxs)
    if qerr > qeps: # percentage error
        warnings.warn(f"The percentage error {qerr} exceeds specified limit {qeps}.")
    
    return np.mean(qxs)


def dp2s(rho, p):
    """
    Using the Sackur-Tetrode equation to calculate specific entropy of a
    gamma=5/3 ideal gas, given pressure and density.
    """
    gamma = 5/3
    sie = p / (rho * (gamma - 1)) # specific internal energy
    s = 1 / u.mh * u.kb * (np.log(u.mh / rho * ((4*np.pi*u.mh**2*sie)/(3*u.h**2))**(3/2)) + 5/2) 
    return s


snap_dir = '/home/hey4/rich_tde/data/raw/ShockTubeN1e3IdealGas'

file_list = []
time_list = []
rich_dissipation_list = []
Tds_list = []

for file in os.listdir(snap_dir):
    if file.endswith('h5'):
        snap = richio.load(os.path.join(snap_dir, file)) # 900

        snap.time

        # Left State
        rho_L = 1.0
        vx_L = 0.0
        P_L = 1.0

        # Right State
        rho_R = 0.125
        vx_R = 0.0
        P_R = 0.1

        # ideal gas gamma
        gamma = 5/3

        # time
        t = snap.time.value

        # Riemann Solver
        rs = RiemannSolver(rho_L, vx_L, P_L, rho_R, vx_R, P_R, gamma, t)
        x, rho, vx, P = rs.solve()

        s = dp2s(snap.rho, snap.P)
        s =( s * snap.density / snap.density).in_base('rich')

        xs = []
        ts = []
        for _file in os.listdir(snap_dir):
            if _file.endswith('h5'):
                _snap = richio.load(os.path.join(snap_dir, _file))
                i_shockfront = np.argmax(_snap.dissipation) # shock front as the maximum of dissipation, good enough for sod shock; be careful in other setups
                x = _snap.x[i_shockfront]
                t = _snap.time[0]

                xs.append(x)
                ts.append(t)

        xs = u.unyt_array(xs)
        ts = u.unyt_array(ts)
        v_shock, b = np.polyfit(ts.value, xs.value, 1)
        v_shock *= richio.units.lscale / richio.units.tscale
        b *= richio.units.lscale
        print(v_shock, b)

        # Total dissipation rate (dissipation rate at shock front as well)
        diss = snap.dissipation * snap.volume
        total_diss = np.sum(diss)
        print(f"The total dissipation rate is {total_diss}")

        # Get shock front position
        i_shock = np.argmax(snap.dissipation)
        x_shock = snap.x[i_shock]

        # Near the neighbor of the shock front
        x1 = x_shock * 1.1  # upstream
        x2 = x_shock * 0.9  # downstream
        i_shock = (snap.x > x2) & (snap.x < x1)
        diss_shock = np.sum(diss[i_shock])
        print(f"Dissipation rate across the shock front is {diss_shock}")
        # Temperature
        T = snap.P / snap.density * u.mh / u.kb
        # Cross section
        A = (snap.box_size[4] - snap.box_size[1]) * (snap.box_size[5] - snap.box_size[2])
        print(f'The cross section is: {A}')

        # T_1 (s_2 - s_1)
        T1 = _get_at_x(snap, T, x=x1)
        T2 = _get_at_x(snap, T, x=x2)
        s1 = _get_at_x(snap, s, x=x1)
        s2 = _get_at_x(snap, s, x=x2)
        rho1 = _get_at_x(snap, snap.rho, x=x1)
        diss_sjump = T1 * (s2 - s1) * rho1 * A * v_shock
        print(f"Dissipation rate across shock front, as seen from entropy jump {diss_sjump}")
        diss_shock / diss_sjump

        file_list.append(file)
        time_list.append(snap.time[0].value)
        rich_dissipation_list.append(diss_shock.value)
        Tds_list.append(diss_sjump.value)
    
        dict2df = {'File': file_list, 'Time': time_list, 'RICH Dissipation': rich_dissipation_list, 'Tds Dissipation': Tds_list}
        df = pd.DataFrame(dict2df)
        df.to_csv('dissipation_RICH_vs_Tds.csv', index=False)



