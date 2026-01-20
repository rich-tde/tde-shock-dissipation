"""
Post processing of the shock tube tests

do
```
python pp.py -h
```
To see how to use the script.
"""
import os
import argparse
import glob

import numpy as np
import unyt as u

import richio
from richio.shockfinder.rs import RiemannSolver
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
        warnings.warn(f"Error is {qerr}.")
    
    return np.mean(qxs)

if __name__ == '__main__':

    ## Constants
    gamma = 5/3

    ## Command line arguments
    parser = argparse.ArgumentParser(
    prog='pp.py',
    description="""Calculate various postprocessing diagnostics for the
    shocktube tests. Output can be read by `unyt.loadtxt(fname, delimiter=',')`.
    """
    )
    parser.add_argument('snap_dir', type=str, help='Directory containing the .h5 snapshots')
    parser.add_argument('output_fname', type=str, help='Output file name (full paths).')
    parser.add_argument('-f', '--overwrite', help='Overwrite existing output file.', action='store_true')
    args = parser.parse_args()

    output_fname = args.output_fname
    overwrite = args.overwrite

    snap_dirs = sorted(glob.glob(args.snap_dir))
    if not snap_dirs:
        raise FileNotFoundError(f"No files found matching pattern: {args.input_pattern}")

    # Initialize
    rho2_to_rho1_arr = []
    diss_dupdv_arr = []
    diss_rich_arr = []
    diss_ie_arr = []

    for snap_dir in snap_dirs:
        ## Find the shock speed from fitting
        ## FUTURE: from mach number jump condition?
        _xs = []
        _ts = []
        for file in os.listdir(snap_dir):
            if file.endswith('h5'):
                _snap = richio.load(os.path.join(snap_dir, file))
                i_shfront = np.argmax(_snap.dissipation) # sh front as the maximum of dissipation, good enough for sod sh; be careful in other setups
                _x = _snap.x[i_shfront]
                _t = _snap.time[0]

                _xs.append(_x)
                _ts.append(_t)

        _xs = u.unyt_array(_xs)
        _ts = u.unyt_array(_ts)

        v_sh, b = np.polyfit(_ts.value, _xs.value, 1)
        v_sh *= richio.units.lscale / richio.units.tscale

        ## Get pre post shock quantities

        snap = richio.load(os.path.join(snap_dir, 'snap_final.h5'))

        # Get sh front position
        i_sh = np.argmax(snap.dissipation)
        x_sh = snap.x[i_sh]

        # Near the neighbor of the sh front
        x1 = x_sh * 1.05  # upstream
        x2 = x_sh * 0.95  # downstream

        # Temperature
        T = snap.P / snap.density * u.mh / u.kb

        # Pre/post shock quantities
        T1 = _get_at_x(snap, T, x=x1)
        T2 = _get_at_x(snap, T, x=x2)

        u1 = _get_at_x(snap, snap.sie, x=x1)
        u2 = _get_at_x(snap, snap.sie, x=x2)

        rho1 = _get_at_x(snap, snap.rho, x=x1)
        rho2 = _get_at_x(snap, snap.rho, x=x2)

        P1 = _get_at_x(snap, snap.P, x=x1)
        P2 = _get_at_x(snap, snap.P, x=x2)

        v1_lab = _get_at_x(snap, snap.vx, x=x1) # lab frame
        v2_lab = _get_at_x(snap, snap.vx, x=x2)

        v1_sh = v_sh - v1_lab
        v2_sh = v_sh - v2_lab

        # Total dissipation rate (dissipation rate at sh front as well)
        diss = snap.dissipation * snap.volume
        diss_tot = np.sum(diss)

        i_sh = (snap.x > x2) & (snap.x < x1)
        diss_rich = np.sum(diss[i_sh])

        # Cross section
        A = (snap.box_size[4] - snap.box_size[1]) * (snap.box_size[5] - snap.box_size[2])

        diss_dupdv = (u2 * rho2 - u1 * (rho2/rho1)**gamma * rho1) * v2_sh * A

        diss_ie = (u2 * rho2 * v2_sh - u1 * rho1 * v1_sh) * A

        rho2_to_rho1 = rho2/rho1

        rho2_to_rho1_arr.append(rho2_to_rho1)
        diss_dupdv_arr.append(diss_dupdv)
        diss_rich_arr.append(diss_rich)
        diss_ie_arr.append(diss_ie)

    
    rho2_to_rho1_arr = u.unyt_array([rho2_to_rho1_arr])
    diss_dupdv_arr = u.unyt_array([diss_dupdv_arr])    
    diss_rich_arr = u.unyt_array([diss_rich_arr])      
    diss_ie_arr = u.unyt_array([diss_ie_arr])          
    
    ## Save using u.write_hdf5
    if os.path.isfile(output_fname) and not overwrite:
        raise FileExistsError(
            f"Output file {output_fname} already exists. Use -f/--overwrite to overwrite."
        )
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_fname)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Write to hdf5 to preserve units
    rho2_to_rho1_arr.write_hdf5(
        filename = output_fname,
        dataset_name = "rho2/rho1"
    )
    diss_rich_arr.write_hdf5(
        filename = output_fname,
        dataset_name = "rich_dissipation"
    )
    diss_dupdv_arr.write_hdf5(
        filename = output_fname,
        dataset_name = "dupdv_dissipation"
    )
    diss_ie_arr.write_hdf5(
        filename = output_fname,
        dataset_name = "internal_energy_jump"
    )