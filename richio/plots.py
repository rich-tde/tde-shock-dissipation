from importlib.resources import files
from typing import Any
import warnings

from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import unyt as u

# from richio.config import FIGURES_DIR, PROCESSED_DATA_DIR

# import typer
# app = typer.Typer()

# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = FIGURES_DIR / "plot.png",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Generating plot from data...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Plot generation complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()


def use_nice_style():
    style_path = files("richio.styles").joinpath("nice.mplstyle")
    plt.style.use(style_path)


class SnapshotPlotter:
    def __init__(self, snap):
        self.snap = snap
        self.peek = self.scatter  # alias for peek

    def scatter(self):
        pass

    def slice(
        self, 
        data: str | ArrayLike, 
        res: int | ArrayLike, 
        x: str | ArrayLike = "X", 
        y: str | ArrayLike = "Y", 
        z: str | ArrayLike = "Z",
        plane: str = "xy",
        slice_coord: float | u.array.unyt_quantity | None = None,
        box_size: ArrayLike | None = None,
        unit_system: str = "cgs",
        volume_selection: bool = True, # select based on volume to speed up calculation
        ax: Any | None = None,
        cmap: str | Colormap = "twilight",
        **kwargs
    ):
        """
        Make a slice plot.
        """
        # TODO: implement star_mask : put data to zero instead of removing them
        # (which is bad if you use nn) and put them to the lowest color in
        # colormap when plotting (something like set_bad...)

        # Fetch data
        data = self._get_data(data)
        x = self._get_data(x)
        y = self._get_data(y)
        z = self._get_data(z)
        if volume_selection:
            volume = self._get_data('volume')

        # Set resolution
        try:
            nx, ny = res[0], res[1]
        except TypeError:
            nx = ny = res

        # Set boxsize
        if box_size is None:
            x0, y0, z0, x1, y1, z1 = self.snap.box  # Load the box size
            x0, y0, z0 = _parse_plane(plane, x0, y0, z0)
            x1, y1, z1 = _parse_plane(plane, x1, y1, z1)
        else:
            x0, y0, x1, y1= box_size
        
        # x_slice, y_slice, z_slice should only have one that is not None
        x, y, z = _parse_plane(plane, x, y, z)      # redefine x y to be the plane, z the sliced direction

        # Make Euclidean grid
        xspace = np.linspace(x0, x1, nx, endpoint=False)
        yspace = np.linspace(x0, y1, ny, endpoint=False)
        zspace = slice_coord

        if volume_selection:
            mask = np.abs(z - slice_coord) < volume**(1/3)
            # assuming spherical cells, V^(1/3)=(4pi/3)^(1/3)R ~ 1.6R, we don't
            # include the factor such that if V is not round enough we won't
            # lose too much accuracy
            data = data[mask]
            x = x[mask]
            y = y[mask]
            z = z[mask]

        X, Y, Z = np.meshgrid(xspace, yspace, zspace, indexing="ij")

        coords = np.stack([x, y, z], axis=-1)  # coordinates of the particles 
        grid_coords = np.stack([X, Y, Z], axis=-1)  # coordinates of the grid (query points)
        grid_coords = np.squeeze(grid_coords)            # remove extra dimension (nx, ny, 1, 3) to (nx, ny, 3)

        i = _kdtree_interpolate(coords=coords, grid_coords=grid_coords)

        sliced_data = data[i]
        sliced_data = sliced_data.in_base(unit_system)

        if ax is None:
            fig, ax = plt.subplots()

        # Plot
        xx, yy = np.meshgrid(xspace, yspace, indexing="ij")
        im = ax.pcolormesh(xx, yy, np.log10(sliced_data), cmap=cmap, **kwargs)
        unit_latex = sliced_data.units.latex_repr
        plt.colorbar(im, ax=ax, label=f"$\\log(\\rho/{unit_latex})$")   # TODO: allow for general plots

        return ax, sliced_data


        

    def projection(
        self,
        data: str | ArrayLike,
        res: int | ArrayLike,
        x: str | ArrayLike = "X",
        y: str | ArrayLike = "Y",
        z: str | ArrayLike = "Z",
        box_size: ArrayLike | None = None,
        unit_system: str = "cgs",
        ax: Any | None = None,
        cmap: str | Colormap = "twilight",
        **kwargs,
    ):
        """
        Make a projection plot. To make use of the unit system, use either str
        keys or unyt_array data for `data`, `x`, `y`, `z`, `box_size`.
        """
        # Fetch data
        data = self._get_data(data)
        x = self._get_data(x)
        y = self._get_data(y)
        z = self._get_data(z)

        # Set boxsize
        if box_size is None:
            x0, y0, z0, x1, y1, z1 = self.snap.box  # Load the box size
        else:
            x0, y0, z0, x1, y1, z1 = box_size

        # Set resolution
        try:
            nx, ny, nz = res[0], res[1], res[2]
        except TypeError:
            nx = ny = nz = res

        # Make Euclidean grid
        xspace = np.linspace(x0, x1, nx, endpoint=False)  # disable endpoints such that dz = (z1-z0)/res instead of (z1-z0)/(res-1)
        yspace = np.linspace(y0, y1, ny, endpoint=False)
        zspace = np.linspace(z0, z1, nz, endpoint=False)  # TODO: add an option to use np.geomspace

        X, Y, Z = np.meshgrid(xspace, yspace, zspace, indexing="ij")

        coords = np.stack([x, y, z], axis=-1)  # coordinates of the particles
        grid_coords = np.stack([X, Y, Z], axis=-1)  # coordinates of the grid (query points)

        i = _kdtree_interpolate(coords=coords, grid_coords=grid_coords)

        grid_data = data[i]

        dz = (z1 - z0) / nz
        projected_data = np.sum(grid_data * dz, axis=-1).in_base(unit_system)

        if ax is None:
            fig, ax = plt.subplots()

        # Plot
        xx, yy = np.meshgrid(xspace, yspace, indexing="ij")
        im = ax.pcolormesh(xx, yy, np.log10(projected_data), cmap=cmap, **kwargs)
        unit_latex = projected_data.units.latex_repr
        plt.colorbar(im, ax=ax, label=f"$\\log(\\Sigma/{unit_latex})$") # TODO: update for non-density cases

        return ax, projected_data

    def _get_data(
        self, data: str | ArrayLike
    ) -> u.unyt_array:  # TODO: add masking option that automatically mask floor gas
        if isinstance(data, str):
            key = data
            data = self.snap[key]
        elif isinstance(data, u.unyt_array):
            pass
        elif isinstance(data, np.ndarray):
            data = data * u.Dimensionless
            warnings.warn("No unit attached, assuming data is dimensionless.")
        else:
            raise TypeError(
                f"Data type {type(data)} unsupported."
                "Use either str, unyt.unyt_array, or numpy.ndarray."
            )

        return data



def _parse_plane(plane, x, y, z):
    """
    Parse a string input "xy" to data x, y, z; "yz" to y, z, x; "zx" to z, x, y,
    etc, in order to specify the slicing plane.
    """

    def _parse_xyz(char, x, y, z):
        if char == 'x':
            return x
        elif char == 'y':
            return y
        elif char == 'z':
            return z

    x1 = _parse_xyz(plane[0], x, y, z)
    x2 = _parse_xyz(plane[1], x, y, z)

    if (x1 is x and x2 is y) or (x1 is y and x2 is x):
        x3 = z
    elif (x1 is x and x2 is z) or (x1 is z and x2 is x):
        x3 = y
    elif (x1 is y and x2 is z) or (x1 is z and x2 is y):
        x3 = x
    else:
        raise Exception(f"Plane {plane} is unrecognizable.")

    return x1, x2, x3



def _kdtree_interpolate(coords, grid_coords, k=1, eps=0, workers=1, **kwargs) -> np.ndaray(dtype=int):

    from scipy.spatial import KDTree

    tree = KDTree(coords)  # build tree
    d, i = tree.query(
        grid_coords, k=1, eps=0.0, p=2, workers=1
    )  # the most time-consuming step

    return i
