"""
Plots output of kelvin_helmholtz.py.

Based heavily on the plot_shear example provided with the dedalus3 code.

Usage:
    plot_snapshots.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]
"""

import h5py
import matplotlib as mpl
from pathlib import joinpath

mpl.use("Agg")
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools


def main(filename: str, start: int, count: int, output: str) -> None:
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    tasks = ["density"]
    scale = 2
    dpi = 200
    title_func = lambda sim_time: f"t = {sim_time:.3f}"
    savename_func = lambda write: f"write_{write:06}.png"
    # Layout
    nrows, ncols = 1, 1
    image = plot_tools.Box(1, 2)
    pad = plot_tools.Frame(0.2, 0, 0, 0)
    margin = plot_tools.Frame(0.2, 0.1, 0, 0)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    # Plot writes
    with h5py.File(filename, mode="r") as file:
        for index in range(start, start + count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = file["tasks"][task]
                plot_tools.plot_bot_3d(
                    dset,
                    0,
                    index,
                    axes=axes,
                    title=task,
                    even_scale=True,
                    visible_axes=False,
                )
            # Add time title
            title = title_func(file["scales/sim_time"][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.45, y=title_height, ha="left")
            # Save figure
            savename = savename_func(file["scales/write_number"][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":
    import pathlib

    from dedalus.tools import post
    from dedalus.tools.parallel import Sync
    from docopt import docopt

    args = docopt(__doc__)

    output_path = pathlib.Path(args["--output"]).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0 and not output_path.exists():
            output_path.mkdir()
    post.visit_writes(args["<files>"], main, output=output_path)
