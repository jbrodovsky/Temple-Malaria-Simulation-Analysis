"""
Utilities for plotting and raster file manipulation.

This module provides functions for visualizing district, population, and prevalence
data from raster arrays, as well as reading and writing raster files in ASCII grid format.
"""

import numpy
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from masim_analysis import configure


def plot_districts(
    districts_raster: numpy.ndarray,
    labels: list[str],
    country_name: str,
    fig_size: tuple[int, int] = (10, 10),
    loc=None,
) -> Figure:
    """
    Plot the district mapping of the country according to the raster array.

    Parameters
    ----------
    districts_raster : numpy.typing.NDArray
        Raster array representing the districts.
    labels : list[str]
        List of labels for the districts.
    country_name : str
        Name of the country for the plot title.
    fig_size : tuple[int, int], optional
        Size of the figure, by default (10, 10).
    loc : str, optional
        Location of the legend, by default None.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    cmap = plt.get_cmap("tab20", len(labels))
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(districts_raster, cmap=cmap)
    ax.set_title(f"{country_name} Districts")
    # create legend handles
    handles = [Patch(color=cmap(i), label=labels[i + 1].replace("_", " ")) for i in range(11)]
    ax.legend(
        # bbox_to_anchor=bbox_to_anchor,
        handles=handles,
        title="Districts",
        loc=loc,
    )
    return fig


def plot_population(
    population_raster: numpy.ndarray, country_name: str, fig_size: tuple[int, int] = (10, 10), population_upper_limit: float = 1000
) -> Figure:
    """
    Plot the population density of the country according to the raster array.

    Parameters
    ----------
    population_raster : numpy.typing.NDArray
        Raster array representing population density.
    country_name : str
        Name of the country for the plot title.
    fig_size : tuple[int, int], optional
        Size of the figure, by default (10, 10).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    img = ax.imshow(population_raster, cmap="turbo")
    img.set_clim(0, population_upper_limit)
    ax.set_title(f"{country_name} Population")
    fig.colorbar(img, ax=ax, label="Population")
    
    return fig


def plot_prevalence(
    prevalence_raster: numpy.ndarray, country_name: str, fig_size: tuple[int, int] = (10, 10)
) -> Figure:
    """
    Plot the prevalence of malaria according to the raster array.

    Parameters
    ----------
    prevalence_raster : numpy.typing.NDArray
        Raster array representing malaria prevalence.
    country_name : str
        Name of the country for the plot title.
    fig_size : tuple[int, int], optional
        Size of the figure, by default (10, 10).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    img = ax.imshow(prevalence_raster, cmap="coolwarm")
    ax.set_title(f"{country_name} Prevalence")
    fig.colorbar(img, ax=ax, label="Prevalence")
    return fig


def read_raster(file: str) -> tuple[numpy.ndarray, dict]:
    """
    Read in a raster file and return the raster array and metadata.

    Parameters
    ----------
    file : str
        Path to the raster file.

    Returns
    -------
    tuple
        A tuple containing the raster array (numpy.typing.NDArray) and metadata dictionary (dict).
    """
    with open(file, "r") as f:
        data = f.read().splitlines()
    metadata = data[:6]
    data = data[6:]
    metadata = {line.split()[0]: float(line.split()[1]) for line in metadata}
    raster = numpy.zeros((int(metadata["nrows"]), int(metadata["ncols"])))
    for i, line in enumerate(data):
        line = line.split()
        line = numpy.asarray(line, dtype=float)
        raster[i, :] = line
    raster[raster == metadata["NODATA_value"]] = numpy.nan
    return raster, metadata


def write_raster(raster: numpy.ndarray, file: str, xllcorner: float, yllcorner: float, cellsize: int = 5000) -> None:
    """
    Write a raster array to a file.

    Parameters
    ----------
    raster : numpy.typing.NDArray
        The raster array to write.
    file : str
        The path to the output file.
    xllcorner : float
        The x-coordinate of the lower left corner of the raster.
    yllcorner : float
        The y-coordinate of the lower left corner of the raster.
    cellsize : int, optional
        The size of each cell in the raster, by default 5000.
    """
    nrows, ncols = raster.shape
    raster = numpy.where(numpy.isnan(raster), configure.NODATA_VALUE, raster)
    with open(file, "w") as f:
        f.write(f"ncols\t{ncols}\n")
        f.write(f"nrows\t{nrows}\n")
        f.write(f"xllcorner\t{xllcorner}\n")
        f.write(f"yllcorner\t{yllcorner}\n")
        f.write(f"cellsize\t{cellsize}\n")
        f.write(f"NODATA_value\t{configure.NODATA_VALUE}\n")
        for row in raster:
            f.write(" ".join([str(value) for value in row]) + "\n")
