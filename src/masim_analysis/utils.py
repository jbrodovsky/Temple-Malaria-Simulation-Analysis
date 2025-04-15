# Country calibration script
import os
from datetime import date

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from masim_analysis import configure


def plot_districts(districts_raster: np.ndarray, labels: list[str], country_name: str) -> plt.Figure:
    """
    Plot the distict mapping of the country according to the raster array.
    """
    cmap = plt.get_cmap("tab20", len(labels))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(districts_raster, cmap=cmap)
    ax.set_title(f"{country_name} Districts")
    # create legend handles
    handles = [Patch(color=cmap(i), label=labels[i + 1].replace("_", " ")) for i in range(11)]
    ax.legend(bbox_to_anchor=(1.75, 1), handles=handles, title="Districts", loc="upper right")
    return fig


def plot_population(population_raster: np.ndarray, country_name: str) -> plt.Figure:
    """
    Plot the population density of the country according to the raster array.
    """
    fig, ax = plt.subplots()
    ax.imshow(population_raster, cmap="coolwarm")
    ax.set_title(f"{country_name} Population")
    fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax, label="Population Density")
    return fig


def plot_prevalence(prevalence_raster: np.ndarray, country_name: str) -> plt.Figure:
    """
    Plot the prevalence of malaria according to the raster array.
    """
    fig, ax = plt.subplots()
    ax.imshow(prevalence_raster, cmap="coolwarm")
    ax.set_title(f"{country_name} Prevalence")
    fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax, label="Prevalence")


def read_raster(file: str) -> np.ndarray:
    """
    Read in a raster file and return the raster array and metadata.

    Args:
        file (str): Path to the raster file.

    Returns:
        tuple: A tuple containing the raster array and metadata.
    """
    with open(file, "r") as f:
        data = f.read().splitlines()
    metadata = data[:6]
    data = data[6:]
    metadata = {line.split()[0]: float(line.split()[1]) for line in metadata}
    raster = np.zeros((int(metadata["nrows"]), int(metadata["ncols"])))
    for i, line in enumerate(data):
        line = line.split()
        line = np.asarray(line, dtype=float)
        raster[i, :] = line
    raster[raster == metadata["NODATA_value"]] = np.nan
    return raster, metadata


def write_raster(raster: np.ndarray, file: str, xllcorner: float, yllcorner: float, cellsize: int = 5000) -> None:
    """
    Write a raster array to a file.

    Args:
        raster (np.ndarray): The raster array to write.
        file (str): The path to the output file.
        xllcorner (float): The x-coordinate of the lower left corner of the raster.
        yllcorner (float): The y-coordinate of the lower left corner of the raster.
        cellsize (int): The size of each cell in the raster. Defaults to 5000.
    """
    nrows, ncols = raster.shape
    raster = np.where(np.isnan(raster), configure.NODATA_VALUE, raster)
    with open(file, "w") as f:
        f.write(f"ncols\t{ncols}\n")
        f.write(f"nrows\t{nrows}\n")
        f.write(f"xllcorner\t{xllcorner}\n")
        f.write(f"yllcorner\t{yllcorner}\n")
        f.write(f"cellsize\t{cellsize}\n")
        f.write(f"NODATA_value\t{configure.NODATA_VALUE}\n")
        for row in raster:
            f.write(" ".join([str(value) for value in row]) + "\n")
