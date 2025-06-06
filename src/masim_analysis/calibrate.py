"""
Calibration scripts for MaSim.

This module provides functions for generating MaSim configurations for calibration,
running calibration simulations, summarizing results, and fitting models (e.g., sigmoid)
to calibration data.
"""

# Country calibration script
import json
import os
from datetime import date
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy
from numpy.typing import NDArray
import pandas
import seaborn as sns
from matplotlib.figure import Figure
from ruamel.yaml import YAML
from ruamel.yaml.emitter import EmitterError

# Curve Fitting (linear and polynomial regression models)
from scipy.optimize import curve_fit
from tqdm import tqdm

from masim_analysis import analysis, commands, configure

yaml = YAML()


def generate_configuration_files(
    country_code: str,
    calibration_year: int,
    population_bins: list[int],
    access_rates: list[float],
    beta_values: list[float],
    birth_rate: float,
    death_rate: list[float],
    initial_age_structure: list[int],
    age_distribution: list[float],
    seasonality_file_name: str = "seasonality",
    strategy_db: dict[int, dict[str, str | list[int]]] = configure.STRATEGY_DB,
    events: Optional[list[dict]] = None,
) -> None:
    """
    Generate MaSim configuration files for a given country and calibration parameters.

    Parameters
    ----------
    country_code : str
        The country code (e.g., "RWA", "MOZ").
    calibration_year : int
        The year for which calibration is being performed.
    population_bins : list[int]
        List of population bins for calibration.
    access_rates : list[float]
        List of treatment access rates for calibration.
    beta_values : list[float]
        List of beta values (transmission intensity) for calibration.
    birth_rate : float
        The birth rate for the simulation.
    death_rate : list[float]
        List representing the age-specific death rates.
    initial_age_structure : list[int]
        List representing the initial age structure of the population.
    age_distribution : list[float]
        List representing the age distribution for certain outputs/calculations.
    seasonality_file_name : str, optional
        Name of the seasonality file, by default "seasonality".
    strategy_db : dict[int, dict[str, str | list[int]]], optional
        Database of intervention strategies, by default configure.STRATEGY_DB.
    events : Optional[list[dict]], optional
        List of scheduled events for the simulation, by default None.
    """
    # configure calibration dates
    comparison = date(calibration_year, 1, 1)
    start = date(calibration_year - 11, 1, 1)
    end = date(calibration_year + 1, 12, 31)
    # Create default execution control dictionary

    # Generate the configuration files
    for pop in tqdm(population_bins):
        for access in access_rates:
            for beta in beta_values:
                execution_control = configure.configure(
                    country_code,
                    birth_rate,
                    initial_age_structure,
                    age_distribution,
                    death_rate,
                    start,
                    comparison,
                    end,
                    strategy_db,
                    f"{pop}_{access}_{beta}",
                    beta,
                    pop,
                    access,
                    True,
                )
                write_pixel_data_files(execution_control["raster_db"], pop)
                output_path = os.path.join("conf", country_code, "calibration", f"cal_{pop}_{access}_{beta}.yml")
                try:
                    yaml.dump(execution_control, open(output_path, "w"))
                except EmitterError as e:
                    print(f"Error writing YAML file {output_path}: {e}")


def write_pixel_data_files(raster_db: dict, population: int):
    """
    Write pixel data files based on raster database and population.

    Parameters
    ----------
    raster_db : dict
        A dictionary containing raster data or paths to raster files.
    population : int
        The population value to use in generating pixel data.
    """
    with open(raster_db["population_raster"], "w") as file:
        file.write(
            f"ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value {configure.NODATA_VALUE}\n{population}"
        )
    with open(raster_db["district_raster"], "w") as file:
        file.write(f"ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value {configure.NODATA_VALUE}\n1")


def generate_command_and_job_files(
    country_code: str,
    population_bins: list[int],
    access_rates: list[float],
    beta_values: list[float],
    repetitions: int = 20,
    cores: int = 28,
    nodes: int = 1,
):
    """
    Generate MaSim command files and job submission scripts for calibration runs.

    Parameters
    ----------
    country_code : str
        The country code (e.g., "RWA", "MOZ").
    population_bins : list[int]
        List of population bins used in calibration.
    access_rates : list[float]
        List of treatment access rates used in calibration.
    beta_values : list[float]
        List of beta values used in calibration.
    repetitions : int, optional
        Number of repetitions for each simulation, by default 20.
    cores : int, optional
        Number of cores to request per node for job submission, by default 28.
    nodes : int, optional
        Number of nodes to request for job submission, by default 1.
    """
    # Generate the command and job files
    for pop in tqdm(population_bins):
        filename = f"{country_code}_{pop}_cmds.txt"
        with open(filename, "w") as f:
            for access in access_rates:
                for beta in beta_values:
                    for j in range(repetitions):
                        f.write(
                            f"./bin/MaSim -i ./conf/{country_code}/calibration/cal_{pop}_{access}_{beta}.yml -o ./output/{country_code}/calibration/cal_{pop}_{access}_{beta}_ -r SQLiteDistrictReporter -j {j + 1}\n"
                        )
        commands.generate_job_file(
            filename,
            job_name=f"{country_code}_{pop}_jobs",
            cores_override=cores,
            nodes_override=nodes,
        )


def summarize_calibration_results(
    country_code: str,
    population_bins: list[int],
    access_rates: list[float],
    beta_values: list[float],
    comparison_year: int,
    output_dir: str,
    repetitions: int = 20,
) -> pandas.DataFrame:
    """
    Summarize the results of MaSim calibration runs.

    This function reads output files from multiple simulation runs,
    aggregates relevant metrics (e.g., PfPR), and returns a summary DataFrame.

    Parameters
    ----------
    country_code : str
        The country code.
    population_bins : list[int]
        Population bins used in calibration.
    access_rates : list[float]
        Access rates used in calibration.
    beta_values : list[float]
        Beta values used in calibration.
    comparison_year : int
        The year used for comparison or validation of results.
    output_dir : str
        Directory containing the MaSim output files.
    repetitions : int, optional
        Number of repetitions run for each parameter set, by default 20.

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarizing the calibration results.
    """
    base_file_path = os.path.join(output_dir, country_code, "calibration")
    summary = pandas.DataFrame(
        columns=["population", "access_rate", "beta", "iteration", "pfprunder5", "pfpr2to10", "pfprall"]
    )
    comparison = date(comparison_year, 1, 1)
    year_end = date(comparison_year + 1, 1, 1)
    # Process summary
    for pop in tqdm(population_bins):
        for access in access_rates:
            for beta in beta_values:
                for i in range(1, repetitions + 1):
                    filename = f"cal_{pop}_{access}_{beta}_monthly_data_{i}"
                    file = os.path.join(base_file_path, f"{filename}.db")
                    try:
                        months = analysis.get_table(file, "monthlydata")
                        monthlysitedata = analysis.get_table(file, "monthlysitedata")
                    except FileNotFoundError as e:
                        print(f"File not found: {e}")
                        continue
                    data = pandas.merge(monthlysitedata, months, left_on="monthlydataid", right_on="id")
                    data["date"] = pandas.to_datetime(data["modeltime"], unit="s")

                    summary.loc[filename] = data[
                        (data["date"] >= comparison.strftime("%Y-%m-%d"))
                        & (data["date"] < year_end.strftime("%Y-%m-%d"))
                    ][["pfprunder5", "pfpr2to10", "pfprall"]].mean()
                    summary.loc[filename, "population"] = pop
                    summary.loc[filename, "access_rate"] = access
                    summary.loc[filename, "beta"] = beta
                    summary.loc[filename, "iteration"] = int(i)

    # summary.to_csv(f"{base_file_path}/calibration_summary.csv")
    return summary


def process_missing_jobs(
    country_code: str,
    population_bins: list[int],
    access_rates: list[float],
    beta_values: list[float],
    output_dir: str,
    repetitions: int = 20,
):
    """
    Identify and potentially re-process missing jobs from a calibration run.

    This function checks for expected output files and may trigger reruns
    or report missing data.

    Parameters
    ----------
    country_code : str
        The country code.
    population_bins : list[int]
        Population bins used in calibration.
    access_rates : list[float]
        Access rates used in calibration.
    beta_values : list[float]
        Beta values used in calibration.
    output_dir : str
        Directory containing the MaSim output files.
    repetitions : int, optional
        Number of repetitions expected for each parameter set, by default 20.
    """
    base_file_path = os.path.join(output_dir, country_code, "calibration")
    for pop in tqdm(population_bins):
        for access in access_rates:
            for beta in beta_values:
                for i in range(repetitions):
                    filename = f"cal_{pop}_{access}_{beta}_monthly_data_{i + 1}"
                    file = os.path.join(base_file_path, f"{filename}.db")
                    try:
                        # Attempt to read the monthly data and monthly site data
                        # If the file exists, it will be read successfully
                        _months = analysis.get_table(file, "monthlydata")
                        _monthlysitedata = analysis.get_table(file, "monthlysitedata")
                    except FileNotFoundError:
                        with open(f"missing_calibration_runs_{pop}.txt", "a") as f:
                            # f.write(f"{e}\n")
                            f.write(
                                f"./bin/MaSim -i ./conf/{country_code}/calibration/cal_{pop}_{access}_{beta}.yml -o ./output/{country_code}/calibration/cal_{pop}_{access}_{beta}_ -r SQLiteDistrictReporter -j {i + 1}\n"
                            )
                        if not os.path.exists(f"missing_calibration_runs_{pop}_job.sh"):
                            with open(f"missing_calibration_runs_{pop}_job.sh", "w") as f:
                                f.write("#!/bin/sh\n")
                                f.write("#PBS -l walltime=48:00:00\n")
                                f.write(f"#PBS -N MissingCalibrationRuns_{pop}\n")
                                f.write("#PBS -q normal\n")
                                f.write("#PBS -l nodes=4:ppn=28\n")
                                f.write("cd $PBS_O_WORKDIR\n")
                                f.write(f"torque-launch missing_calibration_runs_{pop}.txt\n")
                        continue


def sigmoid_fit(x, a, b, c):
    """
    Sigmoid function for curve fitting.

    Equation: y = 1 / (1 + exp(-c * (x - b))) * a

    Parameters
    ----------
    x : array_like
        The independent variable.
    a : float
        The maximum value (amplitude) of the sigmoid.
    b : float
        The x-value of the sigmoid's midpoint.
    c : float
        The steepness of the sigmoid curve.

    Returns
    -------
    array_like
        The calculated y-values of the sigmoid function.
    """
    return a / (1 + numpy.exp(-b * (x - c)))


def find_beta(
    pfpr_target: NDArray,
    linear_model: Any,  # Replace Any with a more specific type if known
    popt: NDArray,
    pfpr_cutoff: float,
) -> NDArray:
    """
    Find Beta values corresponding to an array of PfPR (Plasmodium falciparum
    parasite rate) values using a fitted model.

    This function uses an inverse prediction from a previously fitted model
    (e.g., linear or sigmoid) to estimate the beta value that would produce
    a target PfPR.

    Parameters
    ----------
    pfpr_target : numpy.typing.NDArray
        Target PfPR values.
    linear_model : typing.Any
        The fitted linear model (or its prediction function).
        (Note: Consider replacing `Any` with a more specific type if available for the linear model object)
    popt : numpy.typing.NDArray
        Optimal parameters for a non-linear model (e.g., sigmoid), if applicable.
    pfpr_cutoff : float
        A PfPR cutoff value, potentially used for model selection or extrapolation.

    Returns
    -------
    numpy.typing.NDArray
        Estimated Beta values corresponding to the target PfPRs.
    """
    pfpr_target = numpy.array(pfpr_target)  # Ensure input is a NumPy array
    beta_values = numpy.zeros_like(pfpr_target, dtype=numpy.float64)  # Placeholder for results

    # Linear region: pfpr_target < cutoff
    mask_linear = pfpr_target < pfpr_cutoff
    if numpy.any(mask_linear):
        beta_log_linear = (pfpr_target[mask_linear] - linear_model.intercept_) / linear_model.coef_[0]
        beta_values[mask_linear] = 10 ** (beta_log_linear)  # Convert back from log-space

    # Sigmoid region: pfpr_target >= cutoff
    mask_sigmoid = pfpr_target >= pfpr_cutoff
    if numpy.any(mask_sigmoid):
        a, b, c = popt
        beta_log_sigmoid = c - (1 / b) * numpy.log(a / pfpr_target[mask_sigmoid] - 1)
        beta_values[mask_sigmoid] = 10 ** (beta_log_sigmoid)  # Convert back from log-space

    return beta_values


def plot_log_sigmoid(
    populations: list[int] | NDArray, access_rates: list[float] | NDArray, means: pandas.DataFrame, model_map: dict
) -> Figure:
    """
    Plot the results of log-sigmoid fitting for calibration data.

    This function generates a grid of plots, where each subplot shows the
    relationship between Beta (log-transformed) and PfPR for a specific
    population and access rate, along with the fitted sigmoid curve.

    Parameters
    ----------
    populations : list[int] | numpy.typing.NDArray
        List of unique population values.
    access_rates : list[float] | numpy.typing.NDArray
        List of unique access rate values.
    means : pandas.DataFrame
        DataFrame containing the mean PfPR and Beta values for each
        combination of population and access rate.
        Expected columns: 'population', 'access_rate', 'pfpr2to10', 'beta'.
    model_map : dict
        A nested dictionary containing the parameters of the fitted sigmoid models.
        Structure: {access_rate: {population: model_parameters}}.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the grid of plots.
    """
    # Determine grid size
    num_rows = len(populations)
    num_cols = len(access_rates)
    # Create subplots grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharex=True, sharey=True)
    # Ensure axes is always a 2D list for consistency
    if num_rows == 1:
        axes = numpy.array([axes])  # Convert to 2D array
    if num_cols == 1:
        axes = numpy.array([[ax] for ax in axes])  # Convert to 2D array
    # Define cutoff based on pfpr2to10_mean
    pfpr_cutoff = 0.0  # Set the desired cutoff for pfpr2to10_mean
    # Perform regression for each (Population, TreatmentAccess) group
    for i, population in enumerate(populations):
        for j, treatment_access in enumerate(access_rates):
            ax = axes[i, j]  # Select subplot location
            # Filter the data for the current Population and TreatmentAccess
            group = means[(means["population"] == population) & (means["access_rate"] == treatment_access)]
            if group.empty:
                ax.set_visible(False)  # Hide empty plots
                print(f"No data for population {population} and access rate {treatment_access}")
                continue
            # Plot data
            ax.set_xscale("log")
            sns.scatterplot(
                x=group["beta"].values, y=group["pfpr2to10"].div(100).values, ax=ax, label="Data", color="black"
            )
            # Predictions
            popt = model_map[treatment_access][population]
            if popt is not None:
                pfpr_targets = numpy.linspace(0, 1, 100).reshape(-1, 1)
                X_plot = find_beta(pfpr_targets, None, popt, pfpr_cutoff)
                ax.plot(X_plot, pfpr_targets, color="red")
            # Titles & Labels
            ax.set_title(f"Pop {population}, Access {treatment_access}")
            if j == 0:
                ax.set_ylabel("pfpr2to10_mean")  # Label only on first column
            if i == num_rows - 1:
                ax.set_xlabel("Beta")  # Label only on last row
            ax.legend(fontsize=7)
    # Adjust layout
    plt.suptitle("Curve Fitting for beta vs pfpr2to10 by Population & Treatment Access", fontsize=16)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def log_sigmoid_fit(
    populations: list[int] | NDArray,
    access_rates: list[float] | NDArray,
    means: pandas.DataFrame,
) -> dict[float, dict[int, Any]]:
    """
    Perform log-sigmoid regression on calibration data.

    This function fits a sigmoid model to the relationship between
    log-transformed Beta values and PfPR (Plasmodium falciparum parasite rate)
    for different combinations of population and treatment access rates.

    Parameters
    ----------
    populations : list[int] | numpy.typing.NDArray
        List of unique population values for which to fit the model.
    access_rates : list[float] | numpy.typing.NDArray
        List of unique treatment access rates for which to fit the model.
    means : pandas.DataFrame
        A DataFrame containing the mean PfPR and Beta values from calibration runs.
        It must include columns: 'population', 'access_rate', 'pfpr2to10', and 'beta'.
        'pfpr2to10' should be the mean PfPR in 2-10 year olds.
        'beta' is the transmission parameter.

    Returns
    -------
    dict[float, dict[int, typing.Any]]
        A nested dictionary where the outer keys are access rates, inner keys are
        populations, and values are the parameters (e.g., from `scipy.optimize.curve_fit`)
        of the fitted log-sigmoid model for that combination.

    Raises
    ------
    RuntimeError
        If `curve_fit` fails to converge for a particular data subset.
    """
    # Define cutoff based on pfpr2to10_mean
    pfpr_cutoff = 0.0  # Set the desired cutoff for pfpr2to10_mean
    models_map = {
        access_rate: {population: None for population in populations} for access_rate in access_rates
    }  # stores trained model for every parameter configuration

    # Perform regression for each (Population, TreatmentAccess) group
    for population in populations:
        for treatment_access in access_rates:
            # Filter the data for the current Population and TreatmentAccess
            group = means[(means["population"] == population) & (means["access_rate"] == treatment_access)]
            if group.empty:
                continue

            group = group.copy()  # Create a copy to avoid SettingWithCopyWarning
            group["pfpr2to10"] = numpy.array(group["pfpr2to10"].values) / 100
            group["beta"] = numpy.log10(numpy.array(group["beta"].values))

            X = group["beta"].values  # Log of Predictor (Beta)
            y = group["pfpr2to10"].values  # Response variable

            # Determine cutoff Beta based on pfpr2to10_mean
            if numpy.any(y < pfpr_cutoff):
                cutoff_beta_val = numpy.max(X[y < pfpr_cutoff])  # Largest Beta where pfpr2to10_mean <= cutoff
            else:
                cutoff_beta_val = numpy.min(X)  # Default to min beta if no values are below cutoff

            X_filtered = X
            y_filtered = y

            if len(X_filtered) < 3:  # Check if enough data points for regression
                continue

            try:
                # Perform sigmoid regression
                popt, pcov = curve_fit(
                    sigmoid_fit,
                    X_filtered,
                    y_filtered,
                    maxfev=5000,
                    p0=[
                        numpy.max(y_filtered) if len(y_filtered) > 0 else 0,
                        numpy.median(X_filtered) if len(X_filtered) > 0 else 0,
                        1,
                    ],
                )
                models_map[treatment_access][population] = popt  # Store parameters

            except RuntimeError:
                pass  # Or handle error as needed
            except TypeError:  # Handle cases where curve_fit might receive empty arrays from p0 logic
                pass

    return models_map


def get_beta(
    models_map: dict[float, dict[int, list[float]]], access_rate: float, population: int, pfpr: float
) -> float:
    """
    Get the beta value for a given access rate, population, and pfpr target.

    Parameters
    ----------
    models_map : dict[float, dict[int, list[float]]]
        Nested dictionary containing sigmoid model parameters for each
        access rate and population combination.
    access_rate : float
        Treatment access rate.
    population : int
        Population size.
    pfpr : float
        Target PfPR value.

    Returns
    -------
    float
        Estimated beta value.
    """
    if numpy.isnan(access_rate) or numpy.isnan(population):
        return numpy.nan
    # Find which population key to use by searching for the largest population less than or equal to the given population
    populations = numpy.asarray(list(models_map[access_rate].keys())).squeeze()
    if population <= 10:
        population = 10
    else:
        population_key = numpy.argwhere(populations <= population).squeeze().tolist()
        if type(population_key) is list:
            if len(population_key) > 0:
                population = int(populations[population_key[-1]])
        else:
            population = int(populations[population_key])
    # Get the model
    a = 0.0
    b = 0.0
    c = 0.0
    try:
        a, b, c = models_map[access_rate][population]
    except TypeError:
        coefs = models_map[access_rate][population]
        a = coefs[0]
        b = coefs[1]
        c = coefs[2]
    except KeyError as e:
        print(f"KeyError: {e} for access rate {access_rate} and population {population}")
        return numpy.nan
    except ValueError as e:
        print(f"ValueError: {e} for access rate {access_rate} and population {population}")
        print(f"Received the following coefficients: {models_map[access_rate][population]}")
    # Get the beta value
    try:
        beta_log = c - (1 / b) * numpy.log(a / pfpr - 1)
    except ZeroDivisionError:
        beta_log = 0
    beta = 10**beta_log
    if numpy.isnan(beta):
        return 0
    return beta


def load_beta_model(filename: str) -> dict:
    """
    Load the beta model from a json file.

    Parameters
    ----------
    filename : str
        The name of the json file to load.

    Returns
    -------
    dict
        A dictionary containing the beta model.
    """
    models = json.load(open(filename, "r"))
    numeric = {float(k): {int(float(k2)): v2 for k2, v2 in v.items()} for k, v in models.items()}
    return numeric


def create_beta_map(
    models_map: dict[float, dict[int, list[float]]],
    population_raster: NDArray,
    access_rate_raster: NDArray,
    prevalence_raster: NDArray,
) -> NDArray:
    """
    Create a beta map based on the population, access rate and prevalence rasters.

    Parameters
    ----------
    models_map : dict[float, dict[int, list[float]]]
        Nested dictionary containing sigmoid model parameters for each
        access rate and population combination.
    population_raster : numpy.typing.NDArray
        Population raster.
    access_rate_raster : numpy.typing.NDArray
        Access rate raster.
    prevalence_raster : numpy.typing.NDArray
        Prevalence raster.

    Returns
    -------
    numpy.typing.NDArray
        Beta map.
    """
    # Create a beta map
    beta_map = numpy.zeros_like(population_raster)
    # Naive implementation of beta map
    rows, cols = beta_map.shape
    for r in range(rows):
        for c in range(cols):
            beta_map[r, c] = get_beta(
                models_map, access_rate_raster[r, c], population_raster[r, c], prevalence_raster[r, c]
            )
    return beta_map
