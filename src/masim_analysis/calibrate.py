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
from numpy.typing import NDArray, ArrayLike
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


def sigmoid(x, a, b, c):
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


def inverse_sigmoid(y, a, b, c):
    """
    Inverse sigmoid function for prediction.

    Equation: x = c - (1 / b) * log(a / y - 1)

    Parameters
    ----------
    x : array_like
        The dependent variable (y-values).
    a : float
        The maximum value (amplitude) of the sigmoid.
    b : float
        The steepness of the sigmoid curve.
    c : float
        The x-value of the sigmoid's midpoint.

    Returns
    -------
    array_like
        The calculated x-values of the inverse sigmoid function.
    """
    return c - (1 / b) * numpy.log(a / y - 1)


def fit_log_sigmoid_model(
    betas: ArrayLike,
    pfpr: ArrayLike,
    pfpr_cutoff: float = 0.0,
) -> NDArray[numpy.float64]:
    """
    Fit sigmoid models to calibration data for different populations and treatment access rates.

    This function performs a log-sigmoid regression on the calibration data,
    fitting a model to the relationship between log-transformed Beta values
    and PfPR (Plasmodium falciparum parasite rate) for each combination of
    population and treatment access rate.

    Parameters
    ----------
    populations : list[int] | numpy.typing.NDArray
        List of unique population values for which to fit the model.
    treatment_rate : list[float] | numpy.typing.NDArray
        List of unique treatment access rates for which to fit the model.

    Returns
    -------
    dict[float, dict[int, typing.Any]]
        A nested dictionary where the outer keys are access rates, inner keys are
        populations, and values are the parameters (e.g., from `scipy.optimize.curve_fit`)
        of the fitted log-sigmoid model for that combination.
    """
    # X = beta"].values
    # y = group["pfpr2to10"].values

    # Convert betas and pfpr to numpy arrays for element-wise operations
    betas = numpy.array(betas)
    pfpr = numpy.array(pfpr)

    # Determine cutoff Beta based on pfpr2to10_mean
    if numpy.any(pfpr < pfpr_cutoff):
        cutoff_beta_val = numpy.max(betas[pfpr < pfpr_cutoff])  # Largest Beta where pfpr2to10_mean <= cutoff
        X_filtered = numpy.log(betas[betas < cutoff_beta_val])
        y_filtered = pfpr[betas < cutoff_beta_val]
    else:
        X_filtered = numpy.log(betas)
        y_filtered = pfpr / 100

    if len(X_filtered) < 3:  # Check if enough data points for regression
        print(f"Not enough data points for regression: {len(X_filtered)} points found.")
        return numpy.empty(0)
    try:
        # Perform sigmoid regression
        popt, _ = curve_fit(
            sigmoid,
            X_filtered,
            y_filtered,
            maxfev=5000,
        )
        return numpy.array(popt)  # Store parameters

    except RuntimeError:
        print("Curve fitting failed to converge. Not enough data points or poor initial guess.")
        return numpy.empty(0)  # Or handle error as needed
    except TypeError:  # Handle cases where curve_fit might receive empty arrays from p0 logic
        print("TypeError: Invalid input types for curve fitting. Ensure betas and pfpr are numeric arrays.")
        return numpy.empty(0)


def get_beta_models(
    populations: list[int],
    access_rates: list[float],
    means: pandas.DataFrame,
    pfpr_cutoff: float = 0.0,
) -> dict[float, dict[int, list[float]]]:
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
        access_rate: {population: [] for population in populations} for access_rate in access_rates
    }  # stores trained model for every parameter configuration

    # Perform regression for each (Population, TreatmentAccess) group
    for population in populations:
        for treatment_access in access_rates:
            # Filter the data for the current Population and TreatmentAccess
            group = means[(means["population"] == population) & (means["access_rate"] == treatment_access)]
            if group.empty:
                continue

            group = group.copy()  # Create a copy to avoid SettingWithCopyWarning
            pfpr = group["pfpr2to10"].to_numpy()
            beta = group["beta"].to_numpy()
            coefs = fit_log_sigmoid_model(beta, pfpr, pfpr_cutoff)
            if coefs.size == 0:
                print(
                    f"Skipping fitting for population {population} and access rate {treatment_access} due to insufficient data."
                )
                continue
            # Store the fitted model parameters
            # coefs_as_list = [float(coef) for coef in coefs]  # Convert to list of floats
            models_map[treatment_access][population] = coefs.tolist()  # type: ignore # Convert to list
    return models_map


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
