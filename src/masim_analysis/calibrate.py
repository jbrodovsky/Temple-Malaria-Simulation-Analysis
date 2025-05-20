# Country calibration script
import json
import os
from datetime import date

from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ruamel.yaml import YAML

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
    strategy_db: dict = configure.STRATEGY_DB,
) -> None:
    """
    Generate configuration files for the given country code and date range for calibration.
    Args:
        country_code (str): The country code for the simulation.
        calibration_year (int): The year for calibration.
        population_bins (list[int]): List of population bins.
        access_rates (list[float]): List of access rates.
        beta_values (list[float]): List of beta values.
        birth_rate (float): Birth rate for the simulation.
        death_rate (list[float]): List of death rates.
        age_distribution (list[float]): Age distribution for the simulation.

    Returns:
        None
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
                    True
                )
                write_pixel_data_files(execution_control["raster_db"], pop)
                output_path = os.path.join("conf", country_code, "calibration", f"cal_{pop}_{access}_{beta}.yml")
                yaml.dump(execution_control, open(output_path, "w"))


def write_pixel_data_files(raster_db: dict, population: int):
    """
    Write the pixel data files for the simulation.
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
    Generate command and job files for the given country code and parameters.
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
) -> pd.DataFrame:
    base_file_path = os.path.join(output_dir, country_code, "calibration")
    summary = pd.DataFrame(
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
                    data = pd.merge(monthlysitedata, months, left_on="monthlydataid", right_on="id")
                    data["date"] = pd.to_datetime(data["modeltime"], unit="s")

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
    Writes a unified command and job file for any missing data file.
    """
    base_file_path = os.path.join(output_dir, country_code, "calibration")
    for pop in tqdm(population_bins):
        for access in access_rates:
            for beta in beta_values:
                for i in range(repetitions):
                    filename = f"cal_{pop}_{access}_{beta}_monthly_data_{i + 1}"
                    file = os.path.join(base_file_path, f"{filename}.db")
                    try:
                        months = analysis.get_table(file, "monthlydata")
                        monthlysitedata = analysis.get_table(file, "monthlysitedata")
                    except FileNotFoundError as e:
                        with open(f"missing_calibration_runs_{pop}.txt", "a") as f:
                            # f.write(f"{e}\n")
                            f.write(
                                f"./bin/MaSim -i ./conf/{country_code}/calibration/cal_{pop}_{access}_{beta}.yml -o ./output/{country_code}/calibration/cal_{pop}_{access}_{beta}_ -r SQLiteDistrictReporter -j {i + 1}\n"
                            )
                        # commands.generate_job_file(
                        #     f"missing_calibration_runs_{pop}_{access}.txt",
                        #     f"{country_code}_{pop}_jobs",
                        #     cores_override=28,
                        # )
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


def linear_fit(
    populations: list[int] | NDArray,
    access_rates: list[float] | NDArray,
    beta_values: list[float] | NDArray,
    means: pd.DataFrame,
) -> plt.Figure:
    pass


def sigmoid_fit(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


# Function to find Beta values corresponding to an array of pfpr values
def find_beta(pfpr_target, linear_model, popt, pfpr_cutoff):
    pfpr_target = np.array(pfpr_target)  # Ensure input is a NumPy array
    beta_values = np.zeros_like(pfpr_target, dtype=np.float64)  # Placeholder for results

    # Linear region: pfpr_target < cutoff
    mask_linear = pfpr_target < pfpr_cutoff
    # print(pfpr_target, pfpr_cutoff)
    # print("any mask linear: ", np.any(mask_linear))
    if np.any(mask_linear):
        beta_log_linear = (pfpr_target[mask_linear] - linear_model.intercept_) / linear_model.coef_[0]
        beta_values[mask_linear] = 10 ** (beta_log_linear)  # Convert back from log-space

    # Sigmoid region: pfpr_target >= cutoff
    mask_sigmoid = pfpr_target >= pfpr_cutoff
    if np.any(mask_sigmoid):
        a, b, c = popt
        beta_log_sigmoid = c - (1 / b) * np.log(a / pfpr_target[mask_sigmoid] - 1)
        beta_values[mask_sigmoid] = 10 ** (beta_log_sigmoid)  # Convert back from log-space

    return beta_values


def plot_log_sigmoid(
    populations: list[int] | NDArray, access_rates: list[float] | NDArray, means: pd.DataFrame, model_map: dict
) -> plt.Figure:
    """
    Perform log-sigmoid regression on the given populations and access rates.
    Args:
        populations (list[int] | NDArray): List of populations.
        access_rates (list[float] | NDArray): List of access rates.
        means (pd.DataFrame): DataFrame containing the means with the following
            columns: ['population', 'access_rate', 'pfpr2to10_mean', 'beta'].
    Returns:
        tuple: A tuple containing the figure and a dictionary of models.
    """
    # Determine grid size
    num_rows = len(populations)
    num_cols = len(access_rates)
    # Create subplots grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharex=True, sharey=True)
    # Ensure axes is always a 2D list for consistency
    if num_rows == 1:
        axes = np.array([axes])  # Convert to 2D array
    if num_cols == 1:
        axes = np.array([[ax] for ax in axes])  # Convert to 2D array
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
                x=group["beta"].values, y=group["pfpr2to10"].values / 100, ax=ax, label="Data", color="black"
            )
            # Predictions
            popt = model_map[treatment_access][population]
            if popt is not None:
                pfpr_targets = np.linspace(0, 1, 100).reshape(-1, 1)
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
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def log_sigmoid_fit(
    populations: list[int] | NDArray,
    access_rates: list[float] | NDArray,
    means: pd.DataFrame,
) -> tuple[plt.Figure, dict]:
    """
    Perform log-sigmoid regression on the given populations and access rates.
    Args:
        populations (list[int] | NDArray): List of populations.
        access_rates (list[float] | NDArray): List of access rates.
        means (pd.DataFrame): DataFrame containing the means with the following
            columns: ['population', 'access_rate', 'pfpr2to10_mean', 'beta'].
    Returns:
        tuple: A tuple containing the figure and a dictionary of models.
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
            y = group["pfpr2to10"].values / 100
            X = np.log10(group["beta"].values)

            # X = group["beta"].values  # Log of Predictor (Beta)
            # y = group["pfpr2to10"].values  # Response variable

            # Determine cutoff Beta based on pfpr2to10_mean
            if any(y < pfpr_cutoff):
                cutoff_beta = np.max(X[y < pfpr_cutoff])  # Largest Beta where pfpr2to10_mean <= cutoff
            else:
                cutoff_beta = float("-inf")  # No cutoff

            # Sigmoid Regression on data after cutoff
            mask_sigmoid = X.ravel() >= cutoff_beta

            if np.sum(mask_sigmoid) > 1:
                X_sigmoid = X[mask_sigmoid].flatten()
                y_sigmoid = y[mask_sigmoid]
                try:
                    popt, _ = curve_fit(sigmoid_fit, X_sigmoid, y_sigmoid, maxfev=10000)
                except TypeError as e:
                    print(
                        f"Error in curve_fit: {e} for population {population} and treatment access {treatment_access}"
                    )
                    popt = None
                    continue
                except ValueError as e:
                    print(
                        f"Error in curve_fit: {e} for population {population} and treatment access {treatment_access}"
                    )
                    popt = None
                    continue
                except RuntimeError as e:
                    print(
                        f"Error in curve_fit {e} for population {population} and treatment access {treatment_access}: note enough data points"
                    )
                    popt = None
                    continue
            else:
                popt = None

            models_map[treatment_access][population] = popt  # (linear_model, popt)

    return models_map


def _log_sigmoid_fit(
    populations: list[int] | NDArray,
    access_rates: list[float] | NDArray,
    means: pd.DataFrame,
) -> tuple[plt.Figure, dict]:
    """
    Perform log-sigmoid regression on the given populations and access rates.
    Args:
        populations (list[int] | NDArray): List of populations.
        access_rates (list[float] | NDArray): List of access rates.
        means (pd.DataFrame): DataFrame containing the means with the following
            columns: ['population', 'access_rate', 'pfpr2to10_mean', 'beta'].
    Returns:
        tuple: A tuple containing the figure and a dictionary of models.
    """
    # Determine grid size
    num_rows = len(populations)
    num_cols = len(access_rates)

    # Create subplots grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharex=True, sharey=True)

    # Ensure axes is always a 2D list for consistency
    if num_rows == 1:
        axes = np.array([axes])  # Convert to 2D array
    if num_cols == 1:
        axes = np.array([[ax] for ax in axes])  # Convert to 2D array

    # Define cutoff based on pfpr2to10_mean
    pfpr_cutoff = 0.0  # Set the desired cutoff for pfpr2to10_mean
    models_map = {
        access_rate: {population: None for population in populations} for access_rate in access_rates
    }  # stores trained model for every parameter configuration

    # Perform regression for each (Population, TreatmentAccess) group
    for i, population in enumerate(populations):
        for j, treatment_access in enumerate(access_rates):
            ax = axes[i, j]  # Select subplot location

            # Filter the data for the current Population and TreatmentAccess
            # group = df_final[(df_final['Population'] == population) & (df_final['TreatmentAccess'] == treatment_access)]
            group = means[(means["population"] == population) & (means["access_rate"] == treatment_access)]
            group["pfpr2to10"] = group["pfpr2to10"].values / 100
            group["beta"] = np.log10(group["beta"].values)

            if group.empty:
                ax.set_visible(False)  # Hide empty plots
                continue

            X = group["beta"].values  # Log of Predictor (Beta)
            y = group["pfpr2to10"].values  # Response variable

            # Determine cutoff Beta based on pfpr2to10_mean
            if any(y < pfpr_cutoff):
                cutoff_beta = np.max(X[y < pfpr_cutoff])  # Largest Beta where pfpr2to10_mean <= cutoff
            else:
                cutoff_beta = float("-inf")  # No cutoff

            # Sigmoid Regression on data after cutoff
            mask_sigmoid = X.ravel() >= cutoff_beta

            if np.sum(mask_sigmoid) > 1:
                X_sigmoid = X[mask_sigmoid].flatten()
                y_sigmoid = y[mask_sigmoid]
                try:
                    popt, _ = curve_fit(sigmoid_fit, X_sigmoid, y_sigmoid, maxfev=10000)
                except TypeError as e:
                    print(
                        f"Error in curve_fit: {e} for population {population} and treatment access {treatment_access}"
                    )
                    popt = None
                    continue
                except ValueError as e:
                    print(
                        f"Error in curve_fit: {e} for population {population} and treatment access {treatment_access}"
                    )
                    popt = None
                    continue
            else:
                popt = None

            models_map[treatment_access][population] = popt  # (linear_model, popt)

            # Predictions
            pfpr_targets = np.linspace(0, 1, 100).reshape(-1, 1)
            # pfpr_targets = np.linspace(0.01, 1, 99).reshape(-1, 1) # To avoid divide by 0 error

            X_plot = find_beta(pfpr_targets, None, popt, pfpr_cutoff)

            # X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

            ax.set_xscale("log")

            # sns.scatterplot(x=np.exp(group['Beta']), y=group['pfpr2to10_mean'], ax=ax, label="Data", color='black')
            sns.scatterplot(x=10 ** (group["beta"]), y=group["pfpr2to10"], ax=ax, label="Data", color="black")
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
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig, models_map


def get_beta(models_map: dict[dict[list]], access_rate: float, population: int, pfpr: float) -> float:
    """
    Get the beta value for a given access rate, population, and pfpr target
    """
    if np.isnan(access_rate) or np.isnan(population):
        return np.nan
    # Find which population key to use by searching for the largest population less than or equal to the given population
    populations = np.asarray(list(models_map[access_rate].keys())).squeeze()
    if population <= 10:
        population = 10
    else:
        population_key = np.argwhere(populations <= population).squeeze().tolist()
        # print(f"Population key index: {population_key}")
        if type(population_key) is list:
            if len(population_key) > 0:
                population = populations[population_key[-1]]
        else:
            population = populations[population_key]
    # print(f"Population key: {population}")
    # Get the model
    try:
        a, b, c = models_map[access_rate][population]
    except TypeError as e:
        coefs = models_map[access_rate][population]
        a = coefs[0]
        b = coefs[1]
        c = coefs[2]
    except KeyError as e:
        print(f"KeyError: {e} for access rate {access_rate} and population {population}")
        return np.nan
    # Get the beta value
    # beta = find_beta(pfpr_target, None, model, 0.0)
    beta_log = c - (1 / b) * np.log(a / pfpr - 1)
    beta = 10**beta_log
    if np.isnan(beta):
        return 0
    return beta


def load_beta_model(filename: str) -> dict:
    """
    Load the beta model from a json file
    Args:
        filename (str): The name of the json file to load
    Returns:
        dict: A dictionary containing the beta model
    """
    models = json.load(open(filename, "r"))
    numeric = {float(k): {int(float(k2)): v2 for k2, v2 in v.items()} for k, v in models.items()}
    return numeric


def create_beta_map(
    models_map: dict[dict[list]], population_raster: NDArray, access_rate_raster: NDArray, prevalence_raster: NDArray
) -> NDArray:
    """
    Create a beta map based on the population, access rate and prevalence rasters
    Args:
        population_raster (NDArray): Population raster.
        access_rate_raster (NDArray): Access rate raster.
        prevalence_raster (NDArray): Prevalence raster.
        models_map (dict[dict[list]]): Models map.
    Returns:
        NDArray: Beta map.
    """
    # Create a beta map
    beta_map = np.zeros_like(population_raster)
    # Naive implementation of beta map
    rows, cols = beta_map.shape
    for r in range(rows):
        for c in range(cols):
            beta_map[r, c] = get_beta(
                models_map, access_rate_raster[r, c], population_raster[r, c], prevalence_raster[r, c]
            )
    return beta_map
