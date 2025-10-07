"""
Analysis tools for MaSim outputs.

This module provides functions for interacting with MaSim simulation output
databases, performing data analysis, and generating plots.
"""

from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from matplotlib.figure import Figure

from masim_analysis.configure import CountryParams

table_names = [
    "monthlydata",
    "sqlite_sequence",
    "monthlysitedata",
    "genotype",
    "monthlygenomedata",
]


# Database tools -----------------------------------------------------------------------------
def get_all_tables(db: Path | str) -> list:
    """
    Get all tables in a sqlite3 database.

    Parameters
    ----------
    db : str or Path
        Path to sqlite3 database.

    Returns
    -------
    list
        List of tables in the database.
    """
    # Validate input file path
    db_path = Path(db)
    if not db_path.exists():
        raise FileNotFoundError(f"File not found: {db}")
    with sqlite3.connect(str(db_path)) as conn:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()

    return [table[0] for table in tables]


def get_table(db: Path | str, table: str) -> pd.DataFrame:
    """
    Get a table from a sqlite3 database.

    Parameters
    ----------
    db : str or Path
        Path to sqlite3 database.
    table : str
        Name of table to get.

    Returns
    -------
    pd.DataFrame
        Table as a pandas DataFrame.
    """
    # Validate input file path
    db_path = Path(db)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db}")
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    return df


# Data analysis tools -----------------------------------------------------------------------
def calculate_treatment_failure_rate(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate treatment failures for a given table.

    Parameters
    ----------
    data : pd.DataFrame
        monthlysitedata table to calculate treatment failures.

    Returns
    -------
    pd.DataFrame
        Table with treatment failures.
    """
    # Calculate treatment failures
    data["failure_rate"] = data["treatmentfailures"] / data["treatments"]
    return data


def aggregate_failure_rates(path: Path | str, strategy: str, locationid: int = 0) -> pd.DataFrame:
    """
    Aggregate failure rate data by strategy. This function searches path for all the result
    files for the given strategy and aggregates them.

    Parameters
    ----------
    path : str or Path
        Path to search for result files.
    strategy : str
        Strategy to aggregate data for.
    locationid : int, optional
        locationid to filter by, defaults to 0 which returns all locations.

    Returns
    -------
    pd.DataFrame
        Aggregated data.
    """
    # Get all files for the strategy
    files = list(Path(path).glob(f"{strategy}_*.db"))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for strategy {strategy} in {path}")
    else:
        print(f"Found {len(files)} files for strategy {strategy}")
    summary = pd.DataFrame()
    for file in files:
        try:
            monthlysitedata = get_table(file, "monthlysitedata")
            if locationid != 0:
                monthlysitedata = monthlysitedata[monthlysitedata["locationid"] == locationid]
            monthlysitedata = calculate_treatment_failure_rate(monthlysitedata)
            # Use the filename (without extension and path) as the column name for this run's failure rate
            col_name = file.stem
            summary[col_name] = monthlysitedata.groupby("monthlydataid")["failure_rate"].sum()
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue  # Skip to the next file if an error occurs

    summary["mean"] = summary.mean(axis=1)
    summary["median"] = summary.median(axis=1)
    summary["95th"] = summary.quantile(axis=1, q=0.95)
    summary["5th"] = summary.quantile(axis=1, q=0.05)
    return summary


def save_aggregated_data(
    monthlysitedata: dict[str, pd.DataFrame],
    monthlydataid: pd.Series,
    strategy: str,
    path: Path | str,
):
    """
    Save aggregated data to a file.

    Parameters
    ----------
    monthlysitedata : dict[str, pd.DataFrame]
        Aggregated data.
    monthlydataid : pd.Series
        monthlydataid column.
    strategy : str
        Strategy name.
    path : str or Path
        Path to save the file.
    """
    db_path = Path(path) / f"{strategy}_aggregated.db"
    with sqlite3.connect(str(db_path)) as conn:
        for key, df in monthlysitedata.items():
            df_to_save = df.copy()
            if "monthlydataid" not in df_to_save.columns and monthlydataid is not None:
                df_to_save["monthlydataid"] = monthlydataid
            df_to_save.to_sql(key, conn, if_exists="replace", index=False)


def plot_strategy_treatment_failure(data: pd.DataFrame, strategy: str, figsize: tuple = (18, 3)):
    """
    Plot treatment failure rate for a given strategy.

    Parameters
    ----------
    data : pd.DataFrame
        Treatment failure data to plot from aggregate_failure_rates.
    strategy : str
        Strategy to plot.
    figsize : tuple, optional
        Figure size, by default (18, 3).

    Returns
    -------
    tuple
        A tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # ax.plot(months / 12, data['failure_rate']['mean'], label='Mean')
    ax.plot(data.index / 12, data["median"], label="Median")
    ax.fill_between(
        data["median"].index / 12,
        data["5th"],
        data["95th"],
        color="b",
        alpha=0.15,
        label="5th-95th percentile",
    )
    ax.axhline(y=0.1, color="r", linestyle="--", label="10% threshold")
    ax.set_title(f"{strategy} Treatment Failure Rate")
    ax.set_xlabel("Years")
    ax.set_ylabel("Treatment Failure Rate")
    ax.legend()
    return fig, ax


def get_population_data(file: Path | str, month: int = -1) -> pd.DataFrame:
    """
    Get population data from a MaSim output database.

    Parameters
    ----------
    file : str or Path
        Path to the MaSim output database (.db file).
    month : int, optional
        Month to get data for. Defaults to -1 (all months).

    Returns
    -------
    pd.DataFrame
        DataFrame containing population data.
    """
    # Assert the file exists
    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file}")
    population_data = get_table(file, "monthlysitedata")
    if month > 0:
        population_data = population_data.loc[population_data["monthlydataid"] == month]
    else:
        last_month = population_data.iloc[-1]["monthlydataid"]
        population_data = population_data.loc[population_data["monthlydataid"] == last_month]
    population_data = population_data[["locationid", "population", "infectedindividuals"]]
    population_data = population_data.set_index("locationid")
    return population_data


def get_genome_data(file: Path | str, month: int = -1) -> pd.DataFrame:
    """
    Get genome data from a MaSim output database.

    Parameters
    ----------
    file : str or Path
        Path to the MaSim output database (.db file).
    month : int, optional
        Month to get data for. Defaults to -1 (all months).

    Returns
    -------
    pd.DataFrame
        DataFrame containing genome data.
    """
    # Assert the file exists
    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file}")
    genome_data = get_table(file, "monthlygenomedata")
    if len(genome_data) == 0:
        raise ValueError(f"No genome data found in {file}")
    if month > 0:
        genome_data = genome_data.loc[genome_data["monthlydataid"] == month]
    else:
        last_month = genome_data.iloc[-1]["monthlydataid"]
        genome_data = genome_data.loc[genome_data["monthlydataid"] == last_month]
    genome_data = genome_data[["locationid", "genomeid", "occurrences", "weightedoccurrences"]]
    genome_data = genome_data.set_index("locationid")
    genome_data = genome_data.drop(columns="locationid")
    return genome_data


def calculate_genome_frequencies(file: Path | str, month: int = -1) -> pd.DataFrame:
    """
    Calculate genome frequencies from a MaSim output database.

    Parameters
    ----------
    file : str or Path
        Path to the MaSim output database (.db file).
    month : int, optional
        Month to get data for. Defaults to -1 (all months).

    Returns
    -------
    pd.DataFrame
        DataFrame containing genome frequencies.
    """
    # Assert the file exists
    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file}")
    # Get the genomes
    genomes = get_table(file, "genotype")
    # Get the genome data
    genome_data = get_genome_data(file, month)
    # Get the population data
    population_data = get_population_data(file, month)
    # Calculate the genome frequencies
    genome_frequencies = pd.DataFrame(index=population_data.index, columns=genomes["id"].to_list())
    totals = []
    for genome in genomes["id"].to_list():
        occurrences = genome_data.loc[genome_data["genomeid"] == genome, ["occurrences"]]
        occurrences = occurrences.join(population_data, how="outer").fillna(0)
        genome_frequencies[genome] = occurrences["occurrences"] / occurrences["population"]
        totals.append(occurrences["occurrences"].sum() / occurrences["population"].sum())
    genome_frequencies.loc["totals"] = totals
    return genome_frequencies


def get_resistant_genotypes(genomes: pd.DataFrame, allele: str) -> pd.DataFrame:
    """
    Filter resistant genotypes from a DataFrame of genome data.

    Parameters
    ----------
    genomes : pd.DataFrame
        DataFrame containing genome data (typically from get_genome_data).
    allele : str
        Allele to consider for resistance.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only resistant genotypes.
    """
    resistant_genotypes = genomes.loc[genomes["name"].str.contains(allele)]
    return resistant_genotypes


def calculate_resistant_genome_frequencies(
    file: Path | str, allele: str, month: int = 0, locationid: int = -1
) -> pd.DataFrame:
    """
    Calculate resistant genome frequencies from a MaSim output database.

    Parameters
    ----------
    file : str or Path
        Path to the MaSim output database (.db file).
    allele : str
        Allele to consider for resistance.
    month : int, optional
        Month to get data for. Defaults to 0 (first month).
    locationid : int, optional
        Location ID to filter by. Defaults to -1 (all locations).

    Returns
    -------
    pd.DataFrame
        DataFrame containing resistant genome frequencies.
    """
    # Assert the file exists
    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file}")
    genomes = get_table(file, "genotype")
    resistant_genotypes = genomes.loc[genomes["name"].str.contains(allele)]
    pop = get_table(file, "monthlysitedata")
    parasite = get_table(file, "monthlygenomedata")
    fqy = (
        pop[["monthlydataid", "locationid", "population", "infectedindividuals"]]
        .merge(
            parasite[
                [
                    "monthlydataid",
                    "locationid",
                    "genomeid",
                    "occurrences",
                    "weightedoccurrences",
                ]
            ],
            on=["monthlydataid", "locationid"],
        )
        .fillna(0)
    )
    fqy["frequency"] = fqy["weightedoccurrences"] / fqy["infectedindividuals"]
    # resistant_genome_data = genome_data[genome_data['genomeid'].isin(resistant_genotypes['id'])]
    fqy = fqy[fqy["genomeid"].isin(resistant_genotypes["id"])]
    if locationid > 0:
        fqy = fqy[fqy["locationid"] == locationid]
    if month > 0:
        fqy = fqy[fqy["monthlydataid"] == month]
    elif month == 0:
        fqy = fqy[fqy["monthlydataid"] == fqy["monthlydataid"].max()]
    return fqy


def aggregate_resistant_genome_frequencies(
    path: Path | str, strategy: str, allele: str = "H", month: int = -1, locationid: int = -1
) -> list:
    """
    Aggregate resistant genome frequencies across multiple simulation runs for a strategy.

    Parameters
    ----------
    path : str or Path
        Path to search for result files.
    strategy : str
        Strategy to aggregate data for.
    allele : str, optional
        Allele to consider for resistance, by default "H".
    month : int, optional
        Month to get data for. Defaults to -1 (all months).
    locationid : int, optional
        Location ID to filter by. Defaults to -1 (all locations).

    Returns
    -------
    list
        List of DataFrames, each containing resistant genome frequencies for a run.
    """
    # Get all files for the strategy
    files = list(Path(path).glob(f"{strategy}_*.db"))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for strategy {strategy} in {path}")
    else:
        print(f"Aggregating data for strategy {strategy} with {len(files)} files")
    # Get the monthlysitedata table for the first file to set up aggregated data
    genome_frequencies = []  # pd.DataFrame()
    # Aggregate data for the rest of the files
    for file in files:
        print(f"Aggregating data for {file}")
        # data = calculate_genome_frequencies(file)
        try:
            fqy = calculate_resistant_genome_frequencies(file, allele, month, locationid)
        except TypeError as e:
            print(e)
            continue
        except FileNotFoundError as e:
            print(e)
            continue
        except IndexError as e:
            print(e)
            print(f"Length of data: {len(genome_frequencies)}")
            continue
        except ValueError as e:
            print(e)
            continue
        if len(fqy) > 0:
            # genome_frequencies.append(fqy)
            if locationid > 0:
                genome_frequencies.append(fqy["frequency"].sum())
            else:
                genome_frequencies.append(fqy)
    return genome_frequencies


def aggregate_resistant_genome_frequencies_by_month(
    path: Path | str, strategy: str, allele: str = "H", locationid: int = -1
) -> pd.DataFrame:
    """
    Aggregate resistant genome frequencies by month across multiple simulation runs.

    Parameters
    ----------
    path : str or Path
        Path to search for result files.
    strategy : str
        Strategy to aggregate data for.
    allele : str, optional
        Allele to consider for resistance, by default "H".
    locationid : int, optional
        Location ID to filter by. Defaults to -1 (all locations).

    Returns
    -------
    pd.DataFrame
        DataFrame containing aggregated resistant genome frequencies by month.
    """
    files = list(Path(path).glob(f"{strategy}_*.db"))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for strategy {strategy} in {path}")
    else:
        print(f"Aggregating data for strategy {strategy} with {len(files)} files")

    months = get_table(files[0], "monthlydata")["id"]
    genome_frequencies = pd.DataFrame(index=months)
    for file in files:
        fqy = calculate_resistant_genome_frequencies(file, allele, -1, locationid)
        fqy = fqy.groupby("monthlydataid")["frequency"].sum()
        genome_frequencies[file] = fqy

    genome_frequencies["mean"] = genome_frequencies.mean(axis=1)
    genome_frequencies["median"] = genome_frequencies.median(axis=1)
    genome_frequencies["95th"] = genome_frequencies.quantile(axis=1, q=0.95)
    genome_frequencies["5th"] = genome_frequencies.quantile(axis=1, q=0.05)

    return genome_frequencies


def plot_strategy_results(path: Path | str, strategy: str, locationid: int = 0) -> Figure:
    """
    Plot aggregated results for a given strategy.

    Parameters
    ----------
    path : str or Path
        Path to search for result files.
    strategy : str
        Strategy to plot results for.
    locationid : int, optional
        Location ID to filter by, by default 0 (all locations).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    # Assert that the location is valid
    if locationid < 0:
        raise ValueError("locationid must be a positive integer")
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # Get all results files in the path
    results_files = list(path_obj.glob(f"{strategy}_.db"))
    if len(results_files) == 0:
        raise FileNotFoundError(f"No results files found for strategy {strategy}")
    # Read all results files
    failures = pd.DataFrame()
    for file in results_files:
        data = get_table(file, "monthlysitedata")
        data = calculate_treatment_failure_rate(data)
        failures[file] = data["failure_rate"]

    if locationid > 0:
        failures = failures.loc[failures["locationid"] == locationid]
    # else, aggregate all locations by taking the mean of the failure rates at each month
    else:
        months = failures["monthlydataid"].unique()
        monthly_average = pd.DataFrame()
        for month in months:
            monthly_data = failures.loc[failures["monthlydataid"] == month]
            monthly_average[month, "population"] = monthly_data["population"].sum()
            monthly_average[month, "treatments"] = monthly_data["treatments"].sum()
            monthly_average[month, "treatmentfailures"] = monthly_data["treatmentfailures"].sum()
            monthly_average[month, "failure_rate"] = monthly_data["failure_rate"].mean()

    failures["mean"] = failures.mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot(failures["monthlydataid"])
    return fig


def plot_combined_strategy_aggragated_results(path: Path | str, strategy: str, allele: str = "H", locationid: int = -1):
    """
    Plot combined aggregated results for a strategy, including treatment failure and resistant genome frequencies.

    Parameters
    ----------
    path : str or Path
        Path to search for result files.
    strategy : str
        Strategy to plot results for.
    allele : str, optional
        Allele to consider for resistance in genome frequency plots, by default "H".
    locationid : int, optional
        Location ID to filter by. Defaults to -1 (all locations).
    """
    # Assert that the location is valid
    if locationid < 0:
        raise ValueError("locationid must be a positive integer")
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # Get all results files in the path
    results_files = list(path_obj.glob(f"{strategy}_*.db"))
    if len(results_files) == 0:
        raise FileNotFoundError(f"No results files found for strategy {strategy}")
    agg = aggregate_resistant_genome_frequencies_by_month(path, strategy, allele, locationid)
    treatment_failures = aggregate_failure_rates(path, strategy, locationid)

    fig, ax = plt.subplots()
    ax.plot(agg.index / 12, agg["mean"], label="Mean", color="r")
    ax.plot(agg.index / 12, agg["median"], label="Median", color="r", alpha=0.5)
    ax.fill_between(
        agg.index / 12,
        agg["5th"],
        agg["95th"],
        color="r",
        alpha=0.15,
        label="5th-95th percentile",
    )

    ax.plot(
        treatment_failures.index / 12,
        treatment_failures["mean"],
        label="Mean",
        color="b",
    )
    ax.plot(
        treatment_failures.index / 12,
        treatment_failures["median"],
        label="Median",
        color="b",
        alpha=0.5,
    )
    ax.fill_between(
        treatment_failures.index / 12,
        treatment_failures["5th"],
        treatment_failures["95th"],
        color="b",
        alpha=0.15,
        label="5th-95th percentile",
    )

    ax.set_title(f"{strategy} Treatment Failure Rate and {allele} Resistant Genotype Frequency")
    ax.set_xlabel("Years")
    ax.set_ylabel("Frequency")
    ax.legend()

    return fig


def get_average_summary_statistics(
    path: Path | str, country: CountryParams
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get average summary statistics across all .db files in a given directory.

    Parameters
    ----------
    path : str or Path
        Path to the directory containing .db files.
    country : CountryParams
        Country parameters object.

    Returns
    -------
    tuple
        A tuple containing DataFrames for average population, total clinical episodes, etc.
    """
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_dir():
        raise NotADirectoryError(f"Path {path} is not a valid directory.")
    db_files = list(path_obj.glob("*.db"))
    if len(db_files) == 0:
        raise FileNotFoundError(f"No .db files found in directory {path}.")

    ave_population = pd.DataFrame(columns=["monthlydataid", "locationid", "population"])
    ave_cases = pd.DataFrame(columns=["monthlydataid", "locationid", "clinicalepisodes"])
    ave_prevalence_2_to_10 = pd.DataFrame(columns=["monthlydataid", "locationid", "pfpr2to10"])
    ave_cases_2_to_10 = pd.DataFrame(columns=["monthlydataid", "locationid", "cases2to10"])
    ave_prevalence_under_5 = pd.DataFrame(columns=["monthlydataid", "locationid", "pfprunder5"])
    ave_cases_under_5 = pd.DataFrame(columns=["monthlydataid", "locationid", "casesunder5"])

    for rep in range(20):
        data = get_table(
            Path("output")
            / country.country_code
            / "validation"
            / f"{country.country_code}_validation_monthly_data_{rep}.db",
            "monthlysitedata",
        )
        cases_2_to_10 = data[
            [
                "monthlydataid",
                "locationid",
                "clinicalepisodes_by_age_class_2_3",
                "clinicalepisodes_by_age_class_3_4",
                "clinicalepisodes_by_age_class_4_5",
                "clinicalepisodes_by_age_class_5_6",
                "clinicalepisodes_by_age_class_6_7",
                "clinicalepisodes_by_age_class_7_8",
                "clinicalepisodes_by_age_class_8_9",
                "clinicalepisodes_by_age_class_9_10",
            ]
        ].copy()
        cases_2_to_10["clinicalepisodes_2_to_10"] = cases_2_to_10[
            [
                "clinicalepisodes_by_age_class_2_3",
                "clinicalepisodes_by_age_class_3_4",
                "clinicalepisodes_by_age_class_4_5",
                "clinicalepisodes_by_age_class_5_6",
                "clinicalepisodes_by_age_class_6_7",
                "clinicalepisodes_by_age_class_7_8",
                "clinicalepisodes_by_age_class_8_9",
                "clinicalepisodes_by_age_class_9_10",
            ]
        ].sum(axis=1)
        cases_under_5 = data[
            [
                "monthlydataid",
                "locationid",
                "clinicalepisodes_by_age_class_0_1",
                "clinicalepisodes_by_age_class_1_2",
                "clinicalepisodes_by_age_class_2_3",
                "clinicalepisodes_by_age_class_3_4",
                "clinicalepisodes_by_age_class_4_5",
            ]
        ].copy()
        cases_under_5["clinicalepisodes_under5"] = cases_under_5[
            [
                "clinicalepisodes_by_age_class_0_1",
                "clinicalepisodes_by_age_class_1_2",
                "clinicalepisodes_by_age_class_2_3",
                "clinicalepisodes_by_age_class_3_4",
                "clinicalepisodes_by_age_class_4_5",
            ]
        ].sum(axis=1)
        # Add a column to the ave_* data frames from data
        try:
            ave_population = ave_population.merge(
                data[["monthlydataid", "locationid", "population"]].copy(),
                how="outer",
                on=["monthlydataid", "locationid"],
                suffixes=("", f"_{rep}"),
            )
            ave_cases = ave_cases.merge(
                data[["monthlydataid", "locationid", "clinicalepisodes"]].copy(),
                how="outer",
                on=["monthlydataid", "locationid"],
                suffixes=("", f"_{rep}"),
            )
            ave_prevalence_2_to_10 = ave_prevalence_2_to_10.merge(
                data[["monthlydataid", "locationid", "pfpr2to10"]].copy(),
                how="outer",
                on=["monthlydataid", "locationid"],
                suffixes=("", f"_{rep}"),
            )
            ave_cases_2_to_10 = ave_cases_2_to_10.merge(
                cases_2_to_10[["monthlydataid", "locationid", "clinicalepisodes_2_to_10"]].copy(),
                how="outer",
                on=["monthlydataid", "locationid"],
                suffixes=("", f"_{rep}"),
            )
            ave_prevalence_under_5 = ave_prevalence_under_5.merge(
                data[["monthlydataid", "locationid", "pfprunder5"]].copy(),
                how="outer",
                on=["monthlydataid", "locationid"],
                suffixes=("", f"_{rep}"),
            )
            ave_cases_under_5 = ave_cases_under_5.merge(
                cases_under_5[["monthlydataid", "locationid", "clinicalepisodes_under5"]].copy(),
                how="outer",
                on=["monthlydataid", "locationid"],
                suffixes=("", f"_{rep}"),
            )
        except Exception as e:
            print(f"Error processing replication {rep}: {e}")
    return (
        ave_population,
        ave_cases,
        ave_prevalence_2_to_10,
        ave_cases_2_to_10,
        ave_prevalence_under_5,
        ave_cases_under_5,
    )


def plot_prevalence_trend(
    observed: NDArray | list[float],
    simulated: NDArray | list[float],
    populations: NDArray | list[float] | None = None,
    age_str: str | None = None,
) -> Figure:
    """
    Plot prevalence trend from observed data.

    Parameters
    ----------
    observed : NDArray or list of float
        Observed prevalence data.

    simulated : NDArray or list of float
        Simulated prevalence data.

    populations : NDArray or list of float, optional
        Population data for weighting, by default None.

    age_str : str, optional
        Age group string for title, by default None.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    if populations is None:
        populations = np.ones(len(observed))
    ax.scatter(
        observed,
        simulated,
        s=populations / np.max(populations) * 100,
        marker="o",
        alpha=0.35,
        cmap="viridis",
        c=populations,
        label="Predicted PfPR",
    )
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_label("Population", rotation=270, labelpad=15)
    x = np.linspace(0, 1, 1000)
    ax.plot(x, x, color="red", linestyle="--")
    ax.set_xlim((0, 0.6))
    ax.set_ylim((0, 0.6))
    ax.set_xlabel("Observed PfPR")
    ax.set_ylabel("Predicted PfPR")
    if age_str is None:
        ax.set_title("Observed vs Predicted PfPR")
    else:
        ax.set_title(f"Observed vs Predicted PfPR ({age_str.replace('_', ' ')})")
    ax.set_xticks(np.arange(0, 0.6, 0.1))
    ax.legend()
    return fig
