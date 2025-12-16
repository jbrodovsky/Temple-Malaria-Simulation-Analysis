"""masim_analysis.analysis
=================================

Utilities for reading MaSim SQLite output files, aggregating simulation
results across runs, computing genome frequencies and treatment-failure
statistics, and producing commonly-used diagnostic plots.

This module centralizes the code used by the calibration workflow to extract
tables from MaSim ``.db`` files (see `get_table` / `get_all_tables`), compute
aggregates across many runs (e.g. `aggregate_failure_rates`), and produce
figures used to validate calibration fits.

Notes
-----
- Many functions expect MaSim's schema and table names (``monthlysitedata``,
    ``monthlygenomedata``, ``genotype``) to be present in the `.db` files.
- Outputs from MaSim often express PfPR values as percentages; callers that
    consume PfPR should convert to fractions when necessary.
"""

from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from matplotlib.figure import Figure


table_names = [
    "monthlydata",
    "sqlite_sequence",
    "monthlysitedata",
    "genotype",
    "monthlygenomedata",
]


# Database tools -----------------------------------------------------------------------------
def get_all_tables(db: Path | str) -> list:
    """Return a list of table names in a SQLite database.

    Parameters
    ----------
    db
        Path to a SQLite database file produced by MaSim.

    Returns
    -------
    list
        List of table name strings present in the database.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
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
    """Read a table from a MaSim SQLite database into a DataFrame.

    Parameters
    ----------
    db
        Path to the `.db` file.
    table
        Table name to read (e.g. ``monthlysitedata``, ``genotype``).

    Returns
    -------
    pandas.DataFrame
        Contents of the requested table.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    ValueError
        If the requested table is not present in the database.
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
    """Compute per-row treatment failure rate and append as ``failure_rate``.

    The function expects columns ``treatmentfailures`` and ``treatments`` to
    be present in ``data`` and returns the same DataFrame with a new column
    ``failure_rate`` equal to ``treatmentfailures / treatments``.

    Parameters
    ----------
    data
        DataFrame (typically a ``monthlysitedata`` table) containing
        ``treatmentfailures`` and ``treatments`` columns.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with an added ``failure_rate`` column.
    """
    # Calculate treatment failures
    data["failure_rate"] = data["treatmentfailures"] / data["treatments"]
    return data


def aggregate_failure_rates(path: Path | str, strategy: str, locationid: int = 0) -> pd.DataFrame:
    """Aggregate treatment-failure statistics across multiple runs.

    This function searches ``path`` for files matching ``{strategy}_*.db``,
    computes per-file failure rates (via ``calculate_treatment_failure_rate``),
    and returns a DataFrame containing aggregated statistics (mean, median,
    percentiles) across runs.

    Parameters
    ----------
    path
        Directory containing MaSim `.db` result files.
    strategy
        Strategy name prefix used in filenames (e.g. ``baseline``).
    locationid
        If > 0, filter results to a single location id; default 0 keeps all.

    Returns
    -------
    pandas.DataFrame
        Aggregated series indexed by time (monthlydataid) containing
        summary columns such as ``mean``, ``median``, ``5th``, ``95th``.
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
    """Persist aggregated monthlysitedata into a new SQLite file.

    Parameters
    ----------
    monthlysitedata
        Mapping of run id -> DataFrame (aggregated monthly site data).
    monthlydataid
        Series or index corresponding to monthly time steps.
    strategy
        Strategy name used to name the output file: ``{strategy}_aggregated.db``.
    path
        Directory where the aggregated DB will be written.
    """
    db_path = Path(path) / f"{strategy}_aggregated.db"
    with sqlite3.connect(str(db_path)) as conn:
        for key, df in monthlysitedata.items():
            df_to_save = df.copy()
            if "monthlydataid" not in df_to_save.columns and monthlydataid is not None:
                df_to_save["monthlydataid"] = monthlydataid
            df_to_save.to_sql(key, conn, if_exists="replace", index=False)


def plot_strategy_treatment_failure(data: pd.DataFrame, strategy: str, figsize: tuple = (18, 3)):
    """Plot aggregated treatment failure percentiles and median for a strategy.

    Parameters
    ----------
    data
        DataFrame returned from ``aggregate_failure_rates`` indexed by
        ``monthlydataid`` and containing ``median``, ``5th`` and ``95th``.
    strategy
        Strategy name used in plot title.
    figsize
        Matplotlib figure size tuple.

    Returns
    -------
    matplotlib.figure.Figure
        The created Figure object.
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
    """Extract population-related columns from a MaSim `.db` file.

    Parameters
    ----------
    file
        Path to a MaSim SQLite output `.db`.
    month
        If > 0, restrict to a single monthlydataid; default -1 returns all rows.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``locationid`` with columns ``population`` and
        ``infectedindividuals`` for the requested month(s).
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
    """Return per-location genome occurrence/weighted occurrence table.

    Parameters
    ----------
    file
        Path to MaSim `.db` file.
    month
        If > 0, restrict to a single monthlydataid; default -1 returns all rows.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``locationid`` with columns like ``genomeid``,
        ``occurrences`` and ``weightedoccurrences``.
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
    """Compute genome frequencies (per genotype) normalized by infected count.

    The function merges per-location genome occurrence data with the
    ``monthlysitedata`` to compute frequencies as
    ``weightedoccurrences / infectedindividuals``.

    Parameters
    ----------
    file
        Path to a MaSim `.db` file.
    month
        If > 0, restrict to that monthly snapshot; default -1 uses all.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by location with columns for each genome id and a
        final row `'totals'` containing the column sums.
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
    """Return subset of genome DataFrame whose name contains ``allele``.

    Parameters
    ----------
    genomes
        DataFrame with a textual ``name`` column describing genotype alleles.
    allele
        Substring to match (e.g. ``'H'``) used to select resistant genotypes.

    Returns
    -------
    pandas.DataFrame
        Subset of ``genomes`` containing only rows that match the allele.
    """
    resistant_genotypes = genomes.loc[genomes["name"].str.contains(allele)]
    return resistant_genotypes


def calculate_resistant_genome_frequencies(
    file: Path | str, allele: str, month: int = 0, locationid: int = -1
) -> pd.DataFrame:
    """Compute per-sample frequencies for genomes containing a resistant allele.

    The function computes the fraction of weighted occurrences for genomes
    matching ``allele`` divided by infected individuals at each location and
    time point, optionally filtering to a single month or location.

    Parameters
    ----------
    file
        MaSim `.db` file path.
    allele
        Allele substring used to select resistant genotypes (e.g. ``'H'``).
    month
        Month to restrict to: ``0`` uses the first month, ``-1`` means all.
    locationid
        If > 0, filter results to a single location id.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with columns including ``monthlydataid``,
        ``locationid``, ``frequency`` (fraction of resistant genomes).
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
    """Collect resistant-genome frequency DataFrames from multiple runs.

    For each `.db` file matching ``{strategy}_*.db`` in ``path`` the function
    computes resistant genome frequencies (via
    ``calculate_resistant_genome_frequencies``) and returns the list of per-run
    DataFrames for downstream aggregation/plotting.

    Parameters
    ----------
    path
        Directory containing run `.db` files.
    strategy
        Filename prefix for runs to include.
    allele
        Allele substring to match for resistance.
    month, locationid
        Optional filters forwarded to the per-file calculation function.

    Returns
    -------
    list[pandas.DataFrame]
        One DataFrame per run containing resistant genome frequency rows.
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
    """Aggregate resistant-genome frequencies across runs into monthly summaries.

    Returns a DataFrame indexed by monthlydataid containing summary columns
    such as ``mean``, ``median``, ``5th`` and ``95th`` percentiles.
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
    """Produce a simple matplotlib Figure visualizing strategy-level summaries.

    This helper reads aggregated files for ``strategy`` in ``path`` and
    returns a `Figure` object. It is intentionally lightweight and used by
    notebooks and quick diagnostics.
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
    """Create a combined plot of resistant genotype frequency and failures.

    Returns a `Figure` that overlays genome-frequency summaries and treatment
    failure percentiles to facilitate strategy comparisons.
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
    path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute average time-series DataFrames across all `.db` files in a folder.

    The function returns six DataFrames used by calibration routines:
    ``ave_population``, ``ave_cases``, ``ave_prevalence_2_to_10``,
    ``ave_cases_2_to_10``, ``ave_prevalence_under_5``, ``ave_cases_under_5``.

    Parameters
    ----------
    path
        Directory containing MaSim `.db` run outputs.

    Returns
    -------
    tuple
        Six DataFrames as described above.
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

    rep = 0
    for file in path_obj.glob("*.db"):
        data = get_table(file, "monthlysitedata")
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
        rep += 1
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
    population_plot_scalar: float = 100.0,
    upper_limit: float = 0.6,
) -> Figure:
    """Scatter observed vs simulated prevalence with population-weighted markers.

    Parameters
    ----------
    observed
        1-D array-like of observed PfPR values (fractions).
    simulated
        1-D array-like of simulated PfPR values computed by models.
    populations
        Optional population weights used to scale marker sizes in the plot.
    age_str
        Optional string used in the plot title (e.g. "2-10" or "under5").

    Returns
    -------
    matplotlib.figure.Figure
        Scatter plot Figure comparing observed and simulated prevalence.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    if populations is None:
        populations = np.ones(len(observed))
    ax.scatter(
        observed,
        simulated,
        s=populations / np.nanmax(populations) * population_plot_scalar,
        marker="o",
        alpha=0.35,
        cmap="viridis",
        c=populations,
        label="Predicted PfPR",
    )
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_label("Population", rotation=270, labelpad=15)
    x = np.linspace(0, upper_limit, 1000)
    ax.plot(x, x, color="red", linestyle="--")
    ax.set_xlim((0.0, upper_limit))
    ax.set_ylim((0.0, upper_limit))
    ax.set_xlabel("Observed PfPR")
    ax.set_ylabel("Predicted PfPR")
    if age_str is None:
        ax.set_title("Observed vs Predicted PfPR")
    else:
        ax.set_title(f"Observed vs Predicted PfPR ({age_str.replace('_', ' ')})")
    ax.set_xticks(np.arange(0.0, upper_limit, 0.1))
    ax.legend()
    return fig
