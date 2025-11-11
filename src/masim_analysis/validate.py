"""
Validation scripting and testing tools for MaSim analysis.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from pandas import DataFrame

from masim_analysis import analysis, calibrate, commands, configure, utils
from masim_analysis.configure import CountryParams


from ruamel.yaml import YAML

yaml = YAML()


def _averaging_pass(
    country: CountryParams,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    (
        ave_population,
        ave_cases,
        ave_prevalence_2_to_10,
        ave_cases_2_to_10,
        ave_prevalence_under_5,
        ave_cases_under_5,
    ) = analysis.get_average_summary_statistics(Path("output") / country.country_code / "validation")
    ave_population.to_csv(Path("output") / country.country_code / "validation" / "ave_population.csv")
    ave_cases.to_csv(Path("output") / country.country_code / "validation" / "ave_cases.csv")
    ave_prevalence_2_to_10.to_csv(Path("output") / country.country_code / "validation" / "ave_prevalence_2_to_10.csv")
    ave_cases_2_to_10.to_csv(Path("output") / country.country_code / "validation" / "ave_cases_2_to_10.csv")
    ave_prevalence_under_5.to_csv(Path("output") / country.country_code / "validation" / "ave_prevalence_under_5.csv")
    ave_cases_under_5.to_csv(Path("output") / country.country_code / "validation" / "ave_cases_under_5.csv")

    return (
        ave_population,
        ave_cases,
        ave_prevalence_2_to_10,
        ave_cases_2_to_10,
        ave_prevalence_under_5,
        ave_cases_under_5,
    )


def _prevelance_comparison(
    country: CountryParams,
    ave_cases: DataFrame,
    mean_prevalence_2_to_10: DataFrame,
    mean_prevalence_under_5: DataFrame,
    mean_population: DataFrame,
):
    ave_cases = ave_cases.drop(columns="clinicalepisodes")
    months = ave_cases["monthlydataid"].unique()
    ending_month = months[-13]
    ave_cases_year = (
        ave_cases[ave_cases["monthlydataid"].between(ending_month - 12, ending_month, inclusive="left")]
        .groupby("locationid")
        .sum()
        .drop(columns="monthlydataid")
    )
    ave_cases_year["mean"] = ave_cases_year.mean(axis=1)

    # population, _ = utils.read_raster(Path("data") / country.country_code / f"{country.country_code}_population.asc")
    prevalence_obs, _ = utils.read_raster(Path("data") / country.country_code / f"{country.country_code}_pfpr210.asc")
    prevalence_obs = prevalence_obs.reshape(-1)
    prevalence_comp = mean_prevalence_2_to_10[["mean"]].copy().div(100).rename(columns={"mean": "mean_2_to_10"})
    prevalence_comp["mean_under_5"] = (
        mean_prevalence_under_5[["mean"]].copy().div(100).rename(columns={"mean": "mean_under_5"})
    )
    prev_obs = DataFrame(
        {"obs": prevalence_obs[~np.isnan(prevalence_obs)]},
        index=np.arange(len(prevalence_obs[~np.isnan(prevalence_obs)])),
    )
    prevalence = prevalence_comp.merge(prev_obs, left_index=True, right_index=True, how="outer")
    prevalence.merge(mean_population["mean"].rename("population"), left_index=True, right_index=True, how="outer")

    return prevalence


def post_process(country: CountryParams, params: dict, logger: logging.Logger | None = None):
    if logger is None:
        logger = utils.get_country_logger(country.country_code, "validation")

    # Validation post-processing averaging pass
    (
        ave_population,
        ave_cases,
        ave_prevalence_2_to_10,
        ave_cases_2_to_10,
        ave_prevalence_under_5,
        ave_cases_under_5,
    ) = _averaging_pass(country)

    # Total case count verification
    mean_cases, mean_prevalence_2_to_10, mean_prevalence_under_5, mean_population = calibrate.get_last_year_statistics(
        ave_cases, ave_prevalence_2_to_10, ave_prevalence_under_5, ave_population
    )
    logger.info(
        f"{mean_cases['mean'].sum(): ,.0f} clinical episodes | SCALED: {mean_cases['mean'].sum() / params['artificial_rescaling_of_population_size']: ,.0f}"
    )
    logger.info(
        f"{mean_population['mean'].sum(): ,.0f} population | SCALED: {mean_population['mean'].sum() / params['artificial_rescaling_of_population_size']: ,.0f}"
    )
    # Prevalence comparison
    prevalence = _prevelance_comparison(
        country, ave_cases, mean_prevalence_2_to_10, mean_prevalence_under_5, mean_population
    )
    prevalence["population"] = mean_population["mean"]  # FIXED BUG HERE
    prevalence.to_csv(Path("output") / country.country_code / "validation" / "prevalence_comparison.csv")
    logger.info("Prevalence comparison data saved.")
    prevalence_fit = analysis.plot_prevalence_trend(
        prevalence["obs"].to_numpy(),
        prevalence["mean_2_to_10"].to_numpy(),
        prevalence["population"].to_numpy(),  # BUG HERE
        "2 to 10",
    )
    prevalence_fit.savefig(
        Path("images") / country.country_code / "prevalence_fit_2_to_10.png",
        dpi=300,
        bbox_inches="tight",
    )
    logger.info("Prevalence fit plot saved.")
    logger.info("Validation post-processing completed.")


def validate(country_code: str, repetitions: int = 50, output_dir: Path | str = Path("output")):
    """
    run the validation pipeline for a MaSim model for a given country.

    Parameters
    ----------
    country_code : str
        Country code for calibration (e.g., 'UGA').
    repetitions : int, optional
        Number of repetitions per parameter combination, by default 50.
    output_dir : Path | str, optional
        Directory to store output files, by default Path("output").

    Returns
    -------
    None
    """
    country = CountryParams.load(name=country_code)
    logger = utils.get_country_logger(country_code, "validation")
    # Create validation configuration
    strategy_db = yaml.load((Path("conf") / country_code.lower() / "test" / "strategy_db.yaml").read_text())
    events = yaml.load((Path("conf") / country_code.lower() / "test" / "events.yaml").read_text())
    params = configure.configure(
        country_code=country.country_code,
        birth_rate=country.birth_rate,
        initial_age_structure=country.initial_age_structure,
        age_distribution=country.age_distribution,
        death_rates=country.death_rate,
        starting_date=country.starting_date,
        start_of_comparison_period=country.start_of_comparison_period,
        ending_date=country.ending_date,
        strategy_db=strategy_db,
        calibration_str="",
        calibration=False,
    )
    params["artificial_rescaling_of_population_size"] = 0.25
    params["events"].extend(events)
    with open(Path("conf") / country_code.lower() / "test" / "validation_config.yaml", "w") as f:
        yaml.dump(params, f)

    _, cmds = commands.generate_commands(
        Path("conf") / country_code.lower() / "test" / "validation_config.yaml",
        Path(output_dir) / country_code.lower() / "validation",
        repetitions,
        True,
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("output", country.country_code, "validation")
    os.makedirs(output_dir, exist_ok=True)

    # Execute commands using multiprocessing
    max_workers = utils.get_optimal_worker_count()
    successful, failed = utils.multiprocess(cmds, max_workers, logger)
    logger.info(f"Validation runs completed: {successful} successful, {failed} failed.")
    # Post-processing
    post_process(country, params, logger)


def main():
    parser = argparse.ArgumentParser(description="Validate MaSim model for a given country.")
    parser.add_argument("country_code", type=str, help="Country code for calibration (e.g., 'UGA').")
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=20,
        help="Number of repetitions per parameter combination (default: 50).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="output",
        help="Directory to store output files (default: 'output').",
    )
    args = parser.parse_args()
    validate(args.country_code, args.repetitions, args.output_dir)


if __name__ == "__main__":
    main()
