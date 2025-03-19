"""
Generate input configuration files for MaSim. This module provides functions to generate input configuration YAML files for MaSim. This should be used to generate the appropriate strategy input files and calibration files.
"""

from ruamel.yaml import YAML
import argparse
from datetime import date
import os

SEASONAL_MODEL = {"enable": False}
yaml = YAML()


def create_spatial_model(calibration_mode: bool = False) -> dict:
    if calibration_mode:
        kappa = 0
        alpha = 0
        beta = 0
        gamma = 0
    else:
        kappa = 0.01093251
        alpha = 0.22268982
        beta = 0.14319618
        gamma = 0.83741484
    return {
        "name": "Wesolowski",
        "Wesolowski": {
            "kappa": kappa,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        },
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="MaSim Configuration Tool",
        usage="configure -n <country_code> -s <start_date> -e <end_date> -c <comparison_date> [-sc <start_collecting_day>]",
        description="MaSim input configuration tool. This tool is used to generate input configuration (.yml/.yaml) files for MaSim. The tool primarily sets the required dates and data file pathes for the simulation.",
        add_help=True,
    )
    parser.add_argument(
        "name",
        type=str,
        # required=True,
        help="The country code name (ex: rwa for Rwanda).",
    )
    parser.add_argument(
        "start",
        type=str,
        # required=True,
        help="The starting date for the simulation: YYYY-MM-DD.",
    )
    parser.add_argument(
        "end",
        type=str,
        # required=True,
        help="The ending date for the simulation: YYYY-MM-DD.",
    )
    parser.add_argument(
        "comparison",
        type=str,
        # required=True,
        help="The start of the comparison period: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--sc",
        type=int,
        required=False,
        default=1826,
        metavar="start collecting day",
        help="The day number to start collecting data for the comparison period. Default is 1826.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=False,
        default="baseline",
        metavar="strategy name",
        help="The strategy name to use for the simulation. Default is baseline. Output file will be names <strategy>.yml.",
    )
    parser.add_argument(
        "--br",
        type=float,
        required=False,
        default=0.035,
        metavar="birth rate",
        help="The birth rate for the simulation. Default is 0.0412.",
    )
    parser.add_argument(
        "--dr",
        type=float,
        required=False,
        default=0.0005,
        metavar="death rate",
        nargs="+",
        help="The death rate(s) for the simulation. Default is 0.0005. A single scalar will be applied to all age classes. A list of scalars must correspond to the number of age classes.",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        required=False,
        default=False,
        help="Generate calibration files for the country. Disables movement and creates a set of calibration raster files for the country.",
    )

    args = parser.parse_args()
    # Validate the date strings
    try:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
        comparison_date = date.fromisoformat(args.comparison)
    except ValueError:
        print("Invalid date format in input strings. Use YYYY-MM-DD.")
        return

    args.start = start_date.strftime("%Y/%m/%d")
    args.end = end_date.strftime("%Y/%m/%d")
    args.comparison = comparison_date.strftime("%Y/%m/%d")

    return args


def load_yaml(file_path: str) -> dict:
    """
    Load a YAML file and return the contents as a dictionary.
    """
    yaml = YAML()
    with open(file_path, "r") as file:
        return yaml.load(file)


def validate_raster_files(
    name: str,
    calibration: bool = False,
    calibration_string: str = "",
    access_rate: float = 0.55,
    age_distribution: list[float] = [
        0.0378,
        0.0378,
        0.0378,
        0.0378,
        0.0282,
        0.0282,
        0.0282,
        0.0282,
        0.0282,
        0.029,
        0.029,
        0.029,
        0.029,
        0.029,
        0.169,
        0.134,
        0.106,
        0.066,
        0.053,
        0.035,
        0.0,
    ],
    beta: float = 0.01,
    population: int = 1000,
) -> dict:
    """
    Validate the raster files for the simulation. Optionally, generate calibration raster files.
    """
    conf_root = os.path.join("conf", name)
    data_root = os.path.join("data", name)
    if calibration:
        data_root = os.path.join(data_root, "calibration")
        conf_root = os.path.join(conf_root, "calibration")
    try:
        os.makedirs(data_root)
    except FileExistsError as e:
        pass
    try:
        os.makedirs(conf_root)
    except FileExistsError as e:
        pass

    if not calibration:
        raster_db = {
            "population_raster": os.path.join(data_root, f"{name}_population.asc"),
            "district_raster": os.path.join(data_root, f"{name}_district.asc"),
            "pr_treatment_under5": os.path.join(data_root, f"{name}_treatmentseeking.asc"),
            "pr_treatment_over5": os.path.join(data_root, f"{name}_treatmentseeking.asc"),
            "beta_raster": os.path.join(data_root, f"{name}_beta.asc"),
            "cell_size": 5,
            "age_distribution_by_location": [age_distribution],
            "p_treatment_for_less_than_5_by_location": [-1],
            "p_treatment_for_more_than_5_by_location": [-1],
            "beta_by_location": [-1],
        }
    else:
        raster_db = {
            "population_raster": os.path.join(data_root, f"{name}_{calibration_string}_population.asc"),
            "district_raster": os.path.join(data_root, f"{name}_{calibration_string}_district.asc"),
            "cell_size": 5,
            "age_distribution_by_location": [age_distribution],
            "p_treatment_for_less_than_5_by_location": [access_rate],
            "p_treatment_for_more_than_5_by_location": [access_rate],
            "beta_by_location": [beta],
        }
        if not os.path.exists(raster_db["population_raster"]):
            with open(raster_db["population_raster"], "w") as file:
                file.write(f"ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value -9999\n{population}")
            with open(raster_db["district_raster"], "w") as file:
                file.write("ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value -9999\n1")

    return raster_db


def main(
    name: str,
    start_date: str,
    end_date: str,
    comparison_date: str,
    start_collecting_day: int = 1825,
    birth_rate: float = 0.0412,
    death_rate: list[float] = [
        0.02641,
        0.00202,
        0.00202,
        0.00202,
        0.00198,
        0.00247,
        0.00247,
        0.00247,
        0.00247,
        0.00247,
        0.00247,
        0.00247,
        0.00455,
        0.00455,
        0.05348,
    ],
    calibration: bool = False,
    strategy: str = "baseline",
) -> dict:
    """
    Generate the input configuration file for MaSim."
    """
    execution_control = load_yaml(os.path.join("templates", "config.yml"))
    execution_control["starting_date"] = start_date
    execution_control["start_of_comparison_period"] = comparison_date
    execution_control["ending_date"] = end_date
    execution_control["start_collect_data_day"] = start_collecting_day
    execution_control["birth_rate"] = birth_rate
    execution_control["death_rate_by_age_class"] = death_rate
    execution_control["seasonal_info"] = SEASONAL_MODEL
    execution_control["spatial_model"] = create_spatial_model(calibration)
    execution_control["raster_db"] = validate_raster_files(name, calibration, strategy)
    execution_control["drug_db"] = load_yaml("templates/drug_db.yml")
    execution_control["therapy_db"] = load_yaml("templates/therapy_db.yml")
    execution_control["genotype_info"] = load_yaml("templates/genotype_info.yml")
    if calibration:
        execution_control["events"] = [
            {"name": "turn_off_mutation", "info": {"day": start_date}},
            # {"name": "begin_intervention", "info": {"day": start_date, "strategy_ids": [0]}},
        ]
        output_path = os.path.join("conf", name, "calibration", f"{strategy}.yml")
    else:
        execution_control["events"] = [{"name": "turn_off_mutation", "info": {"day": start_date}}]
        output_path = os.path.join("conf", name, f"{strategy}.yml")
        # Write the configuration files
        yaml.dump(execution_control, open(output_path, "w"))

    return execution_control


if __name__ == "__main__":
    args = parse_args()
    main(args.name, args.start, args.end, args.comparison, args.sc, args.br, args.dr, args.calibration, args.strategy)
