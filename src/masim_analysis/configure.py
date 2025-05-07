"""
Generate input configuration files for MaSim. This module provides functions to generate input configuration YAML files for MaSim. This should be used to generate the appropriate strategy input files and calibration files.
"""

import argparse
import os
from datetime import date
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

from ruamel.yaml import YAML

SEASONAL_MODEL = {"enable": True}
NODATA_VALUE = -9999
yaml = YAML()

genotype_info = {
    "loci": [
        {
            "locus_name": "pfcrt",
            "position": 0,
            "alleles": [
                {
                    "value": 0,
                    "allele_name": "K76",
                    "short_name": "K",
                    "can_mutate_to": [1],
                    "mutation_level": 0,
                    "daily_cost_of_resistance": 0.0,
                },
                {
                    "value": 1,
                    "allele_name": "76T",
                    "short_name": "T",
                    "can_mutate_to": [0],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.0005,
                },
            ],
        },
        {
            "locus_name": "pfmdr1",
            "position": 1,
            "alleles": [
                {
                    "value": 0,
                    "allele_name": "N86 Y184 one copy of pfmdr1",
                    "short_name": "NY--",
                    "can_mutate_to": [1, 2, 4],
                    "mutation_level": 0,
                    "daily_cost_of_resistance": 0.0,
                },
                {
                    "value": 1,
                    "allele_name": "86Y Y184 one copy of pfmdr1",
                    "short_name": "YY--",
                    "can_mutate_to": [3, 0, 5],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.0005,
                },
                {
                    "value": 2,
                    "allele_name": "N86 184F one copy of pfmdr1",
                    "short_name": "NF--",
                    "can_mutate_to": [3, 0, 6],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.0005,
                },
                {
                    "value": 3,
                    "allele_name": "86Y 184F one copy of pfmdr1",
                    "short_name": "YF--",
                    "can_mutate_to": [1, 2, 7],
                    "mutation_level": 2,
                    "daily_cost_of_resistance": 0.00099975,
                },
                {
                    "value": 4,
                    "allele_name": "N86 Y184 2 copies of pfmdr1",
                    "short_name": "NYNY",
                    "can_mutate_to": [0],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.005,
                },
                {
                    "value": 5,
                    "allele_name": "86Y Y184 2 copies of pfmdr1",
                    "short_name": "YYYY",
                    "can_mutate_to": [1],
                    "mutation_level": 2,
                    "daily_cost_of_resistance": 0.0055,
                },
                {
                    "value": 6,
                    "allele_name": "N86 184F 2 copies of pfmdr1",
                    "short_name": "NFNF",
                    "can_mutate_to": [2],
                    "mutation_level": 2,
                    "daily_cost_of_resistance": 0.0055,
                },
                {
                    "value": 7,
                    "allele_name": "86Y 184F 2 copies of pfmdr1",
                    "short_name": "YFYF",
                    "can_mutate_to": [3],
                    "mutation_level": 3,
                    "daily_cost_of_resistance": 0.006,
                },
            ],
        },
        {
            "locus_name": "K13 Propeller",
            "position": 2,
            "alleles": [
                {
                    "value": 0,
                    "allele_name": "R561",
                    "short_name": "R",
                    "can_mutate_to": [1],
                    "mutation_level": 0,
                    "daily_cost_of_resistance": 0.0,
                },
                {
                    "value": 1,
                    "allele_name": "561H",
                    "short_name": "H",
                    "can_mutate_to": [0],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.0005,
                },
            ],
        },
        {
            "locus_name": "Plasmepsin 2-3",
            "position": 3,
            "alleles": [
                {
                    "value": 0,
                    "allele_name": "Plasmepsin 2-3 one copy",
                    "short_name": "1",
                    "can_mutate_to": [1],
                    "mutation_level": 0,
                    "daily_cost_of_resistance": 0.0,
                },
                {
                    "value": 1,
                    "allele_name": "Plasmepsin 2-3 2 copies",
                    "short_name": "2",
                    "can_mutate_to": [0],
                    "mutation_level": 1,
                    "daily_cost_of_resistance": 0.0005,
                },
            ],
        },
    ]
}


drug_db = {
    0: {
        "name": "ART",
        "half_life": 0.0,
        "maximum_parasite_killing_rate": 0.999,
        "n": 25,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [2],
        "selecting_alleles": [[1]],
        "k": 4,
        "EC50": {"..0..": 0.75, "..1..": 1.2},
    },
    1: {
        "name": "ADQ",
        "half_life": 9.0,
        "maximum_parasite_killing_rate": 0.95,
        "n": 19,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [0, 1],
        "selecting_alleles": [[1], [0, 1, 3, 4, 5, 7]],
        "k": 4,
        "EC50": {
            "00...": 0.62,
            "01...": 0.85,
            "02...": 0.5,
            "03...": 0.775,
            "04...": 0.62,
            "05...": 0.85,
            "06...": 0.5,
            "07...": 0.775,
            "10...": 0.7,
            "11...": 0.9,
            "12...": 0.65,
            "13...": 0.82,
            "14...": 0.7,
            "15...": 0.9,
            "16...": 0.65,
            "17...": 0.82,
        },
    },
    2: {
        "name": "SP",
        "half_life": 6.5,
        "maximum_parasite_killing_rate": 0.9,
        "n": 15,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.0,
        "affecting_loci": [],
        "selecting_alleles": [],
        "k": 4,
        "EC50": {".....": 1.08},
    },
    3: {
        "name": "CQ",
        "half_life": 10,
        "maximum_parasite_killing_rate": 0.95,
        "n": 19,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [0, 1],
        "selecting_alleles": [[1], [1, 3, 5, 7]],
        "k": 4,
        "EC50": {
            "00...": 0.72,
            "01...": 0.9,
            "02...": 0.72,
            "03...": 0.9,
            "04...": 0.72,
            "05...": 0.9,
            "06...": 0.72,
            "07...": 0.9,
            "10...": 1.19,
            "11...": 1.35,
            "12...": 1.19,
            "13...": 1.35,
            "14...": 1.19,
            "15...": 1.35,
            "16...": 1.19,
            "17...": 1.35,
        },
    },
    4: {
        "name": "LUM",
        "half_life": 4.5,
        "maximum_parasite_killing_rate": 0.99,
        "n": 20,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [0, 1],
        "selecting_alleles": [[0], [0, 2, 3, 4, 6, 7]],
        "k": 4,
        "EC50": {
            "00...": 0.8,
            "01...": 0.67,
            "02...": 0.9,
            "03...": 0.8,
            "04...": 1.0,
            "05...": 0.87,
            "06...": 1.1,
            "07...": 1.0,
            "10...": 0.75,
            "11...": 0.6,
            "12...": 0.85,
            "13...": 0.75,
            "14...": 0.95,
            "15...": 0.8,
            "16...": 1.05,
            "17...": 0.95,
        },
    },
    5: {
        "name": "PQ",
        "half_life": 28.0,
        "maximum_parasite_killing_rate": 0.9,
        "n": 15,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [3],
        "selecting_alleles": [[1]],
        "resistant_factor": [[1]],
        "k": 4,
        "EC50": {"...0.": 0.58, "...1.": 1.4},
    },
    6: {
        "name": "MF",
        "half_life": 21.0,
        "maximum_parasite_killing_rate": 0.9,
        "n": 15,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.005,
        "affecting_loci": [1],
        "selecting_alleles": [[4, 5, 6, 7]],
        "k": 4,
        "EC50": {
            ".0...": 0.45,
            ".1...": 0.45,
            ".2...": 0.45,
            ".3...": 0.45,
            ".4...": 1.1,
            ".5...": 1.1,
            ".6...": 1.1,
            ".7...": 1.1,
        },
    },
    7: {
        "name": "QUIN",
        "half_life": 18,
        "maximum_parasite_killing_rate": 0.9,
        "n": 3,
        "age_specific_drug_concentration_sd": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
        "mutation_probability": 0.0,
        "affecting_loci": [],
        "selecting_alleles": [],
        "k": 4,
        "EC50": {"0....": 1.41, "1....": 1.41},
    },
}


therapy_db = {
    # ACT - artesunate–amodiaquine (ASAQ)
    0: {"drug_id": [0, 1], "dosing_days": [3]},
    # ACT - artemether–lumefantrine (AL)
    1: {"drug_id": [0, 4], "dosing_days": [3]},
    # ACT - artesunate-mefloquine (ASMQ)
    2: {"drug_id": [0, 6], "dosing_days": [3]},
    # ACT - Dihydroartemisinin-piperaquine (DP)
    3: {"drug_id": [0, 5], "dosing_days": [3]},
    # MONO - Amodiaquine (ADQ)
    4: {"drug_id": [1], "dosing_days": [3]},
    # MONO - Artesunate (AS)
    5: {"drug_id": [0], "dosing_days": [3]},
    # MONO - Chloroquine (CQ)
    6: {"drug_id": [3], "dosing_days": [3]},
    # MONO - Quinine (QUIN)
    7: {"drug_id": [7], "dosing_days": [7]},
    # COMBINATION - Sulfadoxine/pyrimethamine (SP)
    8: {"drug_id": [2], "dosing_days": [3]},
}
relative_infectivity = {
    "sigma": 3.91,
    "ro": 0.00031,
    # on average 1 mosquito take 3 microliters of blood per bloodeal
    "blood_meal_volume": 3,
}

strategy_db = {
    0: {
        "name": "baseline",
        "type": "MFT",
        "therapy_ids": [0],
        "distribution": [1],
    }
}


@dataclass
class ConfigureParams:
    country_code: str
    days_between_notifications: int = 30
    initial_seed_number: int = 0
    connection_string: str = "host=masimdb.vmhost.psu.edu dbname=rwanda user=sim password=sim connect_timeout=60"
    record_genome_db: bool = True
    report_frequency: int = 30
    starting_date: date = date(2005, 1, 1)
    start_of_comparison_period: date = date(2020, 1, 1)
    ending_date: date = date(2024, 1, 1)
    start_collect_data_day: int = 1826
    number_of_tracking_days: int = 11
    transmission_parameter: float = 0.55  # beta value
    age_structure: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 60, 100])
    initial_age_structure: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25, 35, 45, 55, 65, 100]
    )
    artificial_rescaling_of_population_size: float = 0.25
    age_distribution_by_location: List[List[float]] = field(
        default_factory=lambda: [
            [
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
            ]
        ]
    )
    p_treatment_for_less_than_5_by_location: List[float] = field(default_factory=lambda: [-1.0])
    p_treatment_for_more_than_5_by_location: List[float] = field(default_factory=lambda: [-1.0])
    seasonality_toggle: bool = True
    spatial_model_toggle: bool = True
    birth_rate: float = 0.0412
    death_rate_by_age_class: List[float] = field(
        default_factory=lambda: [
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
        ]
    )
    mortality_when_treatment_fail_by_age_class: List[float] = field(
        default_factory=lambda: [
            0.040,
            0.020,
            0.020,
            0.020,
            0.020,
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
            0.001,
            0.001,
            0.001,
            0.001,
        ]
    )
    initial_strategy_id: int = 0
    days_to_clinical_under_five: int = 4
    days_to_clinical_over_five: int = 6
    days_mature_gametocyte_under_five: int = 4
    days_mature_gametocyte_over_five: int = 6
    gametocyte_level_under_artemisinin_action: float = 1.0
    gametocyte_level_full: float = 1.0
    p_relapse: float = 0.01
    relapse_duration: int = 30
    relapseRate: float = 4.4721
    update_frequency: int = 7
    allow_new_coinfection_to_cause_symtoms: bool = True
    using_free_recombination: bool = True
    tf_window_size: int = 60
    tf_testing_day: int = 28
    fraction_mosquitoes_interrupted_feeding: float = 0.0
    inflation_factor: float = 0.01
    using_age_dependent_bitting_level: bool = False
    using_variable_probability_infectious_bites_cause_infection: bool = False
    mda_therapy_id: int = 8
    age_bracket_prob_individual_present_at_mda: List[int] = field(default_factory=lambda: [10, 40])
    mean_prob_individual_present_at_mda: List[float] = field(default_factory=lambda: [0.85, 0.75, 0.85])
    sd_prob_individual_present_at_mda: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3])

    def __post_init__(self):
        self.number_of_age_classes = len(self.age_structure)
        if len(self.death_rate_by_age_class) != len(self.age_structure):
            raise ValueError("The length of death_rate_by_age_class must match the length of age_structure.")
        if len(self.mortality_when_treatment_fail_by_age_class) != len(self.age_structure):
            raise ValueError(
                "The length of mortality_when_treatment_fail_by_age_class must match the length of age_structure."
            )
        # Validate that all non-defaulted values are greater than or equal to zero
        if self.days_between_notifications < 0:
            raise ValueError("days_between_notifications must be greater than or equal to zero.")
        if self.initial_seed_number < 0:
            raise ValueError("initial_seed_number must be greater than or equal to zero.")
        if self.record_genome_db not in [True, False]:
            raise ValueError("record_genome_db must be a boolean value.")
        if self.report_frequency < 0:
            raise ValueError("report_frequency must be greater than or equal to zero.")
        if self.start_collect_data_day < 0:
            raise ValueError("start_collect_data_day must be greater than or equal to zero.")
        if self.number_of_tracking_days < 0:
            raise ValueError("number_of_tracking_days must be greater than or equal to zero.")
        if self.transmission_parameter < 0:
            raise ValueError("transmission_parameter must be greater than or equal to zero.")
        if self.artificial_rescaling_of_population_size < 0:
            raise ValueError("artificial_rescaling_of_population_size must be greater than or equal to zero.")
        if self.birth_rate < 0:
            raise ValueError("birth_rate must be greater than or equal to zero.")
        if any(rate < 0 for rate in self.death_rate_by_age_class):
            raise ValueError("All values in death_rate_by_age_class must be greater than or equal to zero.")
        if any(rate < 0 for rate in self.mortality_when_treatment_fail_by_age_class):
            raise ValueError(
                "All values in mortality_when_treatment_fail_by_age_class must be greater than or equal to zero."
            )
        if self.days_to_clinical_under_five < 0:
            raise ValueError("days_to_clinical_under_five must be greater than or equal to zero.")
        if self.days_to_clinical_over_five < 0:
            raise ValueError("days_to_clinical_over_five must be greater than or equal to zero.")
        if self.days_mature_gametocyte_under_five < 0:
            raise ValueError("days_mature_gametocyte_under_five must be greater than or equal to zero.")
        if self.days_mature_gametocyte_over_five < 0:
            raise ValueError("days_mature_gametocyte_over_five must be greater than or equal to zero.")
        if self.gametocyte_level_under_artemisinin_action < 0:
            raise ValueError("gametocyte_level_under_artemisinin_action must be greater than or equal to zero.")
        if self.gametocyte_level_full < 0:
            raise ValueError("gametocyte_level_full must be greater than or equal to zero.")
        if self.p_relapse < 0:
            raise ValueError("p_relapse must be greater than or equal to zero.")
        if self.relapse_duration < 0:
            raise ValueError("relapse_duration must be greater than or equal to zero.")
        if self.relapseRate < 0:
            raise ValueError("relapseRate must be greater than or equal to zero.")
        if self.update_frequency < 0:
            raise ValueError("update_frequency must be greater than or equal to zero.")
        if self.tf_window_size < 0:
            raise ValueError("tf_window_size must be greater than or equal to zero.")
        if self.tf_testing_day < 0:
            raise ValueError("tf_testing_day must be greater than or equal to zero.")
        if self.fraction_mosquitoes_interrupted_feeding < 0:
            raise ValueError("fraction_mosquitoes_interrupted_feeding must be greater than or equal to zero.")
        if self.inflation_factor < 0:
            raise ValueError("inflation_factor must be greater than or equal to zero.")
        if self.mda_therapy_id < 0:
            raise ValueError("mda_therapy_id must be greater than or equal to zero.")
        if any(prob < 0 for prob in self.mean_prob_individual_present_at_mda):
            raise ValueError("All values in mean_prob_individual_present_at_mda must be greater than or equal to zero.")
        if any(sd < 0 for sd in self.sd_prob_individual_present_at_mda):
            raise ValueError("All values in sd_prob_individual_present_at_mda must be greater than or equal to zero.")

        self.number_of_age_classes = len(self.age_structure)


@dataclass
class ParasiteDensityLevel:
    log_parasite_density_cured: float = -4.699
    log_parasite_density_from_liver: float = -2.000
    log_parasite_density_asymptomatic: float = 3
    log_parasite_density_clinical: float = 4.301
    log_parasite_density_clinical_from: float = 3.301
    log_parasite_density_clinical_to: float = 5.301
    log_parasite_density_detectable: float = 1.000
    log_parasite_density_detectable_pfpr: float = 1.699
    log_parasite_density_pyrogenic: float = 3.398


@dataclass
class ImmuneSystemInformation:
    b1: float = 0.00125
    b2: float = 0.0025
    duration_for_naive: int = 300
    duration_for_fully_immune: int = 60
    mean_initial_condition: float = 0.1
    sd_initial_condition: float = 0.1
    immune_inflation_rate: float = 0.01
    max_clinical_probability: float = 0.99
    immune_effect_on_progression_to_clinical: float = 4
    age_mature_immunity: int = 10
    factor_effect_age_mature_immunity: int = 1
    midpoint: float = 0.4


@dataclass
class CirculationInfo:
    max_relative_moving_value: int = 35
    number_of_moving_levels: int = 100
    moving_level_distribution: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "Gamma",
            "Exponential": {"scale": 0.17},
            "Gamma": {"mean": 5, "sd": 10},
        }
    )
    circulation_percent: float = 0.00336
    length_of_stay: Dict[str, Any] = field(default_factory=lambda: {"mean": 5, "sd": 10})


@dataclass
class ParasiteInfo:
    parasite_type_id: int
    prevalence: float


@dataclass
class InitialParasiteInfoEntry:
    location_id: int
    parasite_info: list[ParasiteInfo]

    def to_dict(self):
        return {"location_id": self.location_id, "parasite_info": [asdict(p) for p in self.parasite_info]}


@dataclass
class RelativeBittingInfo:
    max_relative_biting_value: int = 35
    number_of_biting_levels: int = 100
    biting_level_distribution: Dict[str, Any] = field(
        default_factory=lambda: {
            "distribution": "Gamma",
            "Exponential": {"scale": 0.17},
            "Gamma": {"mean": 5, "sd": 10},
        }
    )


parasite_density_level = ParasiteDensityLevel()
immune_system_information = ImmuneSystemInformation()
circulation_info = CirculationInfo()
initial_parasite_info = [
    InitialParasiteInfoEntry(
        location_id=-1,
        parasite_info=[
            ParasiteInfo(parasite_type_id=32, prevalence=0.05),
            ParasiteInfo(parasite_type_id=36, prevalence=0.05),
        ],
    )
]
relative_bitting_info = RelativeBittingInfo()


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


def create_seasonal_model(enable: bool, country_code: str, period: int = 365) -> dict:
    """
    Create the seasonal model for the simulation.
    """
    return {
        "enable": enable,
        "mode": "rainfall",
        "rainfall": {
            "filename": os.path.join("data", country_code, f"{country_code}_seasonality.csv"),
            "period": period,
        },
    }


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
    access_rate: float = -1.0,
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
    beta: float = -1.0,
    population: int = -1,
) -> dict:
    """
    Validate the raster files for the simulation. Optionally, generate calibration raster files.
    """
    conf_root = os.path.join("conf", name)
    data_root = os.path.join("data", name)
    if calibration:
        data_root = os.path.join(data_root, "calibration")
        conf_root = os.path.join(conf_root, "calibration")
        calibration_string = f"_{calibration_string}"
    try:
        os.makedirs(data_root)
    except FileExistsError as e:
        pass
    try:
        os.makedirs(conf_root)
    except FileExistsError as e:
        pass

    raster_db = {
        "population_raster": os.path.join(data_root, f"{name}{calibration_string}_population.asc"),
        "district_raster": os.path.join(data_root, f"{name}{calibration_string}_districts.asc"),
        "cell_size": 5,
        "pf_treatment_under5": os.path.join(data_root, f"{name}_treatmentseeking.asc"),
        "pf_treatment_over5": os.path.join(data_root, f"{name}_treatmentseeking.asc"),
        "age_distribution_by_location": [age_distribution],
        "p_treatment_for_less_than_5_by_location": [access_rate],
        "p_treatment_for_more_than_5_by_location": [access_rate],
    }
    if calibration:
        raster_db["beta_by_location"] = [beta]

        if not os.path.exists(raster_db["population_raster"]):
            with open(raster_db["population_raster"], "w") as file:
                file.write(
                    f"ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value {NODATA_VALUE}\n{population}"
                )
            with open(raster_db["district_raster"], "w") as file:
                file.write("ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value {NODATA_VALUE}\n1")
    else:
        raster_db["beta_by_location"] = [0.55]
        raster_db["beta_raster"] = os.path.join(data_root, f"{name}_beta.asc")

    return raster_db


def configure(params: ConfigureParams) -> dict:
    """
    Create the configuration dictionary for the simulation. This function allows the user to set all the parameters in the .yml files used by MaSim.
    """
    # execution_control = {
    #    "days_between_notifications": params.days_between_notifications,
    #    "initial_seed_number": params.initial_seed_number,
    #    "connection_string": params.connection_string,
    #    "record_genome_db": params.record_genome_db,
    #    "report_frequency": params.report_frequency,
    #    "starting_date": params.starting_date.strftime("%Y/%m/%d"),
    #    "start_of_comparison_period": params.start_of_comparison_period.strftime("%Y/%m/%d"),
    #    "ending_date": params.ending_date.strftime("%Y/%m/%d"),
    #    "start_collect_data_day": params.start_collect_data_day,
    #    "number_of_tracking_days": params.number_of_tracking_days,
    #    "transmission_parameter": params.transmission_parameter,
    #    "number_of_age_classes": len(params.age_structure),
    #    "age_structure": params.age_structure,
    #    "initial_age_structure": params.initial_age_structure,
    #    "artificial_rescaling_of_population_size": params.artificial_rescaling_of_population_size,
    #    "age_distribution_by_location": params.age_distribution_by_location,
    #    "p_treatment_for_less_than_5_by_location": params.p_treatment_for_less_than_5_by_location,
    #    "p_treatment_for_more_than_5_by_location": params.p_treatment_for_more_than_5_by_location,
    #    "seasonality_toggle": params.seasonality_toggle,
    #    "spatial_model_toggle": params.spatial_model_toggle,
    #    "birth_rate": params.birth_rate,
    #    "death_rate_by_age_class": params.death_rate_by_age_class,
    #    "mortality_when_treatment_fail_by_age_class": params.mortality_when_treatment_fail_by_age_class,
    #    "initial_strategy_id": params.initial_strategy_id,
    #    "days_to_clinical_under_five": params.days_to_clinical_under_five,
    #    "days_to_clinical_over_five": params.days_to_clinical_over_five,
    #    "days_mature_gametocyte_under_five": params.days_mature_gametocyte_under_five,
    #    "days_mature_gametocyte_over_five": params.days_mature_gametocyte_over_five,
    #    "gametocyte_level_under_artemisinin_action": params.gametocyte_level_under_artemisinin_action,
    #    "gametocyte_level_full": params.gametocyte_level_full,
    #    "p_relapse": params.p_relapse,
    #    "relapse_duration": params.relapse_duration,
    #    "relapseRate": params.relapseRate,
    #    "update_frequency": params.update_frequency,
    #    "allow_new_coinfection_to_cause_symtoms": params.allow_new_coinfection_to_cause_symtoms,
    #    "using_free_recombination": params.using_free_recombination,
    #    "tf_window_size": params.tf_window_size,
    #    "fraction_mosquitoes_interrupted_feeding": params.fraction_mosquitoes_interrupted_feeding,
    #    "inflation_factor": params.inflation_factor,
    #    "using_age_dependent_bitting_level": params.using_age_dependent_bitting_level,
    #    "using_variable_probability_infectious_bites_cause_infection": params.using_variable_probability_infectious_bites_cause_infection,
    #    "mda_therapy_id": params.mda_therapy_id,
    #    "age_bracket_prob_individual_present_at_mda": params.age_bracket_prob_individual_present_at_mda,
    #    "mean_prob_individual_present_at_mda": params.mean_prob_individual_present_at_mda,
    #    "sd_prob_individual_present_at_mda": params.sd_prob_individual_present_at_mda,
    # }
    execution_control = asdict(params)
    execution_control["starting_date"] = params.starting_date.strftime("%Y/%m/%d")
    execution_control["start_of_comparison_period"] = params.start_of_comparison_period.strftime("%Y/%m/%d")
    execution_control["ending_date"] = params.ending_date.strftime("%Y/%m/%d")
    execution_control["raster_db"] = validate_raster_files(
        params.country_code, calibration=False, age_distribution=params.initial_age_structure
    )
    execution_control["drug_db"] = drug_db
    execution_control["therapy_db"] = therapy_db
    execution_control["spatial_model"] = create_spatial_model(params.spatial_model_toggle)
    execution_control["seasonal_info"] = create_seasonal_model(params.seasonality_toggle, params.country_code)
    execution_control["events"] = [
        {"name": "turn_off_mutation", "info": [{"day": params.starting_date.strftime("%Y/%m/%d")}]}
    ]
    execution_control["strategy_db"] = strategy_db
    execution_control["parasite_density_level"] = asdict(parasite_density_level)
    execution_control["immune_system_information"] = asdict(immune_system_information)
    execution_control["circulation_info"] = asdict(circulation_info)
    execution_control["initial_parasite_info"] = [entry.to_dict() for entry in initial_parasite_info]
    execution_control["relative_bitting_info"] = asdict(relative_bitting_info)

    execution_control["relative_infectivity"] = relative_infectivity
    execution_control["genotype_info"] = genotype_info

    return execution_control


def main(
    name: str,
    start_date: str,
    end_date: str,
    comparison_date: str,
    start_collecting_day: int = 1825,
    birth_rate: float = 0.0412,
    initial_age_structure: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25, 35, 45, 55, 65, 100],
    death_rate=[
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
    age_distribution=[
        0.037,
        0.132,
        0.161,
        0.142,
        0.090,
        0.086,
        0.070,
        0.052,
        0.044,
        0.044,
        0.031,
        0.041,
        0.024,
        0.017,
        0.013,
        0.017,
    ],
    calibration: bool = False,
    strategy: str = "baseline",
) -> dict:
    """
    Generate the input configuration file for MaSim."
    """

    assert len(death_rate) == len(initial_age_structure) == len(age_distribution), (
        f"The length of death_rate ({len(death_rate)}) and age_distribution ({len(age_distribution)}), must be the same as age_structure ({len(age_distribution)})."
    )
    execution_control = load_yaml(os.path.join("templates", "config.yml"))
    execution_control["starting_date"] = start_date
    execution_control["start_of_comparison_period"] = comparison_date
    execution_control["ending_date"] = end_date
    execution_control["start_collect_data_day"] = start_collecting_day
    execution_control["birth_rate"] = birth_rate
    execution_control["death_rate_by_age_class"] = death_rate
    execution_control["death_rate"] = death_rate
    execution_control["initial_age_structure"] = initial_age_structure
    execution_control["seasonal_info"] = {
        "enable": True,
        "mode": "rainfall",
        "rainfall": {
            "filename": os.path.join("data", name, f"{name}_seasonality.csv"),
            "period": 365,
        },
    }
    execution_control["spatial_model"] = create_spatial_model(calibration)
    execution_control["raster_db"] = validate_raster_files(name, calibration, age_distribution=age_distribution)
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
        execution_control["strategy_db"] = {
            0: {
                "name": "baseline",
                "type": "MFT",
                "therapy_ids": [0],
                "distribution": [1],
            },
        }
        execution_control["initial_strategy_id"] = 0
        output_path = os.path.join("conf", name, f"{strategy}.yml")
        # Write the configuration files
        yaml.dump(execution_control, open(output_path, "w"))

    return execution_control
