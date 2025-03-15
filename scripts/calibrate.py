# Country calibration script
from masim_analysis import configure
from datetime import date
import os

from ruamel.yaml import YAML

yaml = YAML()

# Calibration parameters
populations = [10, 25, 50]
access_rates = [0.001, 0.005, 0.01]
betas = [0.55, 0.65, 0.75, 0.85]
# Input parameters
name = "moz"
start = date(2000, 1, 1)
end = date(2020, 12, 31)
comparison = date(2015, 1, 1)
birth_rate = 31.2 / 1000
death_rate = [
    [
        0.0382,
        0.03019,
        0.02027,
        0.01525,
        0.01248,
        0.00359,
        0.00361,
        0.00365,
        0.00379,
        0.00379,
        0.00386,
        0.00504,
        0.0055,
        0.0174,
        0.0174,
    ]
]
### Execution ------------------------------------------------
execution_control = configure.main(
    name,
    start.strftime("%Y/%m/%d"),
    end.strftime("%Y/%m/%d"),
    comparison.strftime("%Y/%m/%d"),
    calibration=True,
)
execution_control["strategy_db"] = {
    0: {
        "name": "baseline",
        "type": "MFT",
        "therapy_ids": [0],
        "distribution": [1],
    },
}
execution_control["initial_strategy_id"] = 0

execution_control["events"] = [
    {"name": "turn_off_mutation", "info": [{"day": start.strftime("%Y/%m/%d")}]},
]
for pop in populations:
    for access in access_rates:
        for beta in betas:
            execution_control["raster_db"] = configure.validate_raster_files(
                "moz",
                calibration=True,
                calibration_string=f"{pop}_{access}_{beta}",
                access_rate=access,
                beta=beta,
                population=pop,
            )
            output_path = os.path.join("conf", name, "calibration", f"cal_{pop}_{access}_{beta}.yml")
            yaml.dump(execution_control, open(output_path, "w"))
