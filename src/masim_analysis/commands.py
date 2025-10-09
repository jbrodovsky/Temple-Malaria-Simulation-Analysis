"""
Generate commands and job files for running MaSim simulations.

This module includes functions to:
- Generate MaSim execution commands based on configuration files.
- Batch generate commands for multiple configurations.
- Create job script files for submission to a cluster (e.g., PBS).
"""

import argparse
from asyncio import wait_for
import os
from enum import Enum
from math import ceil, floor
from pathlib import Path
from typing import Optional

from numpy.f2py.symbolic import Op


class Cluster(Enum):
    ONE = "nd01"
    TWO = "nd02"
    THREE = "nd03"
    FOUR = "nd04"


# Multiprocessing notes:
# - We can specify the number of repetitions for each strategy using the -j flag
# - Previous implementation seems to have caused the cluster to run out of memory
# Need to investigate adding nodes to the job and maybe automatically calculating
# the number of nodes and clusters needed based on the number of repetitions. If
# you specify multiple nodes, the jobs system will requisition cores equal to the
# number of nodes times the number of cores per node. For example, if you specify
# 2 nodes and 15 cores per node, the job system will requisition 30 cores.
# - There is no reason NOT to parallelize the jobs, so I don't know to what extent
# using the -j flag is useful on the cluster. It might be useful for local testing.

# We can run approximately 16 jobs simultaneously on a single node
# We can run approximatley 10 sequential job-batchs on a single node (48 / 4.5)
# We can run approximately 160 jobs on a single node (16 * 10) for the max wall time


def generate_commands(
    input_configuration_file: Path | str,
    output_directory: Path | str,
    repetitions: int = 1,
    use_pixel_reporter: bool = True,
) -> tuple[str, list[str]]:
    """
    Generate commands for MaSim.

    Parameters
    ----------
    input_configuration_file : Path | str
        The input configuration file path, ex: ./conf/rwa/AL5.yml.
    output_directory : Path | str
        The output directory to store simulation results, ex: ./output/rwa.
    repetitions : int, optional
        The number of repetitions, by default 1.

    Returns
    -------
    tuple[str, list[str]]
        A tuple containing the commands filename and a list of generated commands.
    """
    # Convert to Path objects for consistent handling
    input_path = Path(input_configuration_file)
    output_path = Path(output_directory)

    # Check if the output directory exists
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Parse the strategy name
    strategy_name = input_path.stem
    commands_filename = f"{strategy_name}_{repetitions}reps.txt"
    commands_path = Path(commands_filename)
    if commands_path.exists():
        commands_path.unlink()
    commands = []
    output_file = output_path / strategy_name
    if use_pixel_reporter:
        reporter = "SQLitePixelReporter"
    else:
        reporter = "SQLiteDistrictReporter"
    for i in range(repetitions):
        commands.append(f"./bin/MaSim -i {input_path} -o {output_file}_ -r {reporter} -j {i}\n")
    return commands_filename, commands


def batch_generate_commands(
    input_configuration_directory: Path | str,
    output_directory: Path | str,
    repetitions: int = 1,
) -> list[str]:
    """
    Batch generate commands for MaSim.

    Parameters
    ----------
    input_configuration_directory : Path | str
        The input configuration directory, ex: ./input/rwa.
    output_directory : Path | str
        The output directory, ex: ./output/rwa.
    repetitions : int, optional
        The number of repetitions, by default 1.

    Returns
    -------
    list[str]
        A list of all generated commands.
    """
    # Convert to Path objects for consistent handling
    input_path = Path(input_configuration_directory)
    output_path = Path(output_directory)

    commands = []
    for yml_file in input_path.rglob("*.yml"):
        _, new_commands = generate_commands(
            yml_file,
            output_path,
            repetitions,
            # in_parallel=True,
        )
        commands.extend(new_commands)
    return commands


def generate_job_file(
    commands_filename: str,
    node: Cluster = Cluster.ONE,
    job_name: Path | str = Path("MyJob"),
    cores_override: Optional[int] = None,
    time_override: Optional[int] = 48,
    std_output_location: Optional[Path | str] = Path("."),
    std_error_location: Optional[Path | str] = Path("."),
    email: Optional[str] = None,
) -> None:
    """
    Generate a generic job file for submitting a list of commands to
    execute on the cluster in parallel.

    Parameters
    ----------
    commands_filename : str
        The file path and name of the file containing the shell commands to execute.
    node : Cluster, optional
        The cluster node to use, by default Cluster.ONE (nd01).
    job_name : str, optional
        The name of the job, by default "MyJob".
    cores_override : Optional[int], optional
        Override the number of cores per node, by default None which
        the usage to maximum.
    time_override : Optional[int], optional
        Override the wall time in hours, by default 48.
    std_output_location : Optional[Path | str], optional
        Standard output file location, by default None, which sets the
        output location to '.' and thus writes to ./<job_name>.output.
    std_error_location : Optional[Path | str], optional
        Standard error file location, by default None, which sets the
        error location to '.' and thus writes to ./<job_name>.error.
    email : Optional[str], optional
        Email address for job notifications, by default None.
    """
    # Get the number of lines in commands_filename
    with open(commands_filename, "r") as f:
        num_commands = sum(1 for _ in f)
    # Generate the job file
    job_filename = f"{job_name}.pbs"
    if os.path.exists(job_filename):
        os.remove(job_filename)

    # Normalize available CPU count to a non-None int
    available_cpus = os.cpu_count() or 1

    if cores_override is None:
        cores_requested = available_cpus
    elif cores_override is not None and cores_override <= available_cpus:
        cores_requested = cores_override
    else:
        cores_requested = available_cpus

    wait_for_available_slot = (
        "function wait_for_available_slot() {\n"
        f"while (( $(jobs -r | wc -l) >= {cores_requested} )); do\n"
        "    sleep 0.5 \n"  # <<< check every 0.5s if a slot is free
        "done\n"
        "}\n"
    )
    command_loop = (
        'while IFS= read -r line || [[ -n "$line" ]]; do\n'
        '    if [[ -n "$line" ]]; then\n'
        "        wait_for_available_slot\n"  # <<< wait if too many jobs
        '        echo "Running: $line" \n'
        f'        eval "$line" > {std_output_location}/{job_name}_${{INDEX}}.output 2> {std_error_location}/{job_name}_${{INDEX}}.error & \n'
        "        ((INDEX++))\n"
        "    fi\n"
        f"done < {commands_filename}\n"
    )

    with open(job_filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#PBS -l walltime={time_override}:00:00\n")
        f.write(f"#PBS -N {job_name}\n")
        f.write("#PBS -q nd\n")
        f.write(f"#PBS -l select=1:ncpus={cores_requested}:host={node.value}\n")
        f.write(f"#PBS -o {std_output_location}/{job_name}.output\n")
        f.write(f"#PBS -e {std_error_location}/{job_name}.error\n")
        if email:
            f.write("#PBS -m bae\n")
            f.write(f"#PBS -M {email}\n")
        f.write("\n")
        f.write("cd $PBS_O_WORKDIR\n")
        f.write(wait_for_available_slot)
        f.write("\n")
        f.write(command_loop)
        f.write("\n")
    # Display the commands file
    print(f"Job file generate for {commands_filename} and saved to: {job_filename}")
    run_command = f"qsub {job_filename}"
    print(f"To submit the job, run: {run_command}")
