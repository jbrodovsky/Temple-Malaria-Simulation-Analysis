"""
Generate commands for MaSim
"""

import argparse
import os
from math import ceil, floor

NODE_MEMORY = 128  # GB
NODE_CORES = 28  # Cores
JOB_MEMORY = 8  # GB
MAX_CORES_PER_NODE = floor(NODE_MEMORY / JOB_MEMORY) - 1  # Cores
TIME_PER_JOB = 5  # Hour
MAX_WALL_TIME = 48  # Hour
JOBS_PER_NODE = floor(MAX_WALL_TIME / TIME_PER_JOB) * MAX_CORES_PER_NODE  # Jobs
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
    input_configuration_file: str, output_directory: str, repetitions: int = 1
) -> tuple[str, list[str]]:
    """
    Generate commands for MaSim

    Parameters
    ----------
    input_configuration_file : str
        The input configuration file path, ex: ./conf/rwa/AL5.yml
    output_directory : str
        The output directory to store simulation results, ex: ./output/rwa
    repetitions : int
        The number of repetitions
    """
    # Check if the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Parse the strategy name
    strategy_name = os.path.basename(input_configuration_file).split(".yml")[0]
    commands_filename = f"{strategy_name}_{repetitions}reps.txt"
    if os.path.exists(commands_filename):
        os.remove(commands_filename)
    commands = []
    # with open(commands_filename, "w") as f:
    output_file = os.path.join(output_directory, f"{strategy_name}")

    for i in range(repetitions):
        commands.append(
            f"./bin/MaSim -i {input_configuration_file} -o {output_file}_ -r SQLiteDistrictReporter -j {i}\n"
        )
    return commands_filename, commands


def batch_generate_command_jobs(
    input_configuration_directory: list,
    output_directory: str,
    repetitions: int = 1,
) -> list[str]:
    """
    Batch generate commands for MaSim

    Parameters
    ----------
    input_configuration_directory : list
        The input configuration directory, ex: ./input/rwa
    output_directory : str
        The output directory, ex: ./output/rwa
    repetitions : int
        The number of repetitions
    """
    commands = []
    for root, _, files in os.walk(input_configuration_directory):
        for file in files:
            if file.endswith(".yml"):
                input_configuration_file = os.path.join(root, file)
                _, new_commands = generate_commands(
                    input_configuration_file,
                    output_directory,
                    repetitions,
                    # in_parallel=True,
                )
                commands.extend(new_commands)
    return commands


def generate_job_file(
    commands_filename: str, job_name: str = "MyJob", cores_override: int = None, nodes_override: int = None
) -> None:
    # Get the number of lines in commands_filename
    with open(commands_filename, "r") as f:
        num_commands = sum(1 for _ in f)
    # Generate the job file
    job_filename = f"{job_name}.sh"
    if os.path.exists(job_filename):
        os.remove(job_filename)

    # Read the length of the commands file
    if (cores_override is not None and cores_override > 0 and cores_override <= 28) and (
        nodes_override is not None and nodes_override > 0 and nodes_override <= 10
    ):
        # needed_nodes = ceil(num_commands / cores_override)
        cores_requested = cores_override
        needed_nodes = nodes_override
    else:
        needed_nodes = ceil(num_commands / JOBS_PER_NODE)
        cores_requested = MAX_CORES_PER_NODE

    with open(job_filename, "w") as f:
        f.write("#!/bin/sh\n")
        f.write("#PBS -l walltime=48:00:00\n")
        f.write(f"#PBS -N {job_name}\n")
        f.write("#PBS -q normal\n")
        f.write(f"#PBS -l nodes={needed_nodes}:ppn={cores_requested}\n")
        f.write("cd $PBS_O_WORKDIR\n")
        f.write(f"torque-launch {commands_filename}\n")
    # Display the commands file
    print(f"Commands file: {commands_filename}")
    print(f"Job script: {job_filename}")
    run_command = f"qsub {job_filename}"
    print(f"To submit the job, run: {run_command}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate commands for MaSim")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="The input configuration file path, ex: ./conf/rwa/AL5.yml or /conf/rwa. If it is a directory, it will batch generate commands for all the files in the directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="The output directory to store simulation results, ex: ./output/rwa",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=1,
        help="The number of repetitions to run for each strategy specified by the input configuration file",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="MyJob",
        help="The name of the job to submit to the cluster (default: MyJob)",
    )

    args = parser.parse_args()
    # generate_commands(args.input, args.output, args.repetitions)
    if os.path.isdir(args.input):
        commands = batch_generate_command_jobs(args.input, args.output, args.repetitions)
        filename = "batch_commands.txt"
    else:
        filename, commands = generate_commands(args.input, args.output, args.repetitions)
    with open(filename, "w") as f:
        for command in commands:
            f.write(command)
    generate_job_file(filename, args.name)


if __name__ == "__main__":
    main()
