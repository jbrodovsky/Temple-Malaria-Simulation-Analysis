"""
Generate commands for MaSim
"""

import os
import sys
import argparse


def generate_commands(
    input_configuration_file: str, output_directory: str, repetitions: int = 1
) -> str:
    """
    Generate commands for MaSim

    Parameters
    ----------
    input_configuration_file : str
        The input configuration file path, ex: ./input/rwa/AL5.yml
    output_directory : str
        The output directory, ex: ./output/rwa
    repetitions : int
        The number of repetitions
    """
    # Check if the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Parse the strategy name
    strategy_name = os.path.basename(input_configuration_file).split(".")[0]
    commands_filename = f"{strategy_name}_{repetitions}reps.txt"
    if os.path.exists(commands_filename):
        os.remove(commands_filename)
    with open(commands_filename, "w") as f:
        for i in range(repetitions):
            output_file = os.path.join(output_directory, f"{strategy_name}_{i}.csv")
            f.write(
                f"./bin/MaSim -i {input_configuration_file} -o {output_file}/run_{i}_ -r SQLiteDistrictReporter\n"
            )
    # Generate the job file
    job_filename = f"{strategy_name}_{repetitions}reps.job"
    if os.path.exists(job_filename):
        os.remove(job_filename)
    with open(job_filename, "w") as f:
        f.write("#!/bin/sh")
        f.write("#PBS -l walltime=48:00:00")
        f.write("#PBS -N $(basename $input_file .yml)_${num_runs}reps")
        f.write("#PBS -q normal")
        f.write("#PBS -l nodes=1:ppn=28")
        f.write("cd $PBS_O_WORKDIR")
        f.write("torque-launch $commands_file")
    # Display the commands file
    print("Job file and commands file generated for $input_file with $num_runs runs.")
    print(f"Commands file: {commands_filename}")
    # cat $commands_file
    print(f"Job script: {job_filename}")
    # cat $job_script
    run_command = f"qsub {job_filename}"
    print(f"To submit the job, run: {run_command}")
    return run_command


def batch_generate_command_jobs(
    input_configuration_directory: list, output_directory: str, repetitions: int = 1
):
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
    jobs = []
    for root, _, files in os.walk(input_configuration_directory):
        for file in files:
            if file.endswith(".yml"):
                input_configuration_file = os.path.join(root, file)
                jobs.append(
                    generate_commands(
                        input_configuration_file, output_directory, repetitions
                    )
                )
    print(f"Generated {len(jobs)} jobs.")
    # for job in jobs:
    #     print(job)
    run_jobs = input("Would you like to run the jobs? (y/n): ")
    if run_jobs.lower() == "y" or run_jobs.lower() == "yes":
        for job in jobs:
            os.system(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate commands for MaSim")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="The input configuration file path, ex: ./input/rwa/AL5.yml or /input/rwa. If it is a directory, it will batch generate commands for all the files in the directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="The output directory, ex: ./output/rwa",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=1,
        help="The number of repetitions",
    )
    args = parser.parse_args()
    # generate_commands(args.input, args.output, args.repetitions)
    if os.path.isdir(args.input):
        batch_generate_command_jobs(args.input, args.output, args.repetitions)
    else:
        generate_commands(args.input, args.output, args.repetitions)
