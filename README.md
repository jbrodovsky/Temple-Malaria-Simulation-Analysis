# Temple Malaria Simulation Analysis

This repository contains the code and data for the execution and analysis of the [malaria simulation model](https://github.com/jbrodovsky/Temple-Malaria-Simulation) developed by the Temple University Boni Lab. The model is a stochastic, individual-based model that simulates the transmission of malaria in a population of humans and mosquitoes.

This repository server two related purposes:

1. Code for running the simulation model (both locally for testing and on a cluster for large-scale simulations)
2. Code for analyzing the output of the simulation model

To that end there is both pre-processing code and post-processing code. Exact workflow is to be determined, but the [Owl's Nest cluster](https://www.hpc.temple.edu/) at Temple University is running CentOS whereas the local workstations are running Ubuntu or macOS. Cluster scripts should only be written in `sh`, `bash`, or native `Python`. Local scripts may take advantage of `nu` or any other shell and a managed Python environment. 

The general advised workflow is to develop input files locally that describe a given simulation scenario and then to transfer them to the desired cluster for processing. Use the `generate_commands.py` script to generate a list of commands and job files to run on Owl's Nest. Once complete, transfer the output data back to the local workstation for post processing analysis.

## Installation

This repository is for operation and analysis and requires the binaries compiled by the [malaria simulation model](https://github.com/jbrodovsky/Temple-Malaria-Simulation) repository. To install, simply clone this repository locally. Once cloned, the simulation binaries `MaSim` and `DxGGenerator` that are built from the simulation model should be copied to the `bin` directory of this repository.

These binaries are typically build in the `/build/bin` directory of the model repository. Given that this model runs both locally on a workstation (typically running Ubuntu or macOS) and on a cluster (typically running CentOS), the binaries should be compiled for the appropriate platforms locally and copied to the `bin` directory. A CI/CD pipeline to build these binaries should be implemented in the future and this repository should be updated to instead depend on those binaries.

## Usage

The structure of this repository is intended such that the simulation is called from the root folder: `./bin/MaSim`. For organization, unchanging input data (namely the raster `.asc` files) is placed in the `data` folder and the input `.yml` files in the `input` folder. File paths in the input files should be relative to the root folder. The simulation does have an output folder input flag, however it does not automatically generate the folder if it does not exist. Please pre-generate the output folder before running the simulation.

The simulation can be run locally with the following command:

```shell
./bin/MaSim -i ./input/<input_file>.yml -o ./output/<output_folder> -r SQLiteDistrictReporter
```

The MaSim simulation needs a input `.yml` file that describes the simulation parameters and events. The simulation binary itself cannot run multiple strategies in parallel or multiple repetitions of the same strategy. The simulation will run through one sequence of events (which may or may not contain multiple strategies) and then stop.

In order to do actual comparative science, we need to run multiple repetitions of the simulation for multiple strategies. While these can be done at low scale on a high-powered workstation (each run usually needs around 8GB of memory) with multiple CPU cores, it is beneficial to run these repetitions in parallel on the cluster.

For parallel computing on the cluster, generate a list of commands following the template above and save it to a text file (ex: `commands.txt`). The commands text file is used to launch the jobs on the cluster using a job file as such:

```sh
#!/bin/sh
#PBS -l walltime=24:00:00
#PBS -N MaSim
#PBS -q normal
#PBS -l nodes=1:ppn=28

torque-launch commands.txt
```

Queue the job by running `qsub <job_script>.sh`. This job described above will run on the normal queue for a maximum of 24 hours using 1 node with 28 cores (the maximum available). Please use the `generate_commands.py` script to generate the commands file and job file automatically based on the input file(s) and number of repetitions.

Use the following naming conventions and data organization priciples:

- The `data` folder contains all the country-specific data files (either one country, or in country level folders)
- The `input` folder contains all the input files for the simulation again organized by country if multiple countries are being analyzed.
- Input file names should describe the strategy being run
- Output files should organized by country and strategy and be named as `<strategy>_<repetition>.db`

## Data Transfer

The simulation generates a lot of data. The data is stored in the output folder specified in the input file. Owl's Nest has git, and this repo can be cloned and changes pulled directly to the cluster. Output data can be transfered using the `scp` command.

To transfer to Owl's Nest:

```sh
scp -r <local_folder> <username>@owlsnest.temple.edu:/path/to/destination
```

To transfer from Owl's Nest:

```sh
scp -r <username>@owlsnest.temple.edu:/path/to/destination <local_folder>
```
