# Temple Malaria Simulation Analysis

This repository contains the code and data for the execution and analysis of the [malaria simulation model](https://github.com/jbrodovsky/Temple-Malaria-Simulation) developed by the Temple University Boni Lab. The model is a stochastic, individual-based model that simulates the transmission of malaria in a population of humans and mosquitoes. This analysis toolkit was developed by [James Brodovsky](https://github.com/jbrodovsky) and Sarit Adhikari as part of the calibration efforts for Burkino Faso and Mozambique in 2025.

This repository serves two related purposes:

1. Code for running the simulation model (both locally for testing and on a cluster for large-scale simulations)
2. Code for analyzing the output of the simulation model

To that end there is both pre-processing code and post-processing code. The general workflow is as follows:
1. Pre-process data and create the required input raster and csv files under the corresponding country folder.
2. Use the "Calibration Notebook" to create the additional configuration data files consisting of the country parameters, intervention strategies, and simulation parameters.
3. On the cluster, run the calibration then the validation simulations using the input files created in step 2.
4. Use the "Validation Notebook" to analyze the output of the simulations and generate plots.

## Installation

This repository is for operation and analysis and requires the binaries compiled by the [malaria simulation model](https://github.com/jbrodovsky/Temple-Malaria-Simulation) repository. To install, simply clone this repository locally. Once cloned, the simulation binaries `MaSim` and `DxGGenerator` that are built from the simulation model should be copied to the `bin` directory of this repository.

These binaries are typically built in the `/build/bin` directory of the model repository. Given that this model runs both locally on a workstation (typically running Ubuntu or macOS) and on a cluster (typically running CentOS or Ubuntu), the binaries should be compiled for the appropriate platforms locally and copied to the `bin` directory. A CI/CD pipeline to build these binaries should be implemented in the future and this repository should be updated to instead depend on those binaries.

## Usage

The structure of this repository is intended such that the simulation is called from the root folder: `./bin/MaSim`. For organization, unchanging input data (namely the raster `.asc` files) is placed in the `data` folder and the input `.yml` files in the `conf` folder. File paths in the input files should be relative to the root folder. The simulation does have an output folder input flag, however it does not automatically generate the folder if it does not exist. Please pre-generate the output folder before running the simulation.

The simulation can be run locally with the following command:

```shell
./bin/MaSim -i ./conf/<input_file>.yml -o ./output/<output_folder> -r SQLiteDistrictReporter
```

The MaSim simulation needs a input configuration (`.yml`) file that describes the simulation parameters and events. The simulation binary itself cannot run multiple strategies in parallel or multiple repetitions of the same strategy. The simulation will run through one sequence of events (which may or may not contain multiple strategies) and then stop.

In order to do actual comparative science, we need to run multiple repetitions of the simulation for multiple strategies. While these can be done at low scale on a high-powered workstation (each run usually needs around 8-12GB of memory depending on country size and scaling) with multiple CPU cores, it is beneficial to run these repetitions in parallel on a cluster with additional cores and more memory.

For parallel computing on the cluster, generate a list of commands following the template above and save it to a text file (ex: `commands.txt`). The commands text file is used to launch the jobs on the cluster.

Use the following naming conventions and data organization principles:

- The `data` folder contains all the country-specific data files (either one country, or in country level folders)
- The `input` folder contains all the input files for the simulation again organized by country if multiple countries are being analyzed.
- Input file names should describe the strategy being run
- Output files should organized by country and strategy and be named as `<strategy>_<repetition>.db`

## Data Transfer

The simulation generates a lot of data. The data is stored in the output folder specified in the input file. Owl's Nest has git, and this repo can be cloned and changes pulled directly to the cluster. Output data can be transfered using the `scp` command.

## Style guide

This package follows pretty strict styling guidelines for clarity and consistency. The code is formatted and linted using `ruff` and `pyright`. In particular, `pyright` is configured in standard mode, which means that it will check for type errors and other issues in the code. Warnings and errors reported by either of these tools should be addressed before submitting a pull request. In some cases, this may result in somewhat more verbose code, but it is done to ensure that the code is clear and easy to understand. The goal is to make the code as readable and maintainable as possible.
