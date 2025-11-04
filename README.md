# Temple Malaria Simulation Analysis

This repository contains the code and data for the execution and analysis of the [malaria simulation model](https://github.com/jbrodovsky/Temple-Malaria-Simulation) developed by the Temple University Boni Lab. The model is a stochastic, individual-based model that simulates the transmission of malaria in a population of humans and mosquitoes.

This repository serves two related purposes:

1. Code for running the simulation model (both locally for testing and on a cluster for large-scale simulations)
2. Code for analyzing the output of the simulation model

## Installation

This repository is for operation and analysis and requires the binaries compiled by the [malaria simulation model](https://github.com/jbrodovsky/Temple-Malaria-Simulation) repository. To install, simply clone this repository locally. Once cloned, the simulation binaries `MaSim` and `DxGGenerator` that are built from the simulation model should be copied to the `bin` directory of this repository.

These binaries are typically built in the `/build/bin` directory of the model repository. Given that this model runs both locally on a workstation (typically running Ubuntu or macOS) and on a cluster (typically running CentOS), the binaries should be compiled for the appropriate platforms locally and copied to the `bin` directory. A CI/CD pipeline to build these binaries should be implemented in the future and this repository should be updated to instead depend on those binaries.

### Python Environment Setup

This repository includes a Python package (`masim-analysis`) with command-line tools for calibration, validation, and workflow automation. Install the package and its dependencies:

```bash
# Using pip
pip install -e .

# Or using uv (if available)
uv pip install -e .
```

For cluster deployment, use the provided build script:

```bash
scripts/server_build.bash
source .venv/bin/activate
```

## Workflow Commands

The package provides several command-line tools to streamline calibration, validation, and simulation workflows. These commands are available after installing the package:

### Available Commands

#### 1. `calibrate` - Model Calibration Pipeline

Runs the complete calibration pipeline for a country, including configuration generation, simulation execution, and beta map creation.

```bash
calibrate <country_code> [-r REPETITIONS] [-o OUTPUT_DIR]
```

**Example:**
```bash
calibrate moz -r 20 -o output
```

#### 2. `validate` - Model Validation Pipeline

Executes validation simulations using calibrated parameters and generates validation statistics.

```bash
validate <country_code> [-r REPETITIONS] [-o OUTPUT_DIR]
```

**Example:**
```bash
validate moz -r 50 -o output
```

#### 3. `commands` - Simulation Command Generation

Generates MaSim simulation commands and PBS job files for cluster execution.

**Subcommands:**

- **Generate commands for a single configuration:**
  ```bash
  commands generate -c CONFIG_FILE [-o OUTPUT_DIR] [-r REPETITIONS] [-n NAME]
  ```

- **Batch generate commands from multiple configurations:**
  ```bash
  commands batch -i INPUT_DIR -o OUTPUT_DIR [-r REPETITIONS] [-n COMMANDS_FILE]
  ```

- **Generate PBS job file:**
  ```bash
  commands job -f COMMANDS_FILE [-d NODE] [-n JOB_NAME] [-c CORES] [-t HOURS]
  ```

**Examples:**
```bash
# Generate commands for a single strategy
commands generate -c conf/moz/moz_validation.yml -o output/moz/validation -r 50

# Batch generate commands from all configs in a directory
commands batch -i conf/rwa -o output/rwa -r 100

# Create a job file for cluster execution
commands job -f commands.txt -d nd01 -n MozValidation -t 24
```

#### 4. `masim` - Interactive Terminal Interface

Launches an interactive terminal menu for executing common workflows without remembering command syntax.

```bash
masim
```

This provides a menu-driven interface for:
- Generating simulation commands
- Batch command generation
- Creating PBS job files
- Running calibration pipelines
- Setting up new country directories

## Usage

### Running Simulations with Workflow Commands

The recommended approach for running simulations is to use the provided command-line tools, which handle configuration generation, command creation, and cluster job submission automatically.

**Typical Workflow:**

Note: This requires a preprocessing step done locally to prepare data files before running these commands on the cluster. At a minimum, this includes preparing population, seasonality, and treatment access rate, and prevalence raster data files in the `data` directory and a baseline configuration file in the `conf` directory for the country being simulated.

1. **Calibration**: Generate calibrated beta value map for a country
   ```bash
   calibrate <country_code> -r 20
   ```

2. **Validation**: Validate the calibrated model
   ```bash
   validate <country_code> -r 50
   ```

3. **Strategy Testing**: Generate commands for different treatment strategies
   ```bash
   commands batch -i conf/<country> -o output/<country> -r 100
   ```

4. **Cluster Execution**: Create and submit job files
   ```bash
   commands job -f commands.txt -d nd01 -n JobName -t 48
   qsub job.pbs
   ```

### Direct MaSim Execution (Advanced)

For manual execution or testing, the simulation binaries can be called directly from the root folder:

```shell
./bin/MaSim -i ./conf/<input_file>.yml -o ./output/<output_folder> -r SQLiteDistrictReporter
```

**Important Notes:**

- File paths in configuration files should be relative to the root folder
- Output directories must be created before running simulations
- The MaSim binary runs a single simulation instance; use the `commands` tool and PBS job submission for parallel execution

### Data Organization Conventions

Use the following conventions for organizing data and configurations:

- The `data` folder contains all country-specific data files (organized by country code in subfolders)
- The `conf` folder contains all configuration files for simulations, organized by country
- Configuration file names should describe the strategy being tested
- Output files should be organized by country and strategy, named as `<country code>/<strategy>/<strategy>_<repetition>.db`
- Templates for standard configurations are available in the `templates` folder


## Data Transfer

The simulation generates a lot of data, and at the moment this repo handles both software and data. To transfer source files (code, configurations, raster data --- anything under the `src`, `conf`, or `data` folders --- check them into version control via Git. Transfer using git push/pull. Individual country calibrations should take place on their own branches. Output data can be transferred using the `scp` command.