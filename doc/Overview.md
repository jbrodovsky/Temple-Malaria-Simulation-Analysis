# MaSim Analysis and Calibration Toolkit

This document provides a high-level overview of the MaSim analysis toolkit in this repository, including installation, layout, common workflows, and major Python / CLI entry points. It is based on the Python package `masim-analysis`, the surrounding repo structure, and the calibration workflow documented in `Calibration Notebook v2.ipynb`.

## 1. Repository Overview

This repo wraps the compiled MaSim simulation binary with Python tooling for:

- Setting up per-country directory structures and data
- Running calibration and validation pipelines to fit the transmission parameter (beta)
- Generating command lists and PBS job scripts for cluster execution
- Post-processing MaSim SQLite outputs into CSV tables and plots

### 1.1. Key Directories

- `bin/`
  - Contains compiled binaries from the core MaSim model repo:
    - `MaSim` – main stochastic, individual-based malaria simulator
    - `DxGGenerator` – auxiliary tool for genotype-related inputs (where used)
  - These binaries are not built here; they must be copied in from the model repo.

- `src/masim_analysis/`
  - `analysis.py` – database readers, aggregation functions, plotting helpers
  - `calibrate.py` – calibration pipeline, beta-fitting tools, sinusoidal helpers
  - `commands.py` – command and PBS job generation, country directory setup, CLI `commands`
  - `configure.py` – configuration builder, drug/therapy/genotype databases, `CountryParams`
  - `interactive.py` – menu-driven `masim` TUI for common tasks
  - `utils.py` – raster I/O, plotting helpers, logging, multiprocessing helpers
  - `validate.py` – validation pipeline and post-processing, CLI `validate`

- `conf/`
  - Per-country YAML configuration files for MaSim.
  - Layout:
    - `conf/<country>/` – root per-country config
      - `calibration/` – auto-generated calibration configs (one per beta/access/pop bin)
      - `test/` – scratch or test configs; also used for growth-validation and validation configs

- `data/`
  - Per-country raster ASCII grids and input CSVs.
  - Typical contents for a country `name`:
    - `data/name/name_districts.asc`
    - `data/name/name_initialpopulation.asc`
    - `data/name/name_population.asc`
    - `data/name/name_pfpr210.asc`
    - `data/name/name_traveltime.asc`
    - `data/name/name_treatmentseeking.asc`
    - `data/name/name_drugdistribution.csv`
    - `data/name/name_incidence_data.csv`
    - `data/name/calibration/` – calibration-specific intermediate data

- `output/`
  - Per-country simulation outputs (SQLite `.db` files) and processed CSV summaries.
  - Layout:
    - `output/<country>/calibration/` – calibration runs
    - `output/<country>/validation/` – validation runs

- `images/`
  - Plots saved from calibration and validation workflows, grouped by country.

- `templates/`
  - Historical YAML templates (e.g., `_raster_db.yml`, `config.yml`, `therapy_db.yml`).
  - Being gradually replaced by `configure.py` data classes and configuration builders.

---

## 2. Installation

The Python tools are packaged as `masim-analysis` and defined in `pyproject.toml`.

### 2.1. Python Requirements

- Python ≥ 3.10
- Core dependencies (see `pyproject.toml`):
  - `numpy`, `pandas`, `scipy`, `matplotlib`, `ruamel-yaml`, `jupyter`
  - Optional interactive dependency: `rich` (for the `masim` TUI)

### 2.2. Local Install (Recommended: editable)

From the repository root:

```bash
# Using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Or using uv
uv pip install -e .
```

This installs the package in editable mode and registers CLI entry points.

On a cluster (e.g., Nessun Dorma) use the provided build script (`scripts/server_build.bash`) to create the virtual environment and install or use the previous pip-based commands.

### 2.3. MaSim Binary

The MaSim binary is not built or included in this repository. You must obtain it from the core MaSim model repository or have it available on your system path. At the moment this package is configured to look for `./bin/MaSim`. Please ensure the binary is compiled and placed in that location before running simulations.


## 3. Command-Line Entry Points

The package exposes several console scripts via `pyproject.toml`:

- `calibrate` → `masim_analysis.calibrate:main`
- `commands`  → `masim_analysis.commands:main`
- `masim`     → `masim_analysis.interactive:main`
- `validate`  → `masim_analysis.validate:main`

### 3.1. `calibrate` – Country Calibration Pipeline

This command runs the full calibration pipeline for a country, including:

- Writing calibration MaSim configs under `conf/<country>/calibration/`
- Generating MaSim commands for each beta/access/population bin
- Executing calibration runs in parallel
- Summarizing `.db` outputs and fitting beta vs. PfPR curves

Usage (typical):

```bash
calibrate <country_code> -r 20 -o output
```

Parameters:

- `country_code` – short code (e.g., `moz`, `bf`, `rwa`, `ago`)
- `-r, --repetitions` – number of repetitions per parameter set
- `-o, --output_dir` – base output directory (default: `output`)

### 3.2. `validate` – Country Validation Pipeline

This command runs MaSim validation simulations with calibrated parameters and processes results, including:

- Generate validation MaSim commands via `commands.generate_commands`.
- Execute commands with multiprocessing (`utils.multiprocess`).
- Aggregate results and write CSV + standard validation plots.

Usage:

```bash
validate <country_code> -r 50 -o output
```

Outputs (under `output/<country>/validation/`):

- `ave_population.csv`
- `ave_cases.csv`
- `ave_prevalence_2_to_10.csv`
- `ave_cases_2_to_10.csv`
- `ave_prevalence_under_5.csv`
- `ave_cases_under_5.csv`
- `prevalence_comparison.csv`

And plots (under `images/<country>/`):

- `prevalence_fit_2_to_10.png`

### 3.3. `commands` – Command and Job Generation

Utility CLI for working directly with the MaSim binary.

Subcommands:

- `generate`
  - Create MaSim commands for a single YAML configuration.
  - Writes a `<strategy>_<reps>reps.txt` file with one shell command per line.
  - Usage:
    
    ```bash
    commands generate -c conf/moz/AL5.yml -o output/moz/AL5 -r 50
    ```

- `batch`
  - Discover all `.yml` configs under a directory and generate a single commands file.
  - Usage:
    
    ```bash
    commands batch -i conf/moz/validation -o output/moz/validation -r 100 -n moz_validation_cmds.txt
    ```

- `job`
  - Convert a commands file into a PBS job script with basic load-balancing.
  - Usage:
    
    ```bash
    commands job -f moz_validation_cmds.txt -d nd01 -n MozValidation -t 48
    ```

    This produces `MozValidation.pbs` and prints the appropriate `qsub` command.

- `setup`
  - Create the standard directory layout for a new country and warn about missing rasters.
  - Usage:
    
    ```bash
    commands setup <country_code>
    ```

### 3.4. `masim` – Interactive Terminal UI

Launches a menu-driven interactive interface (requires `rich`).

Usage:

```bash
masim
```

Menu options:

1. Generate simulation commands (single config)
2. Batch generate commands
3. Generate PBS job file
4. Calibrate model (wraps `calibrate.calibrate`)
5. Setup new country directories

This is useful when working on the cluster or in a terminal where you prefer prompts over remembering CLI flags. That said this feature is optional and still in a prototype stage.

---

## 4. Typical Workflows

The `Calibration Notebook v2.ipynb` provides a detailed, guided workflow for calibrating a new country. A snapshot of this notebook from December 2025 is also contained under `doc/Calibration Notebook Example.pdf`. Below is a condensed summary aligned with that notebook and the Python APIs.

### 4.1. Initial Country Setup

1. Choose a short code and long name, e.g.:
   - `long_name = "Angola"`
   - `name = "ago"`

2. Create directory layout (from Python or CLI):

   - Python:
     
     ```python
     from masim_analysis import commands
     commands.setup_directories("ago")
     ```

   - CLI:
     
     ```bash
     commands setup ago
     ```

3. Populate `data/<name>` with raster and CSV inputs (from shared storage):

   - `<name>_districts.asc`
   - `<name>_population.asc`
   - `<name>_initialpopulation.asc`
   - `<name>_pfpr210.asc`
   - `<name>_traveltime.asc`
   - `<name>_treatmentseeking.asc`
   - `<name>_drugdistribution.csv`
   - `<name>_incidence_data.csv`

4. (Optional) Inspect rasters and district mapping using `utils`:

   ```python
   import os
   import pandas as pd
   from masim_analysis import utils

   name = "ago"
   long_name = "Angola"

   districts, _ = utils.read_raster(os.path.join("data", name, f"{name}_districts.asc"))
   population, _ = utils.read_raster(os.path.join("data", name, f"{name}_initialpopulation.asc"))
   prevalence, _ = utils.read_raster(os.path.join("data", name, f"{name}_pfpr210.asc"))

   dist_fig = utils.plot_districts(districts, labels, long_name)
   pop_fig = utils.plot_population(population, long_name)
   pfpr_fig = utils.plot_prevalence(prevalence, long_name)
   ```

### 4.2. Baseline Drug Strategy and Events

From the calibration notebook:

- Clean and normalize `data/<name>/<name>_drugdistribution.csv`.
- Map columns to therapy IDs defined in `configure.THERAPY_DB`.
- Build a `strategy_db` dictionary that encodes a baseline strategy as an MFT with keys:
  - `name`, `type`, `therapy_ids`, `distribution`.
- Construct an `events` list that schedules the strategy to start at the appropriate year.
- Write out to:
  - `conf/<name>/test/strategy_db.yaml`
  - `conf/<name>/test/events.yaml`

These are then consumed by `configure.configure` during calibration and validation.

### 4.3. Birth-Rate / Growth Sanity Check

Before full calibration, the notebook performs a small growth-validation run to make sure the demographic parameters match the target population.

Key steps (see the notebook for full code):

- Define:
  - `age_distribution`, `birth_rate`, `death_rate`, `initial_age_structure`, `target_population`.
- Build a small configuration using `configure.configure` with a fixed population (`pop = 100000`) and `beta_override = 0.0`.
- Write YAML to `conf/<name>/test/<name>_growth_validation_<pop>.yaml`.
- Generate per-pixel inputs with `calibrate.write_pixel_data_files`.
- Run MaSim manually with `os.system("./bin/MaSim ...")`.
- Analyze population growth using `analysis.get_table` and `matplotlib`, compare to `target_population`, and adjust `birth_rate` if necessary.

Once the growth check yields a population within ~2% of the target, serialize parameters to JSON using `CountryParams` (see below).

### 4.4. Seasonality Calibration

Using monthly incidence data (the type and structure of which will likely vary):

- Load data and clean obvious outlier years.
- Compute median/mean/mode and normalize to derive a relative seasonal signal.
- Use `scipy.optimize.curve_fit` on sinusoidal models from `calibrate`:
  - `calibrate.sinusoidal`
  - `calibrate.positive_sinusoidal`
- Inspect fits and choose the positive half-wave form for consistency with lab conventions.
- Extend the fitted curve to 365 days and write to `data/<name>/<name>_seasonality.csv`.

These seasonality scalars are then used in the MaSim configuration via the `SEASONAL_MODEL` machinery in `configure`.

### 4.5. Treatment-Seeking Raster Normalization

MaSim expects treatment access rasters in the range [0, 1]. If the raster is in [0, 100], divide by 100 and write back:

```python
from pathlib import Path
from masim_analysis import utils
from masim_analysis.configure import CountryParams
import numpy as np

country = CountryParams.load("ago")

raster, meta = utils.read_raster(Path("data") / country.country_code / f"{country.country_code}_treatmentseeking.asc")
if np.nanmax(raster) > 1.0:
    raster /= 100.0

utils.write_raster(
    raster,
    Path("data") / country.country_code / f"{country.country_code}_treatmentseeking.asc",
    meta["xllcorner"],
    meta["yllcorner"],
)
```

### 4.6. End-to-End Calibration and Validation (Cluster)

Typical cluster workflow for a new country:

1. On your workstation:
   - Prepare and clean rasters and CSVs.
   - Run the notebook-guided steps (growth check, seasonality, treatment-seeking normalization).
   - Commit updates to Git.

2. On the cluster (Nessun Dorma):
   - Clone or pull this repository.
   - Build the Python environment and install `masim-analysis`.
   - Run full calibration:
     
     ```bash
     calibrate <country_code> -r 20
     ```

   - Run validation:
     
     ```bash
     validate <country_code> -r 50
     ```

3. Use `scp` to copy `output/<country>/calibration`, `output/<country>/validation`, `images/<country>/`, and `data/<country>/<country>_beta.asc` back locally for further analysis.

---

## 5. Core Python APIs

Below are the main programmatic entry points a user is likely to call from notebooks or scripts.

### 5.1. `masim_analysis.configure`

Key contents:

- Constants / small databases:
  - `GENOTYPE_INFO` – genotype locus definitions and mutation structure
  - `DRUG_DB` – mapping from drug index to pharmacokinetic/pharmacodynamic parameters
  - `THERAPY_DB` – mapping from therapy index to constituent drugs and dosing days
  - `STRATEGY_DB` – default baseline MFT strategy
  - `NODATA_VALUE` – sentinel for rasters

- Data classes:
  - `CountryParams`
    - Fields: `country_code`, `country_name`, `age_distribution`, `birth_rate`, `calibration_year`, `death_rate`, `initial_age_structure`, `target_population`, `starting_date`, `ending_date`, `start_of_comparison_period`, `target_count`, bounds, etc.
    - Methods:
      - `to_dict()` – convert to plain dict suitable for JSON/YAML, with dates formatted as `YYYY/MM/DD`.
      - `save(...)` / `load(...)` (via JSON file under `conf/<country>/test/<country>_country_params.json`), used heavily in the notebook.

  - Additional configuration-related dataclasses (for movement, circulation, parasites, etc.) are defined later in the file and ultimately feed into `configure`.

- Functions:
  - `configure.configure(...) -> dict`
    - Central function to build a serializable MaSim config dictionary.
    - Key parameters (see calibration/validation code):
      - `country_code`
      - `birth_rate`
      - `initial_age_structure`
      - `age_distribution`
      - `death_rates`
      - `starting_date`, `start_of_comparison_period`, `ending_date`
      - `strategy_db` – strategies defined for the run
      - `calibration_str` – string suffix for calibration experiments
      - `beta_override` – override for beta when calibrating
      - `population_scalar` – artificial population rescaling factor
      - `treatment_access_rate` (implied by access bins in calibration)
      - `calibration` flag – if `True`, configure for calibration experiments
    - Returns a Python dict with keys such as `raster_db`, `events`, `drug_db`, `genotype_info`, etc., ready to be dumped with `ruamel.yaml` to a `.yml` file and consumed by MaSim.

### 5.2. `masim_analysis.commands`

- `generate_commands(input_configuration_file, output_directory, repetitions=1, use_pixel_reporter=True)`
  - Returns `(commands_filename, commands_list)` for the specified config.
  - Commands look like:
    
    ```bash
    ./bin/MaSim -i conf/moz/AL5.yml -o output/moz/AL5_ -r SQLitePixelReporter -j 0
    ```

- `batch_generate_commands(input_configuration_directory, output_directory, repetitions=1)`
  - Walks a directory tree, collecting all `.yml` configs and returning a flattened list of commands.

- `generate_job_file(commands_filename, node=Cluster.ONE, job_name="MyJob", cores_override=None, time_override=48, ...)` 
  - Writes `job_name.pbs` with bash logic to execute commands in parallel up to `cores_requested`.

- `setup_directories(country_code)`
  - Creates `conf/`, `data/`, `images/`, `log/`, `output/` (and calibration/validation subfolders) for the country.
  - Performs a simple check that required rasters exist in `data/<country>`.

The CLI `commands` simply wraps these functions with `argparse` plus subcommands.

### 5.3. `masim_analysis.calibrate`

This module contains both the end-to-end `calibrate` CLI entry point and many helper functions used in the notebook.

Notable functions:

- `generate_configuration_files(country_code, calibration_year, access_rates, birth_rate, death_rate, initial_age_structure, age_distribution, strategy_db, logger=None)`
  - Generates the grid of calibration YAML configs under `conf/<country>/calibration/` for each combination of population bin, access rate, and beta.
  - Uses `configure.configure` and `write_pixel_data_files`.

- `write_pixel_data_files(raster_db: dict, population: int)`
  - Writes simple 1×1 ASCII rasters for population and district, used for calibration pixel experiments.

- `generate_calibration_commands(country: CountryParams, access_rates, repetitions=20, output_directory=Path("output")) -> list[str]`
  - Calls `generate_configuration_files` and then `batch_generate_commands` to produce commands for `output/<country>/calibration`.

- `check_missing_runs(country_code, access_rates, output_dir, repetitions=20) -> list[str]`
  - Scans for missing `.db` outputs and returns commands to re-run them.

- Fitting helpers:
  - `sinusoidal`, `positive_sinusoidal` – seasonal curve models (used in the notebook).
  - `linear`, `sigmoid`, `inverse_sigmoid`, `fit_log_sigmoid_model` – used to fit beta–PfPR relationships.
  - Additional helpers later in the file summarize calibration results and plot log-sigmoid fits.

- `calibrate(country_code: str, repetitions: int, output_dir: Path | str = Path("output")) -> None`
  - End-to-end calibration pipeline, used by the `calibrate` CLI and by `interactive_calibrate` in `interactive.py`.

### 5.4. `masim_analysis.validate`

- `_averaging_pass(country: CountryParams)`
  - Uses `analysis.get_average_summary_statistics` on `output/<country>/validation` and writes CSVs.

- `_prevelance_comparison(...)`
  - Compares simulated versus observed PfPR2–10 and under-5 prevalence, using the observed raster `data/<country>_pfpr210.asc` and averaging by district.

- `post_process(country: CountryParams, params: dict, logger=None)`
  - Runs `_averaging_pass`, computes last-year statistics via `calibrate.get_last_year_statistics`, and saves prevalence comparison tables and plots.

- `validate(country_code: str, repetitions: int = 50, output_dir: Path | str = Path("output"))`
  - Full validation pipeline (see CLI above).

### 5.5. `masim_analysis.analysis`

High-level categories of functions:

- SQLite helpers:
  - `get_all_tables(db)` – list all tables in a MaSim `.db`.
  - `get_table(db, table)` – read a single table into a DataFrame.

- Treatment failure aggregation:
  - `calculate_treatment_failure_rate(data)` – add `failure_rate` = `treatmentfailures / treatments`.
  - `aggregate_failure_rates(path, strategy, locationid=0)` – aggregate failure rates across runs.
  - `plot_strategy_treatment_failure(data, strategy, figsize=(18, 3))` – basic plotting utility.

- Population and genome data:
  - `get_population_data(file, month=-1)` – extract final-month population and infected counts.
  - `get_genome_data(file, month=-1)` – extract `monthlygenomedata` subset.
  - `calculate_genome_frequencies(file, month=-1)` – compute genotype frequencies.
  - `get_resistant_genotypes(genomes, allele)` – filter by allele string.
  - `calculate_resistant_genome_frequencies(...)`, `aggregate_resistant_genome_frequencies(...)`, `aggregate_resistant_genome_frequencies_by_month(...)` – helper functions for evolution / resistance analysis.

- Summary statistics across runs:
  - `get_average_summary_statistics(path)` – main function used by calibration/validation for averaging across `.db` files.
  - Additional helpers produce combined strategy plots and prevalence trend plots used by `validate.post_process`.

These APIs are primarily consumed by the calibration and validation modules and by Jupyter notebooks.

### 5.6. `masim_analysis.utils`

- Raster utilities:
  - `read_raster(file) -> (array, metadata)` – ASCII raster reader.
  - `write_raster(raster, file, xllcorner, yllcorner, cellsize=5000)` – ASCII raster writer (using `configure.NODATA_VALUE`).

- Plotting helpers (used in the calibration notebook):
  - `plot_districts(districts_raster, labels, country_name, fig_size=(10, 10), loc=None)`
  - `plot_population(population_raster, country_name, fig_size=(10, 10), population_upper_limit=1000)`
  - `plot_prevalence(prevalence_raster, country_name, fig_size=(10, 10))`

- Logging:
  - `get_country_logger(country_code, logger_name)` – logger writing to `log/<country>/<logger_name>.log` and stdout.

- Parallel execution:
  - `get_optimal_worker_count(utilization=1.0)` – derive reasonable worker count from CPU cores.
  - `run_simulation_command(cmd)` – run a single MaSim command, returning `(cmd, success, error_message)`.
  - `multiprocess(cmds, max_workers, logger)` – generic ProcessPoolExecutor wrapper that executes commands in parallel, logs progress, and returns `(successful_runs, failed_entries)`.

### 5.7. `masim_analysis.interactive`

Implements the `masim` TUI:

- Functions to:
  - Show a rich-text menu
  - Prompt for paths, repetition counts, and cluster parameters
  - Call `generate_commands`, `batch_generate_commands`, `generate_job_file`, `calibrate.calibrate`, and `setup_directories` interactively.

This is useful as a front-end to the APIs described above.

---

## 6. Recommended Usage Patterns

### 6.1. For New Country Calibration

1. Run `commands setup <code>` to create base directories.
2. Populate `data/<code>` with required rasters and CSVs.
3. Use a calibration notebook (e.g., `Calibration Notebook v2.ipynb`) to:
   - Inspect rasters via `utils`.
   - Construct `strategy_db.yaml` and `events.yaml`.
   - Sanity-check birth rate / growth via a small MaSim run.
   - Fit seasonality curves and write `<code>_seasonality.csv`.
   - Serialize stable parameters to `CountryParams` JSON.
4. On the cluster, run `calibrate <code> -r 20` and wait for completion.
5. Run `validate <code> -r 50` to assess calibration quality.
6. Use `analysis` and plotting helpers (plus the notebook) to interpret outputs.

### 6.2. For Strategy / Policy Experiments

Once calibration and validation are satisfactory:

1. Create strategy YAMLs under `conf/<code>/` describing alternative treatment policies.
2. Use `commands batch` to generate commands for all strategy configs.
3. Use `commands job` to create a PBS script and submit via `qsub`.
4. After runs finish, use `analysis` (e.g., `aggregate_failure_rates`, genome frequency tools) and `matplotlib` / notebooks to compare strategies.

---

This overview should be sufficient for an experienced user to install the toolkit, understand how the per-country data and configuration flow through calibration and validation, and locate the major Python APIs and CLI tools used in the MaSim analysis workflows.