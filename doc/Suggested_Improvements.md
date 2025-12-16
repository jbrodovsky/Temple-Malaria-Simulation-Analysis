# Suggested Improvements for MaSim Analysis Toolkit

This document summarizes potential improvements to this repository and the
`masim-analysis` package with an eye toward faster experimentation,
more reproducible workflows, and better long‑term maintainability.

The suggestions are intentionally incremental so they can be adopted
gradually without disrupting ongoing calibration work.

---

## 1. Configuration & Data Management

### 1.1. Separate "source data" from the Python package

Current state:
- Large ASCII rasters and CSVs live under `data/<country>` and are tracked
  in git.
- They are not required to install or import `masim-analysis`, but they are
  required for most real calibrations/validations.

Risks:
- Git history will gradually bloat as rasters and input CSVs are revised.
- Cloning the repo becomes increasingly heavy, even for users only
  interested in the analysis code.
- Makes it hard to ship a light-weight, pip/conda-installable package.

Suggested path:
- **Move heavy inputs to a dedicated data repository or storage bucket.**
  - Option A: Private or public repo such as `Temple-MaSim-Data` containing
    versioned `data/<country>` trees.
  - Option B: Object storage (S3, GCS, campus-hosted HTTP/FTP) where
    canonical rasters and CSVs are published.
- **Keep only minimal test fixtures in this repo.**
  - Small (down-sampled) rasters and toy CSVs that allow tests and
    example notebooks to run quickly.
  - Use these fixtures under something like `tests/data` or
    `data/example_country`.

Implementation sketch:
- Introduce a simple data locator module (e.g. `masim_analysis.data_paths`)
  that resolves a logical dataset name to an on-disk path:
  - Looks in `DATA_ROOT` (env var) or a config file (e.g. `~/.masim/config.toml`).
  - Defaults to `./data` for backwards compatibility when present.
- Document a one-time setup command, e.g.
  - `masim data fetch moz` → downloads `data/moz` into the chosen root.
  - This can be a thin wrapper around `wget`/`curl` or `fsspec`.

Benefits:
- Core package stays light-weight.
- Data can be versioned independently (e.g. `v1`, `v2` of each country
  dataset) without rewriting code history.
- Easier to run the same code against multiple data snapshots.

### 1.2. Treat `conf/` as generated artifacts where possible

Current state:
- Many YAMLs under `conf/<country>` are either static scenario configs or
  auto-generated calibration/validation inputs.
- Generated YAMLs are committed, which both bloats git and makes it
  harder to see which parts are truly "source of truth" vs outputs.

Suggested path:
- **Treat most YAMLs as build products of `configure.py`.**
  - Only commit high-level, human-authored configuration specs, such as
    "baseline strategy" or small country parameter TOML/YAML files.
  - Generate full MaSim input YAMLs on demand via CLI or notebook code
    (`configure.configure`, calibration helpers, etc.).
- **Introduce an explicit `build/` or `generated/` area for YAMLs.**
  - For example: `generated/conf/<country>/calibration` and
    `generated/conf/<country>/validation`.
  - Add these to `.gitignore`.
- **Add a small CLI around `configure`**
  - e.g. `masim build-config <country> --scenario baseline` that writes
    the full YAMLs into `generated/`.

Benefits:
- Git history is dominated by small, human-readable parameter files.
- Fewer merge conflicts on bulky YAMLs.
- Clearer mental model of what is input vs. generated.

### 1.3. Explicit environment configuration

Suggested path:
- Standardize on a lightweight runtime config file, e.g.
  `~/.masim/config.toml` or `masim_config.toml` in the repo.
  - Keys like `masim_binary_path`, `data_root`, `output_root`,
    `log_root`.
  - Provide a helper `masim configure` command that creates/edits this
    file with sensible defaults.
- All code that currently hard-codes `./bin/MaSim`, `data/`, or
  `output/` should read from this configuration (with sensible
  fallbacks for backwards compatibility).

Benefits:
- New users and cluster environments can be configured once instead of
  editing many scripts.
- Easier to support multiple environments per user (e.g. dev vs. cluster).

---

## 2. Binaries and Packaging (MaSim, DxGGenerator)

### 2.1. Goal: installable Python + bundled C++ binaries

Target UX:
- `pip install masim-analysis` **or** `conda install masim-analysis`
  yields:
  - Python package (`masim_analysis`)
  - CLI entry points (`calibrate`, `masim`, `validate`, `commands`)
  - A platform-appropriate MaSim binary that tools can find without
    manual copying.

Constraints:
- MaSim itself is built in a separate repository.
- Binaries are platform-specific (Linux x86_64, maybe others later).

### 2.2. Option A: Conda package for MaSim + a thin Python dependency

Approach:
- Create a new conda package (e.g. `masim-core`) in a suitable channel
  (lab channel on Anaconda Cloud, or campus-internal).
  - Build scripts would clone/checkout the MaSim C++ repo, compile, and
    install binaries into `$PREFIX/bin`.
- Update `masim-analysis` to:
  - Depend on `masim-core` (conda) when installed via conda.
  - Look for the MaSim executable on `$PATH` (so `$(which MaSim)` works)
    first, then fall back to `./bin/MaSim` for legacy workflows.

Changes in this repo:
- Replace hard-coded `./bin/MaSim` strings with a small resolver,
  e.g. `masim_analysis.binary.resolve_masim_path()` which:
  1. Checks env var `MASIM_BINARY`.
  2. Looks up `masim_binary_path` in runtime config.
  3. Falls back to `shutil.which("MaSim")`.
  4. Finally checks `./bin/MaSim` for backwards compatibility.
- Document the dependency in the README and Overview.

Pros:
- Leverages conda’s strengths for binary distribution.
- Keeps this repo mostly Python-focused.

Cons:
- Users relying on pure pip environments still have to bring their own
  MaSim binary (although you can still point `MASIM_BINARY` at it).

### 2.3. Option B: Python wheel that vendors the binary

Approach:
- Build platform-tagged wheels (e.g.
  `masim_analysis-0.2.0-py3-none-manylinux_x86_64.whl`) that include:
  - `masim_analysis/` Python sources.
  - `masim_analysis/bin/MaSim` (and optionally `DxGGenerator`).
- Update commands that call `./bin/MaSim` to instead resolve a binary
  path inside the package using `importlib.resources` or
  `pkg_resources`.

Build outline:
- Add a small `meson`/`cmake`/`make` step to the wheel build that:
  - Fetches a tagged release of the MaSim C++ repo.
  - Builds the binary for the target platform.
  - Copies the resulting executable into `src/masim_analysis/bin/` as
    data files.
- Configure `hatchling` to include these binaries:
  - Use `tool.hatch.build.targets.wheel` with `include` rules for
    `masim_analysis/bin/*`.

Runtime changes:
- Create `masim_analysis/binaries.py` with helpers like
  `get_masim_executable()` returning an absolute path.
- Update `commands.generate_commands` to use that helper instead of
  hard-coded `./bin/MaSim`.

Pros:
- Pure `pip install` gives a ready-to-run environment (no separate
  conda package needed).

Cons:
- More complex CI/build matrix (per-platform builds).
- Binary size ends up inside every wheel (acceptable if a few MB, but
  worth tracking).

### 2.4. Hybrid model (recommended longer term)

- Short term: implement **binary path resolution** and support external
  MaSim installations.
- Medium term: build a **conda package** for MaSim for common lab
  platforms.
- Long term: explore **vendored binaries in wheels** for the
  most-used platforms if that simplifies installs for collaborators.

---

## 3. Experimentation & Reproducibility

### 3.1. Standard experiment metadata

Suggested path:
- For each calibration or validation run, write a small JSON/YAML
  metadata file alongside outputs, containing:
  - `git_commit` (from `git rev-parse HEAD` if available).
  - `masim_analysis_version` and `masim_binary_version`.
  - `data_version` (e.g. `moz_v1`, `ago_2024_11`).
  - CLI arguments or notebook parameters.
  - Time stamps and machine/cluster name.
- Introduce a helper such as
  `masim_analysis.utils.write_run_metadata(output_dir, **kwargs)` and
  call it from the main entry points.

Benefits:
- Easier to answer "what code + data produced this CSV/plot?".
- Enables automated comparisons across runs.

### 3.2. YAML 'DB' parameters to data classes

Suggested path:
- Introduce data classes (using `dataclasses` or `pydantic`) to model
  key configuration concepts such as:
  - Drug definitions (from `DRUG_DB` YAML).
  - Genotype properties (from `GENOTYPE_INFO` YAML).
- Add helpers to load these from YAML files into typed objects.
- Update `configure.py` to use these data classes internally rather
  than raw dicts where possible.
Benefits:
- Stronger validation of configuration inputs.
- Better type hints and IDE support when working with these concepts.
- Easier to extend with new fields in the future.
- Clearer documentation of what each config field means.

---

## 4. Testing, CI, and Code Quality

### 4.1. Test structure

Suggested path:
- Add `tests/` with at least:
  - Unit tests for pure-Python helpers (`configure`, `utils`,
    `commands`).
  - "Smoke" tests that run a tiny MaSim job (using a 5×5 raster, a
    small population, and 1–2 time steps) when a binary is available.
- Use small synthetic rasters/CSVs under `tests/data` rather than
  production inputs.

### 4.2. Continuous Integration

Suggested path:
- Set up GitHub Actions to run on pushes/PRs:
  - `python -m pip install -e .[interactive]`.
  - `pytest` (or `python -m pytest`).
  - `ruff` (already configured) or `python -m ruff check src`.
- Optionally: a separate workflow that runs the tiny MaSim smoke test
  only on Linux, skipping when no binary is available.

Benefits:
- Catches regressions in config generation and command building early.
- Gives confidence to refactor internals (e.g. data classes for
  `DRUG_DB`, `GENOTYPE_INFO`).

---

## 5. Developer & User Ergonomics

### 5.1. Command-line UX

- The existing entry points (`calibrate`, `validate`, `commands`,
  `masim`) are a good foundation.
- Consider adding:
  - `masim data list` – list installed countries and data versions.
  - `masim data fetch <country>` – download or sync data.
  - `masim config show` – print resolved paths and versions (binary,
    data root, output root).

### 5.2. Documentation

- Keep using `doc/Overview.md` as the main landing page; link to:
  - A short "Quick Start" for new users (install, configure, run one
    tiny validation example).
  - A page on "Data & Configuration Management" describing the split
    between code, configs, and external data.
  - Notes on how MaSim binaries are obtained (conda package vs. lab
    build scripts).

---

## 6. Adoption Plan (Incremental)

1. **Introduce path resolution utilities** for MaSim binary and data
   roots, while keeping backward-compatible defaults (`./bin/MaSim`,
   `./data`).
2. **Stop committing generated YAMLs and outputs** by adding
   appropriate patterns to `.gitignore` and adjusting notebooks/CLIs to
   regenerate them.
3. **Create a separate data storage location** (repo or bucket) and a
   small `masim data fetch` helper.
4. **Add minimal tests and CI** around configuration generation and
   command building.
5. **Prototype a conda `masim-core` package** that provides the MaSim
   binary, and update docs to prefer this installation path.

These changes should keep the package installable and light, while
reducing git bloat and making large-scale experimentation easier to
reproduce and extend.
