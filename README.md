# Hybrid Quantum-Classical Reservoir Computing

## Overview

This repository implements a hybrid quantum-classical reservoir computing (HRC) architecture for memory and nonlinear regression tasks. The architecture combines:

- a quantum reservoir computing module based on a disordered spin network with transverse field dynamics,
- a classical Echo State Network (ESN) readout layer,

This setup enables us to reproduce memory tasks that are nonlinear in the quantum input (density matrices composed of qubit states).

The code supports quantum-input tasks using density matrices and quantum state observables.

> Note Classical tasks can also be considered (e.g., NARMA, STM, and parity checking); however, they are not used in this implementation.

## Project Structure

- `source/` — Python package implementing QRC, ESN, HRC, and utilities.
- `scripts/` — entry-point scripts for running experiments and analyses.
- `configs/` — YAML configuration files for experiment setup.

## Key Components

### `source/`

- `base.py`
  - Defines `BaseSeededClass`, which sets deterministic random seeds and configures single-threaded BLAS/OpenMP behavior for reproducibility.

- `hamiltonian.py`
  - Implements `Hamiltonian`, a spin-chain reservoir with transverse magnetic field and local disorder.
  - Provides random Hamiltonian generation, eigenvalue decomposition, local spin operators, and two-spin correlation operators.

- `quantum_reservoir.py`
  - Implements `QuantumReservoirDynamics`, a quantum reservoir class that evolves density matrices, computes observables, and handles weak measurement back-action.
  - Supports quantum input injection and quantum task performance evaluation.

- `esn.py`
  - Implements `Esn` and `EsnDynamics`, a classical Echo State Network model.
  - Includes ESN weight initialization, activation functions, state evolution, and support for quantum-input preprocessing.

- `tasks.py`
  - Defines `Tasks`, which generates input signals, target outputs, and quantum observables for both classical and quantum tasks.
  - Includes task support for `NARMA`, `STM`, `PC`, and `Qinp` as well as performance metrics such as Capacity, NMSE, and fidelity-based evaluations.

- `hybrid.py`
  - Implements `HybridDynamics`, combining quantum reservoir observables with ESN readout in a sequential hybrid configuration.
  - Provides series hybrid performance evaluation across delay and parameter sweeps.

- `utils.py`
  - Utility functions for loading result files, quantum state generation, observable indexing, noise modeling, and density-matrix reconstruction.

## Scripts

- `scripts/gap_ratio.py`
  - Computes eigenvalue gap ratio statistics for the Hamiltonian to analyze ergodicity and spectral behavior.

- `scripts/esn_states.py`
  - Computes ESN state performance sweep results for classical or quantum input tasks.

- `scripts/qrc_observables.py`
  - Calculates QRC observables and performance metrics using configuration from `configs/config_qrc.yaml`.
  - Supports parameter sweeps and command-line overrides.

- `scripts/series_hybrid_performance.py`
  - Evaluates the performance of the HRC architecture (QRC+ESN) across delay and parameter sweeps using `configs/config_hybrid.yaml`.



## Configuration

The repository uses YAML configuration files under `configs/`:

- `configs/config_qrc.yaml` — QRC experiments and observable performance settings.
- `configs/config_hybrid.yaml` — HRC performance settings.

These files define the reservoir size, coupling strengths, delays, measurement settings, task names, and output storage options.

## Installation

Recommended: create a Python virtual environment before installing.

```bash
# Clone the repository (if not already done)
git clone <repo-url>
cd Hybrid-Quantum-Classical-Reservoir-Computing

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

> Note: Note: This setup has been tested on Linux. On Windows, some dependencies (e.g. QuSpin) may require additional build tools or a compatible environment (e.g. WSL).

## Usage

Run a QRC observables experiment:

```bash
python3 scripts/qrc_observables.py
```

Run hybrid HRC performance experiments:

```bash
python3 scripts/series_hybrid_performance.py
```

Run ESN performance sweeps:

```bash
pytho3 scripts/esn_states.py
```

Compute Hamiltonian gap ratio statistics:

```bash
python3 scripts/gap_ratio.py
```

### Example

#### Computing HRC Performance

To evaluate the HRC architecture on a specific quantum task with custom parameters:

```bash
# Precompute QRC observables (optional but recommended for faster runs)
python3 scripts/qrc_observables.py -qt Fidelity -ms 0.6 -ax zxy

# Compute HRC performance with measurement strength 0.6 and 5 virtual nodes per input
python3 scripts/series_hybrid_performance.py -qt Fidelity -ms 0.6 -Vmp 5
```

**Parameter explanations:**
- `-qt Fidelity`: Quantum task type (options: `Fidelity`, `Entropy`, `Entanglement`, etc.)
- `-ms 0.6`: Measurement strength (values between 0 and 1, requires `back_action: True` in config)
- `-Vmp 5`: Virtual nodes in QRC dynamics.
- `-ax zxy`: Measurement axis for observables (e.g., `x`, `y`, `z`, `zxy`)

**Workflow notes:**
- Precomputing observables with `qrc_observables.py` is optional but recommended—it significantly speeds up subsequent runs
- If precomputed observables are not found, `series_hybrid_performance.py` will compute QRC dynamics automatically (this can be computationally expensive for 9-qubit systems)
- All unspecified parameters use defaults from the corresponding YAML configuration file (`configs/config_qrc.yaml` or `configs/config_hybrid.yaml`)

#### Command-line options reference

Use `-h` or `--help` with any script for a full list of options:

```bash
python3 scripts/series_hybrid_performance.py -h
python3 scripts/qrc_observables.py -h
python3 scripts/esn_states.py -h
```

Common parameters across scripts:
- `-Vmp`, `--Vmp`: Virtual nodes per input
- `-dt`, `--dt`: Time step for dynamics
- `-ax`, `--axis`: Measurement or correlation axis
- `-ms`, `--meas-str`: Measurement strength


## Notes

- This repository contains the code used to generate the results presented in the associated research paper on quantum–classical reservoir computing.
- The project is developed in a physics research context, with emphasis on reproducibility and flexibility for numerical experiments rather than software engineering practices.
- Some scripts and functions are included for completeness or extended analyses and may not be directly used in the results reported in the paper.

## Citation

If you use this code in your research, please cite:

Coll-Comas, Mateu; Giorgi, Gian Luca; and Zambrini, Roberta, “Temporal processing of quantum states with hybrid quantum-classical reservoirs,” in submission (2026).

A full reference will be provided once the paper is published.
