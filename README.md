# munacco: Model for Uncertainty-Aware Capacity Calculation and Congestion Management

This repository contains the source code and analysis scripts for the **munacco** model, developed as part of the Master's thesis *"Uncertainty-Aware Capacity Calculation and Congestion Management"*.

The model simulates the sequential processes of capacity calculation, validation, market coupling, and redispatch, following the structure of the European Core Capacity Calculation Region (Core CCR). It includes support for both deterministic and uncertainty-aware formulations, including robust validation methods using chance constraints.

---

## Repository Structure

```
munacco/
├── munacco/                # Core Python model (modularized)
│   ├── analysis/           # KPI analysis and visualization
│   ├── input/              # Data loading from CSV/PyPSA
│   ├── model/              # Model logic (capacity calc, validation, MC, redispatch)
│   ├── scenario/           # Scenario generation and uncertainty handling
│   └── tools.py            # Shared utilities
│
├── example/                # Minimal example on a small 4-node test system
│   ├── 00_example_usage.py
│   └── base_model/         # CSV input files for the 4-node test case
│
├── thesis-analysis/        # Reproduction scripts for thesis experiments
│   ├── stylised-4node/     # Scripts for 4-node case study
│   └── pypsa-eur-50node/   # Scripts for 50-node PyPSA-Eur analysis
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository and install the dependencies via pip:

```bash
git clone https://github.com/mvoitl/munacco.git
cd munacco
pip install -r requirements.txt
```


## Data

The required input and result data is too large to be included in this repository. All datasets used in the thesis (including the PyPSA-Eur-50 network, forecast uncertainty profiles, and results) are available via Zenodo.

Download the data archive here:

**Zenodo DOI**: https://doi.org/10.5281/zenodo.17202843
**File**: `thesis-data.zip`

After downloading, extract the archive to recreate the expected folder structure:

```bash
unzip thesis-data.zip
```

This will create the following directory:

```
thesis-data/
├── stylised-4node/         # Input and results for the 4-node test case
├── pypsa-eur-50node/       # Data and results for the 50-node analysis
└── ...
```

All scripts in `thesis-analysis/` assume this structure.

---

## Getting Started

### Run example with 4-node system

```bash
python example/00_example_usage.py
```

This demonstrates a single capacity calculation and market coupling run using CSV input files.

---

### Reproduce stylised 4-node analysis

Navigate to the analysis folder:

```bash
cd thesis-analysis/stylised-4node
```

Run one of the scripts, for example:

```bash
python 02_Single_scenario_analysis.py
```

---

### Reproduce 50-node PyPSA-Eur case study

Run the full capacity calculation and validation across multiple hourly snapshots:

```bash
cd thesis-analysis/pypsa-eur-50node
python 03_pypsa_run_snapshots_w_uncertainty_high_low_res.py
```

This will process multiple hourly snapshots (with and without uncertainty), and save results to `thesis-data/pypsa-eur-50node/result_data`.

---



