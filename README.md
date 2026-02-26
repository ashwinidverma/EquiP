# EquiP: Equilibrium Plateau Pressure Predictor for Metal Hydrides

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

A machine learning framework for predicting the natural log of equilibrium plateau pressure [**ln(P_eq / MPa)**] of metal hydrides as a function of temperature. EquiP generates complete **Van't Hoff plots** (ln P_eq vs. 1/T), enabling rapid estimation of:

- **ΔH** - enthalpy of hydride formation (kJ mol⁻¹)  
- **ΔS** - entropy of hydride formation (J mol⁻¹ K⁻¹)

without the need for expensive experimental measurements.

---

## Background

Metal hydrides are critical to technologies including hydrogen storage, compression, heat pumps, and catalysis. Their thermodynamic behaviour is captured by the Van't Hoff relation:

```
ln(Peq) = ΔH / (R·T) − ΔS / R
```

Determining P_eq experimentally across a range of temperatures is time-consuming. EquiP replaces this with a **Kernel Ridge Regression (KRR)** model trained on a curated dataset of 293 data points from 77 compositions, using domain-informed elemental and hydriding descriptors.

---

## Repository Structure

```
EquiP/
├── equip.py               # Main module: model, training, evaluation, plotting
├── run_demo.py            # Minimal usage example
├── Data/
│   ├── EQUIP_Input.csv        # Full dataset (293 samples, 77 compositions)
│   └── EQUIP_Input_Mg.csv     # Mg-based subset with XRD descriptors
├── output/                    # Generated automatically: figures, metrics, model
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ashwinidverma/EquiP.git
cd EquiP
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

**Command-line**

```bash
python equip.py --data Data/EQUIP_Input.csv --output output --cv-folds 5
```

**Python API**

```python
from equip import EquiPConfig, run_equip_pipeline

cfg = EquiPConfig(cv_folds=5, output_dir="output")
results = run_equip_pipeline("Data/EQUIP_Input.csv", config=cfg)

model   = results["model"]
metrics = results["cv_metrics"]
print(metrics)   # MAE, RMSE, R²
```

**CLI flags**

| Flag | Description |
|---|---|
| `--data PATH` | Path to input CSV (default: `Data/EQUIP_Input.csv`) |
| `--output DIR` | Output directory (default: `output/`) |
| `--cv-folds N` | Number of CV folds (default: 5) |
| `--no-tune` | Skip hyperparameter grid search |
| `--no-loco` | Skip LOCO validation |

---

## Model Details

| Component | Choice |
|---|---|
| Algorithm | Kernel Ridge Regression (KRR) |
| Kernel | Radial Basis Function (RBF) |
| Feature scaling | StandardScaler (zero mean, unit variance) |
| Hyperparameter search | 5-fold GridSearchCV over α ∈ [10⁻⁴, 10], γ ∈ [10⁻⁴, 1] |
| Validation | K-fold CV + Leave-One-Composition-Out (LOCO) |

**Feature set** includes: temperature (K), elemental descriptors, and structural features.

---

## Outputs

All outputs are written to the directory specified by `--output`:

| File | Description |
|---|---|
| `parity_plot.png` | Predicted vs. experimental ln(P_eq) - full-data fit |
| `parity_plot_loco.png` | Predicted vs. experimental ln(P_eq) - LOCO validation |
| `loco_bar.png` | Per-composition LOCO RMSE bar chart |
| `loco_results.csv` | Per-composition MAE, RMSE, R² from LOCO |
| `results_summary.txt` | Human-readable performance report |
| `equip_model.joblib` | Serialised fitted model (reload with `EquiPModel.load()`) |

---

## Reproducing the Paper Results

```bash
# Full dataset
python equip.py --data Data/EQUIP_Input.csv --output output


```

---

## Citation

```bibtex
@article{verma2025equip,
  title   = {What drives property prediction for solid-state hydrogen storage?
             Data or smart features?},
  author  = {Verma, Ashwini and Joshi, Kavita},
  journal = {ChemRxiv},
  year    = {2025},
  doi     = {10.26434/chemrxiv-2025-9cvm9}
}
```

---

## Contact

**Ashwini D. Verma**  
ashwini.dverma@gmail.com · [LinkedIn](https://www.linkedin.com/in/ashwinidverma/)

Issues and pull requests are welcome.
