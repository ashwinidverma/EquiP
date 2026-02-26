"""
run_demo.py — Minimal EquiP usage example
==========================================
Demonstrates the EquiP API for a professor or collaborator reviewing the code.
"""

import numpy as np

from equip import EquiPConfig, EquiPModel, EquiPPlotter, run_equip_pipeline


def demo_pipeline(data_path: str = "Data/EQUIP_Input.csv") -> None:
    """Run the full EquiP pipeline with default settings."""
    cfg = EquiPConfig(cv_folds=5, output_dir="output")
    results = run_equip_pipeline(data_path, config=cfg, tune=True,
                                 run_loco=True, run_shap=True)

    print("\n=== Cross-Validation Metrics ===")
    print(results["cv_metrics"])


def demo_vant_hoff(model: EquiPModel, X_template, composition: str) -> None:
    """Predict a Van't Hoff curve and extract thermodynamic parameters."""
    temperatures = np.linspace(300, 600, 20)  # K
    thermo = model.extract_vant_hoff(composition, temperatures, X_template)

    print(f"\nVan't Hoff — {composition}")
    print(f"  ΔH = {thermo['delta_H_kJ_mol']:.2f} kJ/mol")
    print(f"  ΔS = {thermo['delta_S_J_mol_K']:.2f} J/mol·K")
    print(f"  R² = {thermo['R2_vant_hoff']:.4f}")


def demo_load_saved_model() -> None:
    """Show how to reload a saved model and make predictions."""
    model = EquiPModel.load("output/equip_model.joblib")
    print("\nLoaded model from disk:")
    print("  Features:", model.feature_names_)


if __name__ == "__main__":
    demo_pipeline()
