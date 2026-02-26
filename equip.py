"""
EquiP: Equilibrium Plateau Pressure Predictor for Metal Hydrides
================================================================
A machine learning framework for predicting the natural log of equilibrium
plateau pressure [ln(Peq/MPa)] of metal hydrides as a function of temperature,
enabling rapid estimation of thermodynamic parameters (ΔH, ΔS) from Van't Hoff
analysis.

Model: Kernel Ridge Regression (KRR) with RBF kernel
Features: Temperature, elemental descriptors, and hydriding features
Validation: K-fold cross-validation and Leave-One-Composition-Out (LOCO)

Reference
---------
Verma, A.; Joshi, K. (2025). What drives property prediction for solid-state
hydrogen storage? Data or smart features? ChemRxiv.
https://doi.org/10.26434/chemrxiv-2025-9cvm9

Author: Ashwini D. Verma
Contact: ashwini.dverma@gmail.com
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EquiP")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
FIGURE_DPI = 150
TARGET_COLUMN = "ln_Peq"
TEMPERATURE_COLUMN = "Temperature_K"
COMPOSITION_COLUMN = "Composition"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ModelMetrics:
    """Container for regression performance metrics."""

    mae: float
    rmse: float
    r2: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"MAE  = {self.mae:.4f}\n"
            f"RMSE = {self.rmse:.4f}\n"
            f"R²   = {self.r2:.4f}\n"
            f"N    = {self.n_samples}"
        )


@dataclass
class LOCOResult:
    """Result for a single composition in LOCO validation."""

    composition: str
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: ModelMetrics


@dataclass
class EquiPConfig:
    """Hyperparameter and runtime configuration for EquiP."""

    # KRR hyperparameter search space
    alpha_range: list[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    )
    gamma_range: list[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    )
    kernel: str = "rbf"
    cv_folds: int = 5
    output_dir: Path = Path("output")
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_dataset(filepath: str | Path) -> pd.DataFrame:
    """Load the EquiP dataset from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV data file. Expected columns include the target
        ``ln_Peq``, ``Temperature_K``, ``Composition``, and feature columns.

    Returns
    -------
    pd.DataFrame
        Raw dataset.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns are absent.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info("Loaded dataset: %d rows × %d columns from '%s'", *df.shape, filepath)

    required = {TARGET_COLUMN, TEMPERATURE_COLUMN, COMPOSITION_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def split_features_target(
    df: pd.DataFrame,
    exclude_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate feature matrix, target vector, and composition labels.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including features, target, and metadata.
    exclude_cols : list of str, optional
        Additional non-feature columns to drop. ``Composition`` and
        ``ln_Peq`` are always excluded from the feature matrix.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target values (ln(Peq)).
    compositions : pd.Series
        Composition labels used for LOCO validation.
    """
    meta_cols = {TARGET_COLUMN, COMPOSITION_COLUMN}
    if exclude_cols:
        meta_cols.update(exclude_cols)

    X = df.drop(columns=list(meta_cols & set(df.columns)))
    y = df[TARGET_COLUMN]
    compositions = df[COMPOSITION_COLUMN]

    logger.info(
        "Feature matrix: %d samples × %d features | Target: '%s'",
        *X.shape,
        TARGET_COLUMN,
    )
    return X, y, compositions


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values.

    Returns
    -------
    ModelMetrics
    """
    return ModelMetrics(
        mae=mean_absolute_error(y_true, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=r2_score(y_true, y_pred),
        n_samples=len(y_true),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EquiPModel:
    """Kernel Ridge Regression model for ln(Peq) prediction.


    Parameters
    ----------
    config : EquiPConfig, optional
        Model configuration. Defaults to :class:`EquiPConfig` with standard
        hyperparameter grids.

    Examples
    --------
    >>> cfg = EquiPConfig(cv_folds=5, output_dir="results")
    >>> model = EquiPModel(cfg)
    >>> model.fit(X_train, y_train)
    >>> metrics = model.evaluate(X_test, y_test)
    """

    def __init__(self, config: Optional[EquiPConfig] = None) -> None:
        self.config = config or EquiPConfig()
        self.scaler = StandardScaler()
        self.krr: Optional[KernelRidge] = None
        self._is_fitted = False
        self.feature_names_: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def tune_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, float]:
        """Grid-search KRR hyperparameters using cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target values.

        Returns
        -------
        dict
            Best hyperparameters ``{"alpha": ..., "gamma": ...}``.
        """
        logger.info("Running hyperparameter grid search (CV=%d)...", self.config.cv_folds)

        param_grid = {
            "alpha": self.config.alpha_range,
            "gamma": self.config.gamma_range,
        }
        krr = KernelRidge(kernel=self.config.kernel)
        X_scaled = self.scaler.fit_transform(X)

        gs = GridSearchCV(
            krr,
            param_grid,
            cv=KFold(n_splits=self.config.cv_folds, shuffle=True,
                     random_state=self.config.random_state),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_scaled, y)

        best_params = gs.best_params_
        logger.info("Best hyperparameters: %s", best_params)
        return best_params

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        alpha: float = 0.01,
        gamma: float = 0.1,
    ) -> "EquiPModel":
        """Fit the KRR model on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target values.
        alpha : float
            KRR regularisation strength.
        gamma : float
            RBF kernel width parameter.

        Returns
        -------
        self : EquiPModel
        """
        self.feature_names_ = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        self.krr = KernelRidge(kernel=self.config.kernel, alpha=alpha, gamma=gamma)
        self.krr.fit(X_scaled, y)
        self._is_fitted = True

        logger.info(
            "Model fitted | kernel=%s  alpha=%.4g  gamma=%.4g",
            self.config.kernel, alpha, gamma,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ln(Peq) for new samples.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same columns used during :meth:`fit`.

        Returns
        -------
        np.ndarray
            Predicted ln(Peq) values.
        """
        self._check_is_fitted()
        X_scaled = self.scaler.transform(X)
        return self.krr.predict(X_scaled)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Compute metrics on a held-out set.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Ground-truth target values.

        Returns
        -------
        ModelMetrics
        """
        y_pred = self.predict(X)
        metrics = compute_metrics(np.array(y), y_pred)
        logger.info("Evaluation metrics:\n%s", metrics)
        return metrics

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, alpha: float, gamma: float
    ) -> ModelMetrics:
        """Perform K-fold cross-validation and report aggregate metrics.

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Full feature matrix and target.
        alpha, gamma : float
            KRR hyperparameters.

        Returns
        -------
        ModelMetrics
            Metrics aggregated across all folds.
        """
        kf = KFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        y_true_all, y_pred_all = [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = StandardScaler()
            krr = KernelRidge(kernel=self.config.kernel, alpha=alpha, gamma=gamma)
            krr.fit(scaler.fit_transform(X_tr), y_tr)
            y_pred = krr.predict(scaler.transform(X_val))

            y_true_all.extend(y_val.tolist())
            y_pred_all.extend(y_pred.tolist())

            fold_m = compute_metrics(np.array(y_val), y_pred)
            logger.debug("  Fold %d: RMSE=%.4f  R²=%.4f", fold, fold_m.rmse, fold_m.r2)

        metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
        logger.info("%d-fold CV metrics:\n%s", self.config.cv_folds, metrics)
        return metrics

    # ------------------------------------------------------------------
    # LOCO validation
    # ------------------------------------------------------------------

    def loco_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        compositions: pd.Series,
        alpha: float,
        gamma: float,
    ) -> list[LOCOResult]:
        """Leave-One-Composition-Out (LOCO) cross-validation.

        For each unique composition, the model is retrained on all *other*
        compositions and tested on the held-out composition. This protocol
        strictly evaluates generalisation to unseen alloy systems.

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Full feature matrix and target.
        compositions : pd.Series
            Composition labels aligned with *X* and *y*.
        alpha, gamma : float
            KRR hyperparameters.

        Returns
        -------
        list of LOCOResult
            One entry per unique composition.
        """
        unique_comps = compositions.unique()
        results: list[LOCOResult] = []

        logger.info("LOCO validation | %d compositions ...", len(unique_comps))

        for comp in unique_comps:
            mask = compositions == comp
            X_train = X[~mask]
            y_train = y[~mask]
            X_test = X[mask]
            y_test = y[mask]

            scaler = StandardScaler()
            krr = KernelRidge(kernel=self.config.kernel, alpha=alpha, gamma=gamma)
            krr.fit(scaler.fit_transform(X_train), y_train)
            y_pred = krr.predict(scaler.transform(X_test))

            metrics = compute_metrics(np.array(y_test), y_pred)
            results.append(
                LOCOResult(
                    composition=comp,
                    y_true=np.array(y_test),
                    y_pred=y_pred,
                    metrics=metrics,
                )
            )
            logger.debug(
                "  LOCO [%s]: MAE=%.4f  RMSE=%.4f  R²=%.4f",
                comp, metrics.mae, metrics.rmse, metrics.r2,
            )

        # Aggregate
        all_true = np.concatenate([r.y_true for r in results])
        all_pred = np.concatenate([r.y_pred for r in results])
        agg = compute_metrics(all_true, all_pred)
        logger.info("LOCO aggregate metrics:\n%s", agg)

        return results

    # ------------------------------------------------------------------
    # Thermodynamic extraction
    # ------------------------------------------------------------------

    def extract_vant_hoff(
        self,
        composition: str,
        temperatures: np.ndarray,
        X_template: pd.DataFrame,
    ) -> dict[str, float]:
        """Extract ΔH and ΔS via Van't Hoff analysis.

        Predicts ln(Peq) at each temperature in *temperatures*, then performs
        a linear regression of ln(Peq) vs. 1/T to extract:

        * ΔH = slope × R  [kJ mol⁻¹]
        * ΔS = –intercept × R  [J mol⁻¹ K⁻¹]

        Parameters
        ----------
        composition : str
            Alloy label (informational only).
        temperatures : np.ndarray
            Array of temperatures [K] at which to evaluate the model.
        X_template : pd.DataFrame
            Single-row feature template representing the composition. The
            ``Temperature_K`` column will be updated at each point.

        Returns
        -------
        dict with keys ``composition``, ``delta_H_kJ_mol``, ``delta_S_J_mol_K``,
        ``R2_vant_hoff``.
        """
        self._check_is_fitted()
        R = 8.314e-3  # kJ mol⁻¹ K⁻¹

        inv_T = 1.0 / temperatures
        ln_Peq_vals = []

        for T in temperatures:
            row = X_template.copy()
            row[TEMPERATURE_COLUMN] = T
            ln_Peq_vals.append(float(self.predict(row)[0]))

        ln_Peq_arr = np.array(ln_Peq_vals)

        # Linear fit: ln(Peq) = ΔH/R × (1/T) – ΔS/R
        coeffs = np.polyfit(inv_T, ln_Peq_arr, deg=1)
        slope, intercept = coeffs

        delta_H = slope * R          # kJ mol⁻¹
        delta_S = -intercept * R * 1e3  # J mol⁻¹ K⁻¹

        # Goodness of fit
        ln_Peq_fit = np.polyval(coeffs, inv_T)
        r2 = r2_score(ln_Peq_arr, ln_Peq_fit)

        logger.info(
            "Van't Hoff [%s]: ΔH = %.2f kJ/mol | ΔS = %.2f J/mol·K | R² = %.4f",
            composition, delta_H, delta_S, r2,
        )
        return {
            "composition": composition,
            "delta_H_kJ_mol": round(delta_H, 3),
            "delta_S_J_mol_K": round(delta_S, 3),
            "R2_vant_hoff": round(r2, 6),
        }


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str | Path) -> None:
        """Serialise the fitted model to disk using :mod:`joblib`.

        Parameters
        ----------
        filepath : str or Path
            Destination file path (e.g., ``model/equip_krr.joblib``).
        """
        import joblib
        self._check_is_fitted()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.krr, "scaler": self.scaler,
                     "features": self.feature_names_, "config": self.config},
                    filepath)
        logger.info("Model saved → %s", filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "EquiPModel":
        """Load a previously saved model.

        Parameters
        ----------
        filepath : str or Path
            Path to a ``.joblib`` file produced by :meth:`save`.

        Returns
        -------
        EquiPModel
        """
        import joblib
        data = joblib.load(filepath)
        instance = cls(config=data["config"])
        instance.krr = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names_ = data["features"]
        instance._is_fitted = True
        logger.info("Model loaded ← %s", filepath)
        return instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "The model has not been fitted yet. Call `.fit()` first."
            )


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

class EquiPPlotter:
    """Publication-quality plotting routines for EquiP results.

    Parameters
    ----------
    output_dir : str or Path
        Directory where figures are saved.
    dpi : int
        Output resolution (dots per inch). Default 150.
    """

    def __init__(
        self,
        output_dir: str | Path = "output",
        dpi: int = FIGURE_DPI,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
        })

    def parity_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: ModelMetrics,
        title: str = "Parity Plot",
        filename: str = "parity_plot.png",
    ) -> Path:
        """Plot predicted vs. experimental ln(Peq) values.

        Parameters
        ----------
        y_true, y_pred : array-like
            Ground-truth and predicted target values.
        metrics : ModelMetrics
            Metrics printed in the legend.
        title : str
            Axes title.
        filename : str
            Output filename relative to *output_dir*.

        Returns
        -------
        Path
            Absolute path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(5.5, 5.5))

        lims = [min(y_true.min(), y_pred.min()) - 0.5,
                max(y_true.max(), y_pred.max()) + 0.5]
        ax.plot(lims, lims, "k--", linewidth=1, label="Ideal")

        ax.scatter(
            y_true, y_pred,
            c="steelblue", edgecolors="white", linewidths=0.4,
            s=45, alpha=0.85,
            label=(
                f"KRR predictions\n"
                f"MAE  = {metrics.mae:.3f}\n"
                f"RMSE = {metrics.rmse:.3f}\n"
                f"R²   = {metrics.r2:.4f}"
            ),
        )

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Experimental ln(P$_{eq}$ / MPa)")
        ax.set_ylabel("Predicted ln(P$_{eq}$ / MPa)")
        ax.set_title(title)
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_aspect("equal")

        outpath = self.output_dir / filename
        fig.savefig(outpath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Parity plot saved → %s", outpath)
        return outpath


    def vant_hoff_plot(
        self,
        inv_T: np.ndarray,
        ln_Peq: np.ndarray,
        composition: str,
        delta_H: float,
        delta_S: float,
        filename: Optional[str] = None,
    ) -> Path:
        """Van't Hoff plot (ln Peq vs. 1/T) for a single composition.

        Parameters
        ----------
        inv_T : np.ndarray
            Inverse temperature values [1/K].
        ln_Peq : np.ndarray
            Predicted ln(Peq) values.
        composition : str
            Composition label used in the title.
        delta_H, delta_S : float
            Extracted thermodynamic parameters.
        filename : str, optional
            Output filename; defaults to ``vant_hoff_{composition}.png``.

        Returns
        -------
        Path
        """
        filename = filename or f"vant_hoff_{composition}.png"
        fig, ax = plt.subplots(figsize=(5.5, 4))

        ax.scatter(inv_T * 1e3, ln_Peq, c="firebrick", s=40, zorder=3)

        # Regression line
        coeffs = np.polyfit(inv_T, ln_Peq, deg=1)
        ax.plot(
            inv_T * 1e3, np.polyval(coeffs, inv_T),
            "k--", linewidth=1.2,
            label=(
                f"ΔH = {delta_H:.1f} kJ/mol\n"
                f"ΔS = {delta_S:.1f} J/mol·K"
            ),
        )

        ax.set_xlabel("1/T  (× 10³ K⁻¹)")
        ax.set_ylabel("ln(P$_{eq}$ / MPa)")
        ax.set_title(f"Van't Hoff Plot — {composition}")
        ax.legend()

        outpath = self.output_dir / filename
        fig.savefig(outpath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Van't Hoff plot saved → %s", outpath)
        return outpath

    def loco_bar(
        self,
        results: list[LOCOResult],
        metric: str = "rmse",
        filename: str = "loco_bar.png",
    ) -> Path:
        """Horizontal bar chart of per-composition LOCO errors.

        Parameters
        ----------
        results : list of LOCOResult
            Output of :meth:`EquiPModel.loco_validate`.
        metric : {'mae', 'rmse', 'r2'}
            Metric to visualise.
        filename : str
            Output filename.

        Returns
        -------
        Path
        """
        comps = [r.composition for r in results]
        values = [getattr(r.metrics, metric) for r in results]

        order = np.argsort(values)[::-1]
        comps_sorted = [comps[i] for i in order]
        vals_sorted = [values[i] for i in order]

        fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(comps))))
        ax.barh(comps_sorted, vals_sorted, color="steelblue", edgecolor="white")
        ax.set_xlabel(metric.upper())
        ax.set_title(f"LOCO Validation — {metric.upper()} per Composition")
        ax.invert_yaxis()

        outpath = self.output_dir / filename
        fig.savefig(outpath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("LOCO bar chart saved → %s", outpath)
        return outpath


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_loco_csv(results: list[LOCOResult], filepath: str | Path) -> None:
    """Export per-composition LOCO metrics to CSV.

    Parameters
    ----------
    results : list of LOCOResult
    filepath : str or Path
        Destination CSV path.
    """
    records = [
        {
            "composition": r.composition,
            "n_samples": r.metrics.n_samples,
            "mae": round(r.metrics.mae, 6),
            "rmse": round(r.metrics.rmse, 6),
            "r2": round(r.metrics.r2, 6),
        }
        for r in results
    ]
    df = pd.DataFrame(records).sort_values("rmse")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("LOCO results saved → %s", filepath)


def save_results_summary(
    cv_metrics: ModelMetrics,
    loco_results: list[LOCOResult],
    filepath: str | Path,
    config: EquiPConfig,
) -> None:
    """Write a human-readable model performance report.

    Parameters
    ----------
    cv_metrics : ModelMetrics
        Cross-validation aggregate metrics.
    loco_results : list of LOCOResult
        LOCO validation results.
    filepath : str or Path
        Output text file path.
    config : EquiPConfig
        Model configuration (logged for reproducibility).
    """
    loco_all_true = np.concatenate([r.y_true for r in loco_results])
    loco_all_pred = np.concatenate([r.y_pred for r in loco_results])
    loco_agg = compute_metrics(loco_all_true, loco_all_pred)

    lines = [
        "=" * 60,
        "  EquiP — Model Performance Report",
        "=" * 60,
        "",
        f"  Kernel       : {config.kernel.upper()}",
        f"  CV folds     : {config.cv_folds}",
        "",
        "  Cross-Validation (K-Fold)",
        "  " + "-" * 38,
        f"  MAE    = {cv_metrics.mae:.4f}",
        f"  RMSE   = {cv_metrics.rmse:.4f}",
        f"  R²     = {cv_metrics.r2:.4f}",
        f"  N      = {cv_metrics.n_samples}",
        "",
        "  Leave-One-Composition-Out (LOCO)",
        "  " + "-" * 38,
        f"  MAE    = {loco_agg.mae:.4f}",
        f"  RMSE   = {loco_agg.rmse:.4f}",
        f"  R²     = {loco_agg.r2:.4f}",
        f"  N      = {loco_agg.n_samples}",
        f"  Comps  = {len(loco_results)}",
        "",
        "=" * 60,
    ]

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines))
    logger.info("Summary saved → %s", filepath)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_equip_pipeline(
    data_path: str | Path,
    config: Optional[EquiPConfig] = None,
    tune: bool = True,
    run_loco: bool = True,
    run_shap: bool = True,
) -> dict:
    """End-to-end EquiP training and evaluation pipeline.

    This function orchestrates data loading, optional hyperparameter tuning,
    cross-validation, LOCO validation, SHAP analysis, and result export.

    Parameters
    ----------
    data_path : str or Path
        Path to the CSV dataset.
    config : EquiPConfig, optional
        Model and I/O configuration.
    tune : bool
        Whether to run grid-search hyperparameter tuning (recommended).
    run_loco : bool
        Whether to perform LOCO validation.
    run_shap : bool
        Whether to compute and save SHAP values.

    Returns
    -------
    dict
        Dictionary with keys ``"model"``, ``"cv_metrics"``,
        ``"loco_results"`` (if *run_loco*), and ``"shap"`` (if *run_shap*).
    """
    config = config or EquiPConfig()
    plotter = EquiPPlotter(output_dir=config.output_dir)
    output: dict = {}

    # 1. Load & split -------------------------------------------------------
    df = load_dataset(data_path)
    X, y, compositions = split_features_target(df)

    # 2. Hyperparameter tuning ----------------------------------------------
    model = EquiPModel(config)

    if tune:
        best_params = model.tune_hyperparameters(X, y)
        alpha, gamma = best_params["alpha"], best_params["gamma"]
    else:
        alpha, gamma = 0.01, 0.1
        logger.info("Using default hyperparameters: alpha=%s, gamma=%s", alpha, gamma)

    # 3. Cross-validation ---------------------------------------------------
    cv_metrics = model.cross_validate(X, y, alpha=alpha, gamma=gamma)
    output["cv_metrics"] = cv_metrics

    # 4. Fit final model on full dataset ------------------------------------
    model.fit(X, y, alpha=alpha, gamma=gamma)
    output["model"] = model

    # Parity plot (training fit — informational)
    y_pred_full = model.predict(X)
    full_metrics = compute_metrics(np.array(y), y_pred_full)
    plotter.parity_plot(
        np.array(y), y_pred_full, full_metrics,
        title="EquiP Parity Plot (Full Dataset Fit)",
        filename="parity_plot.png",
    )

    # 5. LOCO validation ----------------------------------------------------
    if run_loco:
        loco_results = model.loco_validate(X, y, compositions, alpha, gamma)
        output["loco_results"] = loco_results
        save_loco_csv(loco_results, config.output_dir / "loco_results.csv")
        plotter.loco_bar(loco_results, metric="rmse", filename="loco_bar.png")

        # Parity: LOCO predictions
        loco_true = np.concatenate([r.y_true for r in loco_results])
        loco_pred = np.concatenate([r.y_pred for r in loco_results])
        loco_agg = compute_metrics(loco_true, loco_pred)
        plotter.parity_plot(
            loco_true, loco_pred, loco_agg,
            title="EquiP Parity Plot (LOCO Validation)",
            filename="parity_plot_loco.png",
        )
        save_results_summary(cv_metrics, loco_results,
                             config.output_dir / "results_summary.txt", config)


    # 7. Save model ---------------------------------------------------------
    model.save(config.output_dir / "equip_model.joblib")

    logger.info("Pipeline complete. Outputs → %s", config.output_dir)
    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EquiP: Equilibrium Plateau Pressure Predictor for Metal Hydrides"
    )
    parser.add_argument(
        "--data", type=str, default="Data/EQUIP_Input.csv",
        help="Path to the input CSV dataset (default: Data/EQUIP_Input.csv)"
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Directory for all output files (default: output/)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip hyperparameter grid search and use defaults"
    )
    parser.add_argument(
        "--no-loco", action="store_true",
        help="Skip Leave-One-Composition-Out validation"
    )
    args = parser.parse_args()

    cfg = EquiPConfig(
        cv_folds=args.cv_folds,
        output_dir=Path(args.output),
    )

    run_equip_pipeline(
        data_path=args.data,
        config=cfg,
        tune=not args.no_tune,
        run_loco=not args.no_loco,
    )
