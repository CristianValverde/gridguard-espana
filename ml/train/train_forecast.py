"""
ml/train/train_forecast.py — Entrena el forecaster de demanda eléctrica.

Modelo: XGBoost (GPU si disponible) con walk-forward y Optuna.
Salida:  ml/models/candidates/load_forecaster.pkl

Estrategia:
  - Walk-forward expansiva: 4 folds de validación (1 año cada uno)
  - Optuna TPE + MedianPruner: 100 trials
  - Objetivo: minimizar MAE medio sobre los 4 folds
  - Reentrenamiento final sobre todos los datos con mejores hiperparámetros
  - Intervalo de confianza: cuantiles 0.025 y 0.975 con XGBRegressor independiente

Features:
  - Temporales: hour_sin/cos, dow_sin/cos, month_sin/cos, is_weekend, is_holiday_es
  - Climáticas: aemet_ta (temperatura), aemet_hr (humedad) si disponibles
  - Lag: demanda en t-1h, t-2h, t-3h, t-24h, t-48h, t-168h (semana anterior)
  - Rolling: media móvil 24h y 168h
  - Generación: renewables_pct, composite_pressure

Target: demanda real en t+1h

Uso:
    python -m ml.train.train_forecast
    python -m ml.train.train_forecast --trials 50 --folds 3
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train.device_config import DEVICE, log_device_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("train_forecast")

DATA_PATH = Path(os.environ.get("DATA_PROCESSED_PATH", "data/processed")) / "dataset_training.parquet"
CANDIDATES_DIR = Path(os.environ.get("ML_MODELS_PATH", "ml/models")) / "candidates"

LAG_HOURS = [1, 2, 3, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]

BASE_FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend", "is_holiday_es",
    "renewables_pct", "composite_pressure",
]
OPTIONAL_COLS = ["aemet_ta", "aemet_hr"]

TARGET = "demand_real_mw"


# ---------------------------------------------------------------------------
# Preparación de features
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye lag features, rolling stats y concatena todo."""
    out = df[[c for c in BASE_FEATURE_COLS if c in df.columns]].copy()

    # Features opcionales de clima
    for col in OPTIONAL_COLS:
        if col in df.columns:
            out[col] = df[col]

    # Lag features del target
    if TARGET in df.columns:
        for lag in LAG_HOURS:
            out[f"demand_lag_{lag}h"] = df[TARGET].shift(lag)
        # Rolling stats sobre el target
        for w in ROLLING_WINDOWS:
            out[f"demand_roll_mean_{w}h"] = df[TARGET].shift(1).rolling(w, min_periods=w // 2).mean()
            out[f"demand_roll_std_{w}h"] = df[TARGET].shift(1).rolling(w, min_periods=w // 2).std()

    out[TARGET] = df[TARGET]  # target siempre al final
    out = out.dropna()
    return out


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------

def walk_forward_splits(index: pd.DatetimeIndex, n_folds: int = 4, val_months: int = 12):
    """Genera (train_mask, val_mask) para walk-forward expansiva."""
    total_months = (index[-1].year - index[0].year) * 12 + (index[-1].month - index[0].month)
    # Reservar el último fold como test hold-out (nunca usado en Optuna)
    usable_months = total_months - val_months
    fold_size = usable_months // (n_folds + 1)

    splits = []
    for fold in range(n_folds):
        val_end_offset = usable_months - fold * fold_size
        val_start_offset = val_end_offset - val_months

        cutoff_val_start = index[0] + pd.DateOffset(months=val_start_offset)
        cutoff_val_end = index[0] + pd.DateOffset(months=val_end_offset)

        train_mask = index < cutoff_val_start
        val_mask = (index >= cutoff_val_start) & (index < cutoff_val_end)

        if train_mask.sum() > 0 and val_mask.sum() > 0:
            splits.append((train_mask, val_mask))

    # Hold-out temporal: el último año (nunca visto durante Optuna)
    cutoff_holdout = index[-1] - pd.DateOffset(months=val_months)
    holdout_mask = index >= cutoff_holdout
    return splits[::-1], holdout_mask  # orden cronológico


# ---------------------------------------------------------------------------
# Optuna objective con walk-forward
# ---------------------------------------------------------------------------

def make_objective(df_feat: pd.DataFrame, feature_cols: list[str], splits):
    """Cierra sobre los datos y los folds. Devuelve la función objetivo para Optuna."""
    import xgboost as xgb

    xgb_device = "cuda" if (DEVICE is not None and str(DEVICE) == "cuda") else "cpu"

    def objective(trial) -> float:
        import optuna
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
            "device": xgb_device,
            "random_state": 42,
            "n_jobs": -1,
        }

        fold_maes = []
        for fold_idx, (train_mask, val_mask) in enumerate(splits):
            X_train = df_feat.loc[train_mask, feature_cols].values
            y_train = df_feat.loc[train_mask, TARGET].values
            X_val = df_feat.loc[val_mask, feature_cols].values
            y_val = df_feat.loc[val_mask, TARGET].values

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            mae = float(np.mean(np.abs(preds - y_val)))
            fold_maes.append(mae)

            # Pruning por MedianPruner tras el primer fold
            trial.report(np.mean(fold_maes), fold_idx)
            if trial.should_prune():
                raise __import__("optuna").exceptions.TrialPruned()

        return float(np.mean(fold_maes))

    return objective


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(n_trials: int = 100, n_folds: int = 4) -> None:
    import optuna
    import xgboost as xgb

    log_device_info()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {DATA_PATH}. "
            "Ejecutar: python -m ml.data.build_dataset"
        )

    log.info("Cargando dataset…")
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.DatetimeIndex(df.index, tz="UTC")

    log.info("Construyendo features…")
    df_feat = build_features(df)
    feature_cols = [c for c in df_feat.columns if c != TARGET]
    log.info("Features: %d | Filas: %d", len(feature_cols), len(df_feat))

    splits, holdout_mask = walk_forward_splits(df_feat.index, n_folds=n_folds)
    log.info("Walk-forward: %d folds de validación + 1 hold-out final", len(splits))

    # ── Optuna ──────────────────────────────────────────────────────────────
    log.info("Optimizando hiperparámetros XGBoost (%d trials)…", n_trials)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="load_forecast_xgb",
    )
    study.optimize(
        make_objective(df_feat, feature_cols, splits),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_params = study.best_params
    log.info("Mejores hiperparámetros: %s", best_params)
    log.info("Mejor MAE (walk-forward): %.1f MW", study.best_value)

    # ── Evaluación en hold-out ───────────────────────────────────────────────
    train_mask = ~holdout_mask
    X_final_train = df_feat.loc[train_mask, feature_cols].values
    y_final_train = df_feat.loc[train_mask, TARGET].values
    X_holdout = df_feat.loc[holdout_mask, feature_cols].values
    y_holdout = df_feat.loc[holdout_mask, TARGET].values

    xgb_device = "cuda" if (DEVICE is not None and str(DEVICE) == "cuda") else "cpu"
    final_model = xgb.XGBRegressor(**best_params, device=xgb_device, random_state=42, n_jobs=-1)
    final_model.fit(X_final_train, y_final_train, verbose=False)

    preds_holdout = final_model.predict(X_holdout)
    mae_holdout = float(np.mean(np.abs(preds_holdout - y_holdout)))
    mape_holdout = float(np.mean(np.abs((preds_holdout - y_holdout) / np.maximum(y_holdout, 1))) * 100)
    log.info("Holdout MAE: %.1f MW | MAPE: %.2f%%", mae_holdout, mape_holdout)

    # ── Guardar candidato ────────────────────────────────────────────────────
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_model,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "metrics": {"mae_holdout": mae_holdout, "mape_holdout": mape_holdout},
    }
    out_path = CANDIDATES_DIR / "load_forecaster.pkl"
    joblib.dump(bundle, out_path)
    log.info("Modelo guardado: %s", out_path)
    log.info("Siguiente paso: python -m ml.train.validate_models")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena forecaster de demanda eléctrica")
    p.add_argument("--trials", type=int, default=100, help="Trials Optuna")
    p.add_argument("--folds", type=int, default=4, help="Folds walk-forward")
    return p.parse_args()


if __name__ == "__main__":
    args = _args()
    run(n_trials=args.trials, n_folds=args.folds)
