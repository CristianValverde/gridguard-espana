"""
ml/train/train_classifier.py — Entrena el clasificador calibrado P(blackout_1h).

Modelo: XGBoost GPU + calibración isotónica.
Salida:  ml/models/candidates/blackout_classifier.pkl

Estrategia:
  - Etiquetado graduado: ml.incidents.build_graded_labels (ventanas por severidad, S3/S4)
  - Leave-One-Event-Out (LOEO): un fold por evento documentado (D4)
  - Optuna TPE + MedianPruner: 100 trials
  - Objetivo: maximizar AUC-ROC medio sobre los folds LOEO
  - Manejo de desbalance: scale_pos_weight + sample_weight por severidad + umbral óptimo vía F1
  - Calibración isotónica post-entrenamiento sobre hold-out temporal (trailing 12 meses)
  - Métrica final: AUC-ROC + Recall@Precision=0.7 (prioriza recall — perder un apagón
    es peor que una falsa alarma)

Uso:
    python -m ml.train.train_classifier
    python -m ml.train.train_classifier --trials 50
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.incidents import build_graded_labels
from ml.train.device_config import DEVICE, log_device_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("train_classifier")

DATA_PATH = Path(os.environ.get("DATA_PROCESSED_PATH", "data/processed")) / "dataset_training.parquet"
CANDIDATES_DIR = Path(os.environ.get("ML_MODELS_PATH", "ml/models")) / "candidates"

FEATURE_COLS = [
    "balance_mw", "reserve_activated_pct", "interchange_delta_mw",
    "renewables_pct", "composite_pressure",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend", "is_holiday_es",
    # Lag features de los proxies (detectan tendencias)
    "demand_real_mw",
]

LAG_FEATURES = ["balance_mw", "composite_pressure", "renewables_pct"]
LAG_HOURS = [1, 3, 6]


# ---------------------------------------------------------------------------
# Etiquetado, splits y preparación de datos
# ---------------------------------------------------------------------------

def leave_one_event_out_splits(
    index: pd.DatetimeIndex, labels: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validación dejando un evento documentado fuera cada vez (D4).

    Cada fold valida sobre la ventana de un evento (± 12 h de margen) y entrena con el resto.
    Garantiza ≥1 positivo en validación, evitando el AUC=0 de walk-forward con eventos raros.
    """
    from ml.incidents import documented_incidents_utc

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for inc in documented_incidents_utc():
        ts = inc["ts"]
        val_start = ts - pd.Timedelta(hours=inc["window_h"] + 12)
        val_end = ts + pd.Timedelta(hours=12)
        val_mask = np.asarray((index >= val_start) & (index <= val_end))
        train_mask = ~val_mask
        if labels[val_mask].sum() > 0 and labels[train_mask].sum() > 0:
            splits.append((train_mask, val_mask))
    return splits


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade lag features de los proxies de estabilidad."""
    out = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
    for col in LAG_FEATURES:
        if col in df.columns:
            for lag in LAG_HOURS:
                out[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return out.dropna()


def walk_forward_splits(index: pd.DatetimeIndex, n_folds: int = 4, val_months: int = 12):
    """Idéntico a train_forecast.py — walk-forward expansiva."""
    total_months = (index[-1].year - index[0].year) * 12 + (index[-1].month - index[0].month)
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

    cutoff_holdout = index[-1] - pd.DateOffset(months=val_months)
    holdout_mask = index >= cutoff_holdout
    return splits[::-1], holdout_mask


# ---------------------------------------------------------------------------
# Optuna objective con walk-forward y AUC-ROC
# ---------------------------------------------------------------------------

def make_objective(X_full: np.ndarray, y_full: np.ndarray, w_full: np.ndarray,
                   splits, scale_pos_weight: float):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    xgb_device = "cuda" if (DEVICE is not None and str(DEVICE) == "cuda") else "cpu"

    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
            # scale_pos_weight también como hiperparámetro (factor multiplicador sobre el base)
            "scale_pos_weight": trial.suggest_float(
                "spw_factor", 0.5, 2.0
            ) * scale_pos_weight,
            "device": xgb_device,
            "eval_metric": "auc",
            "random_state": 42,
        }

        fold_aucs = []
        for fold_idx, (train_mask, val_mask) in enumerate(splits):
            # train_mask/val_mask son booleanos alineados al índice de df_feat
            X_tr, y_tr, w_tr = X_full[train_mask], y_full[train_mask], w_full[train_mask]
            X_vl, y_vl = X_full[val_mask], y_full[val_mask]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            proba = model.predict_proba(X_vl)[:, 1]
            fold_aucs.append(roc_auc_score(y_vl, proba))

            trial.report(np.mean(fold_aucs), fold_idx)
            if trial.should_prune():
                raise __import__("optuna").exceptions.TrialPruned()

        return -float(np.mean(fold_aucs)) if fold_aucs else 0.0  # minimizar = maximizar AUC

    return objective


# ---------------------------------------------------------------------------
# Calibración isotónica
# ---------------------------------------------------------------------------

class _IsotonicCalibrated:
    """Thin wrapper: XGBoost base + isotonic regression sobre P(blackout)."""
    def __init__(self, base_model, ir):
        self._base = base_model
        self._ir = ir

    def predict_proba(self, X):
        raw = self._base.predict_proba(X)[:, 1]
        cal = self._ir.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def feature_importances_(self):
        return self._base.feature_importances_


def calibrate_model(base_model, X_cal: np.ndarray, y_cal: np.ndarray):
    """Calibración isotónica sin CalibratedClassifierCV (incompatible con cv='prefit' en sklearn≥1.8)."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    raw_proba = base_model.predict_proba(X_cal)[:, 1]
    brier_before = brier_score_loss(y_cal, raw_proba)
    log.info("Brier score antes de calibración: %.4f", brier_before)

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_proba, y_cal)

    cal_proba = ir.predict(raw_proba)
    brier_after = brier_score_loss(y_cal, cal_proba)
    log.info("Brier score después de calibración (isotónica): %.4f", brier_after)
    return _IsotonicCalibrated(base_model, ir)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(n_trials: int = 100) -> None:
    import optuna
    import xgboost as xgb
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve, roc_auc_score,
    )

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

    log.info("Construyendo features y etiquetas graduadas…")
    df_feat = build_features(df)
    labels, weights = build_graded_labels(df_feat.index)

    feature_cols = [c for c in df_feat.columns if c != "blackout_label"]
    X_all = df_feat[feature_cols].values.astype(np.float32)
    y_all = labels.reindex(df_feat.index).fillna(0).values.astype(int)
    w_all = weights.reindex(df_feat.index).fillna(1.0).values.astype(np.float32)

    n_neg = (y_all == 0).sum()
    n_pos = (y_all == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    log.info("Desbalance: %d neg / %d pos → scale_pos_weight base %.1f", n_neg, n_pos, scale_pos_weight)

    splits = leave_one_event_out_splits(df_feat.index, y_all)
    log.info("Leave-One-Event-Out: %d folds (uno por evento)", len(splits))
    if not splits:
        raise RuntimeError("Sin folds válidos — revisar ml/incidents.py y el rango del dataset")

    # Hold-out temporal: trailing 12 meses (independiente de LOEO)
    cutoff_holdout = df_feat.index[-1] - pd.DateOffset(months=12)
    holdout_mask = np.asarray(df_feat.index >= cutoff_holdout)

    # ── Optuna ──────────────────────────────────────────────────────────────
    log.info("Optimizando hiperparámetros (%d trials, objetivo: AUC-ROC)…", n_trials)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study = optuna.create_study(
        direction="minimize",  # minimizamos -AUC
        pruner=pruner,
        study_name="blackout_classifier",
    )
    study.optimize(
        make_objective(X_all, y_all, w_all, splits, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    raw_best = study.best_params
    log.info("Mejores hiperparámetros: %s", raw_best)
    log.info("Mejor AUC-ROC (LOEO): %.4f", -study.best_value)

    # Reconstruir best_params para XGBoost: spw_factor es un alias Optuna,
    # hay que convertirlo a scale_pos_weight real antes de pasarlo al modelo.
    best_params = {k: v for k, v in raw_best.items() if k != "spw_factor"}
    best_params["scale_pos_weight"] = raw_best.get("spw_factor", 1.0) * scale_pos_weight

    # ── Entrenamiento final (train) y calibración (holdout) ─────────────────
    train_mask = ~holdout_mask
    idx_positions = np.arange(len(df_feat.index))
    train_idx = idx_positions[train_mask]
    hold_idx = idx_positions[holdout_mask]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_holdout, y_holdout = X_all[hold_idx], y_all[hold_idx]

    xgb_device = "cuda" if (DEVICE is not None and str(DEVICE) == "cuda") else "cpu"
    final_model = xgb.XGBClassifier(
        **best_params, device=xgb_device, eval_metric="auc",
        random_state=42,
    )
    final_model.fit(X_train, y_train, sample_weight=w_all[train_idx], verbose=False)

    # Calibración isotónica sobre el hold-out temporal
    if y_holdout.sum() > 0:
        calibrated_model = calibrate_model(final_model, X_holdout, y_holdout)
    else:
        log.warning("Hold-out sin positivos — saltando calibración isotónica")
        calibrated_model = final_model

    # ── Métricas finales en hold-out ─────────────────────────────────────────
    proba_holdout = calibrated_model.predict_proba(X_holdout)[:, 1]
    if y_holdout.sum() > 0:
        auc = roc_auc_score(y_holdout, proba_holdout)
        ap = average_precision_score(y_holdout, proba_holdout)
        # Recall@Precision=0.7
        precision, recall, thresholds = precision_recall_curve(y_holdout, proba_holdout)
        recall_at_p70 = float(recall[precision >= 0.7][0]) if (precision >= 0.7).any() else 0.0
        log.info("Hold-out → AUC-ROC: %.4f | AP: %.4f | Recall@P=0.7: %.4f",
                 auc, ap, recall_at_p70)
        # Umbral óptimo: maximizar F1
        f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        opt_threshold = float(thresholds[np.argmax(f1_scores[:-1])])
        log.info("Umbral óptimo F1: %.4f", opt_threshold)
    else:
        auc, ap, recall_at_p70, opt_threshold = 0.0, 0.0, 0.0, 0.5
        log.warning("Hold-out sin positivos — métricas no calculables")

    # ── Feature importance ───────────────────────────────────────────────────
    importance = dict(zip(feature_cols, final_model.feature_importances_))
    top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    log.info("Top-5 features: %s", top5)

    # ── Guardar candidato ────────────────────────────────────────────────────
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": calibrated_model,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "optimal_threshold": opt_threshold,
        "scale_pos_weight": scale_pos_weight,
        "metrics": {
            "auc_roc_holdout": auc,
            "avg_precision_holdout": ap,
            "recall_at_precision_70": recall_at_p70,
        },
        "feature_importance": importance,
    }
    out_path = CANDIDATES_DIR / "blackout_classifier.pkl"
    joblib.dump(bundle, out_path)
    log.info("Modelo calibrado guardado: %s", out_path)
    log.info("Siguiente paso: python -m ml.train.validate_models")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena clasificador de riesgo de apagón")
    p.add_argument("--trials", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = _args()
    run(n_trials=args.trials)
