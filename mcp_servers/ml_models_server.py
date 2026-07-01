"""
mcp_servers/ml_models_server.py — MCP: ml-models

Expone los 3 modelos ML de GridGuard como herramientas MCP.
Carga los pesos en memoria al arrancar. Inferencia determinística, sin LLM.
Soporta hot-swap atómico de modelos (para el pipeline de reentrenamiento mensual).

Modelos:
  - Anomaly Detector:      Isolation Forest (sklearn) o LSTM Autoencoder (PyTorch)
  - Load Forecaster:       XGBoost + Prophet
  - Blackout Classifier:   Gradient Boosting calibrado (XGBoost)

Herramientas expuestas:
  get_anomaly_score(window_json)               -> AnomalyResult (JSON)
  get_load_forecast(horizon_hours, context)    -> LoadForecast (JSON)
  get_blackout_probability(features_json)      -> BlackoutRisk (JSON)
  reload_models()                              -> {"ok": true, "models": [...]}

Ruta de modelos: variable de entorno ML_MODELS_PATH (default: ./ml/models)
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from schemas.models import AnomalyResult, BlackoutRisk, LoadForecast

mcp = FastMCP("gridguard-ml-models")

_DEFAULT_MODELS_PATH = "./ml/models"
_lock = threading.Lock()

# Modelos en memoria — swap atómico protegido por _lock
_models: dict[str, Any] = {
    "iforest": None,      # Isolation Forest (sklearn)
    "lstm_ae": None,      # LSTM Autoencoder (PyTorch) — opcional
    "xgb_forecast": None, # XGBoost cargador de demanda
    "xgb_blackout": None, # XGBoost clasificador de apagón
    "device": "cpu",      # "cuda" o "cpu"
}
_load_status: dict[str, str] = {}  # model_name -> "ok" | "missing" | "error"


def _models_dir() -> Path:
    return Path(os.environ.get("ML_MODELS_PATH", _DEFAULT_MODELS_PATH))


def _load_all_models() -> None:
    """Carga (o recarga) todos los modelos desde disco. Thread-safe."""
    import importlib

    new_models: dict[str, Any] = {}
    status: dict[str, str] = {}
    models_dir = _models_dir()

    # ── Isolation Forest (sklearn) ──────────────────────────────────────────
    iforest_path = models_dir / "anomaly_detector.pkl"
    if iforest_path.exists():
        try:
            import joblib
            new_models["iforest"] = joblib.load(iforest_path)
            status["iforest"] = "ok"
        except Exception as exc:
            status["iforest"] = f"error: {exc}"
    else:
        status["iforest"] = "missing"

    # ── LSTM Autoencoder (PyTorch) — opcional ───────────────────────────────
    lstm_path = models_dir / "lstm_ae.pth"
    if lstm_path.exists():
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # El modelo se guarda como state_dict; aquí cargamos el checkpoint completo
            # El archivo debe contener {"model": model_instance, "threshold": float}
            checkpoint = torch.load(lstm_path, map_location=device, weights_only=False)
            new_models["lstm_ae"] = checkpoint.get("model")
            new_models["lstm_threshold"] = checkpoint.get("threshold", 0.5)
            new_models["device"] = device
            status["lstm_ae"] = "ok"
        except Exception as exc:
            status["lstm_ae"] = f"error: {exc}"
    else:
        status["lstm_ae"] = "missing"

    # ── XGBoost Load Forecaster ─────────────────────────────────────────────
    xgb_forecast_path = models_dir / "load_forecaster.pkl"
    if xgb_forecast_path.exists():
        try:
            import joblib
            new_models["xgb_forecast"] = joblib.load(xgb_forecast_path)
            status["xgb_forecast"] = "ok"
        except Exception as exc:
            status["xgb_forecast"] = f"error: {exc}"
    else:
        # También intentar formato nativo XGBoost .json
        xgb_json_path = models_dir / "xgb_forecast.json"
        if xgb_json_path.exists():
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor()
                model.load_model(xgb_json_path)
                new_models["xgb_forecast"] = model
                status["xgb_forecast"] = "ok"
            except Exception as exc:
                status["xgb_forecast"] = f"error: {exc}"
        else:
            status["xgb_forecast"] = "missing"

    # ── XGBoost Blackout Classifier ─────────────────────────────────────────
    xgb_blackout_path = models_dir / "blackout_classifier.pkl"
    if xgb_blackout_path.exists():
        try:
            import joblib
            new_models["xgb_blackout"] = joblib.load(xgb_blackout_path)
            status["xgb_blackout"] = "ok"
        except Exception as exc:
            status["xgb_blackout"] = f"error: {exc}"
    else:
        xgb_json_path = models_dir / "xgb_blackout.json"
        if xgb_json_path.exists():
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(xgb_json_path)
                new_models["xgb_blackout"] = model
                status["xgb_blackout"] = "ok"
            except Exception as exc:
                status["xgb_blackout"] = f"error: {exc}"
        else:
            status["xgb_blackout"] = "missing"

    with _lock:
        _models.update(new_models)
        _load_status.update(status)


# Carga inicial al arrancar el servidor
_load_all_models()


# ---------------------------------------------------------------------------
# Herramientas MCP
# ---------------------------------------------------------------------------

@mcp.tool()
def get_anomaly_score(window_json: str) -> dict:
    """Calcula anomaly score para una ventana de snapshots de red (6h típico).

    Usa Isolation Forest (siempre disponible) o LSTM Autoencoder si está entrenado.
    Devuelve score [0-1] y las features más anómalas.

    Args:
        window_json: JSON array de GridSnapshot (últimas N horas)
    """
    import numpy as np

    with _lock:
        iforest = _models.get("iforest")
        lstm = _models.get("lstm_ae")
        lstm_threshold = _models.get("lstm_threshold", 0.5)

    try:
        window = json.loads(window_json)
        if not window:
            return {"error": "Ventana vacía"}

        # Construir matriz de features a partir de los snapshots
        feature_names = [
            "demand_mw", "reserve_mw",
            "nuclear_mw", "hydro_mw", "wind_mw", "solar_pv_mw",
            "solar_thermal_mw", "gas_cc_mw", "coal_mw",
        ]

        rows = []
        for snap in window:
            gen = snap.get("generation_by_tech", {})
            row = [
                snap.get("demand_mw", 0) or 0,
                snap.get("reserve_mw", 0) or 0,
                gen.get("nuclear_mw", 0) or 0,
                gen.get("hydro_mw", 0) or 0,
                gen.get("wind_mw", 0) or 0,
                gen.get("solar_pv_mw", 0) or 0,
                gen.get("solar_thermal_mw", 0) or 0,
                gen.get("gas_cc_mw", 0) or 0,
                gen.get("coal_mw", 0) or 0,
            ]
            rows.append(row)

        X = np.array(rows, dtype=float)

        # IF y LSTM se calculan como señales DISTINTAS (ya no son alternativos):
        # el ensemble (Task 8) las fusiona. Ambos corren si están disponibles.
        if_score = 0.0
        lstm_recon_norm = 0.0
        is_anomaly = False
        top_features: list = []
        feature_scores: dict = {}
        model_types: list = []

        # ── Isolation Forest (anomalía puntual) ─────────────────────────────
        if iforest is not None:
            scores_raw = -iforest.score_samples(X)  # más alto = más anómalo
            if_score = float(np.clip(
                (scores_raw.mean() - (-0.5)) / (0.5 - (-0.5)), 0, 1
            ))
            is_anomaly = is_anomaly or bool(
                iforest.predict(X[-1:] if len(X) > 0 else X)[0] == -1
            )
            model_types.append("isolation_forest")
            last = X[-1]
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-8
            deviations = np.abs((last - mean) / std)
            top_idx = deviations.argsort()[-3:][::-1]
            top_features = [feature_names[i] for i in top_idx if i < len(feature_names)]
            feature_scores = {
                feature_names[i]: round(float(deviations[i]), 3)
                for i in range(len(feature_names))
            }

        # ── LSTM Autoencoder (reconstrucción temporal) ──────────────────────
        if lstm is not None:
            try:
                import torch
                device = _models.get("device", "cpu")
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
                lstm.eval()
                with torch.no_grad():
                    reconstructed = lstm(X_tensor)
                    mse = float(torch.mean((X_tensor - reconstructed) ** 2).item())
                lstm_recon_norm = float(min(mse / lstm_threshold, 1.0))
                is_anomaly = is_anomaly or (mse > lstm_threshold)
                model_types.append("lstm_ae")
                feature_scores["reconstruction_mse"] = round(mse, 6)
            except Exception as exc:
                return {"error": f"Error LSTM inference: {exc}"}

        if not model_types:
            return {
                "error": (
                    "Modelos no entrenados. Ejecutar: "
                    "python -m ml.train.train_anomaly"
                )
            }

        # anomaly_score legacy = la señal más fuerte disponible (compatibilidad)
        anomaly_score = max(if_score, lstm_recon_norm)
        result = AnomalyResult(
            anomaly_score=round(anomaly_score, 4),
            is_anomaly=is_anomaly,
            top_features=top_features,
            feature_scores=feature_scores,
            model_type="+".join(model_types),
            inference_time_ms=0.0,
        ).model_dump(mode="json")
        # Señales distintas para el ensemble:
        result["if_score"] = round(if_score, 4)
        result["lstm_recon_norm"] = round(lstm_recon_norm, 4)
        return result

    except Exception as exc:
        return {"error": f"get_anomaly_score falló: {exc}"}


@mcp.tool()
def get_ensemble_score(features_json: str, window_json: str) -> dict:
    """Fusiona LSTM + IF + clasificador en una P(blackout) única (riesgo principal).

    El LSTM (no supervisado, inmune a D5) es el detector load-bearing; el IF y el
    clasificador corroboran. Ver tools/ensemble_scorer.py.

    Args:
        features_json: JSON con la fila de features del clasificador (FEATURE_COLS,
            idealmente de ml.serve_features.serve_feature_row para paridad train/serve).
        window_json:   JSON array de snapshots (secuencia para IF/LSTM).

    NOTA de paridad: la construcción del vector de features de get_anomaly_score y
    get_blackout_probability debe alinearse con ml.serve_features durante la
    integración [PADRE] (Task 10), ya que los modelos re-entrenados usan
    ml.features.FEATURE_COLS + el scaler del bundle.
    """
    from tools.ensemble_scorer import fuse

    anomaly = get_anomaly_score(window_json)
    clf = get_blackout_probability(features_json)
    return fuse(
        lstm_recon_norm=float(anomaly.get("lstm_recon_norm", 0.0) or 0.0),
        if_score=float(anomaly.get("if_score", 0.0) or 0.0),
        clf_proba=float(clf.get("blackout_probability", 0.0) or 0.0),
    )


@mcp.tool()
def get_serve_features(window_hours: int = 168) -> dict:
    """Fila de features de serve (FEATURE_COLS) con paridad train/serve.

    Lee las últimas `window_hours` de la tabla `snapshots` (pobladas por
    ml.data.incremental_load.catch_up) y aplica ml.serve_features. Base para el
    ensemble; el rolling queda activo si hay ≥168h de historia.
    """
    import pandas as pd
    from mcp_servers import memory_server
    from ml.serve_features import db_rows_to_snapshots, serve_feature_row

    with memory_server._conn() as con:
        rows = con.execute(
            "SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT ?",
            (window_hours + 1,),
        ).fetchall()
    if not rows:
        return {"error": "Sin snapshots en la DB. Ejecutar: python -m ml.data.incremental_load"}

    snaps = db_rows_to_snapshots(list(reversed(rows)))
    now = pd.Timestamp(snaps[-1]["timestamp"])
    return serve_feature_row(snaps, now)


@mcp.tool()
def get_blackout_probability_ensemble() -> dict:
    """P(blackout) del ensemble (LSTM load-bearing + IF + CLF) con paridad
    train/serve. Pipeline 100% determinista (shift-left): lee la ventana 168h de
    la DB, calcula las features de serve y fusiona los tres modelos. Sustituye a
    get_blackout_probability como fuente del ml_score del orchestrator.
    """
    import json as _json
    from mcp_servers import memory_server
    from ml.serve_features import db_rows_to_snapshots

    feats = get_serve_features()
    if "error" in feats:
        return feats

    with memory_server._conn() as con:
        rows = con.execute(
            "SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT 168"
        ).fetchall()
    window = db_rows_to_snapshots(list(reversed(rows)))
    return get_ensemble_score(_json.dumps(feats), _json.dumps(window))


@mcp.tool()
def get_load_forecast(horizon_hours: int = 6, context_json: str = "{}") -> dict:
    """Predice la demanda eléctrica para las próximas N horas.

    Args:
        horizon_hours: Horas a predecir (1-24)
        context_json:  JSON con features de contexto (hora, temperatura, festivo...)
    """
    import numpy as np

    with _lock:
        model = _models.get("xgb_forecast")

    if model is None:
        return {
            "error": (
                "Modelo load_forecaster no entrenado. Ejecutar: "
                "python -m ml.train.train_forecast"
            )
        }

    try:
        horizon_hours = max(1, min(horizon_hours, 24))
        context = json.loads(context_json) if context_json else {}

        now = datetime.now(timezone.utc)
        timestamps = [now + timedelta(hours=i) for i in range(1, horizon_hours + 1)]

        # Features para cada hora de predicción (simplificado — el modelo real
        # usaría lag features del histórico)
        import math
        forecasts: list[float] = []
        for ts in timestamps:
            features = np.array([[
                math.sin(2 * math.pi * ts.hour / 24),
                math.cos(2 * math.pi * ts.hour / 24),
                math.sin(2 * math.pi * ts.weekday() / 7),
                math.cos(2 * math.pi * ts.weekday() / 7),
                math.sin(2 * math.pi * ts.month / 12),
                math.cos(2 * math.pi * ts.month / 12),
                float(ts.weekday() >= 5),                 # is_weekend
                float(context.get("is_holiday", 0)),
                float(context.get("temperature_c", 18)),
            ]])
            pred = float(model.predict(features)[0])
            forecasts.append(max(pred, 0.0))

        # Intervalo de confianza ±5% (simplificado; el modelo real usa quantile regression)
        lower = [round(f * 0.95, 1) for f in forecasts]
        upper = [round(f * 1.05, 1) for f in forecasts]

        return LoadForecast(
            horizon_hours=horizon_hours,
            timestamps=timestamps,
            demand_mw=[round(f, 1) for f in forecasts],
            lower_bound_mw=lower,
            upper_bound_mw=upper,
            confidence_pct=95.0,
            model_type="xgboost",
        ).model_dump(mode="json")

    except Exception as exc:
        return {"error": f"get_load_forecast falló: {exc}"}


@mcp.tool()
def get_blackout_probability(features_json: str) -> dict:
    """Calcula P(blackout en próxima hora) usando el clasificador calibrado.

    Args:
        features_json: JSON con StabilityProxies + LoadForecast summary:
            {
              "balance_mw": float,
              "reserve_activated_pct": float,
              "interchange_delta_mw": float,
              "renewables_pct": float,
              "composite_pressure": float,
              "ml_forecast_mw": float,     # demanda prevista próxima hora
              "anomaly_score": float        # del get_anomaly_score anterior
            }
    """
    import numpy as np

    with _lock:
        model = _models.get("xgb_blackout")

    if model is None:
        return {
            "error": (
                "Modelo blackout_classifier no entrenado. Ejecutar: "
                "python -m ml.train.train_classifier"
            )
        }

    try:
        feats = json.loads(features_json)

        X = np.array([[
            feats.get("balance_mw", 0),
            feats.get("reserve_activated_pct", 0),
            feats.get("interchange_delta_mw", 0),
            feats.get("renewables_pct", 0),
            feats.get("composite_pressure", 0),
            feats.get("ml_forecast_mw", 30000),
            feats.get("anomaly_score", 0),
        ]])

        proba = float(model.predict_proba(X)[0][1])  # P(clase=1 → blackout)

        if proba < 0.10:
            level = "LOW"
        elif proba < 0.30:
            level = "MEDIUM"
        elif proba < 0.60:
            level = "HIGH"
        else:
            level = "CRITICAL"

        # Feature importance del modelo XGBoost
        feature_names = [
            "balance_mw", "reserve_activated_pct", "interchange_delta_mw",
            "renewables_pct", "composite_pressure", "ml_forecast_mw", "anomaly_score",
        ]
        importance = {}
        if hasattr(model, "feature_importances_"):
            for name, imp in zip(feature_names, model.feature_importances_):
                importance[name] = round(float(imp), 4)

        return BlackoutRisk(
            probability=round(proba, 4),
            risk_level=level,
            feature_importance=importance,
            calibration_method="isotonic_regression",
        ).model_dump(mode="json")

    except Exception as exc:
        return {"error": f"get_blackout_probability falló: {exc}"}


@mcp.tool()
def reload_models() -> dict:
    """Recarga todos los modelos desde disco sin downtime (hot-swap atómico).

    Llamado por reload_mcp_server.py tras el reentrenamiento mensual.
    """
    try:
        _load_all_models()
        return {"ok": True, "models": _load_status, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    mcp.run()
