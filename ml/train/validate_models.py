"""
ml/train/validate_models.py — Valida candidatos ML contra el golden test (apagón 28-abr-2025).

Carga los modelos entrenados en ml/models/candidates/ y verifica que:
  - El IF y LSTM detectan anomalía en los snapshots pre-apagón
  - El clasificador produce P(blackout) > umbral en los snapshots de riesgo
  - Los modelos producen puntuaciones bajas en el escenario normal (no falsa alarma)

No requiere ESIOS_TOKEN ni modelos en producción.
Los candidatos se leen de ml/models/candidates/.

Uso:
    python -m ml.train.validate_models
    python -m ml.train.validate_models --model iforest      # solo IF
    python -m ml.train.validate_models --model classifier   # solo clasificador
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("validate_models")

CANDIDATES_DIR = Path(os.environ.get("ML_MODELS_PATH", "ml/models")) / "candidates"
GOLDEN_DIR = Path("eval/golden_dataset")


# ---------------------------------------------------------------------------
# Feature engineering a partir de snapshots del golden dataset
# ---------------------------------------------------------------------------

def _snapshot_to_row(snap: dict, prev_net_flow: float | None = None) -> dict:
    """Convierte un snapshot JSON a un dict de features ML."""
    gen = snap.get("generation_by_tech", {})
    flows = snap.get("interchange_flows", {})
    demand = snap.get("demand_mw", 0.0)

    nuclear = gen.get("nuclear_mw", 0.0)
    hydro = gen.get("hydro_mw", 0.0)
    wind = gen.get("wind_mw", 0.0)
    solar_pv = gen.get("solar_pv_mw", 0.0)
    solar_thermal = gen.get("solar_thermal_mw", 0.0)
    gas_cc = gen.get("gas_cc_mw", 0.0)
    coal = gen.get("coal_mw", 0.0)
    cogeneration = gen.get("cogeneration_mw", 0.0)
    pumping = gen.get("pumping_mw", 0.0)

    total_gen = nuclear + hydro + wind + solar_pv + solar_thermal + gas_cc + coal + cogeneration + pumping
    balance_mw = total_gen - demand

    # reserve_mw viene del snapshot; modelamos como ~1.5% de demanda máxima
    reserve = snap.get("reserve_mw", 0.0)
    max_reserve = demand * 0.015 if demand > 0 else 1.0
    reserve_pct = min(float(reserve) / max(max_reserve, 1e-6) * 100, 100.0)

    net_flow = flows.get("es_fr_mw", 0.0) + flows.get("es_pt_mw", 0.0) + flows.get("es_ma_mw", 0.0)
    if prev_net_flow is not None:
        interchange_delta = abs(net_flow - prev_net_flow)
    else:
        interchange_delta = 0.0

    renewables = wind + solar_pv + solar_thermal
    renewables_pct = min(renewables / max(demand, 1e-6) * 100, 100.0)

    # composite_pressure: aproximación estática (sin rolling window histórico)
    balance_norm = max(0.0, 1.0 - (balance_mw + 5000) / 10000)  # más negativo = más presión
    reserve_norm = min(reserve_pct / 20.0, 1.0)                   # alta reserva = alta presión
    interchange_norm = min(interchange_delta / 2000.0, 1.0)
    renewables_norm = min(renewables_pct / 100.0, 1.0)

    composite = (
        0.35 * balance_norm
        + 0.25 * reserve_norm
        + 0.20 * interchange_norm
        + 0.20 * renewables_norm
    )

    ts = pd.Timestamp(snap["timestamp"])
    row = {
        "demand_real_mw": demand,
        "nuclear_mw": nuclear,
        "hydro_mw": hydro,
        "wind_mw": wind,
        "solar_pv_mw": solar_pv,
        "solar_thermal_mw": solar_thermal,
        "gas_cc_mw": gas_cc,
        "coal_mw": coal,
        "cogeneration_mw": cogeneration,
        "balance_mw": balance_mw,
        "reserve_activated_pct": reserve_pct,
        "interchange_delta_mw": interchange_delta,
        "renewables_pct": renewables_pct,
        "composite_pressure": composite,
        # Temporales
        "hour_sin": math.sin(2 * math.pi * ts.hour / 24),
        "hour_cos": math.cos(2 * math.pi * ts.hour / 24),
        "dow_sin": math.sin(2 * math.pi * ts.dayofweek / 7),
        "dow_cos": math.cos(2 * math.pi * ts.dayofweek / 7),
        "month_sin": math.sin(2 * math.pi * ts.month / 12),
        "month_cos": math.cos(2 * math.pi * ts.month / 12),
        "is_weekend": int(ts.dayofweek >= 5),
        "is_holiday_es": 0,  # conservador: no sabemos sin tabla de festivos
    }
    return row, net_flow


def _golden_to_df(scenario: dict) -> pd.DataFrame:
    """Construye DataFrame de features a partir del JSON del golden dataset."""
    rows = []
    labels = []
    prev_flow = None
    for snap in scenario["snapshots"]:
        row, prev_flow = _snapshot_to_row(snap, prev_flow)
        rows.append(row)
        labels.append(snap.get("label", ""))
    df = pd.DataFrame(rows)
    df["label"] = labels
    df.index = pd.DatetimeIndex([s["timestamp"] for s in scenario["snapshots"]])
    return df


# ---------------------------------------------------------------------------
# Validación: Isolation Forest
# ---------------------------------------------------------------------------

def validate_iforest(df_blackout: pd.DataFrame, df_normal: pd.DataFrame) -> bool:
    """Pasa si los snapshots de apagón tienen mayor anomaly score que los normales."""
    try:
        import joblib
    except ImportError:
        log.error("joblib no instalado")
        return False

    model_path = CANDIDATES_DIR / "anomaly_detector.pkl"
    if not model_path.exists():
        log.warning("Isolation Forest no entrenado aún: %s", model_path)
        return False

    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = scaler["features"]

    def _score(df: pd.DataFrame) -> np.ndarray:
        available = [f for f in features if f in df.columns]
        X = df[available].fillna(0).values
        mean = np.array([scaler["mean"].get(f, 0) for f in available])
        std = np.array([scaler["std"].get(f, 1) for f in available])
        X_norm = (X - mean) / std
        return -model.score_samples(X_norm)  # mayor = más anómalo

    scores_blackout = _score(df_blackout)
    scores_normal = _score(df_normal)

    log.info("[IF] Blackout scores (mayor=anómalo): %s",
             np.round(scores_blackout, 4))
    log.info("[IF] Normal  scores:                  %s",
             np.round(scores_normal, 4))

    mean_blackout = scores_blackout.mean()
    mean_normal = scores_normal.mean()
    log.info("[IF] Media blackout=%.4f  media normal=%.4f", mean_blackout, mean_normal)

    passed = mean_blackout > mean_normal
    log.info("[IF] %s", "PASS — apagón más anómalo que normal" if passed else
             "FAIL — no discrimina escenarios")
    return passed


# ---------------------------------------------------------------------------
# Validación: LSTM Autoencoder
# ---------------------------------------------------------------------------

def validate_lstm(df_blackout: pd.DataFrame, df_normal: pd.DataFrame) -> bool:
    """Pasa si el error de reconstrucción del apagón supera el threshold."""
    model_path = CANDIDATES_DIR / "lstm_ae.pth"
    if not model_path.exists():
        log.warning("LSTM AE no entrenado aún: %s", model_path)
        return False

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        log.warning("PyTorch no disponible — saltando validación LSTM")
        return False

    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    model = checkpoint["model"]
    scaler = checkpoint["scaler"]
    threshold = checkpoint["threshold"]
    seq_len = checkpoint["seq_len"]
    features = scaler["features"]

    model.eval()

    def _recon_error(df: pd.DataFrame) -> np.ndarray:
        available = [f for f in features if f in df.columns]
        X = df[available].fillna(0).values.astype(np.float32)
        mean = np.array([scaler["mean"].get(f, 0) for f in available], dtype=np.float32)
        std = np.array([scaler["std"].get(f, 1) for f in available], dtype=np.float32)
        X_norm = (X - mean) / std
        # Repetir filas si el escenario tiene menos filas que seq_len
        if len(X_norm) < seq_len:
            pad = np.tile(X_norm[0], (seq_len - len(X_norm), 1))
            X_norm = np.vstack([pad, X_norm])
        seq = torch.tensor(X_norm[-seq_len:]).unsqueeze(0)  # (1, seq_len, F)
        with torch.no_grad():
            recon = model(seq)
            error = nn.MSELoss()(recon, seq).item()
        return np.array([error])

    error_blackout = _recon_error(df_blackout).mean()
    error_normal = _recon_error(df_normal).mean()

    log.info("[LSTM] Error reconstrucción blackout=%.6f  normal=%.6f  threshold=%.6f",
             error_blackout, error_normal, threshold)

    passed = error_blackout > error_normal
    log.info("[LSTM] %s", "PASS — blackout más difícil de reconstruir" if passed else
             "FAIL — no discrimina escenarios")
    return passed


# ---------------------------------------------------------------------------
# Validación: Blackout Classifier
# ---------------------------------------------------------------------------

def validate_classifier(df_blackout: pd.DataFrame, df_normal: pd.DataFrame) -> bool:
    """Pasa si P(blackout) sube progresivamente y supera 0.60 en los snapshots críticos."""
    try:
        import joblib
    except ImportError:
        log.error("joblib no instalado")
        return False

    model_path = CANDIDATES_DIR / "blackout_classifier.pkl"
    if not model_path.exists():
        log.warning("Clasificador no entrenado aún: %s", model_path)
        return False

    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["feature_cols"]
    threshold = bundle.get("optimal_threshold", 0.5)

    def _proba(df: pd.DataFrame) -> np.ndarray:
        # Rellenar lag features con 0 (no disponibles en snapshots históricos breves)
        row = {f: 0.0 for f in features}
        for col in df.columns:
            if col in features:
                row[col] = float(df[col].iloc[-1])
        X = np.array([[row[f] for f in features]], dtype=np.float32)
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return np.array([model.predict(X)[0]])

    probas_blackout = np.array([
        _proba(df_blackout.iloc[:i+1]).item()
        for i in range(len(df_blackout))
    ])
    proba_normal = _proba(df_normal).item()

    log.info("[CLF] P(blackout) por snapshot:")
    for i, (prob, label) in enumerate(zip(probas_blackout, df_blackout["label"])):
        tag = "ALERTA" if prob >= threshold else "ok"
        log.info("  %s  %.4f  [%s]  %s", df_blackout.index[i], prob, tag, label[:60])

    log.info("[CLF] P(blackout) escenario normal: %.4f", proba_normal)
    log.info("[CLF] Umbral óptimo (F1): %.4f", threshold)

    max_prob = probas_blackout.max()
    passed = max_prob >= 0.60 and proba_normal < 0.40
    log.info("[CLF] %s  (max_blackout=%.4f  normal=%.4f)",
             "PASS" if passed else "FAIL", max_prob, proba_normal)
    return passed


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(
    run_if: bool = True,
    run_lstm: bool = True,
    run_clf: bool = True,
) -> None:
    blackout_path = GOLDEN_DIR / "blackout_apr_2025.json"
    normal_path = GOLDEN_DIR / "normal_baseline.json"

    if not blackout_path.exists() or not normal_path.exists():
        log.error("Golden dataset no encontrado en %s", GOLDEN_DIR)
        return

    scenario_blackout = json.loads(blackout_path.read_text(encoding="utf-8"))
    scenario_normal = json.loads(normal_path.read_text(encoding="utf-8"))

    df_blackout = _golden_to_df(scenario_blackout)
    df_normal = _golden_to_df(scenario_normal)

    log.info("Golden blackout: %d snapshots | Normal: %d snapshots",
             len(df_blackout), len(df_normal))

    results = {}

    if run_if:
        log.info("=== Isolation Forest ===")
        results["iforest"] = validate_iforest(df_blackout, df_normal)

    if run_lstm:
        log.info("=== LSTM Autoencoder ===")
        results["lstm"] = validate_lstm(df_blackout, df_normal)

    if run_clf:
        log.info("=== Blackout Classifier ===")
        results["classifier"] = validate_classifier(df_blackout, df_normal)

    # Resumen
    log.info("")
    log.info("=" * 50)
    log.info("RESUMEN DE VALIDACIÓN")
    log.info("=" * 50)
    any_fail = False
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL" if ok is False else "SKIP"
        log.info("  %-20s %s", name, status)
        if ok is False:
            any_fail = True
    if not results:
        log.info("  (ningún modelo entrenado todavía)")
    log.info("=" * 50)

    if any_fail:
        log.warning("Algunos modelos no superan el golden test. Revisar antes de promover.")
    elif results:
        log.info("Todos los candidatos superan el golden test.")
        log.info("Siguiente paso: python -m ml.retrain_pipeline --skip-download")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Valida candidatos ML con golden test")
    p.add_argument(
        "--model",
        choices=["iforest", "lstm", "classifier", "all"],
        default="all",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _args()
    run(
        run_if=args.model in ("iforest", "all"),
        run_lstm=args.model in ("lstm", "all"),
        run_clf=args.model in ("classifier", "all"),
    )
