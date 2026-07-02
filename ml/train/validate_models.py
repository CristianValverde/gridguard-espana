"""
ml/train/validate_models.py — Valida candidatos ML contra el golden test (apagón 28-abr-2025).

Carga los modelos entrenados en ml/models/candidates/ y verifica que:
  - El IF y LSTM detectan anomalía en los snapshots pre-apagón
  - El clasificador produce P(blackout) > umbral en los snapshots de riesgo
  - El ensemble fusionado discrimina golden vs normal y crece hacia el onset
  - Los modelos producen puntuaciones bajas en el escenario normal (no falsa alarma)

Fin del skew de validación: cada snapshot del golden se evalúa con su ventana
trailing REAL (la semana previa extraída del dataset ESIOS) vía ml.serve_features,
de modo que rolling/z-scores/lags NO van a cero (misma canalización que serve).

No requiere ESIOS_TOKEN. Requiere data/processed/dataset_training.parquet.

Uso:
    python -m ml.train.validate_models
    python -m ml.train.validate_models --model iforest      # solo IF
    python -m ml.train.validate_models --models-dir ml/models  # validar producción
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ml import features as F
from ml import serve_features as SF
from tools.ensemble_scorer import fuse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("validate_models")

CANDIDATES_DIR = Path(os.environ.get("ML_MODELS_PATH", "ml/models")) / "candidates"
GOLDEN_DIR = Path("eval/golden_dataset")
DATASET_PATH = Path(os.environ.get("DATA_PROCESSED_PATH", "data/processed")) / "dataset_training.parquet"

_LINKS = ("es_fr_mw", "es_pt_mw", "es_ma_mw")


# ---------------------------------------------------------------------------
# Contexto trailing real + features por snapshot (paridad con serve)
# ---------------------------------------------------------------------------

def _dataset_rows_to_snapshots(df: pd.DataFrame) -> list[dict]:
    """Convierte filas raw del dataset ESIOS en snapshots dict (esquema DB)."""
    def _f(v) -> float:
        return float(v) if pd.notna(v) else 0.0

    snaps = []
    for ts, row in df.iterrows():
        snaps.append({
            "timestamp": ts.isoformat(),
            "demand_mw": _f(row.get("demand_real_mw")),
            "generation_by_tech": {c: _f(row.get(c)) for c in F.GEN_COLS},
            "interchange_flows": {c: _f(row.get(c)) for c in _LINKS},
            "reserve_mw": _f(row.get("reserve_mw")),
        })
    return snaps


def _golden_context_snapshots(scenario: dict, dataset_path: Path = DATASET_PATH) -> list[dict]:
    """Semana previa (168h) al primer snapshot del golden, como snapshots crudos
    reconstruidos del dataset (misma fuente ESIOS que el entrenamiento)."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {dataset_path}. "
            "Ejecutar: python -m ml.data.build_dataset"
        )
    t0 = pd.Timestamp(scenario["snapshots"][0]["timestamp"])
    t0 = t0.tz_localize("UTC") if t0.tz is None else t0.tz_convert("UTC")

    df = pd.read_parquet(dataset_path)
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    ctx = df[(df.index >= t0 - pd.Timedelta(hours=F.ROLLING_WINDOW_H + 1)) & (df.index < t0)]
    if len(ctx) < 24:
        raise RuntimeError(
            f"Contexto trailing insuficiente ({len(ctx)}h) antes de {t0} — "
            "el dataset no cubre la semana previa al golden."
        )
    log.info("Contexto trailing: %dh reales [%s → %s]", len(ctx), ctx.index[0], ctx.index[-1])
    return _dataset_rows_to_snapshots(ctx)


def _prepare_scenario(scenario: dict) -> dict:
    """Contexto + fila de features de serve por snapshot del golden."""
    context = _golden_context_snapshots(scenario)
    golden = scenario["snapshots"]
    rows, ts_list, labels = [], [], []
    for i, snap in enumerate(golden):
        now = pd.Timestamp(snap["timestamp"])
        rows.append(SF.serve_feature_row(context + golden[: i + 1], now))
        ts_list.append(now)
        labels.append(snap.get("label", ""))
    feats = pd.DataFrame(rows, index=pd.DatetimeIndex(ts_list))
    feats["label"] = labels
    return {"context": context, "golden": golden, "feats": feats}


def _scale(feats: pd.DataFrame, scaler: dict) -> tuple[np.ndarray, list[str]]:
    names = [f for f in scaler["features"] if f in feats.columns]
    X = feats[names].fillna(0.0).values.astype(np.float32)
    mean = np.array([scaler["mean"].get(f, 0.0) for f in names], dtype=np.float32)
    std = np.array([scaler["std"].get(f, 1.0) for f in names], dtype=np.float32)
    return (X - mean) / std, names


# ---------------------------------------------------------------------------
# Señales por snapshot (mismas transformaciones que ml_models_server)
# ---------------------------------------------------------------------------

def _if_scores(sc: dict, bundle: dict) -> np.ndarray:
    """Score IF crudo (-score_samples) por snapshot del golden."""
    model, scaler = bundle["model"], bundle["scaler"]
    X, _ = _scale(sc["feats"], scaler)
    return -model.score_samples(X)


def _lstm_errors(sc: dict, checkpoint: dict) -> np.ndarray:
    """Error de reconstrucción por snapshot, con su ventana trailing real."""
    import torch
    import torch.nn as nn

    model = checkpoint["model"]
    scaler = checkpoint["scaler"]
    seq_len = checkpoint.get("seq_len", 24)
    model.eval()

    errors = []
    for i, snap in enumerate(sc["golden"]):
        now = pd.Timestamp(snap["timestamp"])
        frame = SF.serve_feature_frame(sc["context"] + sc["golden"][: i + 1], now)
        X, _ = _scale(frame, scaler)
        if len(X) < seq_len:
            X = np.vstack([np.tile(X[0], (seq_len - len(X), 1)), X])
        seq = torch.tensor(X[-seq_len:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            recon = model(seq)
            errors.append(float(nn.MSELoss()(recon, seq).item()))
    return np.array(errors)


def _clf_probas(sc: dict, bundle: dict) -> np.ndarray:
    model = bundle["model"]
    features = bundle["feature_cols"]
    probas = []
    for _, row in sc["feats"].iterrows():
        X = np.array([[float(row.get(f, 0.0) or 0.0) for f in features]], dtype=np.float32)
        try:
            probas.append(float(model.predict_proba(X)[0][1]))
        except Exception:
            probas.append(float(model.predict(X)[0]))
    return np.array(probas)


def _load_bundle(models_dir: Path, name: str):
    path = models_dir / name
    if not path.exists():
        return None
    if name.endswith(".pth"):
        import torch
        return torch.load(path, weights_only=False, map_location="cpu")
    import joblib
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------

def validate_iforest(sc_blackout: dict, sc_normal: dict, models_dir: Path) -> bool | None:
    bundle = _load_bundle(models_dir, "anomaly_detector.pkl")
    if bundle is None:
        log.warning("Isolation Forest no entrenado aún")
        return None

    scores_b = _if_scores(sc_blackout, bundle)
    scores_n = _if_scores(sc_normal, bundle)
    log.info("[IF] Blackout scores (mayor=anómalo): %s", np.round(scores_b, 4))
    log.info("[IF] Normal  scores:                  %s", np.round(scores_n, 4))
    log.info("[IF] Media blackout=%.4f  media normal=%.4f", scores_b.mean(), scores_n.mean())

    passed = scores_b.mean() > scores_n.mean()
    log.info("[IF] %s", "PASS — apagón más anómalo que normal" if passed
             else "FAIL — no discrimina escenarios")
    return passed


def validate_lstm(sc_blackout: dict, sc_normal: dict, models_dir: Path) -> bool | None:
    checkpoint = _load_bundle(models_dir, "lstm_ae.pth")
    if checkpoint is None:
        log.warning("LSTM AE no entrenado aún")
        return None

    threshold = checkpoint.get("threshold", 0.5)
    err_b = _lstm_errors(sc_blackout, checkpoint)
    err_n = _lstm_errors(sc_normal, checkpoint)
    log.info("[LSTM] Errores blackout: %s", np.round(err_b, 6))
    log.info("[LSTM] Errores normal:   %s", np.round(err_n, 6))
    log.info("[LSTM] media blackout=%.6f  normal=%.6f  threshold(P99 train)=%.6f",
             err_b.mean(), err_n.mean(), threshold)

    passed = err_b.mean() > err_n.mean()
    log.info("[LSTM] %s", "PASS — blackout más difícil de reconstruir" if passed
             else "FAIL — no discrimina escenarios")
    return passed


def validate_classifier(sc_blackout: dict, sc_normal: dict, models_dir: Path) -> bool | None:
    bundle = _load_bundle(models_dir, "blackout_classifier.pkl")
    if bundle is None:
        log.warning("Clasificador no entrenado aún")
        return None

    threshold = bundle.get("optimal_threshold", 0.5)
    probas_b = _clf_probas(sc_blackout, bundle)
    probas_n = _clf_probas(sc_normal, bundle)

    log.info("[CLF] P(blackout) por snapshot:")
    for ts, prob, label in zip(sc_blackout["feats"].index, probas_b,
                               sc_blackout["feats"]["label"]):
        tag = "ALERTA" if prob >= threshold else "ok"
        log.info("  %s  %.4f  [%s]  %s", ts, prob, tag, str(label)[:60])
    log.info("[CLF] P(blackout) escenario normal: %s", np.round(probas_n, 4))
    log.info("[CLF] Umbral óptimo (F1): %.4f", threshold)

    passed = probas_b.max() >= 0.60 and probas_n.max() < 0.40
    log.info("[CLF] %s  (max_blackout=%.4f  max_normal=%.4f)",
             "PASS" if passed else "FAIL", probas_b.max(), probas_n.max())
    return passed


def validate_ensemble(sc_blackout: dict, sc_normal: dict, models_dir: Path) -> bool | None:
    """Score fusionado LSTM(load-bearing)+IF+CLF por snapshot.

    PASS si (a) media(golden) > media(normal) y (b) el score es no-decreciente
    hacia el onset en los últimos 3 snapshots del golden."""
    if_bundle = _load_bundle(models_dir, "anomaly_detector.pkl")
    lstm_ckpt = _load_bundle(models_dir, "lstm_ae.pth")
    clf_bundle = _load_bundle(models_dir, "blackout_classifier.pkl")
    if not (if_bundle and lstm_ckpt and clf_bundle):
        log.warning("Ensemble incompleto — faltan candidatos")
        return None

    lstm_threshold = lstm_ckpt.get("threshold", 0.5)

    def _fused(sc: dict) -> np.ndarray:
        if_raw = _if_scores(sc, if_bundle)
        lstm_err = _lstm_errors(sc, lstm_ckpt)
        clf = _clf_probas(sc, clf_bundle)
        out = []
        for r, e, c in zip(if_raw, lstm_err, clf):
            # mismas normalizaciones que ml_models_server.get_anomaly_score
            if_norm = float(np.clip(r + 0.5, 0, 1))
            lstm_norm = float(min(e / lstm_threshold, 1.0))
            out.append(fuse(lstm_norm, if_norm, c)["blackout_probability"])
        return np.array(out)

    p_b = _fused(sc_blackout)
    p_n = _fused(sc_normal)

    log.info("[ENSEMBLE] P fusionada por snapshot (golden):")
    for ts, p, label in zip(sc_blackout["feats"].index, p_b, sc_blackout["feats"]["label"]):
        log.info("  %s  %.4f  %s", ts, p, str(label)[:60])
    log.info("[ENSEMBLE] P fusionada normal: %s", np.round(p_n, 4))

    discriminates = p_b.mean() > p_n.mean()
    tail = p_b[-3:]
    monotonic = bool(np.all(np.diff(tail) >= -1e-6))
    log.info("[ENSEMBLE] media golden=%.4f > media normal=%.4f: %s | "
             "monotonía últimos 3: %s", p_b.mean(), p_n.mean(), discriminates, monotonic)

    passed = discriminates and monotonic
    log.info("[ENSEMBLE] %s", "PASS — discrimina y crece hacia el onset" if passed
             else "FAIL")
    return passed


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(
    run_if: bool = True,
    run_lstm: bool = True,
    run_clf: bool = True,
    run_ens: bool = True,
    models_dir: Path = CANDIDATES_DIR,
) -> dict:
    blackout_path = GOLDEN_DIR / "blackout_apr_2025.json"
    normal_path = GOLDEN_DIR / "normal_baseline.json"

    if not blackout_path.exists() or not normal_path.exists():
        log.error("Golden dataset no encontrado en %s", GOLDEN_DIR)
        return {}

    scenario_blackout = json.loads(blackout_path.read_text(encoding="utf-8"))
    scenario_normal = json.loads(normal_path.read_text(encoding="utf-8"))

    log.info("Preparando escenarios con contexto trailing real (serve_features)…")
    sc_blackout = _prepare_scenario(scenario_blackout)
    sc_normal = _prepare_scenario(scenario_normal)
    log.info("Golden blackout: %d snapshots | Normal: %d snapshots",
             len(sc_blackout["golden"]), len(sc_normal["golden"]))
    log.info("Validando artefactos de: %s", models_dir)

    results: dict[str, bool | None] = {}
    if run_if:
        log.info("=== Isolation Forest ===")
        results["iforest"] = validate_iforest(sc_blackout, sc_normal, models_dir)
    if run_lstm:
        log.info("=== LSTM Autoencoder ===")
        results["lstm"] = validate_lstm(sc_blackout, sc_normal, models_dir)
    if run_clf:
        log.info("=== Blackout Classifier ===")
        results["classifier"] = validate_classifier(sc_blackout, sc_normal, models_dir)
    if run_ens:
        log.info("=== Ensemble (LSTM load-bearing + IF + CLF) ===")
        results["ensemble"] = validate_ensemble(sc_blackout, sc_normal, models_dir)

    log.info("")
    log.info("=" * 50)
    log.info("RESUMEN DE VALIDACIÓN")
    log.info("=" * 50)
    any_fail = False
    for name, ok in results.items():
        status = "PASS" if ok else ("FAIL" if ok is False else "SKIP")
        log.info("  %-20s %s", name, status)
        if ok is False:
            any_fail = True
    if not results:
        log.info("  (ningún modelo entrenado todavía)")
    log.info("=" * 50)

    if any_fail:
        log.warning("Algunos modelos no superan el golden test. Revisar antes de promover. "
                    "Gate honesto: el LSTM es el detector primario; el CLF puede seguir "
                    "débil por D5 (n=1 evento S4).")
    elif results:
        log.info("Todos los candidatos evaluados superan el golden test.")
    return results


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Valida candidatos ML con golden test")
    p.add_argument(
        "--model",
        choices=["iforest", "lstm", "classifier", "ensemble", "all"],
        default="all",
    )
    p.add_argument("--models-dir", default=None,
                   help="Directorio de artefactos (default: ml/models/candidates)")
    return p.parse_args()


if __name__ == "__main__":
    args = _args()
    run(
        run_if=args.model in ("iforest", "all"),
        run_lstm=args.model in ("lstm", "all"),
        run_clf=args.model in ("classifier", "all"),
        run_ens=args.model in ("ensemble", "all"),
        models_dir=Path(args.models_dir) if args.models_dir else CANDIDATES_DIR,
    )
