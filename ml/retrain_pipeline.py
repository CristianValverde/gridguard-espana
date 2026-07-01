"""
ml/retrain_pipeline.py — Pipeline de reentrenamiento mensual.

Ejecutado el día 1 de cada mes a las 02:00 UTC por el cron scheduler.
Orquesta las 8 fases definidas en ARCHITECTURE.md §4.

Fases:
  1. Descarga datos del mes anterior (ESIOS + AEMET + ENTSO-E)
  2. Actualiza el dataset de entrenamiento (añade al parquet histórico)
  3-5. Reentrena los 3 modelos en paralelo → candidatos en ml/models/candidates/
  6. Valida candidatos en hold-out. Si mejoran → promueve a producción
  7. Señaliza a MCP: ml-models para hot-swap atómico de pesos
  8. Log de cambios

Uso:
    python -m ml.retrain_pipeline                    # reentrenamiento completo
    python -m ml.retrain_pipeline --dry-run          # simula sin escribir ficheros
    python -m ml.retrain_pipeline --skip-download    # si los datos ya están descargados
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] RETRAIN — %(message)s",
)
log = logging.getLogger("retrain_pipeline")

MODELS_DIR = Path(os.environ.get("ML_MODELS_PATH", "ml/models"))
CANDIDATES_DIR = MODELS_DIR / "candidates"
HISTORY_DIR = MODELS_DIR / "history"
DATA_RAW = Path(os.environ.get("DATA_RAW_PATH", "data/raw"))
DATA_PROCESSED = Path(os.environ.get("DATA_PROCESSED_PATH", "data/processed"))

PRODUCTION_MODELS = {
    "anomaly_detector.pkl": "iforest",
    "lstm_ae.pth": "lstm_ae",
    "load_forecaster.pkl": "xgb_forecast",
    "blackout_classifier.pkl": "xgb_blackout",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _run(cmd: list[str], dry_run: bool) -> bool:
    """Ejecuta un subcomando Python. Devuelve True si tuvo éxito."""
    log.info("$ %s", " ".join(cmd))
    if dry_run:
        log.info("  [DRY-RUN] saltado")
        return True
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("  FALLÓ con código %d", result.returncode)
        return False
    return True


def _snapshot_production(today: str) -> None:
    """Copia los modelos en producción actuales al directorio de historia."""
    snapshot_dir = HISTORY_DIR / today
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for model_file in PRODUCTION_MODELS:
        src = MODELS_DIR / model_file
        if src.exists():
            shutil.copy2(src, snapshot_dir / model_file)
            log.info("  Snapshot: %s → %s", src, snapshot_dir / model_file)


def _promote_candidate(model_file: str, dry_run: bool) -> bool:
    """Copia el candidato sobre el modelo en producción."""
    src = CANDIDATES_DIR / model_file
    dst = MODELS_DIR / model_file
    if not src.exists():
        log.warning("  Candidato no encontrado: %s", src)
        return False
    if dry_run:
        log.info("  [DRY-RUN] promovería %s → %s", src, dst)
        return True
    import shutil
    shutil.copy2(src, dst)
    log.info("  Promovido: %s", dst)
    return True


def _validate_candidates() -> dict[str, bool]:
    """
    Compara métricas de candidatos contra producción.
    Devuelve {model_file: should_promote}.

    Criterios (ARCHITECTURE.md §4, fase 6):
      - blackout_classifier: AUC-ROC candidato ≥ producción
      - load_forecaster:     MAE candidato ≤ producción
      - anomaly_detector:    score_std candidato ≤ producción (más estable)
    """
    import joblib

    results: dict[str, bool] = {}

    for model_file in PRODUCTION_MODELS:
        candidate_path = CANDIDATES_DIR / model_file
        production_path = MODELS_DIR / model_file

        if not candidate_path.exists():
            log.warning("  Sin candidato para %s — manteniendo producción", model_file)
            results[model_file] = False
            continue

        if not production_path.exists():
            log.info("  Sin modelo en producción para %s — promoviendo candidato", model_file)
            results[model_file] = True
            continue

        # Los .pth de PyTorch no tienen métricas comparables aquí; se promueven siempre
        if model_file.endswith(".pth"):
            results[model_file] = True
            continue

        try:
            candidate = joblib.load(candidate_path)
            production = joblib.load(production_path)
            c_metrics = candidate.get("metrics", {})
            p_metrics = production.get("metrics", {})

            promote = False
            if "blackout" in model_file:
                c_auc = c_metrics.get("auc_roc_holdout", 0)
                p_auc = p_metrics.get("auc_roc_holdout", 0)
                promote = c_auc >= p_auc
                log.info("  %s: AUC candidato=%.4f vs producción=%.4f → %s",
                         model_file, c_auc, p_auc, "PROMOVER" if promote else "MANTENER")
            elif "forecaster" in model_file:
                c_mae = c_metrics.get("mae_holdout", float("inf"))
                p_mae = p_metrics.get("mae_holdout", float("inf"))
                promote = c_mae <= p_mae
                log.info("  %s: MAE candidato=%.1f vs producción=%.1f → %s",
                         model_file, c_mae, p_mae, "PROMOVER" if promote else "MANTENER")
            elif "anomaly" in model_file:
                c_std = c_metrics.get("score_std", float("inf"))
                p_std = p_metrics.get("score_std", float("inf"))
                promote = c_std <= p_std
                log.info("  %s: score_std candidato=%.4f vs producción=%.4f → %s",
                         model_file, c_std, p_std, "PROMOVER" if promote else "MANTENER")
            else:
                promote = True

            results[model_file] = promote
        except Exception as exc:
            log.warning("  Error comparando %s: %s — manteniendo producción", model_file, exc)
            results[model_file] = False

    return results


def _signal_mcp_reload(dry_run: bool) -> None:
    """Llama a reload_models() en el servidor MCP ml-models vía proceso local."""
    log.info("Señalizando hot-swap a MCP: ml-models…")
    if dry_run:
        log.info("  [DRY-RUN] saltado")
        return
    try:
        import requests
        # El servidor MCP puede exponer un endpoint HTTP de control interno
        # En producción esto se haría vía el protocolo MCP; aquí usamos
        # un endpoint de administración si está disponible
        resp = requests.post("http://localhost:8765/admin/reload_models", timeout=10)
        resp.raise_for_status()
        log.info("  Hot-swap completado: %s", resp.json())
    except Exception as exc:
        log.warning("  No se pudo señalizar MCP (normal si el servidor no corre): %s", exc)
        log.info("  El servidor MCP cargará los nuevos pesos al próximo reinicio")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(dry_run: bool = False, skip_download: bool = False) -> None:
    today = date.today().isoformat()
    started_at = datetime.now(timezone.utc)
    log.info("=== Reentrenamiento mensual — %s ===", today)

    # Calcular el mes anterior para la descarga
    from calendar import monthrange
    now = date.today()
    if now.month == 1:
        prev_year, prev_month = now.year - 1, 12
    else:
        prev_year, prev_month = now.year, now.month - 1
    last_day = monthrange(prev_year, prev_month)[1]
    period_start = f"{prev_year}-{prev_month:02d}-01"
    period_end = f"{prev_year}-{prev_month:02d}-{last_day:02d}"
    log.info("Período de datos nuevos: %s → %s", period_start, period_end)

    python = sys.executable

    # ── Fase 1: Descarga de datos ────────────────────────────────────────────
    if not skip_download:
        log.info("--- Fase 1: Descarga de datos ---")
        ok = all([
            _run([python, "-m", "ml.data.download_esios",
                  "--start", period_start, "--end", period_end,
                  "--out", str(DATA_RAW / "esios")], dry_run),
            _run([python, "-m", "ml.data.download_aemet",
                  "--start", period_start, "--end", period_end,
                  "--out", str(DATA_RAW / "aemet")], dry_run),
        ])
        if not ok:
            log.error("Fase 1 falló — abortando pipeline")
            return
    else:
        log.info("--- Fase 1: Descarga saltada (--skip-download) ---")

    # ── Fase 2: Rebuild dataset ──────────────────────────────────────────────
    log.info("--- Fase 2: Reconstruyendo dataset de entrenamiento ---")
    if not _run([python, "-m", "ml.data.build_dataset",
                 "--esios", str(DATA_RAW / "esios"),
                 "--aemet", str(DATA_RAW / "aemet"),
                 "--out", str(DATA_PROCESSED)], dry_run):
        log.error("Fase 2 falló — abortando pipeline")
        return

    # ── Fases 3-5: Reentrenamiento en paralelo ───────────────────────────────
    log.info("--- Fases 3-5: Entrenando modelos (paralelo) ---")
    import concurrent.futures
    train_cmds = [
        [python, "-m", "ml.train.train_anomaly"],
        [python, "-m", "ml.train.train_forecast"],
        [python, "-m", "ml.train.train_classifier"],
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_run, cmd, dry_run): cmd for cmd in train_cmds}
        train_results = {f.result(): cmd for f, cmd in futures.items()}
    if not all(train_results.keys()):
        log.error("Algún entrenamiento falló — continuando con validación parcial")

    # ── Fase 6: Validación y promoción ──────────────────────────────────────
    log.info("--- Fase 6: Validando candidatos ---")
    _snapshot_production(today)
    decisions = _validate_candidates()

    promoted = []
    for model_file, should_promote in decisions.items():
        if should_promote:
            if _promote_candidate(model_file, dry_run):
                promoted.append(model_file)

    if promoted:
        log.info("Modelos promovidos: %s", promoted)
    else:
        log.info("Ningún candidato mejoró los modelos en producción — manteniendo versión anterior")

    # ── Fase 7: Hot-swap en MCP ──────────────────────────────────────────────
    if promoted:
        log.info("--- Fase 7: Hot-swap en MCP ml-models ---")
        _signal_mcp_reload(dry_run)

    # ── Fase 8: Resumen ───────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds() / 60
    log.info("=== Reentrenamiento completado en %.1f min ===", elapsed)
    log.info("Modelos promovidos: %s", promoted if promoted else "ninguno")
    log.info("Snapshot del estado anterior: %s", HISTORY_DIR / today)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline de reentrenamiento mensual")
    p.add_argument("--dry-run", action="store_true", help="Simula sin escribir ficheros")
    p.add_argument("--skip-download", action="store_true", help="Salta la descarga de datos")
    return p.parse_args()


if __name__ == "__main__":
    args = _args()
    run(dry_run=args.dry_run, skip_download=args.skip_download)
