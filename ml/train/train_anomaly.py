"""
ml/train/train_anomaly.py — Entrena el detector de anomalías de red eléctrica.

Dos modelos entrenados y guardados como candidatos:
  1. Isolation Forest (sklearn)  → ml/models/candidates/anomaly_detector.pkl
  2. LSTM Autoencoder (PyTorch)  → ml/models/candidates/lstm_ae.pth

Estrategia de hiperparámetros:
  - IF:   Random search (20 trials, sin CV — modelo no supervisado)
  - LSTM: Optuna con MedianPruner (50 trials)

Validación:
  - No supervisada: evaluada como std de scores en datos normales (menor = más estable)
  - Golden test manual: horas previas al apagón 28-abr-2025 deben salir como anomalías

Uso:
    python -m ml.train.train_anomaly
    python -m ml.train.train_anomaly --model iforest  # solo IF
    python -m ml.train.train_anomaly --model lstm     # solo LSTM
    python -m ml.train.train_anomaly --if-trials 30 --lstm-trials 80
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
log = logging.getLogger("train_anomaly")

# ── Rutas ──────────────────────────────────────────────────────────────────
DATA_PATH = Path(os.environ.get("DATA_PROCESSED_PATH", "data/processed")) / "dataset_training.parquet"
CANDIDATES_DIR = Path(os.environ.get("ML_MODELS_PATH", "ml/models")) / "candidates"

# Features de entrada (subset de columnas del dataset)
FEATURE_COLS = [
    "demand_real_mw",
    "nuclear_mw", "hydro_mw", "wind_mw",
    "solar_pv_mw", "solar_thermal_mw", "gas_cc_mw", "coal_mw",
    "balance_mw", "reserve_activated_pct",
    "interchange_delta_mw", "renewables_pct", "composite_pressure",
]

# Ventana temporal para el LSTM (horas hacia atrás)
SEQ_LEN_DEFAULT = 24

# Períodos de anomalía conocida — excluidos del entrenamiento (no deben contaminar el modelo)
KNOWN_ANOMALY_PERIODS = [
    ("2025-04-28 06:00", "2025-04-28 14:00"),  # apagón España
    ("2006-11-04 09:00", "2006-11-04 12:00"),  # separación ibérica
]


# ---------------------------------------------------------------------------
# Carga y preparación de datos
# ---------------------------------------------------------------------------

def load_features(path: Path) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """Carga el dataset y devuelve (X, timestamps, feature_names)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {path}. "
            "Ejecutar primero: python -m ml.data.build_dataset"
        )
    log.info("Cargando dataset desde %s…", path)
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index, tz="UTC")

    # Seleccionar solo las columnas disponibles
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        log.warning("Features no disponibles (ignoradas): %s", missing)

    df_feat = df[available].copy()
    df_feat = df_feat.ffill().bfill()  # interpolar pequeños huecos

    # Excluir períodos de anomalía conocida del entrenamiento
    mask = pd.Series(True, index=df_feat.index)
    for start, end in KNOWN_ANOMALY_PERIODS:
        period_mask = (df_feat.index >= pd.Timestamp(start, tz="UTC")) & \
                      (df_feat.index <= pd.Timestamp(end, tz="UTC"))
        mask &= ~period_mask

    df_train = df_feat[mask].dropna()
    log.info("Datos de entrenamiento: %d filas, %d features", len(df_train), len(available))

    # Normalización Z-score (parámetros de la media de entrenamiento)
    means = df_train.mean()
    stds = df_train.std().replace(0, 1)
    X_norm = ((df_train - means) / stds).values.astype(np.float32)

    # Guardar scaler para inferencia
    scaler = {"mean": means.to_dict(), "std": stds.to_dict(), "features": available}

    return X_norm, df_train.index, available, scaler


# ---------------------------------------------------------------------------
# Modelo 1: Isolation Forest
# ---------------------------------------------------------------------------

def train_isolation_forest(X: np.ndarray, n_trials: int = 20) -> dict:
    """Random search de hiperparámetros para Isolation Forest."""
    import optuna
    from sklearn.ensemble import IsolationForest

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        model = IsolationForest(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            contamination=trial.suggest_float("contamination", 0.005, 0.05, log=True),
            max_samples=trial.suggest_categorical("max_samples", ["auto", 512, 1024]),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)
        # Objetivo: minimizar la desviación estándar de scores en datos normales.
        # Un IF bien calibrado asigna scores similares a todos los datos normales.
        scores = -model.score_samples(X)
        return float(scores.std())

    log.info("Buscando hiperparámetros IF (%d trials)…", n_trials)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=42),
        study_name="isolation_forest",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    log.info("Mejores parámetros IF: %s | score_std=%.4f", best, study.best_value)

    # Reentrenar con los mejores parámetros sobre todos los datos de train
    final_model = IsolationForest(**best, random_state=42, n_jobs=-1)
    final_model.fit(X)

    return {
        "model": final_model,
        "best_params": best,
        "score_std": study.best_value,
        "n_features": X.shape[1],
    }


def save_isolation_forest(result: dict, scaler: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {"model": result["model"], "scaler": scaler, "params": result["best_params"]}
    path = out_dir / "anomaly_detector.pkl"
    joblib.dump(bundle, path)
    log.info("IF guardado: %s", path)
    return path


# ---------------------------------------------------------------------------
# Modelo 2: LSTM Autoencoder
# ---------------------------------------------------------------------------

def _build_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Convierte la matriz (T, F) en secuencias (N, seq_len, F)."""
    seqs = []
    for i in range(len(X) - seq_len + 1):
        seqs.append(X[i : i + seq_len])
    return np.array(seqs, dtype=np.float32)


class LSTMAutoencoder:
    """Factory que devuelve un LSTMAutoencoderModule (importable por pickle)."""

    def __new__(cls, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        try:
            from ml.train.lstm_model import LSTMAutoencoderModule  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("PyTorch requerido para LSTM. pip install torch") from exc

        return LSTMAutoencoderModule(n_features, hidden_size, num_layers, dropout)


def train_lstm_autoencoder(X: np.ndarray, n_trials: int = 50, seq_len: int = SEQ_LEN_DEFAULT) -> dict:
    """Optuna + MedianPruner para la arquitectura del LSTM Autoencoder."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        import optuna
    except ImportError as exc:
        raise ImportError("PyTorch y Optuna requeridos. pip install torch optuna") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    seqs = _build_sequences(X, seq_len)
    n_val = max(1, int(len(seqs) * 0.1))
    seqs_train = torch.tensor(seqs[:-n_val], device=DEVICE)
    seqs_val = torch.tensor(seqs[-n_val:], device=DEVICE)
    n_features = X.shape[1]

    def make_loader(data: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=shuffle)

    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        n_epochs = 20

        model = LSTMAutoencoder(n_features, hidden_size, num_layers, dropout).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        loader = make_loader(seqs_train, batch_size, shuffle=True)

        for epoch in range(n_epochs):
            model.train()
            for (batch,) in loader:
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

            # Validación al final de cada época para pruning
            if epoch % 4 == 3 or epoch == n_epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(seqs_val), seqs_val).item()
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        model.eval()
        with torch.no_grad():
            return criterion(model(seqs_val), seqs_val).item()

    log.info("Buscando arquitectura LSTM (%d trials, MedianPruner)…", n_trials)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4)
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="lstm_autoencoder",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    log.info("Mejor arquitectura LSTM: %s | val_mse=%.6f", best, study.best_value)

    # Reentrenar con mejores parámetros durante más épocas
    log.info("Reentrenando LSTM con mejores hiperparámetros (100 épocas)…")
    model = LSTMAutoencoder(
        n_features, best["hidden_size"], best["num_layers"], best["dropout"]
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"])
    criterion = nn.MSELoss()
    loader = make_loader(seqs_train, best["batch_size"], shuffle=True)

    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            log.info("  Época %d/100 — loss: %.6f", epoch + 1, epoch_loss / len(loader))

    # Calcular threshold de anomalía: percentil 99 de errores de reconstrucción en train
    model.eval()
    with torch.no_grad():
        recon_errors = []
        all_seqs = torch.tensor(seqs, device=DEVICE)
        chunk = 512
        for i in range(0, len(all_seqs), chunk):
            batch = all_seqs[i : i + chunk]
            errors = nn.MSELoss(reduction="none")(model(batch), batch).mean(dim=(1, 2))
            recon_errors.extend(errors.cpu().numpy().tolist())
    threshold = float(np.percentile(recon_errors, 99))
    log.info("Umbral de anomalía (percentil 99): %.6f", threshold)

    return {
        "model": model,
        "threshold": threshold,
        "best_params": best,
        "val_mse": study.best_value,
        "seq_len": seq_len,
        "n_features": n_features,
    }


def save_lstm(result: dict, scaler: dict, out_dir: Path) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch requerido para guardar LSTM") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lstm_ae.pth"
    torch.save(
        {
            "model": result["model"],
            "threshold": result["threshold"],
            "scaler": scaler,
            "params": result["best_params"],
            "seq_len": result["seq_len"],
        },
        path,
    )
    log.info("LSTM Autoencoder guardado: %s", path)
    return path


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(train_if: bool = True, train_lstm: bool = True,
        if_trials: int = 20, lstm_trials: int = 50) -> None:
    log_device_info()
    X, timestamps, feature_names, scaler = load_features(DATA_PATH)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    if train_if:
        log.info("=== Entrenando Isolation Forest ===")
        result_if = train_isolation_forest(X, n_trials=if_trials)
        save_isolation_forest(result_if, scaler, CANDIDATES_DIR)

    if train_lstm:
        if DEVICE is None:
            log.warning("PyTorch no disponible — saltando LSTM Autoencoder")
        else:
            log.info("=== Entrenando LSTM Autoencoder (device=%s) ===", DEVICE)
            result_lstm = train_lstm_autoencoder(X, n_trials=lstm_trials)
            save_lstm(result_lstm, scaler, CANDIDATES_DIR)

    log.info("Candidatos guardados en %s", CANDIDATES_DIR)
    log.info("Siguiente paso: python -m ml.train.validate_models")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena detector de anomalías")
    p.add_argument("--model", choices=["iforest", "lstm", "both"], default="both")
    p.add_argument("--if-trials", type=int, default=20)
    p.add_argument("--lstm-trials", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    args = _args()
    run(
        train_if=args.model in ("iforest", "both"),
        train_lstm=args.model in ("lstm", "both"),
        if_trials=args.if_trials,
        lstm_trials=args.lstm_trials,
    )
