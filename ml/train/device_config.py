"""
ml/train/device_config.py — Selección automática de dispositivo CUDA/CPU.

Importar en todos los scripts de entrenamiento:
    from ml.train.device_config import DEVICE, log_device_info
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _select_device():
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("GPU disponible: %s (%.1f GB VRAM)", name, mem_gb)
        else:
            device = torch.device("cpu")
            log.info("GPU no disponible — entrenamiento en CPU")
        return device
    except ImportError:
        log.warning("PyTorch no instalado — solo modelos scikit-learn/XGBoost disponibles")
        return None


DEVICE = _select_device()


def log_device_info() -> None:
    """Imprime resumen del entorno de cómputo al iniciar un script de entrenamiento."""
    import platform
    import sys
    log.info("Python %s | %s", sys.version.split()[0], platform.platform())
    try:
        import torch
        log.info("PyTorch %s | CUDA %s | Device: %s",
                 torch.__version__,
                 torch.version.cuda or "N/A",
                 DEVICE)
    except ImportError:
        log.info("PyTorch no disponible")
    try:
        import xgboost as xgb
        log.info("XGBoost %s", xgb.__version__)
    except ImportError:
        log.warning("XGBoost no instalado")
    try:
        import sklearn
        log.info("scikit-learn %s", sklearn.__version__)
    except ImportError:
        log.warning("scikit-learn no instalado")
