"""conftest.py (raiz) — pre-calienta imports nativos fragiles antes de recoger tests.

Hallazgo 2026-07-06: `pytest -q` crasheaba con un access violation nativo real
de Windows (reproducible en Git Bash y PowerShell) al llegar a
tests/ml/test_train_classifier.py, que importa pyarrow.dataset de forma
perezosa (via pandas.read_parquet) tras una sesion larga de tests. Standalone
(`python -c "import pyarrow.dataset"` o `pd.read_parquet(...)` en un proceso
limpio) funciona bien siempre -- el fallo depende del estado acumulado del
proceso tras correr el resto de la suite.

Mismo patron ya encontrado y corregido antes en este proyecto para chromadb
(mcp_servers/knowledge_base_server.py) y torch (mcp_servers/ml_models_server.py):
un import nativo pesado hecho perezosamente, dentro de un proceso ya "caliente"
con mucha actividad previa (threads, otros extensiones C cargadas), se cuelga o
crashea; importarlo al arranque, en un proceso todavia simple, lo evita.

Se importa aqui, en el conftest.py raiz, que pytest carga antes de recoger
ningun test -- el proceso esta en su estado mas "limpio" posible en ese punto.
"""
from __future__ import annotations

try:
    import pyarrow  # noqa: F401
    import pyarrow.dataset  # noqa: F401
except ImportError:
    pass
