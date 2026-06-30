# GridGuard España — Especificación de Arquitectura

> Sistema multi-agente para monitorización de la estabilidad de la red eléctrica española  
> Capstone Project · 5-Day AI Agents Intensive (Google × Kaggle) · Junio 2026

---

## 1. Principios de Diseño

### Shift-Left: Determinístico primero, LLM solo cuando aporta valor

El principio clave es **no usar un LLM donde una función Python o un modelo ML basta**. Hay tres niveles de inteligencia, cada uno reservado para lo que realmente necesita:

| Si la tarea es... | Usar... |
|---|---|
| Llamada a API + parseo + validación | Function Tool (Python puro) |
| Detección de patrones estadísticos en series temporales | Modelo ML (determinístico en inferencia) |
| Recuperación de contexto técnico / normativo | RAG (ChromaDB) |
| Análisis de múltiples señales con razonamiento | Skill (AgentTool + LLM) |
| Tarea multi-step con estado interno | Sub-agent (LLM) |
| Coordinación global + decisiones de escalado | Orchestrator (Root Agent) |

### ML vs. Conocimiento de Dominio (RAG) — Roles complementarios

- **ML** responde *¿qué está pasando estadísticamente?* — detecta anomalías en series temporales, predice carga, calcula probabilidad de apagón calibrada con datos históricos. Es determinístico en inferencia, reproducible y auditable.
- **RAG** responde *¿qué significa esto?* — recupera normas ENTSO-E, procedimientos de operación de REE, post-mortems de incidentes anteriores. El LLM sin RAG razona en abstracto; con RAG razona sobre la red española real.
- **LLM** responde *¿qué hacemos?* — sintetiza señal ML + contexto RAG y decide severidad, ámbito (nacional/zonal) y recomendación accionable.

### Skills vs Sub-agentes

- **Skill (AgentTool)**: agente envuelto como herramienta síncrona. Sin estado entre llamadas. Ideal para análisis focalizados y reutilizables con resultado estructurado único.
- **Sub-agent**: agente con su propio loop, herramientas y memoria de sesión. Ideal para tareas autónomas multi-step donde el número de pasos es impredecible.

---

## 2. Arquitectura por Capas

```
╔══════════════════════════════════════════════════════════════╗
║  OFFLINE: Pipeline de Entrenamiento y Conocimiento           ║
║  (ESIOS 2014→ · AEMET 2000→ · ENTSO-E 2011→)               ║
║           ↓                        ↓                         ║
║   Modelos ML (.pkl)         Knowledge Base (ChromaDB)        ║
╚═════════════════════╤════════════════════╤═══════════════════╝
                      │                    │
[Capa 0] Disparadores (Cron / Event Queue / Eval Runner)
    ↓
[Capa 1] Function Tools  (Python puro, sin LLM)
    ↓ [Hook: before_tool_callback — rate-limit, .env block, sanitización]
[Capa 2] MCP Servers  (6 servidores: ree-grid · weather · entso-e
                        memory · ml-models · knowledge-base)
    ↓ [Hook: after_tool_callback — schema, credenciales, truncado]
[Capa 3] Skills / AgentTools  (AnomalyDetector · RenewablesForecast)
    ↓ [Hook: before_model_callback — anti-injection]
[Capa 4] Sub-agentes  (ReportAgent · CrossBorderAgent)
    ↓ [Hook: after_model_callback + after_agent_callback — bounds, schema]
[Capa 5] OrchestratorAgent  (Root Agent · gemini-3.5-flash)
    ↓
[Outputs] Alertas · Risk Score · Dashboard HTML · Informe diario
                      │
[Cross-cutting] Tracing (ADK) · Eval (output + trayectoria)
```

---

## 3. Model Registry — Modelos Siempre Actualizados

### Problema resuelto
Los modelos Gemini evolucionan rápido: `gemini-2.0-flash` fue marcado como **Shut Down** en junio 2026 sin apenas aviso. Hardcodear un nombre de modelo en el código es una deuda técnica garantizada.

### Solución: Model Registry con auto-update semanal

El sistema utiliza un fichero `config/model_registry.yaml` como única fuente de verdad de los modelos. Un job semanal lo actualiza automáticamente consultando la documentación oficial de Google.

```yaml
# config/model_registry.yaml — generado automáticamente, NO editar manualmente
# Última actualización: 2026-06-27 | Fuente: ai.google.dev/gemini-api/docs/models

models:
  fast_agent:          "gemini-3.5-flash"        # Stable · agentic tasks optimized
  smart_agent:         "gemini-3.1-pro"           # Preview · complex reasoning · 2M ctx
  embedding:           "gemini-embedding-2"        # Stable · multimodal embeddings for RAG
  fallback_fast:       "gemini-3.1-flash-lite"    # Stable · cost-efficient fallback

registry_meta:
  source_url:   "https://ai.google.dev/gemini-api/docs/models"
  updated_at:   "2026-06-27T10:00:00Z"
  next_update:  "2026-07-04T10:00:00Z"
  stable_only:  true   # nunca usar modelos Experimental en producción
```

```python
# config/model_loader.py — todos los agentes importan de aquí
import yaml

def get_model(role: str) -> str:
    """Lee el registry actualizado. Nunca usa strings hardcoded."""
    with open("config/model_registry.yaml") as f:
        registry = yaml.safe_load(f)
    return registry["models"][role]

# Uso en agentes:
# model = get_model("fast_agent")   → "gemini-3.5-flash"
# model = get_model("smart_agent")  → "gemini-3.1-pro"
```

### Job de auto-actualización (cron semanal)

```python
# tools/model_registry_updater.py
# Ejecutado cada lunes a las 06:00 UTC por el cron scheduler

def update_model_registry():
    """
    1. Fetches ai.google.dev/gemini-api/docs/models
    2. Parsea modelos disponibles (stable / preview / shut_down)
    3. Mapea roles → mejor modelo disponible (prioriza stable)
    4. Valida que los modelos mapeados responden en la API
    5. Sobreescribe config/model_registry.yaml
    6. Loguea cambios detectados (ej: "gemini-2.0-flash → SHUT DOWN, migrado a gemini-3.5-flash")
    """
```

### Modelos actuales (junio 2026)

| Rol en el sistema | Modelo | Estado | Por qué |
|---|---|---|---|
| Agentes rápidos (Orchestrator, Skills) | `gemini-3.5-flash` | **Stable** | Diseñado para tareas agénticas, supera a 3.1 Pro en benchmarks de agentes (MCP Atlas 83.6%) |
| Agente de razonamiento complejo (ReportAgent) | `gemini-3.1-pro` | Preview | Contexto 2M tokens, razonamiento avanzado para síntesis de informes |
| Embeddings RAG | `gemini-embedding-2` | **Stable** | Primer modelo multimodal de embeddings, mejor para RAG heterogéneo |
| Fallback / testing | `gemini-3.1-flash-lite` | **Stable** | Más barato para CI y pruebas de trayectoria |
| ~~gemini-2.0-flash~~ | ~~gemini-2.0-flash~~ | **⛔ SHUT DOWN** | Eliminado de la arquitectura |
| ~~gemini-2.5-pro~~ | ~~gemini-2.5-pro~~ | Generación anterior | Reemplazado por gemini-3.1-pro |

---

## 4. Pipeline Offline — Datos Históricos y Entrenamiento (CUDA) (CUDA)

### Fuentes de datos confirmadas

| Fuente | Desde | Granularidad | Datos clave |
|---|---|---|---|
| **ESIOS / REData** (token gratuito) | 2014-01-01 | 10 min / horaria | Demanda real, generación por tecnología, intercambios, precios, reserva regulación |
| **REData** (sin token) | 1990 | Mensual / anual | Series estadísticas largas, balance eléctrico |
| **AEMET OpenData** (token gratuito) | ~2000 | Horaria / diaria | Viento, radiación solar, temperatura por estación |
| **ENTSO-E Transparency** (token gratuito) | 2011 (legacy CSV) / 2015 (API) | Horaria | Intercambios transfronterizos, generación EU, capacidad interconexión |

**Volumen resultante**: ~10 años × 365 días × 6 lecturas/hora = **~525,000 puntos temporales** a granularidad de 10 min. Más que suficiente para modelos ML robustos.

### Nota crítica: Frecuencia de red no disponible vía API pública

Los datos de **frecuencia de red (Hz)** no se publican en la API de ESIOS — son telemedidas de tiempo real que REE no expone por seguridad operacional. Se utilizan los siguientes **proxies**, que en la literatura académica son los features estándar para predicción de inestabilidad en redes de alta penetración renovable:

| Proxy | Indicador ESIOS | Por qué funciona |
|---|---|---|
| Balance oferta/demanda | Generación real − Demanda real | Diferencia directa → presión sobre frecuencia |
| Reserva de regulación activada | Indicadores de reserva secundaria | Se activa cuando la frecuencia se desvía |
| Urgencia en intercambios | Variación brusca en flujos FR/PT/MA | Importación de emergencia → tensión de frecuencia |
| % renovables no gestionables | (Solar + Eólica) / Demanda total | Baja inercia sincrónica → mayor vulnerabilidad |

### Golden Dataset — Apagón 28 de abril de 2025

El informe post-mortem de REE del apagón del 28 de abril de 2025 está publicado con reconstrucción minuto a minuto. Es el caso de test principal del sistema de Eval.

### Modelos ML a entrenar

| Modelo | Algoritmo | Input | Output | Librería |
|---|---|---|---|---|
| **Anomaly Detector** | Isolation Forest / LSTM Autoencoder | Ventana 6h de métricas (demanda, mix, intercambios, proxies frecuencia) | `anomaly_score` [0–1] + features más anómalas | scikit-learn / PyTorch |
| **Load Forecaster** | XGBoost / Prophet | Hora, día semana, temperatura, festivo, histórico 30 días | Demanda prevista 6h + intervalo confianza | XGBoost / Prophet |
| **Blackout Risk Classifier** | Gradient Boosting calibrado | `anomaly_score` + `load_forecast` + `renewables_pct` + `margin_reserva` | `P(blackout_1h)` [0–1] + feature importance | scikit-learn |

### Documentos para RAG (Knowledge Base)

| Categoría | Documentos | Para qué |
|---|---|---|
| **Normas operacionales** | PO 1.1, PO 1.2, PO 7.4 de REE; Reglamento red ENTSO-E; EN 50160 | El agente sabe qué significa 49.8Hz en contexto normativo europeo |
| **Post-mortems** | Informe REE apagón 28 abril 2025; Separación ibérica 2006; CEER incidentes 2018-2024 | Reconoce patrones de incidentes anteriores y los cita |
| **Conocimiento técnico** | Inercia sincrónica y sintética; Reserva regulación primaria/secundaria; Topología peninsular; Alta penetración renovable | Explica causas y recomienda mitigaciones |

---

### CUDA — Entrenamiento con GPU

Todos los modelos que soportan GPU usan CUDA para el entrenamiento. La selección de device es automática:

```python
# ml/train/device_config.py
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")  # cuda:0 si hay GPU disponible
```

| Modelo ML | Framework | Soporte CUDA | Guardado de pesos |
|---|---|---|---|
| **LSTM Autoencoder** (anomaly) | PyTorch | ✅ Nativo | `torch.save(model.state_dict(), "lstm_ae.pth")` |
| **XGBoost** (load forecast) | XGBoost | ✅ `device="cuda"` | `model.save_model("xgb_forecast.json")` |
| **Gradient Boosting** (blackout) | XGBoost GPU | ✅ `device="cuda"` | `model.save_model("xgb_blackout.json")` |
| **Isolation Forest** (anomaly alt.) | scikit-learn | ❌ CPU only | `joblib.dump(model, "iforest.pkl")` |

> **Nota**: Si la GPU no está disponible (entorno de desarrollo / CI), todos los modelos tienen fallback automático a CPU. Los pesos son agnósticos al device de inferencia.

### Pipeline de Reentrenamiento Mensual

El reentrenamiento se ejecuta **el día 1 de cada mes a las 02:00 UTC**, fuera de las horas pico de la red eléctrica:

```
[Cron: día 1 del mes, 02:00 UTC]
    ↓
1. download_new_data.py  → descarga datos del mes anterior (ESIOS + AEMET + ENTSO-E)
    ↓
2. build_dataset.py      → añade al parquet histórico, recalcula proxies de frecuencia
    ↓
3. train_anomaly.py      → reentrena LSTM Autoencoder en CUDA, guarda lstm_ae_candidate.pth
4. train_forecast.py     → reentrena XGBoost forecast, guarda xgb_forecast_candidate.json
5. train_classifier.py   → reentrena GBM blackout, guarda xgb_blackout_candidate.json
    ↓ (en paralelo, 3+4+5 son independientes)
6. validate_models.py    → evalúa candidatos en hold-out set del mes anterior
                           Si AUC-ROC ≥ actual y MAE ≤ actual → promueve candidato
                           Si no mejora → mantiene pesos anteriores + alerta en log
    ↓
7. reload_mcp_server.py  → señaliza a MCP: ml-models para recargar pesos en memoria
                           sin downtime (carga nuevo → swap atómico → descarga viejo)
    ↓
8. Notificación en log   → "Reentrenamiento completado. Modelos promovidos: [lista]"
```

```
ml/models/
├── lstm_ae.pth               ← pesos en producción (activos)
├── xgb_forecast.json         ← pesos en producción (activos)
├── xgb_blackout.json         ← pesos en producción (activos)
├── history/
│   ├── 2026-06-01/           ← snapshot del mes anterior (rollback si falla)
│   └── 2026-05-01/
└── candidates/               ← modelos en validación (borrados tras swap)
```

---

## 5. Capa 0 — Disparadores

| Componente | Frecuencia | Acción |
|---|---|---|
| Cron Scheduler | cada 10 min | Invoca `ree_poller()` → almacena snapshot |
| Cron Scheduler | cada 1h | Invoca `RenewablesForecastSkill` |
| Cron Scheduler | cada 24h | Invoca `ReportAgent` |
| Event Trigger | cuando `ml_risk_score` > umbral | Invoca `OrchestratorAgent` para análisis profundo |
| Eval Runner | en CI / bajo demanda | Ejecuta golden tests y trajectory evals |

---

## 5. Capa 1 — Function Tools (Determinísticos, sin LLM)

### `ree_poller()`
- **Qué hace**: GET a ESIOS API, parsea respuesta JSON
- **Inputs**: endpoint, token (desde env var `ESIOS_TOKEN`, nunca hardcoded)
- **Outputs**: `dict` normalizado con demanda, generación por tecnología, intercambios, reserva
- **Error handling**: retry con backoff exponencial (3 intentos), log de gaps

### `data_normalizer(raw: dict) -> GridSnapshot`
- Convierte unidades a SI, timestamps a UTC, valida schema con Pydantic
- **Outputs**: `GridSnapshot` (dataclass validada)

### `stability_proxy_calculator(snapshot: GridSnapshot) -> StabilityProxies`
- Calcula los 4 proxies de frecuencia: balance, reserva activada, variación intercambios, % renovables
- Sustituye al antiguo `frequency_checker` — más robusto al no depender de un dato no disponible públicamente
- **Outputs**: `StabilityProxies(balance_mw: float, reserve_activated_pct: float, interchange_delta_mw: float, renewables_pct: float, composite_pressure: float)`

### `risk_scorer(snapshot: GridSnapshot, ml_score: float, baseline: Baseline) -> RiskScore`
- Combina el score ML con métricas actuales: `0.5 * ml_score + 0.3 * balance_pressure + 0.2 * renewables_risk`
- **Outputs**: `RiskScore(value: int [0-100], components: dict, level: Literal['LOW','MEDIUM','HIGH','CRITICAL'])`

### `alert_formatter(risk: RiskScore, anomalies: list, scope: str) -> Alert`
- Construye alerta estructurada con ámbito (nacional / zonal)
- **Outputs**: `Alert(severity, scope, timestamp, summary, details, recommended_action)`

---

## 6. Capa 2 — MCP Servers (6 servidores)

### `MCP: ree-grid`
Wrapper autenticado sobre ESIOS API con caché en memoria (TTL 9 min).
```
get_realtime_demand(region: str) -> DemandSnapshot
get_generation_mix(start: datetime, end: datetime) -> GenerationMix
get_interchange_flows(border: str) -> InterchangeFlow
get_programmed_generation() -> ProgrammedGeneration
get_reserve_activation() -> ReserveStatus
```
**Credencial**: `ESIOS_TOKEN` · **Histórico disponible desde**: 2014-01-01

### `MCP: weather`
Wrapper sobre AEMET OpenData + OpenMeteo como fallback.
```
get_wind_forecast(zone: str, hours: int) -> WindForecast
get_solar_radiation(zone: str, hours: int) -> RadiationForecast
get_aemet_alerts() -> list[MeteoAlert]
```
**Credencial**: `AEMET_API_KEY` · **Rate limit**: 50 req/min · **Histórico desde**: ~2000

### `MCP: entso-e`
Acceso a datos de interconexión europea.
```
get_cross_border_flows(from_country: str, to_country: str) -> CrossBorderFlow
get_eu_generation(country: str) -> EUGenerationSnapshot
get_capacity_margin(country: str) -> CapacityMargin
```
**Credencial**: `ENTSOE_TOKEN` · **Histórico desde**: 2011 (legacy CSV) / 2015 (API)

### `MCP: memory`
Persistencia local con SQLite.
```
store_snapshot(snapshot: GridSnapshot) -> None
get_baseline(window_hours: int) -> Baseline
query_incidents(start: datetime, end: datetime) -> list[Incident]
get_risk_history(hours: int) -> list[RiskScore]
```

### `MCP: ml-models` ← NUEVO
Expone los modelos ML entrenados como tools. Carga `.pkl` en memoria al iniciar. Inferencia determinística, sin LLM.
```
get_anomaly_score(window: list[GridSnapshot]) -> AnomalyResult
  # Isolation Forest: devuelve score [0-1] + features más anómalas

get_load_forecast(horizon_hours: int, context: dict) -> LoadForecast
  # XGBoost/Prophet: demanda prevista + intervalo de confianza

get_blackout_probability(features: StabilityProxies, forecast: LoadForecast) -> BlackoutRisk
  # Gradient Boosting calibrado: P(blackout_1h) [0-1] + feature importance
```
**Sin credenciales externas** · Modelos en `ml/models/*.pkl`

### `MCP: knowledge-base` ← NUEVO
RAG sobre documentos técnicos embebidos con ChromaDB.
```
search_regulations(query: str, top_k: int = 5) -> list[RegulatoryChunk]
  # Normas ENTSO-E, POs de REE, EN 50160

find_similar_incident(description: str, top_k: int = 3) -> list[IncidentRecord]
  # Post-mortems: apagón 2025, separación ibérica 2006, CEER reports

get_mitigation_procedure(situation: str) -> Procedure
  # Procedimientos operacionales de REE para situaciones de emergencia
```
**Embeddings**: `gemini-embedding-2` (multimodal, primer modelo de embeddings de la serie Gemini 3) · **Vector store**: ChromaDB local

---

## 7. Capa 3 — Skills (AgentTools)

### `AnomalyDetectorSkill`
- **Tipo**: `AgentTool` (síncrono, sin estado entre llamadas)
- **Modelo**: `get_model("fast_agent")` → `gemini-3.5-flash` · **Temperature**: 0.1
- **Por qué Skill**: una sola llamada, resultado estructurado, llamado cada 10 min → velocidad crítica
- **Tools disponibles**: `get_anomaly_score` (ML), `stability_proxy_calculator`, `get_baseline`, `search_regulations` (RAG)
- **Flujo**: ML detecta → RAG contextualiza → LLM razona y clasifica
- **Output**: `AnomalyReport(anomalies: list[Anomaly], severity: str, confidence: float, ml_score: float, regulatory_context: str, reasoning: str)`

### `RenewablesForecastSkill`
- **Tipo**: `AgentTool` (síncrono, sin estado)
- **Modelo**: `get_model("fast_agent")` → `gemini-3.5-flash` · **Temperature**: 0.1
- **Por qué Skill**: resultado único y estructurado, reutilizable por Orchestrator y ReportAgent
- **Tools disponibles**: `get_wind_forecast`, `get_solar_radiation`, `get_generation_mix`, `get_load_forecast` (ML), `get_baseline`
- **Output**: `RenewablesForecast(risk_level: str, expected_generation_pct: float, critical_window: str, ml_forecast_mw: float, reasoning: str)`

---

## 8. Capa 4 — Sub-agentes

### `ReportAgent`
- **Modelo**: `get_model("smart_agent")` → `gemini-3.1-pro` · **Temperature**: 0.4 · **Max turns**: 10
- **Por qué Sub-agent**: múltiples pasos (recopilar → analizar → redactar → formatear), número de pasos varía
- **Tools**: `query_incidents`, `get_risk_history`, `get_blackout_probability` (ML), `find_similar_incident` (RAG), escritura en `reports/`
- **Output**: informe HTML + Markdown con resumen ejecutivo, evolución del risk score, incidencias y comparación con eventos históricos similares

### `CrossBorderAgent`
- **Modelo**: `get_model("fast_agent")` → `gemini-3.5-flash` · **Temperature**: 0.2 · **Max turns**: 8
- **Por qué Sub-agent**: razonamiento multi-país con memoria de sesión
- **Tools**: `get_cross_border_flows`, `get_capacity_margin`, `get_eu_generation`, `get_mitigation_procedure` (RAG)
- **Output**: `CrossBorderAnalysis(can_import: bool, available_capacity_mw: float, countries: list[str], risk_mitigation: str)`

---

## 9. Capa 5 — OrchestratorAgent (Root Agent)

- **Modelo**: `get_model("fast_agent")` → `gemini-3.5-flash` · **Temperature**: 0.1 · **Max turns**: 20 · **Context**: 1M tokens
- **Skills**: `AnomalyDetectorSkill`, `RenewablesForecastSkill`
- **Sub-agents**: `ReportAgent`, `CrossBorderAgent`
- **Function tools**: `ree_poller`, `stability_proxy_calculator`, `risk_scorer`, `alert_formatter`
- **MCP directos**: `get_blackout_probability` (para decisiones rápidas de escalado)

**Lógica de decisión basada en `P(blackout_1h)`**:
```
P < 0.10  → logging pasivo, no acción
P 0.10–0.30 → AnomalyDetectorSkill → log con contexto
P 0.30–0.60 → Anomaly + RenewablesForecast → alerta WARNING (ámbito según zona)
P > 0.60  → todo anterior + CrossBorderAgent + alerta CRITICAL nacional + ReportAgent inmediato
```

---

## 10. Guardrails — ADK Hooks

### `before_tool_callback`
```python
def before_tool_callback(tool_name: str, tool_args: dict) -> dict | None:
    # 1. Bloquear acceso a archivos sensibles
    if tool_name in FILE_READ_TOOLS:
        path = tool_args.get("path", "")
        BLOCKED = [".env", "secrets", ".key", ".pem", "credentials", "token"]
        if any(b in path.lower() for b in BLOCKED):
            return {"error": "ACCESS_DENIED: sensitive file path blocked"}

    # 2. Rate limiting en APIs externas
    if tool_name in API_TOOLS:
        last_call = rate_limiter.get(tool_name)
        if last_call and (now() - last_call) < MIN_INTERVAL[tool_name]:
            return {"error": "RATE_LIMITED: too many requests"}

    # 3. Whitelist de tools permitidas por agente
    if tool_name not in ALLOWED_TOOLS[current_agent]:
        return {"error": f"UNAUTHORIZED: {tool_name} not allowed for {current_agent}"}

    return None
```

### `after_tool_callback`
```python
def after_tool_callback(tool_name: str, output: dict) -> dict:
    # Validar schema JSON de respuesta
    # Escanear output por patrones de credenciales (regex)
    # Validar rangos físicos (frecuencia [45-55]Hz, demanda > 0, score [0-1])
    # Truncar si supera 4k tokens antes de pasar al LLM
```

### `before_model_callback`
```python
def before_model_callback(messages: list) -> list:
    INJECTION_PATTERNS = ["ignore previous", "system:", "jailbreak", "forget your",
                          "new instruction", "disregard", "override"]
    for msg in messages:
        if any(p in str(msg.get("content", "")).lower() for p in INJECTION_PATTERNS):
            msg["content"] = "[SANITIZED: potential injection detected and removed]"
    return messages
```

### `after_model_callback`
```python
def after_model_callback(output: str) -> str:
    # Validar que outputs numéricos están en rangos físicamente plausibles
    # Verificar JSON válido si se esperaba JSON estructurado
    # Rechazar outputs que contengan patrones de credenciales
```

### `after_agent_callback`
```python
def after_agent_callback(agent_output: AgentOutput) -> AgentOutput:
    # Validar schema del output final del agente
    # Comprobar risk_score ∈ [0, 100], P(blackout) ∈ [0, 1]
    # Comprobar severity ∈ {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}
    # Comprobar scope ∈ {'nacional', 'peninsular', 'zonal', 'local'}
```

---

## 11. Eval & Tracing

### Tracing (ADK built-in)
- Cada tool call: nombre, args, duración, output
- Cada LLM call: prompt tokens, completion tokens, latencia, modelo
- Árbol de spans parent-child: `Orchestrator → Skill → Tool → ML Model`
- Export a JSON local + opcional Cloud Trace

### Eval de Outputs

| Componente | Métrica | Método |
|---|---|---|
| `AnomalyDetector` ML | Precision / Recall / F1 | Dataset etiquetado histórico ESIOS 2014-2024 |
| `LoadForecaster` ML | MAE, MAPE | Comparar forecast 6h vs demanda real posterior |
| `BlackoutRisk` ML | AUC-ROC, calibración | Hold-out set con eventos conocidos |
| `AnomalyDetectorSkill` | Precision / Recall alertas | Golden dataset apagón 28 abril 2025 |
| `RenewablesForecastSkill` | MAE a 6h | Comparar forecast vs generación real |
| `ReportAgent` | Calidad narrativa | LLM-as-judge (coherencia, precisión, cita correcta de normativa) |
| `OrchestratorAgent` | False positive rate, latencia end-to-end | % alertas CRITICAL sin evento real, tiempo 1ª alerta |

### Eval de Trayectoria

- **Golden dataset**: escenario del apagón del 28 de abril de 2025 (post-mortem REE publicado)
- **Evaluación**: ¿tools correctas? ¿orden correcto? ¿escaló en el momento adecuado? ¿RAG consultado antes de razonar?

```python
def test_critical_scenario_blackout_apr_2025():
    result = orchestrator.run(scenario="blackout_apr_2025")
    trajectory = result.trajectory

    assert "get_anomaly_score" in trajectory          # ML consultado
    assert "search_regulations" in trajectory          # RAG consultado
    assert "AnomalyDetectorSkill" in trajectory
    assert "CrossBorderAgent" in trajectory            # escalado correcto
    assert result.final_alert.severity == "CRITICAL"
    assert result.final_alert.scope == "nacional"
    assert result.risk_score > 80
    assert result.blackout_probability > 0.6
```

---

## 12. Estructura del Proyecto

```
gridguard/
├── agents/
│   ├── orchestrator.py              # Root Agent
│   ├── anomaly_detector.py          # Skill (AgentTool) — ML + RAG + LLM
│   ├── renewables_forecast.py       # Skill (AgentTool) — ML + LLM
│   ├── report_agent.py              # Sub-agent — ML + RAG + LLM
│   └── cross_border.py              # Sub-agent — RAG + LLM
├── tools/
│   ├── ree_poller.py                # Function Tool
│   ├── data_normalizer.py           # Function Tool
│   ├── stability_proxy_calculator.py # Function Tool (sustituye frequency_checker)
│   ├── risk_scorer.py               # Function Tool (combina ML + métricas)
│   └── alert_formatter.py           # Function Tool
├── mcp_servers/
│   ├── ree_grid_server.py           # MCP: ree-grid (ESIOS API)
│   ├── weather_server.py            # MCP: weather (AEMET + OpenMeteo)
│   ├── entsoe_server.py             # MCP: entso-e
│   ├── memory_server.py             # MCP: memory (SQLite)
│   ├── ml_models_server.py          # MCP: ml-models ← NUEVO
│   └── knowledge_base_server.py     # MCP: knowledge-base (ChromaDB) ← NUEVO
├── ml/
│   ├── data/
│   │   ├── download_esios.py        # Descarga histórico ESIOS 2014→
│   │   ├── download_aemet.py        # Descarga histórico AEMET 2000→
│   │   ├── download_entsoe.py       # Descarga histórico ENTSO-E 2011→
│   │   └── build_dataset.py         # Combina fuentes → dataset_training.parquet
│   ├── train/
│   │   ├── train_anomaly.py         # Isolation Forest / LSTM Autoencoder
│   │   ├── train_forecast.py        # XGBoost / Prophet
│   │   └── train_classifier.py      # Gradient Boosting calibrado
│   └── models/
│       ├── anomaly_detector.pkl
│       ├── load_forecaster.pkl
│       └── blackout_classifier.pkl
├── knowledge/
│   ├── documents/                   # PDFs y textos fuente
│   │   ├── po_ree_1_1.pdf
│   │   ├── po_ree_7_4.pdf
│   │   ├── entso_e_network_code.pdf
│   │   ├── postmortem_apagon_2025.pdf
│   │   └── separacion_iberica_2006.pdf
│   ├── build_kb.py                  # Embeddings → ChromaDB
│   └── chroma_db/                   # Vector store local
├── guardrails/
│   └── hooks.py                     # Todos los ADK callbacks
├── eval/
│   ├── golden_dataset/
│   │   ├── blackout_apr_2025.json   # Escenario principal
│   │   └── normal_baseline.json     # Período estable para control
│   ├── test_trajectories.py
│   ├── test_outputs.py
│   └── test_ml_models.py            # Eval offline de modelos ML
├── schemas/
│   └── models.py                    # Pydantic dataclasses
├── .env.example
├── requirements.txt
└── README.md
```

---

## 13. Variables de Entorno

```bash
# .env.example — nunca incluir valores reales en el repo
ESIOS_TOKEN=your_token_here          # Solicitar en consultasios@ree.es
AEMET_API_KEY=your_key_here          # opendata.aemet.es
ENTSOE_TOKEN=your_token_here         # transparency.entsoe.eu
GOOGLE_API_KEY=your_gemini_key_here  # Google AI Studio
DB_PATH=./data/gridguard.db
CHROMA_DB_PATH=./knowledge/chroma_db
ML_MODELS_PATH=./ml/models
LOG_LEVEL=INFO
RISK_ALERT_THRESHOLD=0.30            # P(blackout) para WARNING
RISK_CRITICAL_THRESHOLD=0.60         # P(blackout) para CRITICAL
```

---

## 14. Conceptos ADK demostrados (requisito: mínimo 3)

1. ✅ **Multi-agent system** — Orchestrator (Root) + 2 Skills (AgentTool) + 2 Sub-agents con roles diferenciados
2. ✅ **MCP servers** — 6 servidores MCP custom (ree-grid, weather, entso-e, memory, ml-models, knowledge-base)
3. ✅ **AgentTool (Skills)** — `AnomalyDetectorSkill` y `RenewablesForecastSkill` como AgentTools síncronos
4. ✅ **Callbacks / Guardrails** — 5 hooks con lógica real (anti-.env, rate-limit, anti-injection, bounds check, schema validation)
5. ✅ **Memory / State** — MCP memory (SQLite) para baseline e histórico de snapshots
6. ✅ **Eval framework** — Output eval (ML metrics + LLM-as-judge) + Trajectory eval con golden dataset real
7. ✅ **Tracing** — ADK tracing en todos los agentes con árbol de spans
8. ✅ **ML integrado** — 3 modelos (anomaly detection, forecasting, blackout classifier) expuestos como MCP tools
9. ✅ **RAG** — Knowledge base con normas ENTSO-E, post-mortems y procedimientos REE vía ChromaDB
