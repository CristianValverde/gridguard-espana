# GridGuard España — Architecture Specification

> Multi-agent system for monitoring the stability of the Spanish electrical grid  
> Capstone Project · 5-Day AI Agents Intensive (Google × Kaggle) · June 2026

---

## 1. Design Principles

### Shift-Left: Deterministic First, LLM Only When It Adds Value

The core principle is **not to use an LLM where a Python function or ML model suffices**. Three levels of intelligence, each reserved for what truly needs it:

| If the task is… | Use… |
|---|---|
| API call + parsing + validation | Function Tool (pure Python) |
| Statistical pattern detection in time series | ML Model (deterministic at inference) |
| Technical / regulatory context retrieval | RAG (ChromaDB) |
| Multi-signal analysis with reasoning | Skill (AgentTool + LLM) |
| Multi-step task with internal state | Sub-agent (LLM) |
| Global coordination + escalation decisions | Orchestrator (Root Agent) |

### ML vs. Domain Knowledge (RAG) — Complementary Roles

- **ML** answers *what is statistically happening?* — detects anomalies in time series, forecasts load, computes blackout probability calibrated on historical data. Deterministic at inference, reproducible and auditable.
- **Markov Chain** answers *where is the system heading?* — given the current state S0–S4 and the empirical transition matrix, projects the risk trajectory over the next N hours. Pure Python/NumPy, no LLM.
- **RAG** answers *what does this mean?* — retrieves ENTSO-E standards, REE operating procedures, prior incident post-mortems. The LLM without RAG reasons in the abstract; with RAG it reasons about the real Spanish grid.
- **LLM** answers *what do we do?* — synthesises ML signal + Markov trajectory + RAG context and decides severity, scope (national/zonal) and actionable recommendation.

### Skills vs Sub-agents

- **Skill (AgentTool)**: agent wrapped as a synchronous tool. No state between calls. Ideal for focused, reusable analyses with a single structured result.
- **Sub-agent**: agent with its own loop, tools and session memory. Ideal for autonomous multi-step tasks where the number of steps is unpredictable.

---

## 2. Layered Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║  OFFLINE: Training Pipeline & Knowledge Base                         ║
║  (ESIOS 2014→ · AEMET 2000→ · ENTSO-E 2011→)                       ║
║           ↓                        ↓                    ↓            ║
║   ML Models (.pkl/.json)   Markov Matrix (.npz)  Knowledge Base      ║
║                            A[5×5] + thresholds    (ChromaDB)         ║
╚════════════╤═══════════════════════╤═══════════════════╤═════════════╝
             │                       │                   │
[Layer 0] Triggers (Cron / Event Queue / Eval Runner)
    ↓
[Layer 1] Function Tools  (pure Python, no LLM)
           + markov_state_tracker  (runtime Markov inference, no LLM)
    ↓ [Hook: before_tool_callback — rate-limit, .env block, sanitisation]
[Layer 2] MCP Servers  (5 servers: grid-data · weather ·
                         memory · ml-models · knowledge-base)
    ↓ [Hook: after_tool_callback — schema, credentials, truncation]
           + on_state_transition hook (S→S+1 escalation guard)
[Layer 3] Skills / AgentTools  (AnomalyDetector · RenewablesForecast · RegulatoryCompliance)
    ↓ [Hook: before_model_callback — anti-injection]
           + on_forecast_risk_horizon hook (24h risk window guard)
[Layer 4] Sub-agents  (ReportAgent · CrossBorderAgent)
    ↓ [Hook: after_model_callback + after_agent_callback — bounds, schema]
[Layer 5] OrchestratorAgent  (Root Agent · gemini-3.5-flash)
    ↓
[Outputs] Alerts · Risk Score · HTML Dashboard · Daily Report
                       │
[Cross-cutting] Tracing (ADK) · Eval (output + trajectory)
```

---

## 3. Model Registry — Always Up-to-Date Models

### Problem Solved
Gemini models evolve quickly: `gemini-2.0-flash` was marked **Shut Down** in June 2026 with little warning. Hard-coding a model name is guaranteed technical debt.

### Solution: Model Registry with Weekly Auto-Update

The system uses `config/model_registry.yaml` as the single source of truth for models. A weekly job updates it automatically by querying Google's official documentation.

```yaml
# config/model_registry.yaml — auto-generated, do NOT edit manually
# Last updated: 2026-06-27 | Source: ai.google.dev/gemini-api/docs/models

models:
  fast_agent:          "gemini-3.5-flash"        # Stable · agentic tasks optimised
  smart_agent:         "gemini-3.1-pro"           # Preview · complex reasoning · 2M ctx
  embedding:           "gemini-embedding-2"        # Stable · multimodal embeddings for RAG
  fallback_fast:       "gemini-3.1-flash-lite"    # Stable · cost-efficient fallback

registry_meta:
  source_url:   "https://ai.google.dev/gemini-api/docs/models"
  updated_at:   "2026-06-27T10:00:00Z"
  next_update:  "2026-07-04T10:00:00Z"
  stable_only:  true   # never use Experimental models in production
```

```python
# config/model_loader.py — all agents import from here
import yaml

def get_model(role: str) -> str:
    """Reads the updated registry. Never uses hard-coded strings."""
    with open("config/model_registry.yaml") as f:
        registry = yaml.safe_load(f)
    return registry["models"][role]
```

### Current Models (June 2026)

| Role in the System | Model | Status | Why |
|---|---|---|---|
| Fast agents (Orchestrator, Skills) | `gemini-3.5-flash` | **Stable** | Designed for agentic tasks, outperforms 3.1 Pro on agent benchmarks (MCP Atlas 83.6%) |
| Complex reasoning agent (ReportAgent) | `gemini-3.1-pro` | Preview | 2M token context, advanced reasoning for report synthesis |
| RAG Embeddings | `gemini-embedding-2` | **Stable** | First multimodal embeddings model, best for heterogeneous RAG |
| Fallback / testing | `gemini-3.1-flash-lite` | **Stable** | Cheaper for CI and trajectory tests |
| ~~gemini-2.0-flash~~ | ~~gemini-2.0-flash~~ | **⛔ SHUT DOWN** | Removed from architecture |

---

## 4. Offline Pipeline — Historical Data and Training

### Confirmed Data Sources

| Source | Since | Granularity | Key Data |
|---|---|---|---|
| **ESIOS / REData** (free token) | 2014-01-01 | 10 min / hourly | Real demand, generation by technology, interchanges, prices, regulation reserve |
| **REData** (no token) | 1990 | Monthly / annual | Long statistical series, electricity balance |
| **AEMET OpenData** (free token) | ~2000 | Hourly / daily | Wind, solar radiation, temperature per station |
| **ENTSO-E Transparency** (free token) | 2011 (legacy CSV) / 2015 (API) | Hourly | Cross-border flows, EU generation, interconnection capacity |

**Resulting volume**: ~10 years × 365 days × 6 readings/hour = **~525,000 time points** at 10 min granularity.

### Critical Note: Grid Frequency Not Available via Public API

**Grid frequency (Hz)** data is not published in the ESIOS API — it is real-time telemetry that REE does not expose for operational security reasons. The following **proxies** are used, which are the standard features in academic literature for instability prediction in high-renewable-penetration grids:

| Proxy | ESIOS Indicator | Why It Works |
|---|---|---|
| Supply/demand balance | Real generation − Real demand | Direct difference → frequency pressure |
| Activated regulation reserve | Secondary reserve indicators | Activated when frequency deviates |
| Interchange urgency | Sharp variation in FR/PT/MA flows | Emergency import → frequency stress |
| % unmanageable renewables | (Solar + Wind) / Total demand | Low synchronous inertia → higher vulnerability |

### Golden Dataset — Blackout of 28 April 2025

REE's post-mortem report for the 28 April 2025 blackout is published with minute-by-minute reconstruction. It is the primary test case for the Eval system.

### ML Models to Train

| Model | Algorithm | Input | Output | Library |
|---|---|---|---|---|
| **Anomaly Detector** | Isolation Forest / LSTM Autoencoder | 6h window of metrics (demand, mix, interchanges, frequency proxies) | `anomaly_score` [0–1] + most anomalous features | scikit-learn / PyTorch |
| **Load Forecaster** | XGBoost / Prophet | Hour, weekday, temperature, holiday, 30-day history | Forecast demand 6h + confidence interval | XGBoost / Prophet |
| **Blackout Risk Classifier** | Calibrated Gradient Boosting | `anomaly_score` + `load_forecast` + `renewables_pct` + `reserve_margin` | `P(blackout_1h)` [0–1] + feature importance | scikit-learn |

### Feature Store with Train/Serve Parity (`ml/features.py` + `ml/serve_features.py`)

All history-dependent features are computed by a **single feature layer** consumed by
training (`build_dataset`), serving (`ml_models_server` via `serve_feature_row` /
`serve_feature_frame` over the trailing 168h window from the `snapshots` table) and
validation (`validate_models` feeds the golden with its REAL trailing week from the
ESIOS dataset). Parity is guaranteed by construction and guarded by
`tests/ml/test_serve_features.py` (serve row == train row at 1e-9).

Key derived features (`ml.features.FEATURE_COLS`, 41 columns): physical proxies
(balance, reserve, interchange, renewables), **inertia proxy** (non-synchronous
penetration), **ROCOF proxies** (1h deltas + rolling volatility), **per-link
interconnection stress** (ES-FR/PT/MA vs NTC), trailing 7-day rolling composite
pressure with absolute-formula cold-start fallback, 7-day z-scores, lags and
temporal encodings. Missing technology/link readings contribute 0 MW (matching
serve semantics); missing demand propagates NaN (row not evaluable).

### Ensemble Fusion — Main Risk Signal (`tools/ensemble_scorer.py`)

`P(blackout)` fuses three detectors with distinct inductive biases, LSTM
**load-bearing by design** (weights not tuned on the golden, mitigating D5 n=1):

| Signal | Model | Weight | Role |
|---|---|---|---|
| `lstm_recon_norm` | LSTM Autoencoder (unsupervised) | 0.5 | Primary detector; override floor P≥0.6 if ≥0.9 |
| `if_score` | Isolation Forest via `decision_function` + `normalize_if_signal` | 0.2 | Point-anomaly corroboration |
| `clf_proba` | Calibrated XGBoost (supervised, capped by D5) | 0.3 | Secondary vote |

Exposed as `get_ensemble_score` / `get_blackout_probability_ensemble` (MCP:
ml-models); the orchestrator passes the fused probability as `ml_score` to
`risk_scorer`. Golden validation (2026-07-02, real trailing context): **IF PASS ·
LSTM PASS (reconstruction error 0.043 → 1.53 at T-3min) · ENSEMBLE PASS (monotonic
toward onset) · CLF FAIL (flat 0.0009 — D5, n=1 S4 event, documented honestly)**.

### Markov Chain — Grid State Transition Matrix

The Markov layer is trained offline alongside the ML models and used at runtime without any LLM:

| Component | Description |
|---|---|
| `ml/grid_state_tracker.py` | **Offline**: estimates A[5×5] empirical transition matrix from historical states (S0–S4), blends with expert Dirichlet priors for rare high-severity transitions, persists matrix + thresholds to `ml/models/markov_state.npz` |
| `tools/markov_state_tracker.py` | **Runtime, no LLM**: loads `.npz`, discretises current state (`compute_state`), returns matrix (`get_transition_matrix`), projects trajectory (`get_trajectory`), computes `prob_reach_severity` |

**Grid states:**
- S0 — Normal
- S1 — Stress
- S2 — Notable incident
- S3 — Major incident
- S4 — Blackout

### Documents for RAG (Knowledge Base)

| Category | Documents | Purpose |
|---|---|---|
| **Operating standards** | REE PO 1.1, PO 1.2, PO 7.4; ENTSO-E Network Code; EN 50160 | Agent knows what 49.8 Hz means in the European regulatory context |
| **Post-mortems** | REE report blackout 28 Apr 2025; Iberian separation 2006; CEER incidents 2018–2024 | Recognises prior incident patterns and cites them |
| **Technical knowledge** | Synchronous and synthetic inertia; Primary/secondary regulation reserve; Peninsula topology; High renewable penetration | Explains causes and recommends mitigations |

---

## 5. Layer 0 — Triggers

| Component | Frequency | Action |
|---|---|---|
| Cron Scheduler | every 10 min | Invokes `incremental_load.catch_up()` → stores snapshots (training scale) |
| Cron Scheduler | every 1h | Invokes `RenewablesForecastSkill` |
| Cron Scheduler | every 24h | Invokes `ReportAgent` |
| Event Trigger | when `ml_risk_score` > threshold | Invokes `OrchestratorAgent` for deep analysis |
| Eval Runner | in CI / on demand | Runs golden tests and trajectory evals |

---

## 6. Layer 1 — Function Tools (Deterministic, No LLM)

### `ree_poller()`
- **What it does**: GET to ESIOS API, parses JSON response
- **Inputs**: endpoint, token (from env var `ESIOS_TOKEN`, never hard-coded)
- **Outputs**: normalised `dict` with demand, generation by technology, interchanges, reserve
- **Error handling**: retry with exponential backoff (3 attempts), gap logging

### `data_normalizer(raw: dict) -> GridSnapshot`
- Converts units to SI, timestamps to UTC, validates schema with Pydantic
- **Outputs**: `GridSnapshot` (validated dataclass)

### `stability_proxy_calculator(snapshot: GridSnapshot) -> StabilityProxies`
- Computes the 4 frequency proxies: balance, activated reserve, interchange variation, % renewables
- **Outputs**: `StabilityProxies(balance_mw, reserve_activated_pct, interchange_delta_mw, renewables_pct, composite_pressure)`

### `risk_scorer(snapshot, ml_score, baseline) -> RiskScore`
- Combines ML score with current metrics: `0.5 * ml_score + 0.3 * balance_pressure + 0.2 * renewables_risk`
- **Outputs**: `RiskScore(value: int [0-100], components: dict, level: Literal['LOW','MEDIUM','HIGH','CRITICAL'])`

### `alert_formatter(risk, anomalies, scope) -> Alert`
- Builds structured alert with scope (national / zonal)
- **Outputs**: `Alert(severity, scope, timestamp, summary, details, recommended_action)`

### `markov_state_tracker` (runtime Markov inference, no LLM)
- **`compute_state(features)`** — discretises current state S0–S4 from `{composite_pressure, renewables_pct, balance_mw}` using persisted thresholds
- **`get_transition_matrix()`** — returns A[5×5]
- **`get_trajectory(state, horizon_hours)`** — most probable state sequence for next N hours
- **`prob_reach_severity(state, target, horizon_hours)`** — P(reaching state ≥ target within horizon)

---

## 7. Layer 2 — MCP Servers (5 servers)

### `MCP: grid-data`
Authenticated wrapper over ESIOS (Spanish generation mix) + ENTSO-E (European interconnection and market data). Consolidates the former `ree-grid` and `entso-e` servers into one, in-memory cache (TTL 9 min for ESIOS, 1 h for ENTSO-E).
```
get_generation_mix(start, end) → GenerationMix
get_cross_border_flows(from_country, to_country) → CrossBorderFlow
get_eu_generation(country) → EUGenerationSnapshot
get_capacity_margin(country) → CapacityMargin
get_eu_market_prices(countries, date) → dict  # day-ahead prices (€/MWh, A44)
```
**Credentials**: `ESIOS_TOKEN` + `ENTSOE_TOKEN` (each tool uses its own) · **Historical data from**: 2014-01-01 (ESIOS)

### `MCP: weather`
Wrapper over AEMET OpenData + OpenMeteo fallback.
```
get_wind_forecast(zone, hours) → WindForecast
get_solar_radiation(zone, hours) → RadiationForecast
get_weather_forecast_24h(zones) → dict  # hourly 24h forecast per zone
```
**Credential**: `AEMET_API_KEY` · **Rate limit**: 50 req/min

### `MCP: memory`
Local persistence with SQLite.
```
store_risk_score(risk_score) → dict
get_baseline(window_hours) → Baseline
query_incidents(start, end) → list[Incident]
get_risk_history(hours) → list[RiskScore]
store_grid_state(timestamp, state) → dict    # persist Markov state
get_state_transitions(hours) → list[dict]    # Markov state history
```
**Database**: SQLite (`gridguard.db`)

### `MCP: ml-models`
Exposes trained ML models and the Markov layer as tools. Loads `.pkl`/`.npz` into memory on startup. Deterministic inference, no LLM.
```
get_anomaly_score(window) → AnomalyResult
  # Isolation Forest: score [0-1] + most anomalous features

get_ensemble_score(features, window) → EnsembleResult
  # LSTM + Isolation Forest + Classifier fusion score

get_serve_features(window_hours) → dict
  # Train/serve-parity feature window assembly

get_blackout_probability_ensemble() → BlackoutRisk
  # Ensemble-fused P(blackout) using latest persisted state

get_load_forecast(horizon_hours, context) → LoadForecast
  # XGBoost/Prophet: forecast demand + confidence interval

get_blackout_probability(features, forecast) → BlackoutRisk
  # Calibrated Gradient Boosting: P(blackout_1h) [0-1] + feature importance

reload_models() → dict
  # Hot-reloads all models/weights from disk

get_grid_state(features) → {state, label, confidence, trajectory_6h}
  # Discretises current state S0–S4 + 6h trajectory

get_markov_trajectory(features, horizon_hours) → {trajectory, prob_reach_s3, prob_reach_s4}
  # Projected state trajectory + escalation probabilities
```
**No external credentials** · Models in `ml/models/` (includes `markov_state.npz`)

### `MCP: knowledge-base`
RAG over technical documents embedded with ChromaDB.
```
search_regulations(query, top_k=5) → list[RegulatoryChunk]
  # ENTSO-E standards, REE POs, EN 50160

find_similar_incident(description, top_k=3) → list[IncidentRecord]
  # Post-mortems: blackout 2025, Iberian separation 2006, CEER reports

get_mitigation_procedure(situation) → Procedure
  # REE operational procedures for emergency situations
```
**Embeddings**: `gemini-embedding-2` · **Vector store**: ChromaDB (local)

---

## 8. Layer 3 — Skills (AgentTools)

### `AnomalyDetectorSkill`
- **Type**: `AgentTool` (synchronous, no state between calls)
- **Model**: `get_model("fast_agent")` → `gemini-3.5-flash` · **Temperature**: 0.1
- **Why Skill**: single synchronous call, structured result, called every 10 min → speed is critical
- **Tools**: `get_anomaly_score` (ML), `stability_proxy_calculator`, `get_baseline`, `search_regulations` (RAG), `get_grid_state` (Markov)
- **Flow**: ML detects → Markov projects trajectory → RAG contextualises → LLM reasons and classifies
- **Output**: `AnomalyReport(anomalies, severity, confidence, ml_score, markov_trajectory, regulatory_context, reasoning)`

### `RenewablesForecastSkill`
- **Type**: `AgentTool` (synchronous, no state)
- **Model**: `get_model("fast_agent")` → `gemini-3.5-flash` · **Temperature**: 0.1
- **Why Skill**: single structured result, reusable by Orchestrator and ReportAgent
- **Tools**: `get_wind_forecast`, `get_solar_radiation`, `get_weather_forecast_24h`, `get_load_forecast` (ML), `get_generation_mix`, `get_baseline`
- **Output**: `RenewablesForecast(risk_level, expected_generation_pct, critical_window_24h, ml_forecast_mw, reasoning)`

### `RegulatoryComplianceSkill` ← NEW
- **Type**: `AgentTool` (synchronous, no state)
- **Model**: `get_model("fast_agent")` → `gemini-3.5-flash` · **Temperature**: 0.1
- **Why Skill**: focused regulatory lookup with single structured output; called when anomaly detected
- **Tools**: `search_regulations` (RAG), `get_grid_state` (Markov)
- **Flow**: LLM queries RAG for applicable regulations given current state → returns compliance report
- **Output**: `ComplianceReport(applicable_regulations, mandatory_actions, compliance_status, reasoning)`

---

## 9. Layer 4 — Sub-agents

### `ReportAgent`
- **Model**: `get_model("smart_agent")` → `gemini-3.1-pro` · **Temperature**: 0.4 · **Max turns**: 10
- **Why Sub-agent**: multiple steps (collect → analyse → draft → format), variable number of steps
- **Tools**: `query_incidents`, `get_risk_history`, `get_blackout_probability` (ML), `find_similar_incident` (RAG), `get_state_transitions` (Markov), write to `reports/`
- **Output**: HTML + Markdown report with executive summary, risk score evolution, incidents, and comparison with similar historical events

### `CrossBorderAgent`
- **Model**: `get_model("fast_agent")` → `gemini-3.5-flash` · **Temperature**: 0.2 · **Max turns**: 8
- **Why Sub-agent**: multi-country reasoning with session memory
- **Tools**: `get_cross_border_flows`, `get_capacity_margin`, `get_eu_generation`, `get_eu_market_prices`, `get_mitigation_procedure` (RAG)
- **Output**: `CrossBorderAnalysis(can_import, available_capacity_mw, countries, risk_mitigation)`

---

## 10. Layer 5 — OrchestratorAgent (Root Agent)

- **Model**: `get_model("fast_agent")` → `gemini-3.5-flash` · **Temperature**: 0.1 · **Max turns**: 20 · **Context**: 1M tokens
- **Skills**: `AnomalyDetectorSkill`, `RenewablesForecastSkill`, `RegulatoryComplianceSkill`
- **Sub-agents**: `ReportAgent`, `CrossBorderAgent`
- **Function tools**: `ree_poller`, `stability_proxy_calculator`, `risk_scorer`, `alert_formatter`
- **MCP direct**: `get_blackout_probability` (for fast escalation decisions), `get_markov_trajectory`

**Escalation logic based on P(blackout_1h) and Markov trajectory**:
```
P < 0.10                    → passive logging, no action
P 0.10–0.30                 → AnomalyDetectorSkill → log with context
P 0.30–0.60 or S≥S2 →S3   → Anomaly + RenewablesForecast + WARNING alert (zonal scope)
P > 0.60  or S≥S3 →S4     → all above + CrossBorderAgent + RegulatoryCompliance + CRITICAL national alert + immediate ReportAgent
```

---

## 11. Guardrails — ADK Hooks

### `before_tool_callback`
```python
def before_tool_callback(tool_name, tool_args):
    # 1. Block access to sensitive files
    if tool_name in FILE_READ_TOOLS:
        BLOCKED = [".env", "secrets", ".key", ".pem", "credentials", "token"]
        if any(b in tool_args.get("path", "").lower() for b in BLOCKED):
            return {"error": "ACCESS_DENIED: sensitive file path blocked"}

    # 2. Rate limiting on external APIs
    if tool_name in API_TOOLS:
        if rate_limiter.is_too_soon(tool_name):
            return {"error": "RATE_LIMITED: too many requests"}

    # 3. Tool whitelist per agent
    if tool_name not in ALLOWED_TOOLS[current_agent]:
        return {"error": f"UNAUTHORIZED: {tool_name} not allowed for {current_agent}"}
```

### `after_tool_callback`
```python
def after_tool_callback(tool_name, output):
    # Validate JSON response schema
    # Scan output for credential patterns (regex)
    # Validate physical ranges (demand > 0, score [0-1], state [0-4])
    # Truncate to 4k tokens before passing to LLM
```

### `on_state_transition` ← NEW
```python
def on_state_transition(prev_state, new_state):
    # Triggered when Markov state changes
    # If prev < new (escalation): log + optionally force AnomalyDetectorSkill
    # If new == S4: emit CRITICAL alert regardless of P(blackout) threshold
    # Prevents silent state drift going unnoticed between 10-min cron cycles
```

### `before_model_callback`
```python
def before_model_callback(messages):
    INJECTION_PATTERNS = ["ignore previous", "system:", "jailbreak", "forget your",
                          "new instruction", "disregard", "override"]
    for msg in messages:
        if any(p in str(msg.get("content", "")).lower() for p in INJECTION_PATTERNS):
            msg["content"] = "[SANITIZED: potential injection detected and removed]"
    return messages
```

### `on_forecast_risk_horizon` ← NEW
```python
def on_forecast_risk_horizon(trajectory, horizon_hours=24):
    # Triggered after RenewablesForecastSkill runs
    # If prob_reach_s3 > 0.3 within 24h window → escalate to WARNING
    # If prob_reach_s4 > 0.1 within 24h window → escalate to CRITICAL
    # Bridges Markov trajectory and the alert pipeline
```

### `after_model_callback`
```python
def after_model_callback(output):
    # Validate that numeric outputs are in physically plausible ranges
    # Verify valid JSON if structured JSON was expected
    # Reject outputs containing credential patterns
```

### `after_agent_callback`
```python
def after_agent_callback(agent_output):
    # Validate final agent output schema
    # Check risk_score ∈ [0, 100], P(blackout) ∈ [0, 1]
    # Check severity ∈ {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}
    # Check scope ∈ {'national', 'peninsular', 'zonal', 'local'}
```

---

## 12. Eval & Tracing

### Tracing (ADK built-in)
- Every tool call: name, args, duration, output
- Every LLM call: prompt tokens, completion tokens, latency, model
- Parent-child span tree: `Orchestrator → Skill → Tool → ML Model`
- Export to local JSON + optional Cloud Trace

### Output Eval

| Component | Metric | Method |
|---|---|---|
| `AnomalyDetector` ML | Precision / Recall / F1 | Labelled historical dataset ESIOS 2014–2024 |
| `LoadForecaster` ML | MAE, MAPE | Compare 6h forecast vs actual subsequent demand |
| `BlackoutRisk` ML | AUC-ROC, calibration | Hold-out set with known events |
| `AnomalyDetectorSkill` | Precision / Recall alerts | Golden dataset blackout 28 April 2025 |
| `RenewablesForecastSkill` | MAE at 6h | Compare forecast vs actual generation |
| `RegulatoryComplianceSkill` | Citation accuracy | LLM-as-judge: are cited regulations actually applicable? |
| `ReportAgent` | Narrative quality | LLM-as-judge (coherence, accuracy, correct citation of regulations) |
| `OrchestratorAgent` | False positive rate, end-to-end latency | % CRITICAL alerts without real event, time of 1st alert |

### Trajectory Eval

- **Golden dataset**: scenario from the blackout of 28 April 2025 (REE published post-mortem)
- **Evaluation**: correct tools? correct order? escalated at the right time? RAG consulted before reasoning? Markov trajectory consulted before escalation?

```python
def test_critical_scenario_blackout_apr_2025():
    result = orchestrator.run(scenario="blackout_apr_2025")
    trajectory = result.trajectory

    assert "get_anomaly_score" in trajectory          # ML consulted
    assert "get_markov_trajectory" in trajectory       # Markov consulted
    assert "search_regulations" in trajectory          # RAG consulted
    assert "AnomalyDetectorSkill" in trajectory
    assert "RegulatoryComplianceSkill" in trajectory
    assert "CrossBorderAgent" in trajectory            # correct escalation
    assert result.final_alert.severity == "CRITICAL"
    assert result.final_alert.scope == "national"
    assert result.risk_score > 80
    assert result.blackout_probability > 0.6
```

---

## 13. Project Structure

```
gridguard/
├── agents/
│   ├── orchestrator.py              # Root Agent
│   ├── anomaly_detector.py          # Skill (AgentTool) — ML + Markov + RAG + LLM
│   ├── renewables_forecast.py       # Skill (AgentTool) — ML + LLM
│   ├── regulatory_compliance.py     # Skill (AgentTool) — RAG + Markov + LLM  ← NEW
│   ├── report_agent.py              # Sub-agent — ML + Markov + RAG + LLM
│   └── cross_border.py              # Sub-agent — RAG + LLM
├── tools/
│   ├── ree_poller.py                # Function Tool
│   ├── data_normalizer.py           # Function Tool
│   ├── stability_proxy_calculator.py # Function Tool
│   ├── risk_scorer.py               # Function Tool
│   ├── alert_formatter.py           # Function Tool
│   ├── ensemble_scorer.py           # LSTM+IF+CLF fusion (no LLM)  ← NEW
│   └── markov_state_tracker.py      # Runtime Markov inference (no LLM)  ← NEW
├── mcp_servers/
│   ├── grid_data_server.py          # MCP: grid-data (ESIOS + ENTSO-E, incl. day-ahead market prices)  ← NEW
│   ├── weather_server.py            # MCP: weather (AEMET + OpenMeteo + 24h forecast)
│   ├── memory_server.py             # MCP: memory (SQLite + Markov state history)
│   ├── ml_models_server.py          # MCP: ml-models (incl. Markov state + trajectory)
│   └── knowledge_base_server.py     # MCP: knowledge-base (ChromaDB)
├── ml/
│   ├── data/
│   │   ├── download_esios.py
│   │   ├── download_aemet.py
│   │   ├── download_entsoe.py
│   │   ├── build_dataset.py         # delegates features to ml/features.py (--refresh in-place)
│   │   └── incremental_load.py      # ESIOS → snapshots DB (backfill/catch_up)  ← NEW
│   ├── train/
│   │   ├── train_anomaly.py
│   │   ├── train_forecast.py
│   │   ├── train_classifier.py
│   │   └── validate_models.py       # golden w/ real trailing context + validate_ensemble
│   ├── features.py                  # SINGLE feature layer (train/serve parity)  ← NEW
│   ├── serve_features.py            # serve-time window assembly + DB bridge  ← NEW
│   ├── incidents.py                 # State definitions S0–S4 + thresholds  ← NEW
│   ├── grid_state_tracker.py        # Offline Markov matrix estimation  ← NEW
│   └── models/
│       ├── anomaly_detector.pkl
│       ├── load_forecaster.pkl
│       ├── blackout_classifier.pkl
│       └── markov_state.npz         # A[5×5] + thresholds  ← NEW
├── knowledge/
│   ├── documents/                   # Source PDFs
│   ├── build_kb.py                  # Embeddings → ChromaDB
│   └── chroma_db/                   # Local vector store (ChromaDB)
├── guardrails/
│   └── hooks.py                     # All ADK callbacks (incl. on_state_transition)
├── eval/
│   ├── golden_dataset/
│   │   ├── blackout_apr_2025.json
│   │   └── normal_baseline.json
│   ├── test_trajectories.py
│   ├── test_outputs.py
│   └── test_ml_models.py
├── schemas/
│   └── models.py                    # Pydantic dataclasses (incl. ComplianceReport, MarketPrice)
├── config/
│   ├── model_registry.yaml
│   └── model_loader.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 14. Environment Variables

```bash
# .env.example — never include real values in the repo
ESIOS_TOKEN=your_token_here          # Request at consultasios@ree.es
AEMET_API_KEY=your_key_here          # opendata.aemet.es
ENTSOE_TOKEN=your_token_here         # transparency.entsoe.eu
GOOGLE_API_KEY=your_gemini_key_here  # Google AI Studio
DB_PATH=./data/gridguard.db
CHROMA_DB_PATH=./knowledge/chroma_db
ML_MODELS_PATH=./ml/models
LOG_LEVEL=INFO
RISK_ALERT_THRESHOLD=0.30            # P(blackout) for WARNING
RISK_CRITICAL_THRESHOLD=0.60         # P(blackout) for CRITICAL
```

---

## 15. ADK Concepts Demonstrated (minimum required: 3)

1. ✅ **Multi-agent system** — Orchestrator (Root) + 3 Skills (AgentTool) + 2 Sub-agents with differentiated roles
2. ✅ **MCP servers** — 5 custom MCP servers (grid-data, weather, memory, ml-models, knowledge-base)
3. ✅ **AgentTool (Skills)** — `AnomalyDetectorSkill`, `RenewablesForecastSkill`, `RegulatoryComplianceSkill`
4. ✅ **Callbacks / Guardrails** — 7 hooks with real logic (anti-.env, rate-limit, state transition, injection guard, forecast horizon, bounds check, schema validation)
5. ✅ **Memory / State** — MCP memory (SQLite) for baseline, snapshot history and Markov state history
6. ✅ **Eval framework** — Output eval (ML metrics + LLM-as-judge) + Trajectory eval with real golden dataset
7. ✅ **Tracing** — ADK tracing on all agents with span tree
8. ✅ **ML integrated** — 3 models (anomaly detection, forecasting, blackout classifier) + Markov chain exposed as MCP tools
9. ✅ **RAG** — Knowledge base with ENTSO-E standards, post-mortems and REE procedures via ChromaDB
10. ✅ **Markov chain** — Grid state trajectory model (pure Python/NumPy) integrated across MCP layer and agent decision logic
