# Diseño — Resolución de la limitación de eficacia vía paridad train/serve + ensemble

**Fecha:** 2026-07-01
**Rama:** `agent/brave-kilby-98ceb0`
**Enfoque elegido:** C (feature store con estado + proxies físicos + fusión ensemble)
**Estado:** aprobado en brainstorming; pendiente de plan de implementación.

---

## 1. Contexto y problema

Tras la migración a ADK 2.3.0 y la unificación de `composite_pressure`, el proyecto quedó
con una **limitación de eficacia documentada pero no resuelta**: los detectores supervisado
(clasificador XGBoost) e Isolation Forest **no discriminan** el golden del apagón ibérico del
28-abr-2025; solo el **LSTM autoencoder** lo hace. El riesgo principal del sistema se calcula
hoy con el clasificador (`get_blackout_probability`), que en el golden produce `P≈0.001`.

El ledger atribuía la causa a "features no capturan ROCOF/inercia" + "D5 (3 eventos)". La
investigación de esta sesión confirma esas dos causas y **descubre una tercera**: hay
**skews train/serve** que degradan lo que el modelo ve en producción.

## 2. Análisis de causa raíz

1. **Skew de `composite_pressure` (parcialmente "arreglado", con pérdida de señal).** El fix
   previo (665d9b2) eliminó la normalización min-max rolling 7d y dejó umbrales absolutos para
   matar el train/serve skew. Verificado en git: el rolling antiguo era `df.rolling(7*24)` —
   **trailing/causal, sin leakage de futuro**. Por tanto el AUC 0.73 era **legítimo**, no
   ilusorio; se declaró "no replicable" solo porque en serve no se mantenía histórico. El fix
   sacrificó ~0.11 de AUC (0.73→0.62) para lograr consistencia.

2. **Skew de lag features del clasificador.** `validate_classifier._proba` rellena las lag
   features con **0** ("no disponibles en snapshots breves"). El clasificador entrena con lags
   que llevan buena parte de la señal temporal, pero en serve/validación van a cero → gap
   severo específico del clasificador.

3. **Física ausente.** Los 4 proxies (balance, reserva, intercambio, %renovables) no capturan
   **inercia** ni **ROCOF** (velocidad de cambio de frecuencia). El 28-abr fue un evento de
   **baja inercia** (penetración altísima de inversores solares/eólicos, oscilación de 0,6 Hz,
   cascada en minutos). Sin datos de frecuencia por API, ningún proxy de nivel ve la causa.

4. **D5 — 3 eventos documentados** (2 × S3, 1 × S4). Con un único apagón S4, un clasificador
   supervisado no puede aprender una frontera robusta.

**Consecuencia de diseño:** eliminar los skews (1, 2) recupera señal; añadir proxies físicos
computables (3) ataca la causa dominante dentro de lo posible; y una fusión ensemble con el
LSTM como detector principal neutraliza D5 (4) al no depender del modelo hambriento de datos.

## 3. Objetivos y no-objetivos

**Objetivos**
- Eliminar los tres skews train/serve mediante una **única capa de features** consumida por
  entrenamiento, servicio y validación (paridad por construcción).
- Añadir proxies físicos computables desde los datos que ya recogemos: **inercia**
  (non-synchronous penetration), **ROCOF** (deltas + volatilidad rolling), **estrés de
  interconexión** (ES↔FR, ES↔PT, ES↔MA).
- Restaurar el `composite_pressure` rolling (replicable en serve vía histórico en DB).
- Fusionar LSTM (load-bearing) + IF + clasificador en el **riesgo principal**.
- Un **gate de eficacia honesto**: el ensemble discrimina golden>normal y crece hacia el onset.
- **Grounding de dominio en el RAG:** incorporar contexto de las interconexiones de la red
  española (ES↔FR/PT/MA) para que la capa LLM razone cualitativamente sobre lo que las nuevas
  features numéricas de interconexión miden.

**No-objetivos**
- No se incorporan datos de clima al modelo core (la generación solar ya codifica la
  radiación para el nowcast; el backfill de Open-Meteo se descarta por coste/ganancia marginal).
- No se persiste ENTSO-E direccional (la señal de interconexión sale de los flujos netos ESIOS
  ya presentes). Los `flow_*` de ENTSO-E se **eliminan** del dataset (peso muerto).
- No se pretende demostrar generalización: con **n=1** apagón, cualquier calibración hacia el
  golden es indemostrable. La defensa es el LSTM no supervisado (evento excluido del entreno).

## 4. Principio rector

**Paridad train↔serve vía un feature store con estado.** Toda feature dependiente de historia
(rolling, lags, deltas, volatilidad, z-scores) se calcula con **el mismo código** sobre una
ventana temporal trailing:
- **Train:** la ventana es el dataset horario completo.
- **Serve:** la ventana son las últimas `ROLLING_WINDOW_H=168` horas de la tabla `snapshots`
  (SQLite ya existente; `store_snapshot` ya escribe por poll; `get_baseline` ya lee 168h).
- **Validación:** la ventana es el golden + su semana de contexto real (extraída del dataset
  ESIOS, misma fuente → paridad honesta).

## 5. Componentes

Cada componente se diseña para tener un propósito único, interfaz explícita y dependencias
acotadas.

### 5.1 `ml/features.py` — fuente única de features (NUEVO)

Funciones puras sobre un `DataFrame` horario UTC. Orquestadas en orden fijo por `build_features`.

```python
ROLLING_WINDOW_H = 168
COMPOSITE_WEIGHTS = {"balance":0.35,"reserve":0.25,"interchange":0.20,"renewables":0.20}
INTERCONNECT_CAPACITY_MW = {"es_fr": 2800.0, "es_pt": 3000.0, "es_ma": 900.0}  # NTC típicos; verificar vs REE/ENTSO-E

def add_reserve_pct(df) -> df           # reserva → % (fórmula única, elimina skew de reserva)
def add_inertia_proxy(df) -> df         # non_synchronous_penetration, synchronous_gen_mw
def add_interconnect_stress(df) -> df   # stress_es_fr, stress_es_pt, stress_es_ma, interconnect_stress_total
def add_deltas(df) -> df                # balance_delta_1h, nsp_delta_1h, renewables_delta_1h
def add_volatility(df, windows=(6,24)) -> df   # balance_std_Nh, interchange_std_Nh, nsp_std_Nh
def add_rolling_norm(df) -> df          # composite_pressure (min-max trailing 7d) + balance_z_7d, renewables_z_7d
def add_lags(df, lags=(1,3,6)) -> df    # lags de composite_pressure, balance_mw, nsp
def add_temporal(df) -> df              # hour/dow/month sin-cos, is_weekend, is_holiday_es
def build_features(df) -> df            # orquesta; define orden canónico
FEATURE_COLS: list[str]                 # unión de columnas que ven los modelos
```

**Cold-start / huecos (política única train↔serve):** cuando la ventana es insuficiente
(`min_periods`) o hay huecos, rolling/z/volatilidad/deltas quedan NaN → fallback explícito:
`composite_pressure` → fórmula **absoluta** `compute_composite_pressure_score` (se conserva como
fallback), z-scores→0, volatilidad→0, deltas→0. Las primeras 24h del dataset reciben el mismo
fallback: "arrancar en frío" significa lo mismo entrenando y en producción.

### 5.2 `ml/serve_features.py` — ensamblado en producción (NUEVO)

```python
def assemble_window(snapshots: list[dict], now: pd.Timestamp) -> pd.DataFrame:
    """Reconstruye DataFrame horario UTC (raw cols) desde snapshots crudos de 168h.
    Fuente-agnóstico: acepta filas de la DB o del golden."""

def serve_feature_row(snapshots: list[dict], now) -> dict:
    """assemble_window → build_features → última fila → dict con FEATURE_COLS."""
```

Pasos de `assemble_window`: leer 168h de `snapshots`; reconstruir columnas crudas
(`demand_real_mw`, generación por tecnología, flujos netos `es_fr_mw/es_pt_mw/es_ma_mw`,
reserva); resamplear a índice horario UTC; `ffill` acotado en huecos pequeños; huecos grandes →
cold-start.

**Fuente-agnóstico a propósito:** toma una lista de dicts, no la DB directamente. El runtime le
pasa lo leído de SQLite; la validación le pasa golden + contexto. Mismo código, misma paridad.

### 5.3 `ml/data/build_dataset.py` — modificado

- Sustituye el feature-engineering inline por `ml/features.py::build_features`.
- **Elimina la fusión ENTSO-E muerta** (`flow_*`) y reduce AEMET a lo que consume el forecaster.
- El dataset resultante lleva las features nuevas.

### 5.4 Entrenadores — modificados

- `train_classifier.py`, `train_anomaly.py`: actualizar `FEATURE_COLS` con NSP, deltas,
  volatilidad, rolling composite, z-scores, lags e interconnect stress.
- Re-entrenar IF + LSTM (GPU, `KNOWN_ANOMALY_PERIODS` excluidos) + clasificador (Optuna/GPU) →
  `ml/models/candidates/`.

### 5.5 `ml/grid_state_tracker.py` — umbrales Markov re-derivados

Re-derivar P90/P97 de `composite_pressure` (y los umbrales de estado) sobre la **distribución
rolling nueva**, no la absoluta. Persistir en `markov_state.npz`.

### 5.6 `tools/ensemble_scorer.py` — fusión (NUEVO, corazón de C)

```python
def fuse(lstm_recon_norm: float, if_score: float, clf_proba: float,
         weights: dict | None = None) -> dict:
    """Devuelve {'blackout_probability': float[0,1], 'components': {...}, 'primary': 'lstm'}."""
```

- **LSTM load-bearing:** peso dominante (default `w_lstm=0.5, w_if=0.2, w_clf=0.3`, configurable).
- **Override de anomalía:** si `lstm_recon_norm` supera claramente su umbral, se eleva el suelo de
  la probabilidad (una anomalía temporal fuerte no debe quedar diluida por votos débiles).
- Pesos **dominados por el LSTM por diseño**, NO ajustados al golden (mitigación de overfitting n=1).

### 5.7 Integración de scoring

- `mcp_servers/ml_models_server.py`: exponer LSTM e IF como señales **distintas** (fin del `elif`
  en `get_anomaly_score`) + endpoint que devuelva las tres señales o el score fusionado.
- `tools/risk_scorer.py`: consume el `P` fusionado como `ml_score` (conserva su blend final con
  `composite_pressure` + `renewables_risk`).
- `agents/orchestrator.py`: cablea ensemble → `risk_scorer`.

### 5.8 `ml/train/validate_models.py` — gate de eficacia honesto

- **Golden con contexto trailing:** alimentar `assemble_window` con la semana previa (21–28 abr,
  extraída del dataset ESIOS) para que rolling/lags **no** vayan a cero. Elimina el skew de
  validación (causa raíz #2).
- `validate_ensemble`: PASS si el score fusionado discrimina golden>normal **y** crece hacia el
  onset; LSTM primario. Se documenta que el CLF puede seguir débil por D5 y que el ensemble lo
  sostiene el LSTM.

### 5.9 RAG — contexto de interconexiones (NUEVO corpus)

Complemento cualitativo de las features numéricas de interconexión (§5.1): la capa LLM debe
poder razonar sobre *qué* mide cada enlace y *por qué* importa.

- **Nuevo documento de corpus** `data/kb_seed/procedures/interconexiones_red_espanola.txt` con:
  el carácter de **isla eléctrica** de la península ibérica; ES↔FR como **único enlace AC a la
  zona síncrona continental** (salvavidas de inercia; capacidad NTC típica); ES↔PT como coupling
  **intra-ibérico** (MIBEL, co-víctima el 28-abr); ES↔MA como enlace **HVDC** de menor capacidad
  con rol en la restauración; capacidades NTC de cada enlace; y su papel en el apagón del 28-abr
  y su recuperación.
- **Enrutado (paridad de acceso por agente):** el consumidor principal es `cross_border`, cuya
  whitelist incluye `get_mitigation_procedure` (colección `procedures`) pero **no**
  `search_regulations`. Por eso el documento va a la colección **`procedures`** (reachable por
  cross_border). Alternativa evaluada en el plan: añadir `search_regulations` a la whitelist de
  `cross_border` y ubicarlo en `regulations`. Se elige `procedures` por menor plumbing.
- **Re-seed:** `scripts/seed_knowledge_base.py` (chunking 600/100 existente; idempotente por
  `chunk_id`). No requiere cambios de código si se usa una carpeta ya mapeada en `COLLECTION_MAP`.
- **Enriquecer** `data/kb_seed/procedures/importacion_emergencia.txt` con especificidades de
  mitigación por enlace (import de emergencia desde FR, soporte DC de MA).

## 6. Catálogo de features (nuevas)

| Feature | Definición | Justificación |
|---|---|---|
| `non_synchronous_penetration` | (eólica+solar_pv+solar_térmica)/generación_total | Proxy de inercia — *el* feature del 28-abr (evento de baja inercia). |
| `synchronous_gen_mw` | nuclear+hydro+gas_cc+coal+cogeneración | Base síncrona (inercia disponible). |
| `balance_delta_1h`, `nsp_delta_1h`, `renewables_delta_1h` | `.diff(1)` | Velocidad de cambio — proxy de ROCOF. |
| `balance_std_6h/24h`, `interchange_std_6h/24h`, `nsp_std_6h/24h` | rolling std | Turbulencia/volatilidad — proxy de ROCOF. |
| `composite_pressure` | min-max trailing 7d por proxy + suma ponderada | Restaura AUC 0.73 (causal, replicable vía DB). |
| `balance_z_7d`, `renewables_z_7d` | (x − media_7d)/std_7d | Desviación del normal reciente. |
| `stress_es_fr`, `stress_es_pt`, `stress_es_ma` | \|flujo_neto_enlace\| / capacidad_enlace | Estrés por interconexión; FR = salvavidas síncrono, PT = coupling ibérico, MA = DC. |
| `interconnect_stress_total` | \|flujo neto total\| / capacidad total | Estrés agregado de frontera. |
| lags 1/3/6h de composite/balance/nsp | `.shift(k)` | Tendencia temporal; ahora computable en serve. |

## 7. Fusión ensemble — especificación

Entradas normalizadas a [0,1]: `lstm_recon_norm = min(mse/threshold, 1)`, `if_score` (ya [0,1]),
`clf_proba` (ya [0,1]). Salida: `blackout_probability` [0,1] + `components`.

Regla: `P = clip(w_lstm·lstm + w_if·if + w_clf·clf, 0, 1)`, con override
`P = max(P, lstm_override_floor)` si `lstm_recon_norm ≥ override_threshold`. Todos los parámetros
son constantes documentadas (no ajustadas al golden). El `risk_scorer` aplica su blend posterior
con física.

## 8. Estrategia de testing

- **Unit — paridad (crítico):** `build_features` sobre una ventana ⟹ la fila final es idéntica
  a `serve_feature_row` sobre los mismos snapshots. Este test es el guardián de la paridad.
- **Unit — `ensemble_scorer.fuse`:** LSTM dominante; override; monotonía por componente.
- **Unit — cold-start:** ventana < min_periods ⟹ fallback determinista (== primeras filas del train).
- **Unit — features físicas:** NSP, deltas, stress con casos conocidos.
- **Integración (padre, con dataset + GPU):** `validate_models` verde; `validate_ensemble` PASS.
- **RAG (integración, requiere embeddings):** smoke test de recuperación — una query de
  interconexión (p.ej. "capacidad de la interconexión España-Francia") devuelve el chunk nuevo
  con relevancia alta desde `get_mitigation_procedure`.
- La suite offline debe seguir verde (los tests que requieren dataset/artefactos se auto-omiten).

## 9. Rollout y operación

- Re-build dataset → re-train (candidates) → `validate_models` → promover a `ml/models/`.
- Se conservan los artefactos antiguos como fallback hasta validar los nuevos.
- Cold-start de producción: hasta acumular 168h en la DB, el sistema opera con fallback absoluto;
  la calidad rolling se activa progresivamente conforme se llena la ventana.

## 10. Riesgos y limitaciones honestas

- **n=1.** Un único apagón S4. Pesos de fusión, umbrales Markov y gate se calibran contra un solo
  evento; la generalización es indemostrable con estos datos. Mitigación: LSTM dominante por
  diseño + evento excluido del entreno del LSTM/IF.
- **Física fuera de la API.** Sin datos de frecuencia/ROCOF reales, los proxies (deltas,
  volatilidad, NSP) son aproximaciones. Puede que el golden no se "voltee" del todo; el gate es
  honesto al respecto.
- **Paridad de reserva.** Verificar que el `reserve_mw` del poller sale del mismo indicador ESIOS
  que `secondary_reserve_*` del train. Si divergen: persistir componentes crudas de reserva.
  El cálculo de reserva se unifica en `ml/features.py::add_reserve_pct`.
- **Capacidades de interconexión.** `INTERCONNECT_CAPACITY_MW` son constantes NTC típicas; verificar
  contra valores publicados de REE/ENTSO-E.
- **Coste GPU.** Re-entrenar LSTM+CLF es GPU y usa el dataset del proyecto padre; su verificación
  real ocurre allí. El código y los tests unitarios se verifican en frío.

## 11. Fuera de alcance

- Backfill de radiación Open-Meteo.
- Persistencia de ENTSO-E direccional.
- Datos de clima en el modelo core.
- Cualquier intento de aumentar artificialmente el número de eventos (synthetic events).
