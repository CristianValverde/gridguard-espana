# GridGuard España — Índice maestro: Enriquecimiento de incidentes, capa de Markov y mejoras de agentes

> **Para trabajadores agénticos:** este índice NO se ejecuta. Define la descomposición en 4 planes,
> aloja los artefactos compartidos (registro de incidentes, constraints globales, decisiones de diseño
> transversales) y el orden de ejecución. Cada plan referencia este documento en vez de repetir lo común.
> Ejecuta los planes en orden con `superpowers:subagent-driven-development` o `superpowers:executing-plans`.

**Origen:** `handoff_gridguard_ml_training.md` (sesión previa de entrenamiento ML).

**Objetivo global:** convertir un clasificador de apagón inservible (etiquetas en el instante equivocado,
AUC=0 en todos los folds) en un sistema de riesgo en capas: clasificador supervisado de eventos mayores +
capa de Markov de trayectorias de estado + una capa de agentes que razona sobre causalidad, normativa y
contexto europeo en vez de solo formatear salidas de ML.

---

## Por qué 4 planes y no uno

El handoff cubre **cuatro subsistemas con dependencias estrictas**. Intentarlos como un solo plan viola el
principio de "cada plan produce software funcional y testeable por sí mismo". La cadena de dependencias es:

```
Plan 1 (ML: incidentes + clasificador)
        │  produce: registro de incidentes S0–S4, etiquetas de estado, clasificador re-entrenado
        ▼
Plan 2 (Markov: grid_state_tracker, matriz de transición)   ◄── necesita las etiquetas de estado del Plan 1
        │  produce: compute_state(), get_trajectory(), matriz A[5×5]
        ▼
Plan 3 (Extensiones MCP: weather 24h, entsoe precios, memory estado, markov_server)
        │  parte independiente (weather/entsoe) + parte que envuelve el Plan 2 (markov_server, memory)
        ▼
Plan 4 (Agentes: RegulatoryComplianceSkill + extensiones + hooks)   ◄── necesita los tools MCP del Plan 3
```

**Regla de paralelización:** las tareas de **Plan 3 marcadas "independiente"** (extensiones de `weather_server`
y `entsoe_server`) pueden empezar en cualquier momento, en paralelo con Plan 1/2. Todo lo demás es secuencial.

---

## Constraints globales (aplican a TODOS los planes)

Copiados de `CLAUDE.md` y `ARCHITECTURE.md`. Cada tarea de cada plan los hereda implícitamente.

- **Seguridad NO NEGOCIABLE:** nunca leer `.env`, `*.key`, `secrets.*`, `credentials.*`. Todos los tokens
  (`ESIOS_TOKEN`, `AEMET_API_KEY`, `ENTSOE_TOKEN`, claves Gemini) provienen de variables de entorno.
- **Modelos Gemini:** nunca hardcodear nombres. Siempre `from config.model_loader import get_model` →
  `get_model("fast_agent")` / `get_model("smart_agent")`. (`config/model_loader.py:29`)
- **Shift-left:** Python puro > ML > LLM. No usar un LLM donde una función determinista basta. Markov,
  cálculo de estado y umbrales son **Python puro, sin LLM**.
- **Probe-first (CLAUDE.md global):** antes de escribir CÓDIGO de integración contra una API externa nueva,
  invocar el skill `api-probe`. Afecta a las tareas de Plan 3 que tocan AEMET (predicción horaria) y
  ENTSO-E (precios day-ahead A44). El probe es la PRIMERA tarea de esas integraciones.
- **Zona horaria:** el dataset (`data/processed/dataset_training.parquet`) y toda la persistencia están en
  **UTC**. Cualquier timestamp de incidente o evento se almacena y compara en UTC. Las horas de prensa
  españolas (CET=UTC+1 invierno, CEST=UTC+2 verano) se convierten ANTES de usarse.
- **Tooling:** Python 3.14.4, Windows 11, RTX 4080 (CUDA 12.8). PyTorch 2.11.0+cu128, XGBoost 3.3.0,
  scikit-learn 1.8.0, optuna 4.9.0. `torch.cuda.is_available() == True`.
- **TDD + commits frecuentes:** cada tarea termina en un entregable testeable y un commit.

---

## Artefacto compartido: registro de incidentes verificado (2020–2026)

**Investigado y verificado esta sesión** (fuentes al pie). Este es el insumo central del Plan 1 y la base de
estados observados del Plan 2. **Todas las horas en UTC.**

### Eventos discretos documentados (curados a mano → S3/S4)

| Evento | Hora local | **UTC** | Severidad | Evidencia |
|---|---|---|---|---|
| Separación área síncrona continental europea | 8-ene-2021 14:04:26 CET | **2021-01-08 13:04** | **S3** | Trip en Ernestinovo (Croacia); frecuencia área NW (incl. España) cayó a 49,74 Hz en ~15 s; resincronización 15:07 CET (14:07 UTC). ENTSO-E final report. |
| Desconexión península ↔ Europa (incendio + hidroavión, pérdida doble circuito 400 kV Hernani-Argia) | 24-jul-2021 16:35 CEST | **2021-07-24 14:35** | **S3** | Apagón local 16:45–17:38 CEST (14:45–15:38 UTC); 1,2 M abonados afectados; ENTSO-E ICS escala 2. |
| **Apagón total ibérico** — precursor (oscilación 0,6 Hz, 4 m 42 s) | 28-abr-2025 12:03 CEST | **2025-04-28 10:03** | **S4 (onset)** | Oscilación de 0,6 Hz observada 4 m 42 s antes del colapso. Origen probable planta FV en Badajoz. |
| **Apagón total ibérico** — colapso (−15 000 MW, 60 % generación, en 5 s) | 28-abr-2025 12:33 CEST | **2025-04-28 10:33** | **S4** | Cero energético en España peninsular, Portugal continental y Andorra; >50 M personas. Recuperación tarde del 28; restablecido madrugada del 29. |

> ⚠️ **Corrección crítica.** El handoff y `train_classifier.py:51` usan `"2025-04-28 08:00"`. El colapso real
> fue a las **10:33 UTC** (precursor 10:03 UTC). La etiqueta `08:00` desplazaba la ventana positiva ~2,5 h
> hacia la madrugada tranquila — el clasificador etiquetaba como "pre-apagón" un periodo de red en calma.
> Esto, sumado a la escasez de eventos, explica el AUC=0. **El Plan 1 corrige esto en su primera tarea.**

### Estados de estrés NO discretos (derivados algorítmicamente → S1/S2)

No son eventos puntuales sino **condiciones de red**; se derivan del propio dataset por umbrales físicos
(shift-left), no se curan a mano. Evidencia de su frecuencia (2024): 784 h con precio cero/negativo (~9 % del
año), 696 de ellas con producción solar significativa; vertido económico solar+eólica pasó de 1,6 % (2023) a
6,5 % (2024); penetración renovable instantánea >50 %. Verano 2025: curtailment récord hasta 11 %.

**Implicación de diseño:** S1/S2 son frecuentes (miles de horas); S3/S4 son rarísimos (3 eventos en 6 años).
Esto fija la separación de responsabilidades de la sección siguiente.

---

## Decisiones de diseño transversales (resuelven ambigüedades del handoff)

El handoff ofrecía opciones (A/B/C de etiquetado) sin cerrarlas. Se cierran aquí para mantener consistencia
entre planes:

### D1 — Taxonomía de estados S0–S4 (única para clasificador y Markov)

```
S0  Operación normal
S1  Estrés de red        (alta penetración renovable + desequilibrio moderado)   ── derivado por umbrales
S2  Incidente notable    (presión compuesta severa sostenida / reserva alta)     ── derivado por umbrales
S3  Incidente mayor      (pérdida de interconexión / gran excursión de frecuencia) ── evento documentado
S4  Apagón total         (interrupción total del suministro)                       ── evento documentado
```

### D2 — Separación de responsabilidades de etiquetado (evita circularidad)

El handoff arriesgaba fuga de información si las clases S1/S2 se derivan de las mismas features que entran al
clasificador. Se separa:

- **Clasificador XGBoost (Plan 1):** target = **eventos mayores documentados** (S3/S4) con ventanas graduadas.
  NO usa S1/S2 derivados como target. Su trabajo: "¿hay un evento mayor inminente?".
- **Capa de Markov (Plan 2):** estados observados = taxonomía completa S0–S4, donde S1/S2 se derivan por
  umbrales y S3/S4 de eventos. Su trabajo: "dado el estado actual y la trayectoria, ¿hacia dónde evoluciona?".

Cada capa usa el etiquetado apropiado a su pregunta. La derivación de S1/S2 vive en una función compartida
(`derive_stress_states`, Plan 1 Task 3) consumida por ambos planes.

### D3 — El clasificador es **binario** con vista de severidad, no multi-clase ingenuo

Con solo 2 eventos S3 + 1 S4 en 6 años, un `multi:softprob` de 5 clases es estadísticamente insostenible
(clases S1/S2 contaminarían y S3/S4 tienen ~puñado de ejemplos). Decisión:

- Target binario `evento_mayor ∈ {0,1}` con **ventanas graduadas** (S4: 6 h previas; S3: 3 h previas).
- Se conserva la severidad del evento como **peso de muestra** (`sample_weight`: S4 > S3) para que el modelo
  priorice el patrón del apagón total.
- Salida pública retrocompatible: `get_blackout_probability` sigue devolviendo `P(blackout)` ∈ [0,1]
  (`schemas/models.py:285` `BlackoutRisk`). La capa de Markov, no el clasificador, aporta las clases finas.

### D4 — Validación: **Leave-One-Event-Out**, no walk-forward de 4 folds

Walk-forward expansiva con 4 folds de 1 año garantiza folds sin positivos cuando hay 3 eventos → AUC=0
(causa raíz del handoff). Se sustituye por validación **dejando un evento fuera** cada vez, de modo que cada
pliegue de validación contiene exactamente un evento documentado y su ventana. Detalle en Plan 1 Task 5.

### D5 — Honestidad estadística (va en el README de modelos)

Con 3 eventos mayores, el clasificador supervisado es intrínsecamente débil y se documenta como tal. El valor
de detección recae en tres patas complementarias: (1) detección de anomalías no supervisada (IF/LSTM, ya
entrenados), (2) la capa de Markov sobre estados frecuentes S1/S2, (3) el clasificador como señal adicional.
Ningún plan debe afirmar métricas de apagón "buenas" derivadas de 3 eventos.

### D6 — Nombre del forecaster: NO hay bug (verificado en ejecución 2026-06-30)

El handoff afirmaba una discrepancia (`demand_forecaster.pkl` vs `load_forecaster.pkl`). **Falso.** La
verificación en ejecución muestra que `load_forecaster.pkl` es el nombre canónico, consistente en TODO el
código: `train_forecast.py:240` (produce), `retrain_pipeline.py:46` (promueve), `ml_models_server.py:99`
(carga para inferencia), y `ARCHITECTURE.md:544`. `demand_forecaster.pkl` no aparece en ningún punto del
código — solo en el handoff. **Plan 1 Task 8 queda CANCELADA**: renombrar rompería la consistencia existente
(YAGNI). El nombre canónico es `load_forecaster.pkl`.

---

## Resumen de los 4 planes

### [Plan 1 — Enriquecimiento de incidentes + re-entrenamiento del clasificador](2026-06-30-plan1-incident-enrichment-classifier.md)
**Prioridad máxima. Bloquea todo lo demás.** Crea `ml/incidents.py` (registro S0–S4 verificado), corrige el
timestamp del apagón, reescribe el etiquetado con ventanas graduadas + `sample_weight` por severidad, añade
`derive_stress_states`, sustituye walk-forward por Leave-One-Event-Out, verifica los modelos de fondo
(`lstm_ae.pth`, `load_forecaster.pkl` — ya presentes), re-entrena el clasificador y valida con el golden test.
(Task 8/D6 cancelada: el nombre `load_forecaster.pkl` ya es consistente — ver D6.)
**Entregable:** clasificador re-entrenado con etiquetas correctas + `validate_models` en verde + registro de
incidentes reutilizable.

### [Plan 2 — Capa de cadena de Markov (Python puro)](2026-06-30-plan2-markov-layer.md)
Crea `tools/markov_state_tracker.py` (`compute_state`, `get_trajectory`, `get_transition_matrix`) y
`ml/grid_state_tracker.py` (estimación empírica de la matriz A[5×5] con suavizado de Laplace + priors
bayesianos ENTSO-E para transiciones raras S2→S3→S4). Sin LLM.
**Depende de:** Plan 1 (`derive_stress_states`, registro de incidentes).
**Entregable:** matriz de transición persistida + API de trayectoria testeada.

### [Plan 3 — Extensiones de servidores MCP](2026-06-30-plan3-mcp-extensions.md)
Cuatro extensiones: `weather_server.get_weather_forecast_24h` (AEMET horaria, **probe-first**),
`entsoe_server.get_eu_market_prices` (ENTSO-E A44 day-ahead, **probe-first**),
`memory_server.store_grid_state` + `get_state_transitions` (SQLite), y el **nuevo** `markov_server.py` que
envuelve el Plan 2.
**Depende de:** Plan 2 (para `markov_server` y el estado en memoria). Weather/entsoe son independientes.
**Entregable:** 4 tools MCP nuevos con tests de contrato.

### [Plan 4 — Mejoras de la capa de agentes + hooks](2026-06-30-plan4-agent-enhancements.md)
Crea `agents/regulatory_compliance.py` (nuevo Skill) y extiende instrucciones de `anomaly_detector`,
`renewables_forecast`, `report_agent` (tipo `post_mortem`), `cross_border` (parámetro `mode`) y
`orchestrator` (estado de Markov + nuevos umbrales). Añade los hooks `on_state_transition` y
`on_forecast_risk_horizon` y actualiza `_ALLOWED_TOOLS` en `guardrails/hooks.py` para los tools nuevos.
**Depende de:** Planes 2 y 3.
**Entregable:** capa de agentes que razona sobre causalidad, normativa y trayectoria; tests de integración en verde.

---

## Orden de ejecución recomendado

1. **Plan 1** completo (desbloquea el resto). En paralelo, opcionalmente, las dos tareas "independiente" de Plan 3.
2. **Plan 2** completo.
3. **Plan 3** restante (`markov_server`, memory state).
4. **Plan 4** completo.
5. Tras cada plan: `python -m ml.train.validate_models` y `pytest eval/ -m integration -v` según aplique.
6. Demo final del orquestador con el golden dataset (apagón 28-abr-2025).

---

## Fuentes de la investigación de incidentes

- [Apagón en la península ibérica de 2025 — Wikipedia](https://es.wikipedia.org/wiki/Apag%C3%B3n_en_la_pen%C3%ADnsula_ib%C3%A9rica_de_2025)
- [Red Eléctrica — Informe del incidente del 28 de abril de 2025 (PDF)](https://www.ree.es/sites/default/files/2025-06/NP_RedElectrica_InformeOS_180625.pdf)
- [REE — Electricity supply restored after 24 July 2021 French grid incident](https://www.ree.es/en/press-office/news/press-release/2021/07/electricity-supply-restored-in-those-areas-of-spain-affected-by-an-incidente-in-the-french-power-grid-that-temporarily-disconnected-the-mainland-electricity-system-from-the%20rest-of-Europe)
- [ENTSO-E — Final report on the separation of the Continental Europe power system on 8 January 2021](https://www.entsoe.eu/news/2021/07/15/final-report-on-the-separation-of-the-continental-europe-power-system-on-8-january-2021/)
- [ENTSO-E — Factual report on the 24 July 2021 Continental Europe synchronous area separation](https://www.entsoe.eu/news/2021/11/12/factual-report-on-the-separation-of-the-continental-europe-synchronous-area-on-24-july-2021/)
- [Gridradar — System split, grid frequency below 49.75 Hz, 08.01.2021](https://gridradar.net/en/blog/post/underfrequency_january_2021)
- [El Español/Invertia — Renovables a precio cero/negativo en 2024](https://www.elespanol.com/invertia/empresas/energia/20241002/renovables-dejan-rentables-tiempo-produccion-va-ano/890161491_0.html)
- [Strategic Energy — Curtailment récord verano 2025](https://strategicenergy.eu/curtailment-record-en-verano-2025-hasta-el-11-de-la-energia-renovable-no-se-integro-a-la-red/)
