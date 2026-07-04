# GridGuard España — Memoria del Proyecto

**Capstone · 5-Day AI Agents Intensive (Google × Kaggle) · Junio–Julio 2026**

> Sistema multi-agente para la monitorización y predicción de riesgo de apagón en la red
> eléctrica peninsular española, construido con Google ADK 2.3.0 + Gemini, ML determinista
> (Isolation Forest, LSTM Autoencoder, Gradient Boosting calibrado), una cadena de Markov de
> estados de red, y RAG sobre normativa REE/ENTSO-E.

---

## 1. Resumen ejecutivo

GridGuard es un sistema que combina tres tipos de inteligencia — Python puro, Machine Learning
y LLM — para vigilar la red eléctrica española y escalar avisos operativos antes de que
ocurra un incidente. El diseño sigue el principio **shift-left**: usar la herramienta más
simple que resuelva el problema, y reservar el LLM únicamente para razonamiento que de verdad
lo requiere (síntesis multi-señal, redacción de informes, decisiones de escalación
contextual).

El sistema se validó contra un **golden dataset real**: el apagón peninsular del 28 de abril
de 2025, reconstruido minuto a minuto a partir del post-mortem publicado por REE. La validación
del pipeline ML (ensemble LSTM + Isolation Forest + clasificador) dio resultados honestos:
2 de 3 señales predicen correctamente el evento, la tercera falla por escasez de datos
(1 solo evento S4 histórico) — limitación documentada, no oculta.

La parte más instructiva del proyecto no fue construir el sistema, sino **medirlo con
rigor contra un LLM real** en vez de conformarse con tests offline: la validación live reveló
tres capas de problemas reales — infraestructura (tormenta de subprocesos, imports
bloqueantes), robustez (excepciones sin capturar de terceros) y comportamiento del agente
(no-determinismo del LLM al seguir una política de escalación) — cada una diagnosticada con
evidencia y corregida o documentada honestamente cuando no era corregible en el tiempo
disponible.

---

## 2. Arquitectura del sistema

Ver [`ARCHITECTURE.md`](ARCHITECTURE.md) para la especificación técnica completa (capas,
schemas, MCP servers, hooks). Resumen de las 6 capas:

```
Capa 0  Triggers (cron 10min/1h/24h, cola de eventos, eval runner)
Capa 1  Function Tools — Python puro (ree_poller, data_normalizer,
        stability_proxy_calculator, risk_scorer, alert_formatter,
        escalation_planner, markov_state_tracker)
Capa 2  MCP Servers — 5 servers: grid-data, weather, memory, ml-models, knowledge-base
Capa 3  Skills (AgentTool) — AnomalyDetector, RenewablesForecast, RegulatoryCompliance
Capa 4  Sub-agentes — ReportAgent, CrossBorderAgent
Capa 5  OrchestratorAgent — Root Agent, coordina todo lo anterior
```

Cada capa sube en "peso" de inteligencia solo cuando la anterior no basta:

| Si la tarea es... | Se usa... |
|---|---|
| Llamada a API + parseo + validación | Function Tool (Python puro) |
| Detección de patrones estadísticos en series temporales | Modelo ML (determinista en inferencia) |
| Recuperación de contexto técnico/normativo | RAG (ChromaDB) |
| Análisis multi-señal con razonamiento | Skill (AgentTool + LLM) |
| Tarea multi-paso con estado interno | Sub-agente (LLM) |
| Coordinación global + decisiones de escalación | Orchestrator (Root Agent) |

### 2.1 Transporte MCP compartido (evolución arquitectónica, 2026-07-03)

Originalmente cada agente creaba sus propias instancias `McpToolset` — 17 instancias para 5
servers, lo que disparaba hasta ~17 subprocesos Python simultáneos por ciclo de monitorización
(cada uno importando `torch`/`chromadb` a la vez). Se rediseñó a **5 toolsets singleton
compartidos** (`agents/mcp_toolsets.py`), reduciendo el máximo a 5 subprocesos por corrida.
Spec: `docs/superpowers/specs/2026-07-03-mcp-shared-transport-design.md`.

### 2.2 Consolidación de servers (Plan 7, 2026-07-02)

Los antiguos servers `ree-grid` y `entso-e` se fusionaron en un único `grid-data` (ESIOS +
ENTSO-E), y la capa Markov se plegó dentro de `ml-models`. Resultado: 7 → 5 MCP servers.

---

## 3. Aplicación del principio shift-left (ejemplos concretos)

El shift-left no es solo un eslogan del diseño — se aplicó de forma repetida y verificable
a lo largo del proyecto, incluso para *corregir* comportamiento de agentes:

- **`tools/markov_state_tracker.py`**: la trayectoria de estado de la red (S0–S4, próximas
  6h) se calcula con una matriz de transición Markov entrenada offline (NumPy puro) — cero
  LLM en la inferencia de "hacia dónde va la red".
- **`tools/ensemble_scorer.py`**: la fusión de 3 señales ML (LSTM Autoencoder + Isolation
  Forest + clasificador calibrado) en `P(blackout)` es aritmética ponderada determinista, no
  una síntesis del LLM.
- **`tools/escalation_planner.py`** (2026-07-04): la política de escalación (qué skills
  invocar según banda de riesgo y estado Markov) vivía originalmente en **prosa dentro del
  prompt del orquestador**. Se extrajo a una función Python pura y determinista con 17 tests
  de tabla de verdad — el LLM ya no *decide* la política, solo *ejecuta* lo que el código le
  manda. Este es el ejemplo más claro de shift-left aplicado retroactivamente: cuando se
  detectó que el LLM no seguía la política de forma fiable, la respuesta no fue "mejorar el
  prompt" (ya se intentó dos veces sin éxito) sino **quitarle esa decisión al LLM por
  completo**.
- **`agents/escalation_enforcer.py`** (2026-07-04): red de seguridad determinista que, tras
  la corrida del LLM, recalcula la política con `escalation_planner` y **fuerza por código**
  (sin pasar por el LLM del orquestador) cualquier skill/sub-agente obligatorio que falte.
- **`agents/orchestrator.py::_consume_events_safely`** (2026-07-04): incluso el manejo de
  errores se resolvió con código determinista (un `try/except` que conserva eventos
  parciales) en vez de intentar que el LLM "se comporte mejor".

La memoria de lecciones aprendidas (`store_lesson`/`get_lessons`, ver §7) también sigue este
principio: es una tabla SQLite consultada/escrita por tools MCP, no una memoria vectorial con
coste de embeddings — decisión razonada explícitamente por volumen de datos y determinismo
(ver spec `2026-07-04-lessons-memory-design.md`).

---

## 4. Datos, golden dataset y limitaciones de las fuentes

### 4.1 Fuentes de datos históricos

| Fuente | Desde | Granularidad | Datos clave |
|---|---|---|---|
| ESIOS/REData (token gratuito) | 2014-01-01 | 10 min / horaria | Demanda real, generación por tecnología, intercambios, precios, reserva de regulación |
| REData (sin token) | 1990 | Mensual/anual | Series estadísticas largas |
| AEMET OpenData (token gratuito) | ~2000 | Horaria/diaria | Viento, radiación solar, temperatura por estación |
| ENTSO-E Transparency (token gratuito) | 2011/2015 | Horaria | Flujos transfronterizos, generación UE, capacidad de interconexión |

Volumen resultante: ~10 años × 365 días × 6 lecturas/hora ≈ **525.000 puntos temporales** a
granularidad de 10 minutos.

### 4.2 Limitación crítica: la frecuencia de red NO está disponible vía API pública

**La frecuencia de red (Hz)** — la variable físicamente más directa para detectar
inestabilidad — **no se publica** en la API de ESIOS: es telemetría en tiempo real que REE no
expone por motivos de seguridad operacional. El sistema usa en su lugar **4 proxies físicos**,
estándar en la literatura académica para predicción de inestabilidad en redes con alta
penetración renovable:

| Proxy | Indicador ESIOS | Por qué funciona |
|---|---|---|
| Balance oferta/demanda | Generación real − Demanda real | Diferencia directa → presión sobre frecuencia |
| Reserva de regulación activada | Indicadores de reserva secundaria | Se activa cuando la frecuencia se desvía |
| Urgencia de intercambio | Variación brusca de flujos FR/PT/MA | Importación de emergencia → estrés de frecuencia |
| % renovables no gestionables | (Solar + Eólica) / Demanda total | Baja inercia síncrona → mayor vulnerabilidad |

Esta es una limitación estructural, no un descuido: **ningún actor externo a REE tiene acceso
a la frecuencia en tiempo real**. El proyecto lo documenta explícitamente en vez de aparentar
una precisión que la fuente de datos no permite.

### 4.3 Limitación: datos en tiempo real vs. históricos

Los endpoints de ESIOS/ENTSO-E tienen **latencia de publicación** (no son telemetría
instantánea) y los tokens gratuitos tienen límites de tasa (ver `grid_data_server.py`: TTL de
caché de 9 min para ESIOS, 1h para ENTSO-E, cota dura de 45s por build tras medir un caso real
de ~125s bloqueando el event loop del server — ver §6.3). El sistema **no puede** operar con
la latencia sub-segundo de un SCADA real; está diseñado para monitorización de minutos, no
para protección de red en tiempo real.

### 4.4 Golden dataset — apagón del 28 de abril de 2025

`eval/golden_dataset/blackout_apr_2025.json`: 6 snapshots reconstruidos minuto a minuto desde
6h antes del evento hasta T-3min, con metadata del post-mortem oficial de REE (causa raíz:
pérdida súbita de generación convencional + penetración renovable muy alta sin inercia
síncrona suficiente, 15.000 MW afectados, 6.5h de duración). Contrapartida:
`eval/golden_dataset/normal_baseline.json`, con valores ESIOS reales del 14 de mayo de 2025
(sustituyó una versión sintética inicial que producía saltos de datos irreales — hallazgo de
sesiones previas, ver `efficacy-ensemble-work.md`).

**Limitación honesta del clasificador (CLF):** con un único evento S4 histórico en los datos
de entrenamiento, el clasificador de blackout da AUC LOEO de 0.6875 (no alcanza el 0.73
esperado) y una señal plana (0.0009) en el golden test — documentado como **D5** (limitación
de datos, no de diseño) en vez de forzar un resultado artificialmente mejor. El ensemble
compensa haciendo el **LSTM load-bearing por diseño** (peso 0.5, no ajustado sobre el golden
para evitar sobreajustar a n=1): PASS monótono hacia el evento (error de reconstrucción
0.043 → 1.53 en T-3min).

---

## 5. Evals — outputs y paths

### 5.1 Estructura y comandos

```
eval/
├── golden_dataset/
│   ├── blackout_apr_2025.json      # escenario CRITICAL (post-mortem REE real)
│   └── normal_baseline.json        # escenario VERDE (ESIOS real, 14-may-2025)
├── conftest.py                     # fixtures: blackout_scenario, normal_scenario, snapshots
├── common/                         # utilidades compartidas de eval
└── agents/
    ├── _harness.py                 # run_agent_live(agent, message) — reusable para cualquier agente
    ├── orchestrator/{test_trigger,test_trajectory,test_output}.py
    ├── anomaly_detector/test_trigger.py
    ├── renewables_forecast/test_trigger.py
    ├── regulatory_compliance/test_trigger.py
    ├── cross_border/test_trigger.py
    └── report_agent/test_trigger.py
```

Cada agente se evalúa en **3 ejes**: `trigger` (¿dispara las tools correctas y solo esas?),
`output` (¿el resultado es correcto?), `trayectoria` (¿el orden fue el correcto, shift-left
Python→ML→LLM?). Dos niveles: **offline** (determinista, `python -m pytest eval -q`) y
**live/integration** (LLM real, `-m integration`, consume cuota Gemini).

Comandos exactos y advertencias operativas completas en [`eval/README.md`](eval/README.md).

### 5.2 Estado medido de la suite (2026-07-04, última corrida)

- **Suite offline: 296 passed, 16 deselected** (`python -m pytest -q`).
- **Suite live (12 tests `-m integration`)**: mejor resultado alcanzado 7/12 en 9m41s
  (post-fixes de infraestructura), **0 timeouts, 0 crashes**. Los 5 tests restantes fallan
  por comportamiento no determinista del LLM (ver §6).
- 123 commits en `main`, 8 specs de diseño (`docs/superpowers/specs/`), 13 planes de
  implementación (`docs/superpowers/plans/`) — todo el proceso de diseño→plan→implementación
  queda documentado y trazable, no solo el código final.

### 5.3 Otros outputs generados por el sistema

- `reports/YYYY-MM-DD_HHMM_emergency_report.{html,md}`: informes de emergencia redactados por
  `ReportAgent` (modelo `smart_agent`, contexto 2M tokens) al escalar a CRITICAL.
- `data/gridguard.db` (SQLite): historial de risk scores, snapshots, incidentes, estados
  Markov, y lecciones aprendidas (tabla `lessons`, ver §7).
- `knowledge/chroma_db/`: base vectorial RAG (normativa REE/ENTSO-E, post-mortems).

---

## 6. Lo más instructivo del proyecto: 3 capas de problemas reales encontradas al validar contra LLM real

Un hallazgo central de este proyecto es que **la suite offline en verde no garantiza que el
sistema funcione con un LLM real**. La validación live sucesiva (varias sesiones) destapó tres
categorías de problemas completamente distintas, cada una diagnosticada con evidencia antes de
proponer un fix — nunca se parcheó a ciegas.

### 6.1 Infraestructura: tormenta de subprocesos + imports bloqueantes (2026-07-03/04)

**Síntoma:** timeouts MCP de 60s en cascada durante corridas live.
**Diagnóstico con `py-spy` (perfilador de stacks en vivo):** el kb server importaba
`chromadb` (que arrastra `numpy`) de forma perezosa dentro del primer tool call, con el event
loop de `anyio` ya corriendo — la carga de la extensión C de numpy se colgaba minutos.
**Fix:** import eager al arranque del módulo (mismo patrón que `torch` en `ml_models_server`).
Repro determinista sin LLM: >90s colgado → 0.8s. Además: 17 instancias `McpToolset` → 5
singleton compartidos (§2.1); IO externo sin acotar (embeddings Gemini, ENTSO-E, forecast
secuencial) → todo con timeout explícito; caché de `grid-data` con cota dura de 45s por build
tras medir un caso real de ~125s.

### 6.2 Robustez: excepción de terceros sin capturar (2026-07-04)

**Síntoma:** una skill (`AgentTool` con `output_schema` estricto) respondía texto libre en vez
de JSON; ADK 2.3.0 no captura la `pydantic.ValidationError` resultante y aborta **todo**
`runner.run()` sin producir ningún resultado para ese ciclo.
**Fix:** `_consume_events_safely()` — consume el generador de eventos del Runner en un bucle
con `try/except`, conservando lo ya capturado en vez de perder el ciclo completo. Verificado
sin regresión en una corrida live posterior (0 ocurrencias del error original).

### 6.3 Comportamiento del agente: no-determinismo del LLM al seguir instrucciones vinculantes

**Síntoma:** aun con `escalation_planner` (política 100% determinista) invocándose
correctamente en el orden esperado, el LLM del orquestador **no ejecutaba consistentemente**
lo que el plan mandaba — en escenarios críticos se detenía tras una sola skill; en escenarios
normales disparaba skills de más. Dos intentos de reforzar el prompt (prosa → "VINCULANTE"
explícito) no cambiaron el comportamiento.
**Respuesta:** se construyó `escalation_enforcer.py`, una red de seguridad que repara por
código lo que el LLM omite — probado con 5 tests unitarios mockeados (offline, deterministas).
**Límite honesto encontrado:** la validación end-to-end de esta red de seguridad reveló un
problema *más profundo* aún: el tool de ML real (`get_blackout_probability_ensemble`) lee la
**base de datos en vivo** (ventana de 168h), no el snapshot del escenario de evaluación — el
pipeline ML determinista está desacoplado del golden dataset que se usa para poner a prueba al
LLM. Verificado empíricamente en este mismo proyecto (`get_blackout_probability_ensemble()`
devolvió P=0.12 al ejecutarlo directamente, muy lejos del escenario crítico simulado).
Corregirlo de raíz requeriría un fixture que cargue temporalmente los snapshots del golden
dataset en la base de datos antes de cada corrida live — cambio de infraestructura de evals
compartida por todos los agentes, identificado y documentado como trabajo futuro, no
implementado por restricción de tiempo.

**Por qué esto importa más que "arreglarlo todo":** demuestra el patrón de trabajo del
proyecto — investigar con evidencia (logs, `py-spy`, repros deterministas sin LLM,
verificación empírica de hipótesis antes de comprometerlas a un diseño), corregir lo que es
corregible en el tiempo disponible, y **documentar honestamente** lo que no lo es, en vez de
ocultarlo o fingir que un test en verde significa que el problema desapareció.

---

## 7. Memoria de lecciones aprendidas (persistencia entre ciclos)

Cada ciclo de monitorización del orquestador es una sesión **efímera** (un `Runner` +
`InMemorySessionService` nuevos por invocación; sin scheduler propio en el repo — la cadencia
la fija quien lo invoque). Entre ciclos sobrevive estado operativo en `gridguard.db`
(`risk_scores`, `grid_states`, `incidents`), pero originalmente **no había ningún mecanismo de
aprendizaje** entre sesiones.

Se añadió (2026-07-04) una tabla `lessons` + tools MCP `store_lesson`/`get_lessons`:
- El orquestador consulta lecciones relevantes al inicio de cada ciclo (filtradas por estado
  Markov/banda de escalación).
- Registra una lección nueva al cierre **solo** si el ciclo fue relevante (banda
  NARANJA/ROJO, fallo de tool crítica, o predicción contradicha) — evita ruido en ciclos
  VERDE rutinarios.
- **Decisión de diseño (SQLite, no ChromaDB):** a este volumen (decenas de lecciones, no
  miles), la búsqueda semántica es innecesaria — el agente filtra estructuralmente por estado/
  banda. SQLite es local, gratis, determinista y testeable offline; ChromaDB exigiría un
  embedding Gemini por escritura/lectura (cuota compartida, y justo se acababa de arreglar un
  cuelgue en esa ruta exacta — ver §6.1).
- **Verificado funcionando end-to-end contra LLM real**: en la corrida live del 2026-07-04,
  `get_lessons` y `store_lesson` aparecen en la trayectoria capturada sin errores de
  serialización — el mecanismo de aprendizaje funciona, independientemente del resultado (en
  rojo) de la escalación misma.

---

## 8. Cambios de arquitectura — línea de tiempo resumida

| Fecha | Cambio | Motivo |
|---|---|---|
| 2026-06-29 | IDs ESIOS corregidos (11/13 estaban mal) | Verificación contra producción real |
| 2026-07-01/02 | Paridad train/serve, ensemble LSTM load-bearing, golden validado | Eficacia del sistema ML |
| 2026-07-02 | Plan 7: 7→5 MCP servers (grid-data unificado, Markov plegado en ml-models) | Consolidación, evals por agente, prompts versionables |
| 2026-07-02 | Pipeline de ingesta RAG con gate humano | Escalar la base de conocimiento sin curación 100% manual |
| 2026-07-03 | Transporte MCP compartido: 17→5 subprocesos | Tormenta de subprocesos bloqueando evals live |
| 2026-07-04 | Import eager de chromadb, IO externo acotado, cliente MCP 120s | Causa raíz real de los timeouts (no la tormenta, un import bloqueante) |
| 2026-07-04 | `escalation_planner`: política de escalación de prosa → función Python | LLM no seguía fielmente la política en prosa |
| 2026-07-04 | Memoria de lecciones (`lessons` + `store_lesson`/`get_lessons`) | El orquestador no aprendía nada entre ciclos |
| 2026-07-04 | `escalation_enforcer`: red de seguridad post-LLM | El LLM seguía sin obedecer el plan incluso "vinculante" |
| 2026-07-04 | `_consume_events_safely`: resiliencia ante ValidationError de ADK | Un solo turno mal formateado abortaba el ciclo completo |

---

## 9. Trabajo pendiente (documentado, no implementado)

- **Fixture de evals que cargue el golden dataset en la DB real** antes de la corrida live,
  para que `get_blackout_probability_ensemble` refleje el escenario bajo prueba (ver §6.3).
- **Extender `escalation_enforcer` a `run()`** (ciclo real de producción) — hoy solo cubre
  `run_scenario()` (el path de evals); `run()` no expone el snapshot de forma tan directa.
- **4 temas pendientes** en `data/kb_seed/topics.yaml` (pipeline de ingesta RAG, repetible).
- **Corrección de la sobre-escalación en VERDE** — el LLM a veces dispara skills de más; se
  aceptó como residual de sobre-cautela (no es un riesgo de seguridad, a diferencia de
  no escalar cuando toca).
- **Scheduler real** para `run()` — hoy el ciclo de monitorización se invoca externamente, sin
  cron/loop propio en el repo.

---

## 10. Conclusión

GridGuard demuestra los conceptos centrales del curso — sistema multi-agente, MCP servers
personalizados, Skills/Sub-agentes diferenciados, guardrails con lógica real, memoria/estado,
framework de evals con golden dataset real, tracing, ML integrado y RAG — pero su aporte más
valioso como ejercicio de ingeniería fue **tratar la validación live como una fuente de verdad
más exigente que los tests offline**, encontrar causas raíz reales con evidencia (perfiles de
stack, repros deterministas, verificación empírica de hipótesis) en cada una de las tres capas
del sistema (infraestructura, robustez, comportamiento del agente), y documentar con la misma
honestidad los problemas que se corrigieron y los que quedaron identificados pero pendientes
por restricciones de tiempo.
