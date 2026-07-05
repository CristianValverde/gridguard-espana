<!--
Borrador para pegar en el Kaggle Writeup (Track: Agents for Good).
Cuenta de palabras del CUERPO (sin título/subtítulo): ver nota al final del fichero.
Adjuntar en la Media Gallery: imagen de portada + vídeo de YouTube (≤5 min).
Adjuntar como Project Link: https://github.com/CristianValverde/gridguard-espana
(verificado 2026-07-04: repo TODAVÍA PRIVADO — hacerlo público antes de enviar el Writeup)
-->

# TITLE

GridGuard España: A Multi-Agent System Built to Catch the Next Blackout Before It Hits

# SUBTITLE

Six layers of Python, deterministic ML, RAG, and Gemini-powered agents watching the Spanish
grid in real time — validated against a real historical blackout, and honest about exactly
where an LLM still can't be trusted.

# BODY

## The problem

On 28 April 2025, a real blackout took down the Spanish peninsular grid for 6.5 hours,
affecting up to 15,000 MW. REE's (Red Eléctrica de España, the Spanish grid operator)
official post-mortem shows the warning signs were visible
hours in advance: extreme renewable penetration, falling synchronous inertia, a progressively
worsening supply/demand balance. But there's a catch — **grid frequency, the single most
direct instability signal, is not published by any public API.** REE withholds it for
operational security reasons. Anyone building a monitoring system for the Spanish grid has to
work with proxies, not ground truth, and with real data that is messy, inconsistently
documented, and sometimes just wrong.

GridGuard asks: can a multi-agent system, using only publicly available data, catch that kind
of situation early enough to matter — and can we be honest, in public, about exactly where an
agent built on an LLM (Large Language Model) can and can't be trusted to do that reliably, and
where our own data pipeline has real gaps?

## The solution: three levels of intelligence, each reserved for what it's actually good at

GridGuard follows one rule throughout: never use an LLM where Python or an ML (Machine
Learning) model is enough.

- **Pure Python** computes physical proxies for grid stress (supply/demand balance, activated
  regulation reserve, interconnection stress, % non-manageable renewables) and formats alerts.
- **Deterministic ML** — an ensemble of an Isolation Forest, an LSTM (Long Short-Term Memory)
  Autoencoder, and a calibrated Gradient Boosting classifier — fuses into a single P(blackout)
  estimate. A Markov chain over five grid states (S0 Normal → S4 Blackout), trained offline on
  historical transitions, projects where the grid is heading over the next few hours. Zero LLM
  calls in either path.
- **RAG (Retrieval-Augmented Generation)** over REE/ENTSO-E (Europe's grid-operator
  transparency platform) operating procedures and past-incident post-mortems (Iberian
  separation 2006, this same 2025 blackout) gives the system real regulatory grounding instead
  of generic LLM knowledge — built and maintained through a repeatable, human-gated pipeline,
  not a one-off seed script (see below).
- **The LLM** only enters where genuine reasoning is needed: synthesizing multiple signals into
  a severity judgment, drafting an emergency report, evaluating cross-border import options.

### Architecture — six layers

```
Layer 0  Triggers (cron / event queue / eval runner)
Layer 1  Function Tools — pure Python
Layer 2  5 MCP Servers — grid-data · weather · memory · ml-models · knowledge-base
Layer 3  Skills (AgentTool) — AnomalyDetector · RenewablesForecast · RegulatoryCompliance
Layer 4  Sub-agents — ReportAgent · CrossBorderAgent
Layer 5  OrchestratorAgent (Root Agent) — coordinates everything above
```

Two structural choices here are themselves shift-left decisions, not stylistic ones. First,
Layer 2 is built as **MCP (Model Context Protocol) servers rather than plain function
imports**: several agents call the
same tool (both the orchestrator and the anomaly-detection skill need the same blackout
probability), so defining it once behind a shared protocol boundary — with credentials
(`ESIOS_TOKEN`, `AEMET_API_KEY`) living only inside that server's process, never in agent code
or prompt context — beats every agent importing and re-implementing its own copy. Second, three
agents are **Skills** (`AgentTool`, stateless, single structured result) while two are **full
sub-agents** with their own multi-turn loop and memory — reserved specifically for tasks whose
step count can't be predicted in advance (`ReportAgent` synthesizing several sources into a
report, `CrossBorderAgent` evaluating a variable number of import options). Giving every agent
a full autonomous loop "just in case" would be handing the LLM more control than most of these
tasks actually need — the same shift-left instinct that later produced `escalation_planner`.

Built on **Google ADK (Agent Development Kit) 2.3.0 + Gemini**, with **5 custom MCP servers**
(consolidated from an
original 7 as the design matured), **3 Skills** (`AgentTool`-wrapped, single structured output
each), **2 Sub-agents** with their own tool loops, and a **guardrails layer**
(`guardrails/hooks.py`) enforcing a per-agent tool whitelist, rate-limiting, `.env`/secret-path
blocking, prompt-injection sanitisation, and output-schema validation before anything reaches
the user.

## Turning four public APIs into one dataset you can actually train on

Before any agent could reason about the grid, the project had to build a real data pipeline:
ESIOS/REData (REE's own grid-data API; 10-minute granularity back to 2014), AEMET OpenData
(Spain's national weather agency; hourly weather back to ~2000), and ENTSO-E Transparency
(hourly EU interconnection data back to 2011) for
training — roughly 525,000 ten-minute readings once assembled. A fourth API, Open-Meteo
(token-free), feeds live wind and solar radiation forecasts to `RenewablesForecastSkill` in
production, separate from AEMET's role as the historical training source. This surfaced its own class of bugs, unglamorous
but load-bearing: **11 of the 13 ESIOS indicator IDs we'd initially used were simply wrong**
(mis-derived from the API's naming conventions), caught only by cross-checking against real
production usage; the demand indicator's raw units turned out to be double the real MW value,
silently skewing every downstream feature until corrected; and annual API queries were timing
out until we chunked them into 28-day windows. None of this is exotic engineering, but a
model is only as honest as the numbers it's trained on, and every one of these would have
quietly degraded the whole system if left unfixed. To guarantee the model never sees a
different feature computation at training time than at inference time, every history-dependent
feature is computed by one single function used by training, serving, *and* validation —
enforced by a test asserting the two paths agree to 9 decimal places.

## Validating the ML ensemble against a real blackout, not a synthetic one

The golden test case is the same 28 April 2025 blackout, reconstructed minute-by-minute from
REE's real, published post-mortem. Against it: the Isolation Forest passed, the LSTM
Autoencoder passed (its reconstruction error climbed from 0.043 to 1.53 as the event
approached), and the fused ensemble passed with a monotonically rising P(blackout) — three
independent models converging on the same real event, computed with zero LLM calls. The
calibrated Gradient Boosting classifier, however, **failed** — a flat, uninformative 0.0009
score, because the training data contains exactly one historical full-blackout (S4) event, and
walk-forward validation on a single positive example is close to meaningless (AUC — Area Under
the ROC Curve — collapses to 0 on that fold). Rather than quietly drop the classifier or overfit it into looking better than
it is, we kept the failure visible, made the LSTM detector *load-bearing by design* in the
ensemble (its weight fixed, not tuned against the one golden event, specifically so it
couldn't be overfit to n=1), and documented the classifier's real limitation as a data-scarcity
problem rather than a modelling one. The same honesty applies to the Markov chain over grid
states: transitions into rare, high-severity states are blended with an expert Dirichlet prior
precisely because there isn't enough historical data to estimate them empirically alone.

## A knowledge base that keeps growing, indexed in ChromaDB, without losing human oversight

The RAG layer indexes REE operating procedures, ENTSO-E standards, and incident post-mortems
in a local **ChromaDB** vector store, queried at runtime through three MCP tools —
`search_regulations`, `find_similar_incident`, `get_mitigation_procedure` — that the
AnomalyDetector and RegulatoryCompliance skills call directly; retrieval itself is pure
similarity search, no LLM in that step. But a one-time embedding script doesn't scale, and an
LLM freely researching and inserting "facts" into a safety-relevant knowledge base without a
human ever reviewing them is exactly the kind of unchecked autonomy this project tries to
avoid. So the ingestion pipeline is a repeatable loop: a list of pending topics is picked up by
a research agent (which explicitly tags anything it can't verify as unverified rather than
presenting it with false confidence), a validator cross-checks candidate facts against known
ground truth, and nothing reaches the vector store without a **human approval gate** in
between. Only after that does an idempotent embedding step run — plain Python except for the
one call to **`gemini-embedding-2`** that turns approved text into vectors, deliberately not
wrapped in any agent reasoning, since embedding text needs no judgment once a human already
approved it — with pruning and a manifest file as the single source of truth for what's
actually indexed. The first real batch through this pipeline processed 10 documents and added
42 chunks to the knowledge base — small, but proof that the loop works end-to-end, and that it
can be repeated safely as new regulations or incidents need to be added.

## Stress-testing the agent against a real LLM, not just green offline tests

Building the six-layer architecture was half the project; the other half was **validating it
against real Gemini calls** and following every surprise to its root cause instead of patching
symptoms. Four distinct findings, each investigated with evidence before touching a fix:

**Finding 1 — infrastructure.** Live evals kept timing out. Profiling with `py-spy` (a live
Python stack sampler) showed an agent lazily importing `chromadb` (which pulls in numpy) *inside*
a tool call, with the event loop already running — the C-extension load hung for minutes. Moving
the import to module load time fixed it: a deterministic repro went from >90 seconds to 0.8
seconds. Alongside it: what had been 17 separate MCP subprocess instances (one per agent per
server) were consolidated into 5 shared singletons, cutting the maximum subprocess count per
monitoring cycle by more than two-thirds, and every external I/O call (embeddings, ENTSO-E,
weather) got an explicit timeout after one was measured hanging for ~125 seconds.

**Finding 2 — robustness.** A skill occasionally answers with free text instead of the JSON its
`output_schema` requires. ADK 2.3.0 doesn't catch the resulting `pydantic.ValidationError` — it
aborts the *entire* orchestration run, producing no result at all for that cycle. The fix,
`_consume_events_safely()`, consumes the agent's event stream defensively, keeping whatever was
already captured instead of losing the whole cycle to one malformed turn.

**Finding 3 — agent behaviour.** Even with a fully deterministic escalation policy
(`tools/escalation_planner.py` — pure Python, 17 truth-table unit tests, computing exactly which
skills a given risk level requires), the orchestrator's LLM sometimes didn't follow it: stopping
early in a genuinely critical scenario, or over-triggering in a calm one. Reinforcing the prompt
twice made no measurable difference. The fix wasn't linguistic — it was architectural:
**`escalation_enforcer.py`** recomputes the policy after the LLM's turn and, by code, forces any
mandatory step the model skipped. This is the project's clearest shift-left moment: not just
replacing an LLM call with ML, but *removing a decision from the LLM entirely* once live
evidence showed it couldn't be trusted with it.

**Finding 4 — an empirical benchmark, not a bug.** `config/model_registry.yaml` is the project's
single source of truth for which Gemini model each role uses, built so cost/latency-vs-capability
can be tuned in one place with no model name ever hardcoded. Exercising that design, we first
pointed the agentic roles at `gemini-2.5-flash`, a cheaper prior-generation model. Live testing
turned that into a real comparison: with it, the orchestrator intermittently returned
`finish_reason: MALFORMED_FUNCTION_CALL` — a silently truncated cycle, no exception — and didn't
reliably follow the deterministic escalation policy. Switching the registry to `gemini-3.5-flash`
(confirmed live via `client.models.list()`) eliminated `MALFORMED_FUNCTION_CALL` entirely and made
the orchestrator execute its full 9–11-tool chain end-to-end, following `escalation_planner`'s
verdict exactly. We went one level deeper for `ReportAgent`'s 10-turn tool loop, switching it to
`gemini-3.1-pro-preview-customtools` — a variant the provider describes as optimized for custom
tool usage — over the generic `gemini-3.1-pro-preview` flagship: a genuine, reproducible data
point about the current agentic-capability gap across Gemini generations, found because the
registry let us swap models and re-test in seconds, with zero code changes to any agent. One
smaller bug (`gemini-3.5-flash` wraps its JSON reply in a markdown code fence) surfaced right
after and was fixed the same session. Still open, documented rather than hidden: the eval
harness's ML ensemble reads live database state rather than the golden scenario's own snapshot,
so a "critical" golden scenario doesn't automatically make the ensemble report a high
P(blackout) — an eval-infrastructure gap, not an agent-behaviour one.

Finding this deeper cause did **not** retire `escalation_enforcer` from Finding 3. The two fixes
address different failure modes: the model swap resolved the *observed* cause of non-compliance,
while the enforcer stays as unconditional defense-in-depth — a deterministic function is a no-op
when the LLM already followed the plan, and a monitoring system for grid stability shouldn't rely
on one clean live run as proof an LLM will always behave.

## Results

- **300 deterministic offline tests pass**, with zero network calls — every Python/ML/Markov
  layer, every guardrail, and the escalation policy itself are unit-tested independently of any
  LLM.
- **Over 120 commits**, 8 written design specs, and 13 implementation plans give this project a
  full, inspectable paper trail: every non-trivial feature went through design → written spec →
  implementation plan → independent review before merging, across data engineering, ML training,
  RAG pipeline design, MCP infrastructure, and agent reliability.
- 11 of 13 mis-derived ESIOS indicator IDs found and corrected; train/serve feature parity
  enforced by test, not assumed.
- Ensemble ML validated against a real, published historical blackout, with an honest,
  documented failure (the classifier, on n=1 positive data) rather than a hidden one.
- A repeatable, human-gated RAG ingestion pipeline, not a one-time script — first live batch
  processed and indexed successfully.
- Live agent evals against real Gemini calls, and a live-discovered empirical finding about
  agentic capability across Gemini model generations that changed the project's own
  configuration mid-session.
- A memory of learned lessons (`store_lesson`/`get_lessons`, backed by SQLite — chosen over a
  vector store because the volume doesn't justify embedding cost, a deliberate shift-left call)
  lets the orchestrator consult and record operational lessons across otherwise-stateless
  monitoring cycles; with `gemini-3.5-flash`, we watched it cite a specific stored lesson while
  reasoning about a new cycle.

## Why this matters beyond one grid

Every claim in this project is checked against evidence, not assumed: the golden test scenario
is a real, published incident, not a synthetic one; every "it works" claim is backed by a
specific test count or a specific log line; and every limitation encountered — data quality,
model honesty, infrastructure, robustness, or LLM behaviour — is documented with its exact
reproduction, not smoothed over. For a technology as new and unpredictable as LLM-driven agents,
we think that discipline — building real safety nets around what an LLM gets wrong, keeping ML
failures visible instead of overfitting them away, gating a growing knowledge base with a human
in the loop, and saying so clearly when something still doesn't work — is at least as valuable a
demonstration of "agent design" as a demo that always works on the first try.

Full architecture: `ARCHITECTURE.md`. Full engineering retrospective and every finding above, in
detail: `MEMORIA.md`. Every eval command and measured result: `eval/README.md`. All in the linked
repository.
