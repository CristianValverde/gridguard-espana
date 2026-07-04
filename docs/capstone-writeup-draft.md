<!--
Borrador para pegar en el Kaggle Writeup (Track: Agents for Good).
Cuenta de palabras del CUERPO (sin título/subtítulo): ver nota al final del fichero.
Adjuntar en la Media Gallery: imagen de portada + vídeo de YouTube (≤5 min).
Adjuntar como Project Link: https://github.com/CristianValverde/gridguard-espana (ya PÚBLICO)
-->

# TITLE

GridGuard España: A Multi-Agent System That Learns When Not to Trust the LLM

# SUBTITLE

Detecting blackout risk in the Spanish power grid with ADK + Gemini, deterministic ML, and a
safety net for the moments the agent doesn't behave.

# BODY

## The problem

On 28 April 2025, a real blackout took down the Spanish peninsular grid for 6.5 hours,
affecting up to 15,000 MW. REE's official post-mortem shows the warning signs were visible
hours in advance: extreme renewable penetration, falling synchronous inertia, a progressively
worsening supply/demand balance. But there's a catch — **grid frequency, the single most
direct instability signal, is not published by any public API.** REE withholds it for
operational security reasons. Anyone building a monitoring system for the Spanish grid has to
work with proxies, not ground truth.

GridGuard asks: can a multi-agent system, using only publicly available data, catch that kind
of situation early enough to matter — and can we be honest, in public, about exactly where an
LLM-based agent can and can't be trusted to do that reliably?

## The solution: three levels of intelligence, each reserved for what it's actually good at

GridGuard follows one rule throughout: never use an LLM where Python or an ML model is enough.

- **Pure Python** computes physical proxies for grid stress (supply/demand balance, activated
  regulation reserve, interconnection stress, % non-manageable renewables) and formats alerts.
- **Deterministic ML** — an ensemble of an Isolation Forest, an LSTM Autoencoder, and a
  calibrated Gradient Boosting classifier — fuses into a single P(blackout) estimate. A Markov
  chain over five grid states (S0 Normal → S4 Blackout), trained offline on historical
  transitions, projects where the grid is heading over the next few hours. Zero LLM calls in
  either path.
- **RAG** over REE/ENTSO-E operating procedures and past-incident post-mortems (Iberian
  separation 2006, this same 2025 blackout) gives the system real regulatory grounding instead
  of generic LLM knowledge.
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

Built on **Google ADK 2.3.0 + Gemini**, with **5 custom MCP servers**, **3 Skills**
(`AgentTool`-wrapped, single structured output each), **2 Sub-agents** with their own tool
loops, and a **guardrails layer** (`guardrails/hooks.py`) enforcing a per-agent tool whitelist,
rate-limiting, `.env`/secret-path blocking, prompt-injection sanitisation, and output-schema
validation before anything reaches the user.

## The real engineering story: what happens when the LLM doesn't cooperate

The most valuable part of this project wasn't building the system — it was **validating it
against a real LLM instead of settling for green offline tests**, and following every surprise
to its root cause instead of patching symptoms.

**Finding 1 — infrastructure.** Live evals kept timing out. Profiling with `py-spy` (a live
Python stack sampler) showed a single agent lazily importing `chromadb` (which pulls in numpy)
*inside* a tool call, with the event loop already running — the C-extension load hung for
minutes. Moving the import to module load time fixed it: a deterministic repro went from
>90 seconds to 0.8 seconds. Alongside it: 17 separate MCP subprocess instances were consolidated
into 5 shared singletons, and every external I/O call (embeddings, ENTSO-E, weather) got an
explicit timeout after one was measured hanging for ~125 seconds.

**Finding 2 — robustness.** A skill occasionally answers with free text instead of the JSON its
`output_schema` requires. ADK 2.3.0 doesn't catch the resulting `pydantic.ValidationError` — it
aborts the *entire* orchestration run, producing no result at all for that cycle. The fix,
`_consume_events_safely()`, consumes the agent's event stream defensively, keeping whatever was
already captured instead of losing the whole cycle to one malformed turn.

**Finding 3 — agent behaviour.** Even with a fully deterministic escalation policy
(`tools/escalation_planner.py` — pure Python, 17 truth-table unit tests, computing exactly
which skills a given risk level requires), the orchestrator's LLM sometimes didn't follow it:
stopping early in a genuinely critical scenario, or over-triggering in a calm one. Reinforcing
the prompt twice made no measurable difference. The fix wasn't linguistic — it was
architectural: **`escalation_enforcer.py`** recomputes the policy after the LLM's turn and, by
code, forces any mandatory step the model skipped. This is the project's clearest shift-left
moment: not just replacing an LLM call with ML, but *removing a decision from the LLM
entirely* once live evidence showed it couldn't be trusted with it.

**Finding 4 — an empirical benchmark, not a bug.** `config/model_registry.yaml` is the
project's single source of truth for which Gemini model each role uses — built precisely so a
cost/latency-vs-capability tradeoff can be tuned in one place, with no model name ever
hardcoded in the agents themselves. As part of exercising that design, we deliberately ran the
agentic roles against a more cost-efficient prior-generation model (`gemini-2.5-flash`) first.
Live testing turned that into a real, reproducible comparison: with the cost-efficient model,
the orchestrator intermittently returned Gemini's `finish_reason: MALFORMED_FUNCTION_CALL` —
no exception, just a silently truncated cycle — and didn't reliably follow the fully
deterministic `escalation_planner` policy. Pointing the registry at the current frontier model
(`gemini-3.5-flash`, confirmed available and stable via a live `client.models.list()` query)
eliminated `MALFORMED_FUNCTION_CALL` entirely in the next run, and the orchestrator executed
its full 9–11-tool chain end-to-end, following `escalation_planner`'s verdict exactly for the
risk level it actually computed — the first time all project the complete loop worked without
the enforcer needing to step in. This is a genuine empirical result about the current gap in
agentic capability (specifically, function-calling reliability and instruction-following) between
cost-efficient and frontier Gemini models for a multi-tool agent loop like this one — exactly
the kind of tradeoff the registry design exists to let us measure and act on quickly. A second,
smaller bug (the frontier model wraps its JSON reply in a markdown code fence, which our parser
didn't originally strip) surfaced immediately after and was fixed the same session. What remains
open, and is documented as such rather than hidden: the eval harness's ML ensemble reads live
database state rather than the golden scenario's own snapshot, so a "critical" golden scenario
doesn't automatically make the ensemble report a high `P(blackout)` — an eval-infrastructure
gap, not an agent-behaviour one, and one we chose not to patch under deadline pressure rather
than force a misleading green test.

## Results

- **300 deterministic offline tests pass**, with zero network calls — every Python/ML/Markov
  layer, every guardrail, and the escalation policy itself are unit-tested independently of any
  LLM.
- **123 commits**, 8 written design specs, and 13 implementation plans give this project a full,
  inspectable paper trail: every non-trivial feature went through design → spec → plan →
  implementation → independent review before merging.
- Live agent evals against real Gemini calls have reached **7/12 passing** in the best run, with
  the infrastructure causes (timeouts, crashes) fully eliminated — the remaining gap is LLM
  behavioural non-determinism, tracked and documented rather than hidden.
- A memory of learned lessons (`store_lesson`/`get_lessons`, backed by SQLite — chosen over a
  vector store because the volume doesn't justify embedding cost, a deliberate shift-left call)
  lets the orchestrator consult and record operational lessons across otherwise-stateless
  monitoring cycles, verified working end-to-end against the live LLM.

## Why this matters beyond one grid

Every claim in this project is checked against evidence, not assumed: the golden test scenario
is a real, published incident, not a synthetic one; every "it works" claim is backed by a
specific test count or a specific log line; and every limitation encountered — infrastructure,
robustness, or LLM behaviour — is documented with its exact reproduction, not smoothed over.
For a technology as new and unpredictable as LLM-driven agents, we think that discipline —
building real safety nets around what an LLM gets wrong, and saying so clearly when something
still doesn't work — is at least as valuable a demonstration of "agent design" as a demo that
always works on the first try.

Full architecture: `ARCHITECTURE.md`. Full engineering retrospective and every finding above,
in detail: `MEMORIA.md`. Every eval command and measured result: `eval/README.md`. All in the
linked repository.
