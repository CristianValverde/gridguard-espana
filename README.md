# GridGuard España

**A multi-agent AI system that watches the Spanish electrical grid and escalates warnings
before a blackout happens.**

> Capstone Project · 5-Day AI Agents Intensive (Google × Kaggle) · Track: **Agents for Good**

Built with Google **ADK 2.3.0** + **Gemini**, deterministic **ML** (Isolation Forest, LSTM
Autoencoder, calibrated Gradient Boosting), a **Markov chain** of grid states, and **RAG**
over Spanish/EU grid regulations — following one rule throughout: *never use an LLM where
Python or an ML model is enough.*

---

## The problem

On 28 April 2025, a real blackout took down the Spanish peninsular grid for 6.5 hours,
affecting up to 15,000 MW. The official REE post-mortem shows the warning signs were there
hours in advance: extreme renewable penetration, falling synchronous inertia, a progressively
worsening supply/demand balance. **Grid frequency itself — the most direct signal — is not
published by any public API** (REE withholds it for operational security). This project asks:
can a multi-agent system, working only from the data that *is* publicly available, catch that
kind of situation early enough to matter?

## The solution — three layers of intelligence, only when each is needed

| If the task is… | GridGuard uses… |
|---|---|
| API call + parsing + validation | Function Tool (pure Python) |
| Statistical pattern detection in time series | ML model (deterministic at inference) |
| "Where is the grid heading?" | Markov chain over 5 grid states (S0–S4), pure NumPy |
| Regulatory/technical context | RAG (ChromaDB) over REE/ENTSO-E procedures & past incidents |
| Multi-signal reasoning | Skill (LLM `AgentTool`, single structured output) |
| Multi-step autonomous task | Sub-agent (LLM with its own tool loop) |
| Escalation policy across all of the above | **Deterministic Python** — see below |

```
Layer 0  Triggers (cron / event queue / eval runner)
Layer 1  Function Tools — pure Python (ree_poller, data_normalizer,
         stability_proxy_calculator, risk_scorer, alert_formatter,
         escalation_planner, markov_state_tracker)
Layer 2  5 MCP Servers — grid-data · weather · memory · ml-models · knowledge-base
Layer 3  Skills (AgentTool) — AnomalyDetector · RenewablesForecast · RegulatoryCompliance
Layer 4  Sub-agents — ReportAgent · CrossBorderAgent
Layer 5  OrchestratorAgent — Root Agent, coordinates everything above
```

Full technical specification, MCP tool signatures, hooks and schemas: **[ARCHITECTURE.md](ARCHITECTURE.md)**.
Project retrospective — engineering journey, findings, honest limitations: **[MEMORIA.md](MEMORIA.md)**.

### The most interesting engineering decision: taking the LLM's decision away from it

The orchestrator's escalation policy (*which skills to call, at which risk band*) originally
lived as prose inside the Gemini system prompt. Live testing against the real LLM showed it
didn't reliably follow that policy — it would stop escalating partway through a critical
scenario, or over-trigger in a normal one. Reinforcing the prompt twice made no difference.
The fix was architectural, not linguistic: **[`tools/escalation_planner.py`](tools/escalation_planner.py)**
extracts the entire policy into a pure, unit-tested Python function (17 truth-table tests),
and **[`agents/escalation_enforcer.py`](agents/escalation_enforcer.py)** is a deterministic
safety net that runs *after* the LLM turn and forces, by code, any mandatory step the model
skipped. This — and two other robustness layers found the same way (see §*Findings* in
[MEMORIA.md](MEMORIA.md)) — is the project's real shift-left story: not just using ML instead
of an LLM for scoring, but *removing the LLM from a decision entirely* once live evidence
showed it couldn't be trusted with it.

## Key course concepts demonstrated (of the 6 listed in the rubric, we cover 4)

| Concept | Where |
|---|---|
| Multi-agent system (ADK) | `agents/orchestrator.py` (Root) + 3 Skills (`AgentTool`) + 2 Sub-agents |
| MCP Servers | `mcp_servers/` — 5 custom servers, shared stdio transport (`agents/mcp_toolsets.py`) |
| Security features | `guardrails/hooks.py` — tool whitelist per agent, rate-limiting, `.env`/secret-path blocking, prompt-injection sanitisation, output-schema & numeric-range validation |
| Agent skills | `agents/anomaly_detector.py`, `renewables_forecast.py`, `regulatory_compliance.py` — `AgentTool`-wrapped, single structured output |

*Not used in this project: "Antigravity". Development ran on a different code assistant
with its own IDE, via an existing active subscription — a sunk cost — rather than a
Gemini-based agentic IDE: the free tier of the Gemini API was already exhausted by the
time heavy iteration started, and every paid token of API quota went deliberately toward
what actually needed it — live-testing the agents themselves against real Gemini calls
(see `eval/README.md`), not toward planning or writing the code. Also not used: a live
deployed endpoint (see "Deployability" note below).*

## What's genuinely working vs. what's an honest limitation

- **300 offline tests pass**, deterministically, no network calls (`pytest -q`).
- The deterministic layers (`escalation_planner`, ML ensemble, Markov chain, all MCP tools)
  work reliably and are unit/integration tested.
- **The single most valuable finding of this project was a deliberate empirical benchmark.**
  `config/model_registry.yaml` exists precisely so the cost/latency-vs-capability tradeoff for
  each agent role can be tuned in one place, with no model name ever hardcoded. As part of
  exercising that design, we deliberately ran the agentic roles against a more cost-efficient
  prior-generation model (`gemini-2.5-flash`) first, specifically because it was cheaper. Live
  testing turned that choice into a real comparison: with the cost-efficient model, the
  orchestrator intermittently returned `MALFORMED_FUNCTION_CALL` (a silently truncated cycle, no
  exception) and didn't reliably follow the fully deterministic `escalation_planner` policy.
  Pointing the registry at the current frontier model (`gemini-3.5-flash`, confirmed live via
  `client.models.list()`) eliminated `MALFORMED_FUNCTION_CALL` entirely and made the orchestrator
  follow the escalation policy exactly, in the very next run — the first time the full loop
  worked end-to-end. We went one level deeper for the sub-agent role too: instead of the generic
  flagship model, `smart_agent` uses `gemini-3.1-pro-preview-customtools`, a variant explicitly
  described by the provider as optimized for custom tool usage — a better fit for `ReportAgent`'s
  10-turn, 6-tool loop. This is a genuine, reproducible data point about the current
  agentic-capability gap between cost-efficient and frontier Gemini models for a multi-tool agent
  loop, discovered because the registry design let us swap models and re-test in seconds. One
  smaller bug surfaced immediately after (the frontier model wraps its JSON reply in a markdown
  code fence) and was fixed the same session. See [`eval/README.md`](eval/README.md) for the full
  before/after evidence, every other finding along the way, and what still remains a documented,
  not-yet-fixed limitation (the eval harness's ML ensemble reads live database state rather than
  the golden scenario's own snapshot, so a golden "critical" scenario doesn't automatically
  produce a "critical" `P(blackout)` — an eval-infrastructure gap, not an agent-behaviour one).
  We treat this transparency as part of the technical quality of the submission, not a gap to
  paper over.

## Setup

**Requirements:** Python 3.14, a Google AI Studio API key (free tier works), and free tokens
for ESIOS/AEMET/ENTSO-E if you want live grid data (the offline test suite needs none of these).

```bash
git clone https://github.com/CristianValverde/gridguard-espana.git
cd gridguard-espana
pip install -r requirements.txt
cp .env.example .env   # fill in GOOGLE_API_KEY at minimum; see .env.example for the rest
```

Environment variables (`.env`, never committed — see `.gitignore`):

```bash
GOOGLE_API_KEY=...      # aistudio.google.com → "Get API Key" — required for any live agent run
ESIOS_TOKEN=...         # consultasios@ree.es — optional, only for live Spanish grid data
AEMET_API_KEY=...       # opendata.aemet.es — optional, weather data
ENTSOE_TOKEN=...        # transparency.entsoe.eu — optional, EU interconnection data
```

Note: some RAG code paths read `GEMINI_API_KEY` instead of `GOOGLE_API_KEY` for the embeddings
call (`mcp_servers/knowledge_base_server.py`) — set both to the same value if you plan to
exercise the knowledge-base server.

### Run the offline test suite (no API keys needed)

```bash
python -m pytest -q
# 300 passed, 16 deselected
```

### Run a live monitoring cycle against a golden scenario

```bash
python -c "
import json
from agents.orchestrator import run_scenario
scenario = json.load(open('eval/golden_dataset/blackout_apr_2025.json', encoding='utf-8'))
print(run_scenario(scenario))
"
```

Requires `GOOGLE_API_KEY`. This now runs the full 9–11 tool chain end-to-end (data
normalisation → ML ensemble → Markov state → deterministic escalation → lessons memory →
skills) — see the "single most valuable bug" note above for why it didn't a few hours before
this line was written. `config/model_registry.yaml` is the single source of truth for which
Gemini model is used; if you see `MALFORMED_FUNCTION_CALL` or an empty result, check that the
registry actually points at a currently-available model first (`python -c "from google import
genai, os; print([m.name for m in genai.Client(api_key=os.environ['GOOGLE_API_KEY']).models.list()])"`).

### Run the live agent evals (consumes real Gemini quota, ~10–25 min)

```bash
python -m pytest eval/agents -m integration -v
```

See [`eval/README.md`](eval/README.md) for the full breakdown per agent, offline vs. live
commands, and the measured results of every validation run this project has done.

## Deployability

Participants are not required to deploy to a live endpoint for judging. This project runs
locally by design for the demo (MCP servers as local stdio subprocesses, SQLite for
persistence, ChromaDB as a local vector store) — no cloud deployment was set up, for
reasons of both cost and time given the project's deadline. All code needed to run it
end-to-end is in this repository.

Running this 24/7 would require an infrastructure substitution, not a redesign of the agent
logic itself:

- **SQLite → Cloud SQL / managed Postgres** — local SQLite has no concurrent-write story and
  no managed backups; a 24/7 monitoring loop needs both.
- **Local ChromaDB → a managed vector store** (Vertex AI Vector Search, or a hosted Chroma
  server) — the local persistent client isn't built for concurrent access from multiple
  long-running processes.
- **MCP servers as stdio subprocesses → persistent services** — each server as its own Cloud
  Run service (HTTP/SSE transport instead of stdio), so agents don't pay subprocess-startup
  cost per cycle and can scale independently.
- **APScheduler cron (Layer 0) → Cloud Scheduler + Cloud Run Jobs** — replaces in-process
  scheduling with infrastructure that survives restarts.
- **`.env` → Secret Manager** — same secrets, a store built for production rather than a
  local file.

## Project structure

```
agents/            Root agent, Skills, Sub-agents, shared MCP toolsets, escalation logic
tools/              Pure-Python Function Tools (no LLM)
mcp_servers/        5 MCP servers (grid-data, weather, memory, ml-models, knowledge-base)
ml/                 Data pipeline, feature engineering, model training, Markov chain
knowledge/          RAG documents + ChromaDB build script
guardrails/         ADK callback hooks (security, validation)
eval/               Per-agent evals (trigger/output/trajectory, offline + live) + golden dataset
schemas/            Pydantic models — single source of truth for all data shapes
config/             Model registry + loader (never hardcode a Gemini model name)
docs/superpowers/   Design specs and implementation plans for every feature (full paper trail)
```

## Security

- API tokens are read exclusively from environment variables (`config/model_loader.py`,
  `.env`, never committed — see `.gitignore`).
- `guardrails/hooks.py` blocks any tool from reading `.env`/`*.key`/`secrets.*`/`credentials.*`,
  enforces a per-agent tool whitelist, rate-limits external API calls, sanitises prompt-injection
  patterns, and validates numeric ranges and output schemas before anything reaches the LLM or
  the user.
- No API keys or secrets are present anywhere in this codebase — verified against the full git
  history before submission.

## License

Code in this repository is licensed under **CC-BY 4.0**, per the competition rules.
