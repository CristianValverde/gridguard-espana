# GridGuard España — Submission Video Script (≤5 min, public YouTube)

Internal working document for recording the capstone video. Not part of the repo deliverable
itself — this is your recording aid. Timed to **4:45** to leave a 15s margin. Spoken lines are
in English (the video, like the Writeup, is submitted in English); production notes are in
English too so the whole document reads consistently while you're recording.

**Golden rule for recording:** this is a recording, not a live broadcast — if a take doesn't
come out well, redo it. The centerpiece demo is the Orchestrator's full cycle via
`run_scenario()` against the real April 2025 blackout scenario: a single call, ~20-40s, minimal
quota cost. Do 2-3 unrecorded practice takes before the real one, to know the timing and the
kind of result you'll be narrating.

## How to record: in blocks, not one continuous take

Record each section of this script (Problem / Why Agents / Architecture / Demo / How It Was
Built / Close) as an independent clip and stitch them together afterward with a free editor
(**DaVinci Resolve** — most complete; **Shotcut** — simpler; or **Clipchamp**, already built
into Windows 11, the fastest to learn if you only need to cut and join). Concrete advantages:

- Memorizing 45-85s of text per block is far more manageable than 5 minutes straight through.
- If one block comes out badly, you only redo that block — the good takes elsewhere are safe.
- Hitting the exact 5:00 limit becomes an editing trim, not a live-pacing exercise.

**Treat the human-approval block as a special case** (if you record BONUS (a) with the real
pipeline, not just `cat manifest.json`): only **4 topics remain pending** in
`data/kb_seed/topics.yaml` (`entsoe_emergency_restoration`, `procedimientos_deslastre_reposicion`,
`post_mortem_apagon_2025`, `separacion_iberica_2006`), and actually approving one embeds it
permanently — unlike `run_scenario()`, which you can rerun almost for free. Record it with
margin: let the research agent finish, read the draft calmly, and only then approve and run the
embedding step. If the first attempt is weak, you have others to retry with, but not
unlimited ones — don't leave this block for the end of the day if you're short on time before
the deadline.

Keep the same window/terminal framing and the same microphone across all blocks so the cuts
don't stand out when you edit.

## Visual support, at a glance

| Block | On-screen asset |
|---|---|
| Problem | Cover image/title, then one English-language source on the blackout (see below) |
| Why Agents | The "3 levels of intelligence" table from `README.md` / `MEMORIA.md` §3 |
| Architecture | `architecture_diagram.html`, open in a browser |
| Demo | Terminal inside Antigravity (or your IDE of choice), running the commands live |
| Demo — BONUS | Same terminal: `manifest.json` and/or the `search_regulations` one-liner |
| How It Was Built | `docs/superpowers/specs/` + `docs/superpowers/plans/` folders, or `MEMORIA.md` |
| Close | `README.md` landing view, or a title card |

**On the English-language source for the Problem block:** don't just describe the blackout —
show a real page about it, since the whole point is that this is a real, published event, not
a synthetic scenario. Good options, all in English:
- **ENTSO-E's own official blackout page** — `https://www.entsoe.eu/publications/blackout/28-april-2025-iberian-blackout/`.
  This is a strong choice because it's the *same organization* whose procedures your RAG
  layer indexes — a nice visual callback.
- **NPR's coverage from the day of the event** — `https://www.npr.org/2025/04/28/g-s1-63175/power-outage-spain-portugal-blackout-france`.
  More of a "breaking news" feel if you want that framing for the opening.
- **Wikipedia's "2025 Iberian Peninsula blackout"** article — clean, easy to screenshot, good
  if you just need a stable, scrollable overview page.

Pick whichever fits the framing of your opening line best; ENTSO-E is the most thematically
tied to the rest of the project.

**On using Antigravity for the demo — IDE, not 2.0:** these are two different products, not two
versions of the same thing. **Antigravity IDE** (the original) is editor + terminal + file
explorer in one VS Code-based window — exactly the combined view this demo needs, for both the
live commands and the file walkthroughs. **Antigravity 2.0** (Google I/O 2026) is a separate,
standalone desktop app for orchestrating multiple autonomous agents in parallel across
projects — it has no built-in editor/file view, and its whole point (parallel agent
orchestration) doesn't match what this demo shows (running one script, reading files). Use only
the **IDE** for the whole recording; there's no reason to bring in 2.0 here.

One honest caveat, carried over from an earlier discussion: recording *in* Antigravity IDE
doesn't by itself demonstrate the "Antigravity" concept from the rubric (that would require it
to have been part of actually *building* the agent). Use it here simply as your recording
environment, not as a claim about what built GridGuard.

---

## 0:00–0:45 — Problem (45s)

**What to say (script, adapt it to your voice):**

> "On 28 April 2025, Spain suffered a real blackout that lasted six and a half hours and
> affected up to 15,000 megawatts. REE's official post-mortem shows the warning signs were
> there hours in advance: extreme renewable penetration, falling synchronous inertia, a
> progressively worsening supply/demand balance. But there's a catch: **grid frequency — the
> most direct signal — isn't published by any public API.** REE withholds it for operational
> security reasons. This project asks: can a multi-agent system, working only with what's
> actually publicly available, catch that kind of situation in time?"

**On-screen visual:** the project title + the cover image, then cut to the English-language
blackout source you picked above (ENTSO-E / NPR / Wikipedia) for a few seconds while you say
"REE's official post-mortem" — no terminal needed yet.

---

## 0:45–1:30 — Why Agents (45s)

**What to say:**

> "GridGuard combines three kinds of intelligence, each only where it's actually needed: pure
> Python for what's deterministic, Machine Learning to detect statistical patterns — an
> ensemble of an Isolation Forest, an LSTM autoencoder, and a calibrated classifier — and a
> Markov chain projecting where the grid is heading. The LLM only steps in when real reasoning
> is needed: synthesizing signals, checking regulations, deciding how to escalate. This
> principle is called shift-left, and we applied it retroactively too: when we found the LLM
> wasn't reliably following the escalation policy, the fix wasn't 'improve the prompt' — it was
> taking that decision away from the LLM entirely and handing it to a deterministic Python
> function with 17 tests."

**On-screen visual:** the "3 levels of intelligence" table (read it from `README.md` or
`MEMORIA.md` §3, or turn it into a slide).

---

## 1:30–2:45 — Architecture (75s)

**What to say:**

> "Six layers. Triggers kick off the cycle. Layer 1 is pure Function Tools — no LLM. Layer 2 is
> 5 custom MCP servers: grid data, weather, persistent memory, ML models, and a RAG knowledge
> base over REE and ENTSO-E regulations. Layer 3 is Skills — agents wrapped as synchronous
> tools, each with a single structured result. Layer 4 is sub-agents with their own loop — one
> drafts emergency reports, another evaluates cross-border energy imports. And on top,
> the Orchestrator, the root agent that coordinates all of this and decides how to escalate.
>
> All of this runs locally — we didn't deploy it to the cloud, for reasons of both cost and
> time given the project's deadline. In production the infrastructure would change, not the
> agent logic — the README has the full detail."

**On-screen visual:** the architecture diagram (`architecture_diagram.html`, up to date) or the
ASCII layer diagram from `README.md`/`ARCHITECTURE.md`.

---

## 2:45–4:00 — Demo (75s)

**The centerpiece: the full Orchestrator, live, against the real April 2025 blackout.**

Before starting, a quick flash (5s, no narration) of the offline suite passing — no need to say
anything over it, just let it be visible for a second:

```bash
python -m pytest -q
```

And the centerpiece — the Orchestrator's full cycle over the real blackout snapshot:

```bash
python -c "
import json
from agents.orchestrator import run_scenario
scenario = json.load(open('eval/golden_dataset/blackout_apr_2025.json', encoding='utf-8'))
print(run_scenario(scenario))
"
```

**What to say while it runs (~20-40s, real time to narrate):**

> "This is exactly what a cron job would trigger every 10 minutes in production — I'm just
> firing it manually, now, for the camera. I'm feeding it the real Spanish grid snapshot from
> the hours before the April 2025 blackout, and the Orchestrator decides on its own, with real
> Gemini: it normalizes the data, consults the ML ensemble and the Markov chain, computes the
> risk band, and decides which agents to escalate — with no intervention from me at any point."

**What to say once the result is on screen:**

> "There it is: [read the risk band and which skills it escalated]. This is exactly what
> `escalation_planner` — our deterministic policy, not the LLM — dictates for that band. We're
> showing this live, no pre-recorded captures, because we trust the system we built."

**On-screen visual:** the terminal (inside Antigravity or your IDE of choice) with the full
result (the dict `run_scenario` returns), keeping the `tool_names` field or the tool/agent count
visible for a couple of seconds before cutting.

**Practical note:** the exact result (band, escalated skills) can vary slightly between takes
due to the LLM's own non-determinism — that's expected, not a failure; just read it as it comes
out. If a take doesn't illustrate the point well, redo it: 20-40s and minimal quota cost per
attempt.

**Timing note:** with the silent `pytest -q` flash, this section's spoken script (~140 words)
fits comfortably even at the low end of the 20-40s window.

### OPTIONAL BONUS — only if time is left after practice takes

Two short closers, in priority order (the first is free, the second costs ~1-3s of quota).
Neither has a fixed slot in the timing — they only go in if a practice take of the Orchestrator
comes in shorter than expected.

**a) The human-approval gate (zero cost, 3-4s, no long narration):**

```bash
cat data/kb_seed/manifest.json
```

> "And the one place a human intervenes in the entire system: every fact that enters the
> knowledge base gets recorded here, with who reviewed and approved it — in this case, me."

Show on screen the entry with `"reviewed_by": "Cristian Valverde"` and `"status": "indexed"`.

**b) A direct semantic query to ChromaDB, bypassing any agent (10-15s, ~1-3s of quota):**

```bash
python -c "
from mcp_servers.knowledge_base_server import search_regulations
import json
print(json.dumps(search_regulations('low synchronous inertia in the grid', top_k=2), indent=2, ensure_ascii=False))
"
```

> "And this is how `search_regulations` — the same tool the agent just called — finds the
> right fragment: pure vector search, no LLM reasoning in this step."

If only one fits, prioritize (a): zero cost, zero timing risk, and concrete visual proof of the
human gate that the rest of the script only mentions in passing.

---

## 4:00–4:45 — How It Was Built (45s)

**What to say:**

> "It was built with Google ADK 2.3.0 and Gemini. Development leaned on a different code
> assistant with its own IDE, already under an active subscription — the free tier of the
> Gemini API was already used up, so every paid token was reserved for what actually needed it:
> testing the agents against the real model, not planning or writing code. Every feature went
> through a process of design → written spec → implementation plan → execution with independent
> review at every step — not just 'code that works', but documented, traceable decisions. When
> live validation revealed unexpected LLM behavior, we didn't ignore it: we corrected each
> issue with evidence, not blind patches. What we're still polishing is documented too."

**On-screen visual:** the `docs/superpowers/specs/` and `docs/superpowers/plans/` folders (to
show volume — 8 specs, 13 plans), or `MEMORIA.md` open at the findings section.

---

## 4:45–5:00 — Close (15s)

**What to say:**

> "GridGuard España: six layers of Python, ML, and Gemini-powered agents watching the Spanish
> grid in real time, with its own safety net for the moments the LLM doesn't behave. Full code
> and documentation in the repository."

**On-screen visual:** the repo's `README.md` landing view, or a closing title card.

---

## Checklist before recording

- [x] Architecture diagram updated and visually verified (2026-07-05)
- [ ] GitHub repo already public (otherwise the video/Writeup link won't work)
- [ ] `python -m pytest -q` run right before recording (confirms 300 passed)
- [ ] `GOOGLE_API_KEY` configured; 2-3 unrecorded practice takes of `run_scenario()` to know
      timing and result type before the real take
- [ ] English-language blackout source picked and open/bookmarked for the Problem block
- [ ] Screen recording ready (Win+G, or your usual recorder) + your voice or TTS audio
- [ ] Final video ≤ 5:00 exactly, uploaded to YouTube as **public** (not "unlisted")
