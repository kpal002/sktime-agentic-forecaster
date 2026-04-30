# ESoC 2026 — sktime Agentic Track

**Project:** Agentic Forecaster + sktime-mcp tool surface
**Hub / Sponsor:** German Center for Open Source AI (GC.OS)
**Mentoring org:** sktime
**Applicant:** Kuntal Pal
**Repository (prototype):** this folder — `sktime_agentic_forecaster/`
**Target ideas:** Project Idea **#2 (Agentic forecaster or estimator)** with the **bonus** integration into [`sktime/sktime-mcp`](https://github.com/sktime/sktime-mcp). Upstream anchor: [`sktime/sktime#9721`](https://github.com/sktime/sktime/issues/9721).

## Summary

Most "LLM picks a model" demos work as one-shot prompts that return a free-form code snippet. That isn't shippable in a library — it can't be reviewed, can't be reproduced, and can't compose with the rest of `sktime`'s machinery.

This project ships a different shape: an **`AgenticForecaster`** that fits cleanly into `sktime`'s `BaseForecaster` contract. Its `fit` method delegates *forecaster selection and configuration* to an LLM running a **tool-constrained ReAct loop**. The agent works on a small audited surface — `summarize_data`, `list_forecasters`, `inspect_forecaster`, `fit_candidate`, `score`, `commit` — and never executes free-form code. Once it `commit`s, the resulting forecaster behaves like any other `sktime` forecaster: it composes inside `ForecastingPipeline`, `EvaluateGridSearchCV`, etc.

The same tool surface is exposed as a `FastMCP` server (`sktime_agentic.mcp_server`), so the agent can run **in-process** for library use or be driven **over MCP** from Claude Desktop / Cursor / any MCP-aware client. That second mode is a direct contribution back to the existing `sktime-mcp` project — the bonus path described in the ESoC ideas list.

## Why this fits the call

The ideas list calls out three classes of contribution:

1. **`sktime-mcp`** contributions — code, tools, docs, CI.
2. **Agentic forecaster / estimator** — issue [#9721](https://github.com/sktime/sktime/issues/9721). *Bonus for using `sktime-mcp` tools*.
3. **Foundation models** (issue #6177) and **own ideas**.

This proposal sits exactly at the intersection of (1) and (2). The prototype already implements the agent and ships a thin MCP server using the same tool surface. Each milestone produces concrete, mergeable PRs into either `sktime/sktime-mcp` or `sktime/sktime`.

## What the prototype already shows

The accompanying repo is a working POC:

- `AgenticForecaster` runs end-to-end against a deterministic `MockLLMClient` (no API key required) and against Anthropic Claude when an API key is present.
- The full ReAct loop is wired up — `summarize → list → inspect → fit → score → commit`, with the agent's transcript exposed as `forecaster.transcript_`.
- `FastMCP` server stub at `sktime_agentic.mcp_server` exposes the same six tools.
- Smoke tests pass without `sktime` installed (the registry falls back to a mini-set of `Naive` / `Mean` / `SeasonalNaive` forecasters); when `sktime` is available the registry binds to real forecasters.
- BSD 3-Clause license, matching `sktime` and `sktime-mcp`.

This is intended to demonstrate an *executable design*, not the final product. The ESoC milestones below scale it up.

## Milestones (12 weeks)

| Phase  | Outcome                                                                                                    |
|--------|------------------------------------------------------------------------------------------------------------|
| Wk 1–2 | **Upstream the MCP tool surface.** Land tool definitions + `FastMCP` server in `sktime/sktime-mcp`. CI, docs, basic packaging. |
| Wk 3–4 | **Land `AgenticForecaster` v0** in a feature branch of `sktime` behind a `genai` extra. Univariate forecasting only. Reference issue #9721. |
| Wk 5–6 | **Probabilistic + exogenous.** Support `predict_interval` / `predict_quantiles` and `X` for the inner forecaster. Pipeline composition (the agent's commit can be a `ForecastingPipeline`). |
| Wk 7–8 | **`AgenticPipeline`** — the agent composes `Imputer` → `Detrender` → `Forecaster` rather than picking a single forecaster. Tutorial notebook in `sktime/examples`. |
| Wk 9–10| **Eval harness.** Benchmark vs. `AutoARIMA` / `AutoETS` / `AutoTheta` on `M3`, `M4`-yearly, `tourism`. Publish a notebook with reproducible numbers. |
| Wk 11–12 | Hardening, docs, second-pass review, mentor handoff.                                                     |

Each PR is intended to be ≤ 600 LOC of net new code in `sktime` so reviewers can land it incrementally.

## Required pre-application contribution

The ideas list says applicants should make at least one substantial PR to `sktime` before applying, ideally on **detection / TS classification / TS regression**. Plan:

- File an issue in `sktime/sktime-mcp` reporting the prototype findings (gaps in the existing tool surface, suggested additions: `summarize_data`, `score`, `commit`).
- Open a draft PR to `sktime/sktime-mcp` upstreaming the tool definitions from this prototype.
- Open a draft PR to `sktime/sktime` for one of the open "good-first-issue" tickets in the detection / TS classification track (separate from the project itself, to satisfy the substantive-PR requirement).

## Risks and mitigations

| Risk                                                                                  | Mitigation                                              |
|---------------------------------------------------------------------------------------|---------------------------------------------------------|
| LLM cost / token budget for benchmarks.                                               | Use the deterministic mock client in CI; gate API-using tests behind an env var. Request a token budget on `dev-chat` Discord per the ideas list. |
| Curated registry vs. full `all_estimators`.                                           | Make the registry a YAML config so mentors / users can adjust without code changes. |
| Diverging from upstream `sktime-mcp` design.                                          | Open the design RFC in week 1 and post on Discord `dev-chat` before writing PR code. |
| Probabilistic forecasting tag complexity.                                             | Punt to week 5 once v0 lands; keep v0 univariate point forecasts. |
| Reproducibility of agent runs.                                                        | Persist the full transcript on the fitted forecaster; expose `seed` / temperature controls. |

## Why me

- I work day-to-day with Python ML and have built systems on top of LLM APIs and MCP. The prototype in this repo was built end-to-end in one sitting, including the BaseForecaster-compatible class, the ReAct loop, and the MCP server stub.
- I have a working understanding of `sktime`'s tag-based registry and `BaseForecaster` extension contract — the prototype already uses both.
- The prototype proves I can frame the work as *mergeable PRs into existing repos* rather than a parallel project.

## License

BSD 3-Clause, matching `sktime` and `sktime-mcp`. Code in this repo is released under the same license.
