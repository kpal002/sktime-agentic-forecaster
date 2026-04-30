# sktime-agentic-forecaster

> **An LLM-driven forecaster that constructs an `sktime` pipeline from data and an English prompt — over MCP.**

Prototype contribution proposed for the [GC.OS / ESoC 2026 sktime agentic track](https://github.com/gc-os-ai/mentoring-projects/blob/main/2026/ideas_list.md). Targets project idea **#2 (Agentic Forecaster)** with the **bonus** integration into [`sktime-mcp`](https://github.com/sktime/sktime-mcp). Anchored to the upstream design discussion in [`sktime/sktime#9721`](https://github.com/sktime/sktime/issues/9721).

## What it does

`AgenticForecaster` is a drop-in `sktime` forecaster — same `fit(y, X, fh)` / `predict(fh)` interface — except its "fit" step delegates *forecaster selection and configuration* to an LLM running a ReAct loop over a constrained tool surface.

```python
from sktime_agentic import AgenticForecaster
from sktime.datasets import load_airline

y = load_airline()

f = AgenticForecaster(
    prompt="Monthly airline passengers. Strong yearly seasonality, "
           "growing trend. Pick a forecaster that handles both.",
    backend="anthropic",                # or "mock" for offline tests
    transport="in-process",             # or "mcp" to talk to a sktime-mcp server
)
f.fit(y, fh=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y_pred = f.predict()

print(f.rationale_)        # natural-language explanation of the choice
print(f.selected_)         # e.g. "ExponentialSmoothing(trend='add', seasonal='mul', sp=12)"
```

## Why this design

The agentic-forecaster idea has many shapes. This prototype takes a stance:

1. **Tool-constrained, not free-form.** The LLM never writes raw Python. It plans over a small, audited tool surface (`list_forecasters`, `inspect_forecaster`, `fit_candidate`, `score`, `commit`). This makes the agent's actions reproducible and reviewable — a hard requirement for a library contribution.
2. **MCP-native.** The same tool surface is exposed via a `FastMCP` server (`sktime_agentic.mcp_server`). The agent can run **in-process** for local use or **over MCP** for use from Claude Desktop / Cursor / any MCP client. This is the bonus path described in the ideas list and a direct contribution back to `sktime-mcp`.
3. **`BaseForecaster`-compatible.** The class fits cleanly into existing `sktime` pipelines, ensembles, and tuners — i.e. an `AgenticForecaster` can be a step inside a `ForecastingPipeline` or wrapped in `EvaluateGridSearchCV`. This is what makes it a *library contribution*, not a script.
4. **Rationale as a first-class output.** `f.rationale_` is the agent's English-language justification of the chosen pipeline. This addresses the ideas-list note: *"... possibly also returning an English language prompt."*

## Architecture

```
                        ┌──────────────────────────────────┐
   English prompt ───►  │   AgenticForecaster (BaseForecaster)
   y, X, fh        ───► │                                  │
                        │   ┌──────────────────────────┐   │
                        │   │  ReAct loop              │   │
                        │   │  plan → tool → observe   │   │
                        │   │  → revise → commit       │   │
                        │   └────────────┬─────────────┘   │
                        │                │                 │
                        │   tools (in-process or MCP)      │
                        │   ┌────────────▼─────────────┐   │
                        │   │ list_forecasters         │   │
                        │   │ inspect_forecaster       │   │
                        │   │ fit_candidate / score    │   │
                        │   │ commit                   │   │
                        │   └──────────────────────────┘   │
                        │                │                 │
                        │           sktime registry        │
                        └──────────────────────────────────┘
                                   │
                          y_pred, rationale_, selected_
```

## Status

This is a **working prototype** intended to back a proposal for ESoC 2026. It runs end-to-end against the bundled `MockLLMClient` (no API key needed) and against Anthropic Claude when an `ANTHROPIC_API_KEY` is set. The `sktime` integration is gated behind a soft import — the smoke tests work without `sktime` installed.

| Component                                | State          |
|------------------------------------------|----------------|
| `AgenticForecaster` (sktime-compatible)  | Working POC    |
| ReAct loop                               | Working POC    |
| In-process tool registry                 | Working POC    |
| `MockLLMClient` (deterministic policy)   | Working POC    |
| Anthropic Claude backend (with prompt caching) | Implemented, requires API key + `anthropic` SDK |
| `FastMCP` server (`sktime_agentic.mcp_server`) | Implemented, requires `mcp` SDK |
| `transport='mcp'` in `AgenticForecaster` | Implemented (sync→async bridge via `anyio`, requires `mcp` SDK) |
| `examples/01_basic_usage.py`             | Runs end-to-end |
| `examples/02_natural_language_query.py`  | Runs end-to-end |
| `examples/03_mcp_mode.py`                | Demonstrates MCP server via stdio (requires `mcp` SDK) |

## Install / run

```bash
# Core (offline / mock LLM)
pip install -e .

# With Anthropic Claude
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...

# With MCP server
pip install -e ".[mcp]"

# Run the smoke test (no sktime, no API key needed)
python -m pytest tests/

# Run the airline-passengers demo
python examples/01_basic_usage.py
```

## Live output

### Gemini 2.0 Flash (`--backend gemini`)

Running `examples/02_natural_language_query.py` against a synthetic daily retail series
(2 years, weekly seasonality + mild upward trend):

```
Using backend: gemini

Selected : ExponentialSmoothing  params={'seasonal': 'add', 'trend': 'add', 'sp': 7}

Rationale:
ExponentialSmoothing (Holt-Winters) is a simple and fast forecaster that can model
both additive trend and additive weekly seasonality, as requested by the user.
It achieved a satisfactory MAE of 6.74 on the holdout set.

Forecast (next 14 days):
  + 1: 111.42
  + 2:  96.55
  + 3:  69.71
  + 4:  54.35
  + 5:  59.39
  + 6:  85.02
  + 7: 108.28
  + 8: 113.73   ← weekly pattern repeats with slight upward drift
  + 9:  98.85
  +10:  72.01
  +11:  56.65
  +12:  61.69
  +13:  87.32
  +14: 110.58
```

Gemini correctly identifies the weekly cycle (sp=7) and upward trend, selects
Holt-Winters with additive components, and produces a forecast with realistic
weekly rhythm and ~2-unit upward drift per cycle.

### Mock backend (offline, no API key)

```
Using backend: mock
Selected: SeasonalNaiveForecaster {'sp': 7}

Rationale:
Picked SeasonalNaiveForecaster with params {'sp': 7}. Series length=730,
candidate seasonal period=7. It scored best on mape (0.1440) on the holdout.

Forecast (next 14 days):
  + 1: 101.02     + 8: 101.02  ← weekly pattern repeats (flat)
  + 2:  89.77     + 9:  89.77
  + 3:  83.20    +10:  83.20
  + 4:  57.94    +11:  57.94
  + 5:  68.00    +12:  68.00
  + 6:  76.59    +13:  76.59
  + 7: 100.60    +14: 100.60
```

## ESoC 2026 milestones (if accepted)

| Phase | Milestone |
|-------|-----------|
| Weeks 1–2  | Upstream this repo as `sktime/sktime-mcp` PRs: tool definitions, `FastMCP` server, CI. Open `sktime/sktime` issue #9721 follow-up RFC. |
| Weeks 3–4  | Land `AgenticForecaster` in a feature branch of `sktime` behind a `genai` extra. Cover univariate forecasting only. |
| Weeks 5–6  | Add support for exogenous variables (X), probabilistic forecasts (`predict_interval`, `predict_quantiles`), and pipeline composition. |
| Weeks 7–8  | Add an `AgenticPipeline` that lets the agent compose `Imputer` → `Detrender` → `Forecaster`. Tutorial notebook. |
| Weeks 9–10 | Eval harness — agentic forecaster vs. `AutoARIMA` / `AutoETS` on `M3`, `M4`-yearly, `tourism`. Publish as a benchmark notebook. |
| Weeks 11–12| Hardening, docs, second-pass review. |

## Files

```
sktime_agentic_forecaster/
├── README.md                          # this file
├── PROPOSAL.md                        # application-ready proposal
├── forecasters.yaml                   # YAML-configurable forecaster registry (8 built-in)
├── pyproject.toml                     # package metadata
├── src/sktime_agentic/
│   ├── __init__.py
│   ├── forecaster.py                  # AgenticForecaster (BaseForecaster-compatible)
│   ├── react_loop.py                  # the agent loop
│   ├── tools.py                       # list_forecasters / fit_candidate / score / commit
│   ├── llm_client.py                  # Anthropic (prompt-cached) + Mock backends
│   ├── prompts.py                     # system prompt
│   └── mcp_server.py                  # FastMCP server + MCPClientRegistry
├── examples/
│   ├── 01_basic_usage.py
│   ├── 02_natural_language_query.py
│   └── 03_mcp_mode.py                 # async MCP demo via stdio
├── tests/
│   ├── test_forecaster_smoke.py       # 7 tests incl. explain()
│   ├── test_react_loop.py
│   ├── test_tools.py                  # 6 tests incl. seasonality
│   ├── test_mcp_transport.py          # 4 tests (gated on mcp SDK)
│   └── test_yaml_registry.py          # 4 tests
└── docs/
    └── design.md                      # architecture, transport modes, prompt caching
```

## License

BSD 3-Clause, matching `sktime` and `sktime-mcp`.
