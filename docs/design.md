# Design notes — sktime-agentic-forecaster

## Overview

`AgenticForecaster` is a sktime `BaseForecaster` subclass. Its `fit()` runs a
ReAct loop in which an LLM selects a concrete forecaster from the sktime registry
by making constrained tool calls. The agent never writes or executes free-form
Python code — every action is a call to one of the six audited tools.

```
fit(y, X, fh)
 └─ ReActLoop.run()
     ├─ step 1: summarize_data   → data fingerprint
     ├─ step 2: list_forecasters → available names
     ├─ step 3-n: inspect_forecaster / fit_candidate / score (repeated)
     └─ final: commit(name, params, rationale)
         └─ refits on full y, stores as inner_forecaster_
predict(fh)
 └─ inner_forecaster_.predict(fh)
```

## Transport modes

### in-process (default)

`ToolRegistry` lives in the same process. Tool calls are direct Python method
calls. No network, no serialization. Used for library integration and tests.

```
AgenticForecaster
  └─ ReActLoop
       └─ ToolRegistry.call("fit_candidate", name="ExponentialSmoothing", ...)
```

### mcp (planned — Week 1 milestone)

The same six tools are exposed by `sktime_agentic.mcp_server` via FastMCP over
stdio. `MCPClientRegistry` (not yet implemented) will proxy every `call()` to
the running server over the MCP protocol. This lets any MCP-aware client — Claude
Desktop, Cursor, a custom agent — drive sktime forecaster selection directly.

**Wiring plan:**

1. Start `sktime-agentic-mcp` as a subprocess (or connect to a running instance).
2. `MCPClientRegistry.__init__` opens a stdio `ClientSession` via `mcp.client.stdio`.
3. Each `ToolRegistry.call(name, **kwargs)` becomes an `await session.call_tool(name, kwargs)`.
4. Results are JSON-decoded and returned identically to the in-process path.
5. `bind_data_from_json` (already implemented in the server) is called once at
   the start of `fit()` to push the series to the server-side registry.

The MCP server exposes an additional `bind_data_from_json` tool that is not in
the core tool surface — it exists only to support this remote-bind pattern.

## Tool surface

| Tool | Description |
|---|---|
| `summarize_data` | Length, min/max/mean/std, missing values, trend slope, candidate seasonal period |
| `list_forecasters` | Names in the registry, optional tag filter |
| `inspect_forecaster` | Tags + default params + one-line summary |
| `fit_candidate` | Fit a named forecaster on the in-sample slice |
| `score` | Score a fitted candidate on the holdout window (MAPE / MAE / RMSE) |
| `commit` | Lock in the chosen forecaster; refits on the full series |

## Registry design

The registry is a plain dict keyed by name:

```python
{
    "ExponentialSmoothing": {
        "cls": sktime.forecasting.exp_smoothing.ExponentialSmoothing,
        "tags": {"scitype:y": "univariate"},
        "params": {"trend": "add"},
        "summary": "Holt-Winters exponential smoothing.",
    },
    ...
}
```

When sktime is installed, `_build_sktime_registry()` constructs this dict from
a curated subset (not `all_estimators` — too large for the prototype). `MeanForecaster`
and `SeasonalNaiveForecaster` are kept as stable aliases for `NaiveForecaster` with
appropriate params so the `MockLLMClient` policy works identically with or without sktime.

When sktime is not installed, the `_BUILTIN_REGISTRY` provides three mini-forecasters
so CI and offline smoke tests run without any extra dependencies.

## Prompt caching

`AnthropicClient` caches the system prompt across all steps in a ReAct loop via
`cache_control: {type: ephemeral}`. A single `fit()` call makes 5-12 API calls;
caching the system prompt (which is constant across all of them) saves ~60-80% of
input-token cost after the first step.

## Extending

To add a forecaster to the registry, add an entry to `_build_sktime_registry()`.
To change the agent's strategy, subclass `MockLLMClient` or implement the
`LLMClient` Protocol with a different policy. To expose new tools, add a method
to `ToolRegistry`, extend `TOOL_SCHEMAS`, and add to `_DISPATCH`.
