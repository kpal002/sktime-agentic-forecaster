#### Reference Issues/PRs

Ref #`<YOUR_ISSUE_NUMBER>` — Agentic forecaster workflow: missing tools for iterative candidate evaluation.

#### What does this implement/fix? Explain your changes.

Adds `describe_data_tool(dataset, target_col?)` — a series fingerprinting tool that returns summary statistics for a named sktime dataset:

- `length`, `n_missing`, `min`, `max`, `mean`, `std`
- `trend_slope_per_step` — OLS slope of the series vs time index
- `candidate_seasonal_period` — detected via ACF of the **first-differenced** series
- `frequency` — inferred from the pandas DatetimeIndex where available

**Why first-differencing matters:** raw ACF on a trending series (e.g. airline passengers) is dominated by trend autocorrelation at short lags and returns spurious periods like `sp=2`. First-differencing removes the trend so seasonal peaks surface correctly (`sp=12` for airline data).

**Motivation:** an LLM agent running a model-selection loop needs to inspect data characteristics before deciding which estimators to try. Without this tool, the agent must call `evaluate_estimator` or `fit_predict` blind. `describe_data` provides the "look at the data first" step that makes iterative agentic selection meaningful. See the linked issue for the full workflow gap analysis.

**Changes:**
- `src/sktime_mcp/tools/describe_data.py` — new tool (pure numpy, no new deps)
- `src/sktime_mcp/tools/__init__.py` — export added
- `src/sktime_mcp/server.py` — `Tool()` schema + dispatcher case added

#### Does your contribution introduce a new dependency? If yes, which one?

No. All helpers use only `numpy` and `pandas`, both already core dependencies.

#### What should a reviewer concentrate their feedback on?

- **Dataset loader pattern** — I used named dataset strings (`"airline"`, `"sunspots"`, `"lynx"`) to match the convention in `evaluate_estimator`. If the preferred pattern is data handles from `load_data_source`, happy to change the interface.
- **Seasonality threshold** — ACF threshold is `0.2` on the differenced series. Happy to tune or expose as a parameter.
- **Scope** — read-only fingerprinting tool, no side effects, no handles created.

#### Any other comments?

This is a draft PR accompanying a proposal for the ESoC 2026 sktime agentic track. The tool is part of a larger agentic forecaster prototype (sktime/sktime#9721) where the same fingerprinting logic has been validated against real sktime datasets including airline passengers.

```python
from sktime_mcp.tools.describe_data import describe_data_tool
result = describe_data_tool("airline")
assert result["success"] is True
assert result["candidate_seasonal_period"] == 12
assert result["frequency"] == "ME"
```

#### PR checklist

##### For all contributions
- [ ] I've added myself to the [list of contributors](https://github.com/alan-turing-institute/sktime/blob/main/.all-contributorsrc).
- [ ] Optionally, I've updated sktime's [CODEOWNERS](https://github.com/alan-turing-institute/sktime/blob/main/CODEOWNERS) to receive notifications about future changes to these files.
- [x] I've added unit tests and made sure they pass locally.
