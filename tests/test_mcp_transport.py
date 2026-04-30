"""Integration tests for transport='mcp'.

These tests start a real MCP server subprocess and exercise the full
AgenticForecaster(transport='mcp') path. They are skipped automatically when
the `mcp` SDK is not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

mcp = pytest.importorskip("mcp", reason="mcp SDK not installed")


def _seasonal_series(n: int = 60, sp: int = 12) -> pd.Series:
    t = np.arange(n)
    values = 100 + 0.5 * t + 10 * np.sin(2 * math.pi * t / sp)
    idx = pd.date_range("2020-01", periods=n, freq="ME")
    return pd.Series(values, index=idx, name="y")


# --------------------------------------------------------------------------- #
# MCPClientRegistry unit tests
# --------------------------------------------------------------------------- #


def test_mcp_registry_bind_and_summarize():
    """Registry starts, accepts data, and returns a valid summary."""
    from sktime_agentic.mcp_server import MCPClientRegistry

    reg = MCPClientRegistry()
    try:
        y = _seasonal_series()
        reg.bind_data(y=y, fh=list(range(1, 13)), holdout=12)

        result = reg.call("summarize_data")
        assert isinstance(result, dict)
        assert result["length"] == len(y)
        assert result["mean"] is not None
    finally:
        reg.close()


def test_mcp_registry_list_forecasters():
    """list_forecasters returns at least NaiveForecaster."""
    from sktime_agentic.mcp_server import MCPClientRegistry

    reg = MCPClientRegistry()
    try:
        y = _seasonal_series()
        reg.bind_data(y=y, fh=list(range(1, 13)), holdout=12)

        names = reg.call("list_forecasters")
        assert isinstance(names, list)
        assert "NaiveForecaster" in names
    finally:
        reg.close()


def test_mcp_registry_fit_score_commit():
    """Full exploration loop: fit → score → commit."""
    from sktime_agentic.mcp_server import MCPClientRegistry

    reg = MCPClientRegistry()
    try:
        y = _seasonal_series()
        reg.bind_data(y=y, fh=list(range(1, 13)), holdout=12)

        fit_res = reg.call("fit_candidate", name="NaiveForecaster", params={})
        assert fit_res.get("ok") is True

        score_res = reg.call("score", name="NaiveForecaster", metric="mape")
        assert score_res.get("ok") is True
        assert isinstance(score_res["value"], float)

        commit_res = reg.call(
            "commit",
            name="NaiveForecaster",
            params={},
            rationale="test commit",
        )
        assert commit_res.get("ok") is True
        # Local refit must have produced a real forecaster object.
        assert reg._committed is not None
        assert reg._committed["forecaster"] is not None
    finally:
        reg.close()


# --------------------------------------------------------------------------- #
# Full AgenticForecaster(transport='mcp') end-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.timeout(60)
def test_agentic_forecaster_mcp_fit_predict():
    """AgenticForecaster(transport='mcp') produces predictions identical in
    shape to the in-process backend."""
    from sktime_agentic import AgenticForecaster

    y = _seasonal_series()
    fh = list(range(1, 13))

    f = AgenticForecaster(
        prompt="Synthetic seasonal series.",
        backend="mock",
        transport="mcp",
        holdout=12,
    )
    f.fit(y, fh=fh)

    assert f.selected_ in {"NaiveForecaster", "MeanForecaster", "SeasonalNaiveForecaster"}
    assert isinstance(f.rationale_, str) and f.rationale_
    assert len(f.transcript_) > 0

    y_pred = f.predict()
    assert len(y_pred) == 12
    assert all(np.isfinite(np.asarray(y_pred, dtype=float)))
