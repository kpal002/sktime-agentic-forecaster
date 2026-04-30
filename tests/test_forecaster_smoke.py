"""Smoke tests for AgenticForecaster.

These run with no extras — no sktime, no anthropic, no mcp. They use the
bundled mini-forecasters and the deterministic MockLLMClient.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sktime_agentic import AgenticForecaster


def _seasonal_series(n: int = 60, sp: int = 12) -> pd.Series:
    t = np.arange(n)
    values = 100 + 0.5 * t + 10 * np.sin(2 * math.pi * t / sp)
    idx = pd.date_range("2020-01", periods=n, freq="ME")
    return pd.Series(values, index=idx, name="y")


def test_fit_predict_round_trip():
    y = _seasonal_series()
    f = AgenticForecaster(
        prompt="Synthetic seasonal series.",
        backend="mock",
        holdout=12,
    )
    f.fit(y, fh=list(range(1, 13)))

    assert f.selected_ in {
        "NaiveForecaster",
        "MeanForecaster",
        "SeasonalNaiveForecaster",
    }
    assert isinstance(f.rationale_, str) and f.rationale_
    assert hasattr(f, "transcript_") and len(f.transcript_) > 0

    y_pred = f.predict()
    assert len(y_pred) == 12
    assert all(np.isfinite(y_pred))


def test_seasonal_picks_seasonal_naive():
    """Mock LLM should choose the seasonal candidate when the period is clear and
    the series has no trend (so trend-blind candidates are competitive)."""
    n = 120
    sp = 12
    t = np.arange(n)
    y = pd.Series(
        100 + 20 * np.sin(2 * math.pi * t / sp),
        index=pd.date_range("2010-01", periods=n, freq="ME"),
        name="y",
    )
    f = AgenticForecaster(
        prompt="Strongly seasonal, no trend.",
        backend="mock",
        holdout=12,
    )
    f.fit(y, fh=list(range(1, 13)))

    # On a pure seasonal series SeasonalNaive should beat Naive and Mean.
    assert f.selected_ == "SeasonalNaiveForecaster"
    assert f.selected_params_.get("sp") == sp


def test_predict_requires_fit():
    f = AgenticForecaster(prompt="x", backend="mock")
    try:
        f.predict(fh=[1, 2, 3])
    except Exception as e:
        assert "fitted" in str(e).lower()
    else:
        raise AssertionError("expected an unfitted error")


def test_explain_structure():
    """explain() returns the right schema and populates explanation_."""
    y = _seasonal_series()
    f = AgenticForecaster(prompt="seasonal series", backend="mock", holdout=12)
    f.fit(y, fh=list(range(1, 13)))

    result = f.explain()

    # Top-level keys
    assert "summary" in result
    assert "steps" in result
    assert isinstance(result["summary"], str) and result["summary"]

    # One step per forecast horizon
    assert len(result["steps"]) == 12

    # Each step has the right shape
    for step in result["steps"]:
        assert "fh" in step
        assert "value" in step
        assert "sentence" in step
        assert isinstance(step["sentence"], str) and step["sentence"]
        assert np.isfinite(step["value"])

    # Result is also stored on the forecaster
    assert hasattr(f, "explanation_")
    assert f.explanation_ is result


def test_explain_values_match_predict():
    """Step values in explain() match what predict() returns."""
    y = _seasonal_series()
    f = AgenticForecaster(prompt="x", backend="mock", holdout=12)
    f.fit(y, fh=list(range(1, 13)))

    y_pred = np.asarray(f.predict(), dtype=float)
    explanation = f.explain()

    for i, step in enumerate(explanation["steps"]):
        assert abs(step["value"] - float(y_pred[i])) < 1e-6


def test_explain_requires_fit():
    """explain() raises RuntimeError before fit."""
    f = AgenticForecaster(prompt="x", backend="mock")
    try:
        f.explain(fh=[1, 2, 3])
    except Exception as e:
        assert "fit" in str(e).lower()
    else:
        raise AssertionError("expected error before fit")


def test_transcript_has_commit():
    y = _seasonal_series()
    f = AgenticForecaster(prompt="x", backend="mock", holdout=12)
    f.fit(y, fh=list(range(1, 13)))

    flat = []
    for entry in f.transcript_:
        for action in entry.get("actions", []):
            if action.get("type") == "tool_use":
                flat.append(action.get("name"))

    assert "summarize_data" in flat
    assert "list_forecasters" in flat
    assert "fit_candidate" in flat
    assert "score" in flat
    assert "commit" in flat
