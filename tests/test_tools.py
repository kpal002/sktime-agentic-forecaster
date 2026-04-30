"""Direct tests for the ToolRegistry — no LLM involved."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sktime_agentic.tools import ToolRegistry, summarize_data


def test_summarize_data_basic():
    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5)
    summary = summarize_data(y)
    assert summary["length"] == 50
    assert summary["min"] == 1.0
    assert summary["max"] == 10.0
    assert summary["mean"] == 5.5


def test_seasonality_detection_pure():
    """A pure 12-period sinusoid (no trend) should return sp=12."""
    n = 240
    t = np.arange(n)
    y = pd.Series(np.sin(2 * math.pi * t / 12))
    summary = summarize_data(y)
    sp = summary["candidate_seasonal_period"]
    assert sp is not None
    assert sp == 12


def test_seasonality_detection_trending():
    """A trending + seasonal series (airline-like) must also return sp=12.

    Before the first-differencing fix, trend autocorrelation dominated the raw
    ACF and the detector returned sp=2 instead of sp=12.
    """
    n = 144
    t = np.arange(n)
    # Strong linear trend + yearly seasonality, mimicking airline passengers.
    y = pd.Series(100 + 2.0 * t + 40 * np.sin(2 * math.pi * t / 12))
    summary = summarize_data(y)
    sp = summary["candidate_seasonal_period"]
    assert sp == 12


def test_tool_registry_full_flow():
    reg = ToolRegistry()
    n = 60
    t = np.arange(n)
    y = pd.Series(100 + 0.5 * t + 5 * np.sin(2 * math.pi * t / 12))
    reg.bind_data(y=y, fh=list(range(1, 13)), holdout=12)

    summary = reg.summarize_data()
    assert summary["length"] == n

    forecasters = reg.list_forecasters()
    assert "NaiveForecaster" in forecasters
    assert "MeanForecaster" in forecasters

    info = reg.inspect_forecaster("NaiveForecaster")
    assert info["name"] == "NaiveForecaster"
    assert "summary" in info

    fit_res = reg.fit_candidate("NaiveForecaster")
    assert fit_res["ok"] is True

    score_res = reg.score("NaiveForecaster", metric="mape")
    assert score_res["ok"] is True
    assert score_res["metric"] == "mape"

    commit_res = reg.commit(
        "NaiveForecaster", params={}, rationale="testing"
    )
    assert commit_res["ok"] is True
    assert reg._committed is not None
    assert reg._committed["name"] == "NaiveForecaster"


def test_unknown_forecaster_errors():
    reg = ToolRegistry()
    reg.bind_data(y=pd.Series([1, 2, 3, 4, 5]), holdout=2)
    try:
        reg.fit_candidate("DoesNotExist")
    except KeyError as e:
        assert "DoesNotExist" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_score_without_fit_returns_error():
    reg = ToolRegistry()
    reg.bind_data(y=pd.Series([1, 2, 3, 4, 5]), holdout=2)
    res = reg.score("NaiveForecaster")
    assert res["ok"] is False
    assert "fit_candidate" in res["error"]
