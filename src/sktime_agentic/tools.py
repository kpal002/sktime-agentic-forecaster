"""Tool definitions for the agentic forecaster.

These are the *only* operations the LLM is allowed to take. They wrap a small
slice of the sktime registry so the agent's plan is reproducible and reviewable.

The same registry is exposed two ways:
  * In-process — `ToolRegistry().call("name", **kwargs)` (used by the local agent).
  * Over MCP   — see `sktime_agentic.mcp_server`.

If `sktime` is installed, real forecasters are used. Otherwise the registry falls
back to a tiny built-in set (`NaiveForecaster`, `MeanForecaster`) so the smoke
tests run without `sktime`.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Soft sktime import. Everything below works without sktime — falls back to
# the built-in mini-forecasters defined further down.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only when sktime is installed
    from sktime.registry import all_estimators, all_tags  # type: ignore
    from sktime.performance_metrics.forecasting import (  # type: ignore
        mean_absolute_percentage_error,
    )

    SKTIME_AVAILABLE = True
except Exception:  # pragma: no cover
    SKTIME_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Built-in mini-forecasters (used when sktime is unavailable).
# These shadow a tiny subset of the sktime BaseForecaster contract.
# --------------------------------------------------------------------------- #


class _MiniForecaster:
    """Minimal stand-in for sktime BaseForecaster, used in offline smoke tests."""

    def __init__(self, **params):
        self.params = params
        self._fitted = None

    def fit(self, y, X=None, fh=None):  # noqa: ARG002 - X / fh accepted for parity
        self._fitted = np.asarray(y, dtype=float)
        return self

    def predict(self, fh):
        raise NotImplementedError


class _NaiveForecaster(_MiniForecaster):
    """Last-value forecaster."""

    def predict(self, fh):
        if self._fitted is None:
            raise RuntimeError("not fitted")
        last = float(self._fitted[-1])
        return np.full(len(list(fh)), last)


class _MeanForecaster(_MiniForecaster):
    """Historical-mean forecaster."""

    def predict(self, fh):
        if self._fitted is None:
            raise RuntimeError("not fitted")
        mean = float(np.nanmean(self._fitted))
        return np.full(len(list(fh)), mean)


class _SeasonalNaiveForecaster(_MiniForecaster):
    """Repeats the last `sp` observations."""

    def predict(self, fh):
        if self._fitted is None:
            raise RuntimeError("not fitted")
        sp = int(self.params.get("sp", 1))
        sp = max(1, min(sp, len(self._fitted)))
        season = self._fitted[-sp:]
        n = len(list(fh))
        out = np.empty(n)
        for i in range(n):
            out[i] = season[i % sp]
        return out


_BUILTIN_REGISTRY: dict[str, dict[str, Any]] = {
    "NaiveForecaster": {
        "cls": _NaiveForecaster,
        "tags": {"scitype:y": "univariate", "capability:pred_int": False},
        "params": {},
        "summary": "Repeats the last observed value.",
    },
    "MeanForecaster": {
        "cls": _MeanForecaster,
        "tags": {"scitype:y": "univariate", "capability:pred_int": False},
        "params": {},
        "summary": "Forecasts the historical mean.",
    },
    "SeasonalNaiveForecaster": {
        "cls": _SeasonalNaiveForecaster,
        "tags": {"scitype:y": "univariate", "capability:pred_int": False},
        "params": {"sp": 1},
        "summary": "Repeats the last `sp` observations (seasonal naive).",
    },
}


# --------------------------------------------------------------------------- #
# Data fingerprinting
# --------------------------------------------------------------------------- #


def _detect_seasonality(y: np.ndarray) -> int | None:
    """Seasonality detection via ACF of the first-differenced series.

    First-differencing removes linear (and near-linear) trends so that seasonal
    autocorrelation peaks are not buried under trend autocorrelation.  Without
    this step a trending series like airline passengers returns sp=2 instead of
    sp=12 because the trend dominates the raw ACF at every lag.

    Returns a candidate seasonal period or None. Intentionally simple — no heavy
    deps, just numpy.
    """
    if len(y) < 24:
        return None
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=np.nanmean(y))

    # First-difference to remove trend before computing ACF.
    d = np.diff(y)          # length n-1
    d = d - d.mean()        # centre

    n = len(d)
    max_lag = min(60, n // 2)
    if max_lag < 4:
        return None

    denom = float(np.dot(d, d))
    if denom == 0:
        return None

    acfs = []
    for lag in range(2, max_lag + 1):
        acf = float(np.dot(d[:-lag], d[lag:]) / denom)
        acfs.append((lag, acf))

    # Pick the lag with the highest positive ACF, if it clears the noise floor.
    acfs.sort(key=lambda t: t[1], reverse=True)
    best_lag, best_acf = acfs[0]
    if best_acf > 0.2:          # slightly lower threshold on differenced series
        return best_lag
    return None


def summarize_data(y, X=None) -> dict[str, Any]:
    """Compute a small fingerprint of the target series for the agent."""
    arr = np.asarray(y, dtype=float)
    finite = arr[np.isfinite(arr)]
    summary: dict[str, Any] = {
        "length": int(len(arr)),
        "n_missing": int(np.sum(~np.isfinite(arr))),
        "min": float(np.nanmin(arr)) if finite.size else None,
        "max": float(np.nanmax(arr)) if finite.size else None,
        "mean": float(np.nanmean(arr)) if finite.size else None,
        "std": float(np.nanstd(arr)) if finite.size else None,
        "trend_slope_per_step": None,
        "candidate_seasonal_period": _detect_seasonality(arr),
        "n_exog_features": 0 if X is None else int(np.asarray(X).shape[1]) if np.asarray(X).ndim > 1 else 1,
    }
    if finite.size > 2:
        # OLS slope of finite values vs index — quick & dirty trend estimate.
        idx = np.arange(arr.size, dtype=float)
        mask = np.isfinite(arr)
        try:
            slope, _ = np.polyfit(idx[mask], arr[mask], 1)
            summary["trend_slope_per_step"] = float(slope)
        except Exception:
            pass
    if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
        try:
            summary["frequency"] = pd.infer_freq(y.index)
        except Exception:
            summary["frequency"] = None
    return summary


# --------------------------------------------------------------------------- #
# YAML registry loader
# --------------------------------------------------------------------------- #


def load_registry_from_yaml(path: str | pathlib.Path) -> dict[str, dict[str, Any]]:
    """Load a forecaster registry from a YAML config file.

    Each entry must have ``name``, ``module``, ``class``, ``params``, and
    ``summary`` keys.  Entries whose class fails to import are silently skipped
    so optional dependencies (pmdarima, statsmodels) don't break offline use.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file (e.g. ``"forecasters.yaml"``).

    Returns
    -------
    dict[str, dict]
        Registry dict in the same format as ``_BUILTIN_REGISTRY``.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    FileNotFoundError
        If the path does not exist.

    Examples
    --------
    >>> reg = load_registry_from_yaml("forecasters.yaml")
    >>> f = AgenticForecaster(prompt="...", registry_config="forecasters.yaml")
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "load_registry_from_yaml requires PyYAML. "
            "Install with: pip install pyyaml"
        ) from exc

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry config not found: {path}")

    with path.open() as fh:
        raw = yaml.safe_load(fh)

    entries = raw.get("forecasters", [])
    registry: dict[str, dict[str, Any]] = {}

    for entry in entries:
        name = entry.get("name")
        module_path = entry.get("module")
        class_name = entry.get("class")
        params = entry.get("params") or {}
        summary = entry.get("summary", "")

        if not (name and module_path and class_name):
            continue  # malformed entry — skip silently

        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
        except Exception:
            continue  # optional dep not installed — skip silently

        registry[name] = {
            "cls": cls,
            "tags": {"scitype:y": "univariate"},
            "params": dict(params),
            "summary": summary,
        }

    return registry


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #


@dataclass
class ToolRegistry:
    """In-process registry of the agent's tools.

    The registry holds mutable state for the current `fit` invocation:
      * the data (`_y`, `_X`, `_fh`)
      * the per-candidate fitted forecasters
      * the agent's final committed choice
    """

    _y: Any = None
    _X: Any = None
    _fh: Any = None
    _holdout: int = 0
    _fitted_candidates: dict[str, Any] = field(default_factory=dict)
    _scores: dict[str, float] = field(default_factory=dict)
    _committed: dict[str, Any] | None = None
    # Optional override: caller may supply a custom registry of forecasters.
    _registry_override: dict[str, dict[str, Any]] | None = None

    # ------------- registry plumbing -------------

    def _registry(self) -> dict[str, dict[str, Any]]:
        if self._registry_override is not None:
            return self._registry_override
        if SKTIME_AVAILABLE:  # pragma: no cover - exercised when sktime present
            return _build_sktime_registry()
        return _BUILTIN_REGISTRY

    def bind_data(self, y, X=None, fh=None, holdout: int = 0):
        """Stash the current fit invocation's data for later tool calls."""
        self._y = y
        self._X = X
        self._fh = fh
        self._holdout = int(holdout)
        self._fitted_candidates.clear()
        self._scores.clear()
        self._committed = None

    # ------------- the tools -------------

    def summarize_data(self) -> dict[str, Any]:
        """Return a small fingerprint of the target series."""
        return summarize_data(self._y, self._X)

    def list_forecasters(self, tag_filter: dict[str, Any] | None = None) -> list[str]:
        """List forecasters in the registry, optionally filtered by tag."""
        reg = self._registry()
        names = []
        for name, meta in reg.items():
            tags = meta.get("tags", {})
            if tag_filter:
                if all(tags.get(k) == v for k, v in tag_filter.items()):
                    names.append(name)
            else:
                names.append(name)
        return sorted(names)

    def inspect_forecaster(self, name: str) -> dict[str, Any]:
        """Return tags + default params + a short summary for a forecaster."""
        reg = self._registry()
        if name not in reg:
            raise KeyError(f"unknown forecaster: {name}")
        meta = reg[name]
        return {
            "name": name,
            "tags": dict(meta.get("tags", {})),
            "default_params": dict(meta.get("params", {})),
            "summary": meta.get("summary", ""),
        }

    def fit_candidate(self, name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fit a candidate forecaster on the in-sample portion of the data."""
        reg = self._registry()
        if name not in reg:
            raise KeyError(f"unknown forecaster: {name}")
        cls = reg[name]["cls"]
        params = dict(params or {})
        try:
            f = cls(**params)
        except TypeError as e:
            return {"name": name, "ok": False, "error": f"bad params: {e}"}

        y = self._y
        if y is None:
            raise RuntimeError("call bind_data before fit_candidate")
        n = len(y)
        if self._holdout and self._holdout < n:
            y_train = y[: n - self._holdout] if not isinstance(y, pd.Series) else y.iloc[: n - self._holdout]
        else:
            y_train = y

        try:
            f.fit(y_train, fh=list(range(1, max(1, self._holdout) + 1)) if self._holdout else self._fh)
        except Exception as e:
            return {"name": name, "ok": False, "error": f"fit failed: {e}"}

        self._fitted_candidates[name] = (f, params)
        return {"name": name, "ok": True, "params": params}

    def score(self, name: str, metric: str = "mape") -> dict[str, Any]:
        """Score a previously-fitted candidate on the holdout window."""
        if name not in self._fitted_candidates:
            return {"name": name, "ok": False, "error": "fit_candidate first"}
        if not self._holdout:
            return {"name": name, "ok": False, "error": "no holdout configured"}

        f, _ = self._fitted_candidates[name]
        y = self._y
        n = len(y)
        h = self._holdout
        y_holdout = y.iloc[n - h :] if isinstance(y, pd.Series) else y[n - h :]
        try:
            y_pred = f.predict(list(range(1, h + 1)))
        except Exception as e:
            return {"name": name, "ok": False, "error": f"predict failed: {e}"}

        y_true = np.asarray(y_holdout, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if metric == "mape":
            if SKTIME_AVAILABLE:
                value = float(mean_absolute_percentage_error(y_true, y_pred, symmetric=False))
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    err = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9)
                value = float(np.nanmean(err))
        elif metric == "mae":
            value = float(np.nanmean(np.abs(y_true - y_pred)))
        elif metric == "rmse":
            value = float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))
        else:
            return {"name": name, "ok": False, "error": f"unknown metric: {metric}"}

        if not math.isfinite(value):
            value = float("inf")
        self._scores[name] = value
        return {"name": name, "ok": True, "metric": metric, "value": value}

    def commit(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        rationale: str = "",
    ) -> dict[str, Any]:
        """Lock in the chosen forecaster. Refits on the *full* training series."""
        reg = self._registry()
        if name not in reg:
            return {"ok": False, "error": f"unknown forecaster: {name}"}
        cls = reg[name]["cls"]
        params = dict(params or {})
        try:
            f = cls(**params)
            f.fit(self._y, fh=self._fh)
        except Exception as e:
            return {"ok": False, "error": f"final fit failed: {e}"}
        self._committed = {
            "name": name,
            "params": params,
            "forecaster": f,
            "rationale": rationale,
        }
        return {"ok": True, "name": name, "params": params}

    # ------------- dispatch -------------

    def schema(self) -> list[dict[str, Any]]:
        """JSON-schema descriptions of every tool, for the LLM."""
        return TOOL_SCHEMAS

    def call(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in _DISPATCH:
            raise KeyError(f"unknown tool: {tool_name}")
        return _DISPATCH[tool_name](self, **kwargs)


# --------------------------------------------------------------------------- #
# Sktime registry adapter (only used when sktime is importable)
# --------------------------------------------------------------------------- #


def _build_sktime_registry() -> dict[str, dict[str, Any]]:  # pragma: no cover
    """Build a tiny curated subset of the real sktime forecaster registry.

    Curated rather than `all_estimators(...)` so the agent isn't drowned in
    300+ classes during the prototype phase. Once shipped this list moves
    into a config file.

    The names `MeanForecaster` and `SeasonalNaiveForecaster` are kept as
    stable aliases so the `MockLLMClient` policy (and any user code) works
    identically whether sktime is installed or not.
    """
    from sktime.forecasting.naive import NaiveForecaster as _SKNaive  # type: ignore

    registry: dict[str, dict[str, Any]] = {
        "NaiveForecaster": {
            "cls": _SKNaive,
            "tags": {"scitype:y": "univariate"},
            "params": {"strategy": "last"},
            "summary": "sktime NaiveForecaster (strategy='last').",
        },
        # Stable aliases so MockLLMClient and built-in tests work identically
        # regardless of whether sktime is installed.
        "MeanForecaster": {
            "cls": _SKNaive,
            "tags": {"scitype:y": "univariate"},
            "params": {"strategy": "mean"},
            "summary": "Forecasts the historical mean (NaiveForecaster strategy='mean').",
        },
        "SeasonalNaiveForecaster": {
            "cls": _SKNaive,
            "tags": {"scitype:y": "univariate"},
            "params": {"strategy": "last", "sp": 1},
            "summary": "Repeats the last sp observations (NaiveForecaster strategy='last', sp=sp).",
        },
    }
    # Best-effort additions — skip any that fail to import.
    _try_add(
        registry,
        "PolynomialTrendForecaster",
        "sktime.forecasting.trend",
        summary="Fits a polynomial of given degree.",
        params={"degree": 1},
    )
    _try_add(
        registry,
        "ExponentialSmoothing",
        "sktime.forecasting.exp_smoothing",
        summary="Holt-Winters exponential smoothing.",
        params={"trend": "add"},
    )
    _try_add(
        registry,
        "ThetaForecaster",
        "sktime.forecasting.theta",
        summary="Theta method.",
        params={},
    )
    _try_add(
        registry,
        "AutoARIMA",
        "sktime.forecasting.arima",
        summary="Auto-tuned ARIMA via pmdarima.",
        params={"sp": 1},
    )
    _try_add(
        registry,
        "AutoETS",
        "sktime.forecasting.ets",
        summary="Auto-tuned ETS via statsmodels.",
        params={},
    )
    return registry


def _try_add(  # pragma: no cover
    registry: dict[str, dict[str, Any]],
    name: str,
    module: str,
    summary: str,
    params: dict[str, Any],
) -> None:
    try:
        mod = __import__(module, fromlist=[name])
        cls = getattr(mod, name)
    except Exception:
        return
    registry[name] = {
        "cls": cls,
        "tags": {"scitype:y": "univariate"},
        "params": params,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Tool schema (used by Anthropic tool-use API and the MCP server)
# --------------------------------------------------------------------------- #


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "summarize_data",
        "description": "Return a fingerprint of the target series: length, min/max/mean/std, "
        "missing values, candidate seasonal period, trend slope.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_forecasters",
        "description": "List sktime forecasters available in the registry. Optionally filter by tags.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tag_filter": {
                    "type": "object",
                    "description": "Mapping of tag name → required value, e.g. "
                    '{"capability:pred_int": true}.',
                }
            },
            "required": [],
        },
    },
    {
        "name": "inspect_forecaster",
        "description": "Read tags, default parameters, and a one-line summary for a named forecaster.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "fit_candidate",
        "description": "Fit a forecaster (by name) on the in-sample portion of the data. "
        "Hyperparameters are passed in `params`.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "score",
        "description": "Score a previously-fitted candidate on the holdout window. Metric "
        "is one of mape | mae | rmse.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "metric": {"type": "string", "enum": ["mape", "mae", "rmse"]},
            },
            "required": ["name"],
        },
    },
    {
        "name": "commit",
        "description": "Lock in the chosen forecaster + params. Must be called exactly "
        "once. `rationale` is shown to the user as forecaster.rationale_.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "params": {"type": "object"},
                "rationale": {"type": "string"},
            },
            "required": ["name"],
        },
    },
]


_DISPATCH: dict[str, Callable[..., Any]] = {
    "summarize_data": ToolRegistry.summarize_data,
    "list_forecasters": ToolRegistry.list_forecasters,
    "inspect_forecaster": ToolRegistry.inspect_forecaster,
    "fit_candidate": ToolRegistry.fit_candidate,
    "score": ToolRegistry.score,
    "commit": ToolRegistry.commit,
}
