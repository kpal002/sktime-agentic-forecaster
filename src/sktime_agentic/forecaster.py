"""AgenticForecaster — sktime BaseForecaster-compatible wrapper.

When `sktime` is installed, AgenticForecaster inherits from `BaseForecaster`
so it composes cleanly with the rest of sktime (pipelines, ensembles, tuners).
When `sktime` is not installed (e.g. in CI), it falls back to a duck-typed
parent so the smoke tests still run.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sktime_agentic.llm_client import (
    LLMClient,
    MockLLMClient,
    AnthropicClient,
    OpenAIClient,
    GeminiClient,
    _template_explain,
)
from sktime_agentic.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from sktime_agentic.react_loop import ReActLoop, ReActResult
from sktime_agentic.tools import ToolRegistry, summarize_data, SKTIME_AVAILABLE

# --------------------------------------------------------------------------- #
# Parent class — sktime's BaseForecaster if available, otherwise a duck-typed
# stand-in with the methods we rely on.
# --------------------------------------------------------------------------- #

if SKTIME_AVAILABLE:  # pragma: no cover - requires sktime
    from sktime.forecasting.base import BaseForecaster  # type: ignore
else:

    class BaseForecaster:  # type: ignore[no-redef]
        """Tiny stand-in BaseForecaster for offline use."""

        _tags: dict[str, Any] = {}

        def fit(self, y, X=None, fh=None):
            self._fit(y=y, X=X, fh=fh)
            self._is_fitted = True
            return self

        def predict(self, fh=None, X=None):
            if not getattr(self, "_is_fitted", False):
                raise RuntimeError("Forecaster is not fitted.")
            return self._predict(fh=fh, X=X)

        def get_params(self, deep: bool = True):  # pragma: no cover
            return {k: v for k, v in self.__dict__.items() if not k.endswith("_")}


# --------------------------------------------------------------------------- #
# AgenticForecaster
# --------------------------------------------------------------------------- #


class AgenticForecaster(BaseForecaster):
    """Forecaster whose 'fit' delegates choice-of-model to an LLM agent.

    Parameters
    ----------
    prompt : str
        English-language description of the user's goal. Becomes part of the
        agent's user prompt.
    backend : {"anthropic", "mock"}, default "mock"
        Which `LLMClient` backend to use. "mock" runs offline.
    transport : {"in-process", "mcp"}, default "in-process"
        "in-process" calls the tools directly. "mcp" routes them through a
        running `sktime_agentic.mcp_server`. (mcp transport is implemented as
        a thin wrapper — see docs/design.md.)
    max_steps : int, default 12
        Hard cap on agent tool calls per fit.
    holdout : int, default 12
        Number of trailing observations to hold out for candidate scoring.
    metric : str, default "mape"
        Metric the agent uses to compare candidates.
    llm : LLMClient, optional
        Pre-built LLM client. Overrides `backend` if given.
    tool_registry : ToolRegistry, optional
        Pre-built registry. Overrides the default if given (used by the MCP
        transport).
    model : str, default "claude-sonnet-4-6"
        Anthropic model name when `backend="anthropic"`.

    Attributes (after `fit`)
    ------------------------
    selected_ : str
        Name of the chosen forecaster.
    selected_params_ : dict
        Params the agent committed.
    rationale_ : str
        Plain-English rationale from the agent.
    transcript_ : list of dict
        Step-by-step transcript of the agent's tool calls.
    inner_forecaster_ : object
        The fitted concrete forecaster the agent picked.
    explanation_ : dict
        Set by ``explain()``. Keys: ``"summary"`` (str) and ``"steps"``
        (list of ``{"fh", "value", "sentence"}`` dicts).
    """

    _tags: dict[str, Any] = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "X-y-must-have-same-index": True,
    }

    def __init__(
        self,
        prompt: str,
        backend: str = "mock",
        transport: str = "in-process",
        max_steps: int = 12,
        holdout: int = 12,
        metric: str = "mape",
        llm: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        model: str | None = None,
        registry_config: str | None = None,
    ):
        self.prompt = prompt
        self.backend = backend
        self.transport = transport
        self.max_steps = max_steps
        self.holdout = holdout
        self.metric = metric
        self.llm = llm
        self.tool_registry = tool_registry
        self.model = model
        self.registry_config = registry_config
        # sktime's BaseForecaster.__init__ sets up state we want to keep.
        try:  # pragma: no cover - only matters under real sktime
            super().__init__()
        except TypeError:
            pass
        self._is_fitted = False

    # --------------------------- fit / predict ---------------------------

    def _fit(self, y, X=None, fh=None):
        registry = self._build_registry()
        registry.bind_data(y=y, X=X, fh=fh, holdout=self.holdout)

        client = self._build_llm()

        fingerprint = summarize_data(y, X)
        user_prompt = USER_TEMPLATE.format(
            user_prompt=self.prompt,
            fingerprint=_pretty(fingerprint),
            fh=_render_fh(fh),
        )
        system_prompt = SYSTEM_PROMPT.format(max_steps=self.max_steps)

        loop = ReActLoop(
            client=client,
            registry=registry,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_steps=self.max_steps,
        )
        result: ReActResult = loop.run()

        if registry._committed is None:
            raise RuntimeError(
                "Agent finished without committing a forecaster. "
                f"Transcript: {result.transcript}"
            )

        self.selected_ = result.selected
        self.selected_params_ = result.params
        self.rationale_ = result.rationale
        self.transcript_ = result.transcript
        self.inner_forecaster_ = registry._committed["forecaster"]
        self._registry_ = registry
        return self

    def explain(self, fh=None) -> dict[str, Any]:
        """Generate a natural-language explanation of the forecast.

        Makes a single LLM call (no tool loop) and returns a structured
        explanation: one sentence per forecast step plus an overall summary.
        The result is also stored as ``forecaster.explanation_``.

        Unlike ``rationale_`` (which explains *why a forecaster was chosen*),
        ``explanation_`` explains *why each predicted value is what it is* —
        pointing to the specific trend, seasonal, or mean-reversion driver.

        Parameters
        ----------
        fh : forecast horizon, optional
            Defaults to the horizon supplied at ``fit`` time.

        Returns
        -------
        dict
            ``{"summary": str, "steps": [{"fh": int, "value": float,
            "sentence": str}, ...]}``
        """
        if not hasattr(self, "inner_forecaster_"):
            raise RuntimeError("Call fit() before explain().")

        # Resolve fh
        if fh is None:
            fh = getattr(self._registry_, "_fh", None)
        if fh is None:
            raise ValueError("explain() requires a forecast horizon `fh`.")

        # Predictions — normalise to a plain list of floats.
        y_pred = self.predict(fh)
        pred_values: list[float] = list(np.asarray(y_pred, dtype=float))

        # Use sequential step numbers (1, 2, …) as the display horizon so the
        # explanation reads the same regardless of whether fh is relative ints,
        # a ForecastingHorizon, or absolute timestamps.
        predictions: list[tuple[int, float]] = [
            (i + 1, v) for i, v in enumerate(pred_values)
        ]

        # Data fingerprint for context.
        fingerprint: dict[str, Any] = {}
        if hasattr(self, "_registry_"):
            try:
                fingerprint = summarize_data(
                    self._registry_._y, self._registry_._X
                )
            except Exception:
                pass

        # Delegate to the client's explain method, or fall back to the
        # built-in template if the client doesn't implement it.
        client = self._build_llm()
        if hasattr(client, "explain"):
            result = client.explain(
                forecaster_name=self.selected_,
                forecaster_params=self.selected_params_,
                rationale=self.rationale_,
                fingerprint=fingerprint,
                predictions=predictions,
            )
        else:
            result = _template_explain(
                forecaster_name=self.selected_,
                forecaster_params=self.selected_params_,
                rationale=self.rationale_,
                fingerprint=fingerprint,
                predictions=predictions,
            )

        self.explanation_ = result
        return result

    def _predict(self, fh=None, X=None):  # noqa: ARG002 - X kept for parity
        if not hasattr(self, "inner_forecaster_"):
            raise RuntimeError("AgenticForecaster is not fitted.")
        if fh is None:
            # Re-use the fh provided at fit-time when possible.
            fh = getattr(self._registry_, "_fh", None)
        if fh is None:
            raise ValueError("predict requires a forecast horizon `fh`.")
        return self.inner_forecaster_.predict(fh)

    # --------------------------- helpers ---------------------------

    def _build_registry(self) -> ToolRegistry:
        if self.tool_registry is not None:
            return self.tool_registry
        if self.transport == "in-process":
            if self.registry_config is not None:
                from sktime_agentic.tools import load_registry_from_yaml
                override = load_registry_from_yaml(self.registry_config)
                return ToolRegistry(_registry_override=override)
            return ToolRegistry()
        if self.transport == "mcp":
            try:
                from sktime_agentic.mcp_server import MCPClientRegistry  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "transport='mcp' requires `pip install sktime-agentic-forecaster[mcp]`."
                ) from e
            return MCPClientRegistry()
        raise ValueError(f"unknown transport: {self.transport}")

    # Default model names per backend
    _DEFAULT_MODELS: dict[str, str] = {
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-4o",
        "gemini": "gemini-2.5-flash",
    }

    def _build_llm(self) -> LLMClient:
        if self.llm is not None:
            return self.llm
        if self.backend == "mock":
            return MockLLMClient(metric=self.metric)
        model = self.model or self._DEFAULT_MODELS.get(self.backend)
        if self.backend == "anthropic":
            return AnthropicClient(model=model)
        if self.backend == "openai":
            return OpenAIClient(model=model)
        if self.backend == "gemini":
            return GeminiClient(model=model)
        raise ValueError(
            f"unknown backend: {self.backend!r}. "
            f"Choose from: 'mock', 'anthropic', 'openai', 'gemini'."
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _render_fh(fh) -> str:
    if fh is None:
        return "<not specified at fit time>"
    try:
        return str(list(fh))
    except Exception:
        return repr(fh)


def _pretty(d: dict) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, float):
            parts.append(f"  {k}: {v:.4g}")
        else:
            parts.append(f"  {k}: {v}")
    return "\n".join(parts)
