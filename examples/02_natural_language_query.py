"""Natural-language query demo — works with any backend.

Run with Gemini (free key from https://aistudio.google.com/apikey)::

    export GEMINI_API_KEY=AIza...
    python examples/02_natural_language_query.py --backend gemini

Run with OpenAI::

    export OPENAI_API_KEY=sk-...
    python examples/02_natural_language_query.py --backend openai

Run with Anthropic Claude::

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/02_natural_language_query.py --backend anthropic

Run offline (no key needed)::

    python examples/02_natural_language_query.py --backend mock

Without --backend, auto-detects from env vars (Gemini → OpenAI → Anthropic → mock).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from sktime_agentic import AgenticForecaster


def daily_retail() -> pd.Series:
    """Synthetic daily retail sales: weekly seasonality, mild trend."""
    rng = np.random.default_rng(42)
    n = 365 * 2
    t = np.arange(n)
    weekly = 50 + 30 * np.sin(2 * np.pi * t / 7)
    yearly = 20 * np.sin(2 * np.pi * t / 365)
    trend = 0.05 * t
    noise = rng.normal(0, 8, size=n)
    return pd.Series(
        np.clip(weekly + yearly + trend + noise, 0, None),
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
        name="units",
    )


def _auto_backend() -> str:
    """Pick a backend from available env vars."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"


def main() -> None:
    parser = argparse.ArgumentParser(description="AgenticForecaster natural-language demo")
    parser.add_argument(
        "--backend",
        choices=["mock", "anthropic", "openai", "gemini"],
        default=None,
        help="LLM backend to use (default: auto-detect from env vars)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default model name for the chosen backend",
    )
    args = parser.parse_args()

    backend = args.backend or _auto_backend()
    print(f"Using backend: {backend}" + (f"  model: {args.model}" if args.model else ""))

    y = daily_retail()

    f = AgenticForecaster(
        prompt=(
            "Daily retail units sold. There is a strong weekly cycle "
            "(weekend > weekday) and a mild upward trend. I want a 14-day "
            "point forecast. Speed matters — keep it simple."
        ),
        backend=backend,
        model=args.model,
        transport="in-process",
        holdout=28,
        metric="mape",
        max_steps=10,
    )
    f.fit(y, fh=list(range(1, 15)))

    print(f"\nSelected : {f.selected_}  params={f.selected_params_}")
    print(f"\nRationale:\n{f.rationale_}")
    print("\nForecast (next 14 days):")
    pred = np.asarray(f.predict())
    for i, v in enumerate(pred, 1):
        print(f"  +{i:>2}: {v:.2f}")


if __name__ == "__main__":
    main()
