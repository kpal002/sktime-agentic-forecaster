"""Basic AgenticForecaster demo (works offline with the MockLLMClient).

Run::

    python examples/01_basic_usage.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sktime_agentic import AgenticForecaster


def make_synthetic_airline(n: int = 144) -> pd.Series:
    """Synthesize a monthly series with linear trend + yearly seasonality.

    Mimics the famous airline-passengers dataset shape so the demo runs
    without sktime installed.
    """
    rng = np.random.default_rng(0)
    t = np.arange(n)
    trend = 100 + 0.8 * t
    seasonal = 25 * np.sin(2 * math.pi * t / 12) + 10 * np.cos(2 * math.pi * t / 12)
    noise = rng.normal(0, 5, size=n)
    values = trend + seasonal + noise
    idx = pd.date_range("2010-01", periods=n, freq="ME")
    return pd.Series(values, index=idx, name="passengers")


def main() -> None:
    y = make_synthetic_airline()
    print(f"Series: length={len(y)}, last 3 = {y.tail(3).tolist()}")

    f = AgenticForecaster(
        prompt=(
            "Monthly airline passenger counts with a clear yearly seasonality "
            "and a growing trend. Pick a simple forecaster that captures both."
        ),
        backend="mock",          # change to 'anthropic' with an API key
        transport="in-process",
        holdout=12,
        metric="mape",
    )
    f.fit(y, fh=list(range(1, 13)))

    print()
    print(f"Selected forecaster : {f.selected_}")
    print(f"Selected params     : {f.selected_params_}")
    print(f"Steps taken         : {len(f.transcript_)}")
    print()
    print("Rationale:")
    print(f.rationale_)
    print()

    y_pred = f.predict()
    print(f"First 5 predictions : {np.asarray(y_pred)[:5].tolist()}")


if __name__ == "__main__":
    main()
