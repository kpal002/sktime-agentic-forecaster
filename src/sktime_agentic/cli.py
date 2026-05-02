"""Command-line interface for sktime-agentic-forecaster.

Usage
-----
    sktime-agentic fit data.csv \\
        --prompt "Weekly retail sales, strong weekend effect" \\
        --backend gemini \\
        --holdout 12 \\
        --fh 12 \\
        --output forecast.csv

    sktime-agentic fit data.csv --backend mock --fh 6

The input CSV must have:
  - a datetime index column (first column, or named 'date' / 'time' / 'index')
  - a single numeric target column (second column, or named via --target)

The output CSV contains columns: fh, forecast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_series(path: str, target: str | None):
    """Load a univariate pd.Series from a CSV file."""
    import pandas as pd

    df = pd.read_csv(path)

    # Identify index column
    date_cols = [c for c in df.columns if c.lower() in ("date", "time", "index", "timestamp")]
    if date_cols:
        df = df.set_index(date_cols[0])
        df.index = pd.to_datetime(df.index)
    elif df.dtypes.iloc[0] == object:
        df = df.set_index(df.columns[0])
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # Identify target column
    if target:
        if target not in df.columns:
            print(f"[error] column '{target}' not found. Available: {list(df.columns)}")
            sys.exit(1)
        return df[target].dropna()
    if df.shape[1] == 1:
        return df.iloc[:, 0].dropna()
    # Try to pick a numeric column
    numeric = df.select_dtypes("number").columns.tolist()
    if not numeric:
        print(f"[error] no numeric columns found in {path}")
        sys.exit(1)
    if len(numeric) > 1:
        print(f"[info] multiple numeric columns found, using '{numeric[0]}'. "
              f"Use --target to specify.")
    return df[numeric[0]].dropna()


def cmd_fit(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from sktime_agentic import AgenticForecaster

    print(f"Loading data from: {args.data}")
    y = _load_series(args.data, args.target)
    print(f"  Series length : {len(y)}")
    print(f"  Date range    : {y.index[0]} → {y.index[-1]}")
    print(f"  Backend       : {args.backend}")
    print(f"  Prompt        : {args.prompt}")
    print()

    fh = list(range(1, args.fh + 1))

    f = AgenticForecaster(
        prompt=args.prompt,
        backend=args.backend,
        model=args.model,
        holdout=args.holdout,
        metric=args.metric,
        max_steps=args.max_steps,
    )

    print("Running agentic fit...\n")
    f.fit(y, fh=fh)

    print(f"✓ Selected    : {f.selected_}  {f.selected_params_}")
    print(f"\nRationale:\n{f.rationale_}\n")

    y_pred = f.predict()
    pred_arr = np.asarray(y_pred, dtype=float)

    print("Forecast:")
    for i, v in enumerate(pred_arr, 1):
        print(f"  +{i:>3}: {v:.4f}")

    if args.output:
        out_df = pd.DataFrame({"fh": fh, "forecast": pred_arr})
        out_df.to_csv(args.output, index=False)
        print(f"\nForecast saved to: {args.output}")

    if args.explain:
        print("\nGenerating explanation...")
        exp = f.explain(fh=fh)
        print(f"\nSummary: {exp['summary']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sktime-agentic",
        description="LLM-driven time series forecaster",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── fit ──────────────────────────────────────────────────────────────────
    fit_p = sub.add_parser("fit", help="Fit an agentic forecaster on a CSV file")

    fit_p.add_argument("data", help="Path to input CSV file")
    fit_p.add_argument(
        "--prompt", "-p", required=True,
        help='English description of the forecasting task, e.g. "Weekly sales, seasonal"',
    )
    fit_p.add_argument(
        "--backend", "-b",
        choices=["mock", "anthropic", "openai", "gemini"],
        default="mock",
        help="LLM backend (default: mock)",
    )
    fit_p.add_argument("--model", default=None, help="Override default model name")
    fit_p.add_argument("--target", "-t", default=None, help="Target column name")
    fit_p.add_argument(
        "--fh", type=int, default=12,
        help="Forecast horizon (number of steps ahead, default: 12)",
    )
    fit_p.add_argument(
        "--holdout", type=int, default=12,
        help="Holdout window for candidate scoring (default: 12)",
    )
    fit_p.add_argument(
        "--metric", choices=["mape", "mae", "rmse"], default="mape",
        help="Scoring metric (default: mape)",
    )
    fit_p.add_argument(
        "--max-steps", type=int, default=12, dest="max_steps",
        help="Max agent steps (default: 12)",
    )
    fit_p.add_argument("--output", "-o", default=None, help="Save forecast to CSV")
    fit_p.add_argument(
        "--explain", action="store_true",
        help="Generate a natural-language explanation after forecasting",
    )
    fit_p.set_defaults(func=cmd_fit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
