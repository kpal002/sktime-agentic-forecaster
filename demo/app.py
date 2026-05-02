"""Gradio demo for sktime-agentic-forecaster.

Run locally:
    pip install gradio
    python demo/app.py

Deploy to Hugging Face Spaces:
    - Create a new Space (SDK: Gradio)
    - Push this file as app.py
    - Add requirements.txt with: sktime-agentic-forecaster, gradio, sktime, openai
    - Set GEMINI_API_KEY secret in Space settings
"""

from __future__ import annotations

import io
import os
import textwrap

import gradio as gr
import numpy as np
import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────

EXAMPLE_CSV = textwrap.dedent("""\
    date,passengers
    1949-01,112
    1949-02,118
    1949-03,132
    1949-04,129
    1949-05,121
    1949-06,135
    1949-07,148
    1949-08,148
    1949-09,136
    1949-10,119
    1949-11,104
    1949-12,118
    1950-01,115
    1950-02,126
    1950-03,141
    1950-04,135
    1950-05,125
    1950-06,149
    1950-07,170
    1950-08,170
    1950-09,158
    1950-10,133
    1950-11,114
    1950-12,140
    1951-01,145
    1951-02,150
    1951-03,178
    1951-04,163
    1951-05,172
    1951-06,178
    1951-07,199
    1951-08,199
    1951-09,184
    1951-10,162
    1951-11,146
    1951-12,166
    1952-01,171
    1952-02,180
    1952-03,193
    1952-04,181
    1952-05,183
    1952-06,218
    1952-07,230
    1952-08,242
    1952-09,209
    1952-10,191
    1952-11,172
    1952-12,194
    1953-01,196
    1953-02,196
    1953-03,236
    1953-04,235
    1953-05,229
    1953-06,243
    1953-07,264
    1953-08,272
    1953-09,237
    1953-10,211
    1953-11,180
    1953-12,201
""")


def _load_series(file_obj, date_col: str, target_col: str) -> pd.Series:
    if file_obj is None:
        df = pd.read_csv(io.StringIO(EXAMPLE_CSV))
    else:
        df = pd.read_csv(file_obj.name)

    # auto-detect index
    if date_col and date_col in df.columns:
        df = df.set_index(date_col)
    else:
        date_candidates = [c for c in df.columns if c.lower() in ("date","time","index","timestamp")]
        if date_candidates:
            df = df.set_index(date_candidates[0])
        else:
            df = df.set_index(df.columns[0])

    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    if target_col and target_col in df.columns:
        return df[target_col].dropna()
    numeric = df.select_dtypes("number").columns.tolist()
    if not numeric:
        raise ValueError("No numeric columns found in the CSV.")
    return df[numeric[0]].dropna()


def _auto_backend() -> str:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"


def run_forecast(
    file_obj,
    date_col: str,
    target_col: str,
    prompt: str,
    backend: str,
    fh: int,
    holdout: int,
    metric: str,
    cv_strategy: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sktime_agentic import AgenticForecaster

    # resolve backend
    if backend == "auto":
        backend = _auto_backend()

    try:
        y = _load_series(file_obj, date_col.strip(), target_col.strip())
    except Exception as e:
        return None, f"❌ Error loading data: {e}", ""

    if len(y) < fh + holdout + 4:
        return None, f"❌ Series too short ({len(y)} obs) for fh={fh} + holdout={holdout}.", ""

    try:
        f = AgenticForecaster(
            prompt=prompt,
            backend=backend,
            holdout=holdout,
            metric=metric,
            max_steps=12,
        )
        f.fit(y, fh=list(range(1, fh + 1)))
    except Exception as e:
        return None, f"❌ Fit failed: {e}", ""

    try:
        y_pred = f.predict()
    except Exception as e:
        return None, f"❌ Predict failed: {e}", ""

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0F1117")
    ax.set_facecolor("#0F1117")

    n_show = min(len(y), 48)
    y_plot = y.iloc[-n_show:]
    ax.plot(range(len(y_plot)), y_plot.values,
            color="#4F8EF7", linewidth=2, label="Historical")

    pred_arr = np.asarray(y_pred, dtype=float)
    x_pred = range(len(y_plot) - 1, len(y_plot) + len(pred_arr) - 1)
    ax.plot(list(x_pred), [y_plot.values[-1], *pred_arr[:-1]],
            color="#34D399", linewidth=2, linestyle="--")
    ax.plot(range(len(y_plot) - 1, len(y_plot) + len(pred_arr)),
            [y_plot.values[-1], *pred_arr],
            color="#34D399", linewidth=2, linestyle="--", label="Forecast")

    ax.axvline(len(y_plot) - 1, color="#6B7280", linestyle=":", linewidth=1)
    ax.set_title(f"AgenticForecaster — {f.selected_}", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    ax.legend(facecolor="#1F2937", labelcolor="white", fontsize=10)
    plt.tight_layout()

    # ── result text ──────────────────────────────────────────────────────────
    result_md = (
        f"**✓ Selected:** `{f.selected_}`  \n"
        f"**Params:** `{f.selected_params_}`  \n"
        f"**Backend:** `{backend}`  \n\n"
        f"**Rationale:**\n\n{f.rationale_}"
    )

    forecast_text = "\n".join(
        f"+{i+1:>3}:  {v:.4f}" for i, v in enumerate(pred_arr)
    )

    return fig, result_md, f"```\n{forecast_text}\n```"


# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Base(), title="sktime-agentic-forecaster") as demo:
    gr.Markdown(
        """
        # 🤖 sktime-agentic-forecaster
        **An LLM agent that reads your prompt and picks its own forecasting model.**

        Upload a CSV (or use the built-in airline example), describe your data in English,
        and the agent will select, evaluate, and commit to the best forecaster automatically.

        > Source: [kpal002/sktime-agentic-forecaster](https://github.com/kpal002/sktime-agentic-forecaster)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Data")
            file_input = gr.File(label="Upload CSV (leave empty for airline example)", file_types=[".csv"])
            date_col   = gr.Textbox(label="Date column name", placeholder="date (auto-detected if blank)")
            target_col = gr.Textbox(label="Target column name", placeholder="auto-detected if blank")

            gr.Markdown("### Agent settings")
            prompt = gr.Textbox(
                label="Describe your data (English prompt)",
                value="Monthly airline passengers. Strong yearly seasonality and upward trend.",
                lines=3,
            )
            with gr.Row():
                backend = gr.Dropdown(
                    ["auto", "mock", "gemini", "openai", "anthropic"],
                    value="auto", label="Backend",
                )
                metric = gr.Dropdown(["mape", "mae", "rmse"], value="mape", label="Metric")
            with gr.Row():
                fh      = gr.Slider(1, 36, value=12, step=1, label="Forecast horizon (steps)")
                holdout = gr.Slider(4, 36, value=12, step=1, label="Holdout window")
            cv_strategy = gr.Radio(
                ["holdout", "expanding"], value="holdout",
                label="CV strategy",
                info="'expanding' is more reliable but slower",
            )
            run_btn = gr.Button("▶ Run AgenticForecaster", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Forecast")
            result_out  = gr.Markdown(label="Agent decision")
            forecast_out = gr.Markdown(label="Forecast values")

    run_btn.click(
        fn=run_forecast,
        inputs=[file_input, date_col, target_col, prompt,
                backend, fh, holdout, metric, cv_strategy],
        outputs=[plot_out, result_out, forecast_out],
    )

    gr.Markdown(
        """
        ---
        **Backends:** `auto` detects from env vars (`GEMINI_API_KEY` → `OPENAI_API_KEY` → `ANTHROPIC_API_KEY`).
        `mock` runs offline with no API key — good for testing.
        """
    )

if __name__ == "__main__":
    demo.launch()
