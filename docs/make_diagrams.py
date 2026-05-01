"""Generate blog-style diagrams for the agentic forecaster post."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "docs/images"

# ── palette ────────────────────────────────────────────────────────────────
BG      = "#0F1117"   # dark background
C1      = "#4F8EF7"   # blue  – user / you
C2      = "#A78BFA"   # purple – AgenticForecaster
C3      = "#34D399"   # green  – LLM
C4      = "#FBBF24"   # amber  – tools / registry
C5      = "#F87171"   # red    – result / output
GREY    = "#374151"
LIGHT   = "#E5E7EB"
WHITE   = "#FFFFFF"

def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  saved {OUT}/{name}")

def box(ax, x, y, w, h, color, text, fontsize=11, radius=0.04, text_color=WHITE, bold=False):
    fc = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle=f"round,pad=0,rounding_size={radius}",
                        linewidth=0, facecolor=color, zorder=3)
    ax.add_patch(fc)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, weight=weight, zorder=4,
            wrap=True, multialignment="center")

def arrow(ax, x0, y0, x1, y1, color=LIGHT, lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)

def label(ax, x, y, text, color=LIGHT, fontsize=9, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, zorder=5, style="italic")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 — Big picture: You ↔ AgenticForecaster ↔ LLM
# ═══════════════════════════════════════════════════════════════════════════
def diagram1():
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    ax.axis("off")

    fig.text(0.5, 0.95, "Level 1 — The Big Picture",
             ha="center", color=WHITE, fontsize=15, weight="bold")

    # three actors
    box(ax, 1.5, 2.5, 2.2, 0.9, C1, "You\n(Python script)", bold=True)
    box(ax, 6.0, 2.5, 2.6, 0.9, C2, "AgenticForecaster", bold=True)
    box(ax, 10.5, 2.5, 2.2, 0.9, C3, "LLM\n(Gemini / GPT / Claude)", bold=True)

    # You → AgenticForecaster
    arrow(ax, 2.6, 2.7, 4.7, 2.7, C1)
    label(ax, 3.65, 2.95, "fit(y, fh=[1..12])", color=C1)
    arrow(ax, 4.7, 2.3, 2.6, 2.3, C5)
    label(ax, 3.65, 2.05, "predict() → y_pred", color=C5)

    # AgenticForecaster → LLM
    arrow(ax, 7.3, 2.7, 9.4, 2.7, C2)
    label(ax, 8.35, 3.05, "prompt + tools", color=C2, fontsize=8.5)
    arrow(ax, 9.4, 2.3, 7.3, 2.3, C3)
    label(ax, 8.35, 2.05, "tool_call actions", color=C3, fontsize=8.5)

    # tool loop annotation
    ax.annotate("", xy=(7.3, 1.5), xytext=(9.4, 1.5),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2,
                                connectionstyle="arc3,rad=0"))
    label(ax, 8.35, 1.25, "ReAct loop  (up to N steps)", color=GREY, fontsize=8)

    # bottom note
    fig.text(0.5, 0.03,
             "The LLM never sees raw data — only a fingerprint summary and tool results.",
             ha="center", color=GREY, fontsize=9)

    save(fig, "01_big_picture.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 — Inside fit(): 5 stages
# ═══════════════════════════════════════════════════════════════════════════
def diagram2():
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off")

    fig.text(0.5, 0.96, "Level 2 — Inside fit(y, fh)",
             ha="center", color=WHITE, fontsize=15, weight="bold")

    stages = [
        (5.0, 6.0, C1,  "fit(y, fh)",             "entry point"),
        (5.0, 5.0, C4,  "① Build Registry",       "load forecasters from sktime / YAML"),
        (5.0, 4.0, C4,  "② Bind Data",            "store y, X, fh, holdout slice"),
        (5.0, 3.0, C2,  "③ Build Prompt",         "fingerprint → system + user prompt"),
        (5.0, 2.0, C3,  "④ ReAct Loop",           "LLM ↔ tools  (the interesting part)"),
        (5.0, 1.0, C5,  "⑤ Store Results",        "selected_  rationale_  inner_forecaster_"),
    ]

    for i, (x, y, col, title, sub) in enumerate(stages):
        box(ax, x, y, 5.5, 0.65, col, title, fontsize=12, bold=True)
        label(ax, x, y - 0.27, sub, color=WHITE, fontsize=8.5)
        if i < len(stages) - 1:
            arrow(ax, x, y - 0.35, x, stages[i+1][1] + 0.35, color=LIGHT, lw=2)

    # highlight ReAct
    rect = FancyBboxPatch((2.1, 1.65), 5.9, 0.7,
                          boxstyle="round,pad=0,rounding_size=0.06",
                          linewidth=2, edgecolor=C3, facecolor="none", zorder=5)
    ax.add_patch(rect)

    save(fig, "02_fit_stages.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 — ReAct loop (step by step, airline example)
# ═══════════════════════════════════════════════════════════════════════════
def diagram3():
    fig, ax = plt.subplots(figsize=(13, 8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13); ax.set_ylim(0, 8)
    ax.axis("off")

    fig.text(0.5, 0.97, "Level 3 — The ReAct Loop (Airline Example)",
             ha="center", color=WHITE, fontsize=15, weight="bold")

    # column headers
    cols = {"Messages": 1.7, "LLM": 5.5, "Tool": 9.5, "Result": 12.2}
    for name, x in cols.items():
        col = {"Messages": C1, "LLM": C3, "Tool": C4, "Result": C5}[name]
        ax.text(x, 7.5, name, ha="center", color=col, fontsize=12, weight="bold")

    steps = [
        # (y,  msg_text,              llm_text,           tool_text,           result_text)
        (6.5, "user: Monthly\nairline data...",
               "summarize\nthe data",
               "summarize_data()",
               "sp=12\ntrend=1.8/mo"),
        (5.1, "+ assistant:\ntool_call(summarize)\n+ user: {sp:12}",
               "try Exponential\nSmoothing sp=12",
               "fit_candidate(\nExpSmoothing)",
               "ok ✓"),
        (3.7, "+ assistant:\ntool_call(fit_ES)\n+ user: ok",
               "score it",
               "score(\nExpSmoothing)",
               "MAPE\n0.048"),
        (2.3, "+ assistant:\ntool_call(score)\n+ user: 0.048",
               "try Seasonal\nNaive sp=12\nthen compare",
               "fit + score\n(SeasonalNaive)",
               "MAPE\n0.100"),
        (0.9, "all scores known",
               "ExpSmoothing\nwins → commit",
               "commit(\nExpSmoothing)",
               "✓ done"),
    ]

    for (y, msg, llm, tool, res) in steps:
        box(ax, 1.7, y, 2.8, 0.85, GREY,  msg,  fontsize=7.5)
        box(ax, 5.5, y, 2.4, 0.85, C3,    llm,  fontsize=8.5)
        box(ax, 9.5, y, 2.6, 0.85, C4,    tool, fontsize=8.5)
        box(ax, 12.2, y, 1.8, 0.85, C5,   res,  fontsize=8.5)
        # arrows
        arrow(ax, 6.7, y, 8.2, y, C3, lw=1.2)
        arrow(ax, 10.8, y, 11.3, y, C4, lw=1.2)

    # vertical message growth arrow
    ax.annotate("", xy=(1.7, 1.35), xytext=(1.7, 7.05),
                arrowprops=dict(arrowstyle="-|>", color=C1, lw=1.5))
    label(ax, 0.55, 4.2, "grows\neach step", color=C1, fontsize=8)

    save(fig, "03_react_loop.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 4 — Two transports: in-process vs MCP
# ═══════════════════════════════════════════════════════════════════════════
def diagram4():
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6)
    ax.axis("off")

    fig.text(0.5, 0.96, "Level 4 — Two Transports: in-process vs MCP",
             ha="center", color=WHITE, fontsize=15, weight="bold")

    # ── LEFT: in-process ───────────────────────────────────────────────────
    ax.text(3.25, 5.35, "transport = \"in-process\"",
            ha="center", color=C4, fontsize=11, weight="bold")

    box(ax, 3.25, 4.5,  4.0, 0.7, C2, "AgenticForecaster", bold=True)
    arrow(ax, 3.25, 4.15, 3.25, 3.55, C2, lw=2)
    label(ax, 4.2, 3.85, "direct call", color=LIGHT, fontsize=8)
    box(ax, 3.25, 3.2,  4.0, 0.7, C4, "ToolRegistry\n(Python object)", fontsize=10)
    arrow(ax, 3.25, 2.85, 3.25, 2.25, C4, lw=2)
    box(ax, 3.25, 1.9,  4.0, 0.7, C3, "sktime model\nfitted in memory", fontsize=10)

    ax.text(3.25, 1.2, "✓ Fast   ✓ No extra process",
            ha="center", color=GREY, fontsize=9)

    # divider
    ax.plot([6.5, 6.5], [0.5, 5.7], color=GREY, lw=1, linestyle="--")
    ax.text(6.5, 0.2, "OR", ha="center", color=GREY, fontsize=11, weight="bold")

    # ── RIGHT: MCP ─────────────────────────────────────────────────────────
    ax.text(9.75, 5.35, "transport = \"mcp\"",
            ha="center", color=C3, fontsize=11, weight="bold")

    box(ax, 9.75, 4.5,  4.0, 0.7, C2, "AgenticForecaster", bold=True)
    arrow(ax, 9.75, 4.15, 9.75, 3.55, C2, lw=2)
    label(ax, 10.9, 3.85, "JSON over stdio", color=LIGHT, fontsize=8)
    box(ax, 9.75, 3.2,  4.0, 0.7, C3, "MCPClientRegistry", fontsize=10)
    arrow(ax, 9.75, 2.85, 9.75, 2.25, C3, lw=2)
    box(ax, 9.75, 1.9,  4.0, 0.7, C4,
        "sktime_agentic.mcp_server\n(separate process)", fontsize=9)
    arrow(ax, 9.75, 1.55, 9.75, 0.95, C4, lw=2)
    box(ax, 9.75, 0.65,  4.0, 0.5, GREY,
        "any MCP client can use these tools", fontsize=8.5)

    ax.text(9.75, 0.1, "✓ Shareable   ✓ Works with Claude Desktop / Cursor",
            ha="center", color=GREY, fontsize=9)

    save(fig, "04_transports.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 5 — Multi-backend message translation
# ═══════════════════════════════════════════════════════════════════════════
def diagram5():
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6)
    ax.axis("off")

    fig.text(0.5, 0.96, "Level 5 — Multi-Backend Translation",
             ha="center", color=WHITE, fontsize=15, weight="bold")

    # centre: internal format
    box(ax, 6.5, 3.8, 5.5, 1.5, GREY,
        "Internal Format  (Anthropic canonical)\n\n"
        "{role: assistant, content: [{type: tool_use, name: fit_candidate, ...}]}\n"
        "{role: user,      content: [{type: tool_result, content: 'ok'}]}",
        fontsize=8, radius=0.06)

    ax.text(6.5, 5.1, "ReAct Loop always speaks this format",
            ha="center", color=LIGHT, fontsize=9, style="italic")

    # three backends below
    backends = [
        (2.0,  1.5, C2, "AnthropicClient\n(Claude)",    "no translation\nneeded"),
        (6.5,  1.5, C3, "OpenAIClient\n(GPT-4o)",       "tool_use → tool_calls\ntool_result → role:tool"),
        (11.0, 1.5, C4, "GeminiClient\n(Gemini Flash)", "uses OpenAI-compatible\nendpoint — same as OpenAI"),
    ]

    for bx, by, col, name, note in backends:
        box(ax, bx, by, 3.6, 1.3, col, name, fontsize=10, bold=True)
        label(ax, bx, by - 0.5, note, color=WHITE, fontsize=8)
        arrow(ax, 6.5, 3.05, bx, by + 0.7, LIGHT, lw=1.2)

    # mock
    box(ax, 6.5, 0.4, 3.2, 0.55, GREY, "MockLLMClient  (no API call — deterministic policy)", fontsize=8.5)
    arrow(ax, 6.5, 3.05, 6.5, 0.7, GREY, lw=1, style="->")

    save(fig, "05_backends.png")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 6 — Full end-to-end summary
# ═══════════════════════════════════════════════════════════════════════════
def diagram6():
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)
    ax.axis("off")

    fig.text(0.5, 0.975, "Full End-to-End Flow",
             ha="center", color=WHITE, fontsize=16, weight="bold")

    # top: entry
    box(ax, 7, 8.2, 5, 0.75, C1, "f.fit(y, fh)   →   f.predict()", bold=True, fontsize=12)

    # stage boxes (left-aligned pipeline)
    stages = [
        (7, 7.1, C4, "① Build Registry",  "sktime forecasters + forecasters.yaml"),
        (7, 6.1, C4, "② Bind Data",       "y, X, fh, holdout slice stored"),
        (7, 5.1, C2, "③ Build Prompt",    "fingerprint  +  your English prompt"),
    ]
    for x, y, col, title, sub in stages:
        box(ax, x, y, 9, 0.65, col, title, bold=True, fontsize=11)
        label(ax, x, y-0.25, sub, fontsize=8.5)
        arrow(ax, x, y-0.35, x, y-0.65+0.02, LIGHT, lw=1.8)

    # ReAct loop box
    loop_box = FancyBboxPatch((2.0, 2.3), 10, 2.3,
                              boxstyle="round,pad=0,rounding_size=0.1",
                              linewidth=1.5, edgecolor=C3, facecolor="#1a1f2e", zorder=2)
    ax.add_patch(loop_box)
    ax.text(7, 4.45, "④  ReAct Loop", ha="center", color=C3,
            fontsize=12, weight="bold", zorder=3)

    # loop internals
    internals = [
        (3.5, 3.5, C3,  "LLM.step(\nmessages, tools)"),
        (7.0, 3.5, C4,  "registry.call(\ntool_name)"),
        (10.5, 3.5, C2, "messages\n.append(result)"),
    ]
    for x, y, col, txt in internals:
        box(ax, x, y, 2.8, 0.85, col, txt, fontsize=9)
    arrow(ax, 4.9, 3.5, 5.6, 3.5, LIGHT, lw=1.5)
    arrow(ax, 8.4, 3.5, 9.1, 3.5, LIGHT, lw=1.5)
    # loop back arrow
    ax.annotate("", xy=(3.5, 3.08), xytext=(10.5, 3.08),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2,
                                connectionstyle="arc3,rad=-0.3"))
    label(ax, 7.0, 2.55, "repeat until commit  (max N steps)", color=GREY, fontsize=8.5)

    # arrow into loop
    arrow(ax, 7, 4.78, 7, 4.62, LIGHT, lw=1.8)

    # stage 5
    arrow(ax, 7, 2.3, 7, 1.85, LIGHT, lw=1.8)
    box(ax, 7, 1.55, 9, 0.65, C5, "⑤ Store Results", bold=True, fontsize=11)
    label(ax, 7, 1.3, "selected_   selected_params_   rationale_   inner_forecaster_", fontsize=8.5)

    # predict
    arrow(ax, 7, 1.22, 7, 0.75, LIGHT, lw=1.8)
    box(ax, 7, 0.5, 5, 0.55, C1, "predict()  →  y_pred", bold=True, fontsize=11)

    save(fig, "06_full_flow.png")


if __name__ == "__main__":
    print("Generating diagrams...")
    diagram1()
    diagram2()
    diagram3()
    diagram4()
    diagram5()
    diagram6()
    print("Done — all images saved to docs/images/")
