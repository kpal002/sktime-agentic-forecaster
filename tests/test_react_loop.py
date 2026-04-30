"""Tests for the ReAct loop with the MockLLMClient."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sktime_agentic.llm_client import MockLLMClient
from sktime_agentic.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from sktime_agentic.react_loop import ReActLoop
from sktime_agentic.tools import ToolRegistry, summarize_data


def test_react_loop_commits():
    n = 60
    t = np.arange(n)
    y = pd.Series(100 + 0.5 * t + 5 * np.sin(2 * math.pi * t / 12))

    reg = ToolRegistry()
    reg.bind_data(y=y, fh=list(range(1, 13)), holdout=12)

    fp = summarize_data(y)
    user_prompt = USER_TEMPLATE.format(
        user_prompt="seasonal series",
        fingerprint=str(fp),
        fh="1..12",
    )
    loop = ReActLoop(
        client=MockLLMClient(),
        registry=reg,
        system_prompt=SYSTEM_PROMPT.format(max_steps=10),
        user_prompt=user_prompt,
        max_steps=12,
    )
    result = loop.run()

    assert result.selected is not None
    assert reg._committed is not None
    assert reg._committed["name"] == result.selected
    # Mock client emits one tool per step.
    assert result.n_steps >= 5
