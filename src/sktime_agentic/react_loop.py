"""ReAct loop that drives an LLMClient over a ToolRegistry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sktime_agentic.llm_client import LLMClient
from sktime_agentic.tools import ToolRegistry


@dataclass
class ReActResult:
    selected: str | None
    params: dict[str, Any]
    rationale: str
    transcript: list[dict[str, Any]] = field(default_factory=list)
    n_steps: int = 0


@dataclass
class ReActLoop:
    """Drives an LLMClient over a ToolRegistry to pick a forecaster.

    Stops on the first successful `commit` tool call, on `stop`, or after
    `max_steps` iterations.
    """

    client: LLMClient
    registry: ToolRegistry
    system_prompt: str
    user_prompt: str
    max_steps: int = 12

    def run(self) -> ReActResult:
        transcript: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self.user_prompt}
        ]
        tools = self.registry.schema()

        rationale = ""
        for step_idx in range(1, self.max_steps + 1):
            actions = self.client.step(self.system_prompt, messages, tools)
            transcript.append({"step": step_idx, "actions": actions})

            # If the model returned no actions, stop.
            if not actions:
                break

            # Append the assistant turn to the message list. We assemble the
            # turn from `actions`, mirroring Anthropic's content-block schema.
            assistant_blocks: list[dict[str, Any]] = []
            tool_uses: list[dict[str, Any]] = []
            for a in actions:
                if a["type"] == "stop":
                    if a.get("rationale"):
                        rationale = a["rationale"]
                    assistant_blocks.append({"type": "text", "text": a.get("rationale", "")})
                elif a["type"] == "tool_use":
                    assistant_blocks.append(
                        {
                            "type": "tool_use",
                            "id": a["id"],
                            "name": a["name"],
                            "input": a.get("input", {}),
                        }
                    )
                    tool_uses.append(a)
            if assistant_blocks:
                messages.append({"role": "assistant", "content": assistant_blocks})

            if not tool_uses:
                # No tool calls means the model is done.
                break

            # Run each tool call and append a `tool_result` block in a single
            # user turn (the Anthropic-canonical layout).
            tool_result_blocks: list[dict[str, Any]] = []
            committed_action: dict[str, Any] | None = None
            for tu in tool_uses:
                name = tu["name"]
                inputs = tu.get("input", {}) or {}
                try:
                    result = self.registry.call(tool_name=name, **inputs)
                    is_error = False
                except Exception as e:  # noqa: BLE001 — surface to LLM
                    result = {"ok": False, "error": str(e)}
                    is_error = True

                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "is_error": is_error,
                        "content": json.dumps(_jsonable(result)),
                    }
                )

                if name == "commit" and isinstance(result, dict) and result.get("ok"):
                    committed_action = tu

            if tool_result_blocks:
                messages.append({"role": "user", "content": tool_result_blocks})

            if committed_action is not None:
                # Use the rationale from the commit call itself, falling back to
                # any text the model emitted.
                if not rationale:
                    rationale = committed_action.get("input", {}).get("rationale", "")
                else:
                    extra = committed_action.get("input", {}).get("rationale", "")
                    if extra and extra not in rationale:
                        rationale = (rationale + "\n\n" + extra).strip()
                break

        committed = self.registry._committed or {}
        return ReActResult(
            selected=committed.get("name"),
            params=committed.get("params", {}),
            rationale=committed.get("rationale", rationale),
            transcript=transcript,
            n_steps=len(transcript),
        )


def _jsonable(obj: Any) -> Any:
    """Best-effort coerce arbitrary objects into JSON-friendly values."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # NumPy scalars / arrays.
    try:
        import numpy as np

        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    return repr(obj)
