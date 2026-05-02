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

    # How many steps before the limit to inject a "commit now" warning.
    _WARN_STEPS_BEFORE_LIMIT = 3

    def run(self) -> ReActResult:
        transcript: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self.user_prompt}
        ]
        tools = self.registry.schema()

        rationale = ""
        for step_idx in range(1, self.max_steps + 1):
            # ── Step-budget warning ──────────────────────────────────────────
            # When the agent is close to the limit and hasn't committed yet,
            # prepend a hard nudge before the next LLM call.
            steps_left = self.max_steps - step_idx
            if (
                getattr(self.registry, "_committed", None) is None
                and steps_left <= self._WARN_STEPS_BEFORE_LIMIT
                and steps_left > 0
            ):
                scored = getattr(self.registry, "_scores", {})
                if scored:
                    best = min(scored, key=scored.__getitem__)
                    hint = (
                        f"WARNING: Only {steps_left} step(s) remaining. "
                        f"You MUST call `commit` now. "
                        f"Best candidate so far: '{best}' "
                        f"(score={scored[best]:.4f}). "
                        "Call commit immediately — no more fitting or scoring."
                    )
                else:
                    hint = (
                        f"WARNING: Only {steps_left} step(s) remaining. "
                        "You MUST call `commit` now with the best candidate you have. "
                        "No more fitting or inspecting — commit immediately."
                    )
                messages.append({"role": "user", "content": hint})

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
                # Model stopped without committing — inject a strong reminder
                # and give it one more chance rather than failing hard.
                if getattr(self.registry, "_committed", None) is None and step_idx < self.max_steps:
                    messages.append({
                        "role": "user",
                        "content": (
                            "IMPORTANT: You stopped without calling `commit`. "
                            "You MUST call the `commit` tool now with the best "
                            "forecaster you have evaluated so far. "
                            "Do not produce any text — call `commit` immediately."
                        ),
                    })
                    continue
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

        # ── Auto-commit fallback ─────────────────────────────────────────────
        # If the agent exhausted all steps without committing, pick a sensible
        # candidate and commit automatically rather than surfacing a hard
        # error.  Three-tier strategy:
        #   1. Best by score, if any candidate was successfully scored.
        #   2. Try fast holdout-score on every fitted candidate, pick the best.
        #   3. Just commit any fitted candidate.
        _scores = getattr(self.registry, "_scores", {})
        _fitted = getattr(self.registry, "_fitted_candidates", {})

        if getattr(self.registry, "_committed", None) is None:
            best_name: str | None = None
            best_score: float | None = None

            if _scores:
                best_name = min(_scores, key=_scores.__getitem__)
                best_score = _scores[best_name]
            elif _fitted:
                # Tier 2: try a quick holdout score on each fitted candidate.
                for name in list(_fitted.keys()):
                    try:
                        result = self.registry.call(
                            tool_name="score", name=name,
                            metric="mape", cv="holdout",
                        )
                        if isinstance(result, dict) and result.get("ok"):
                            v = result.get("value")
                            if v is not None and (best_score is None or v < best_score):
                                best_name, best_score = name, float(v)
                    except Exception:  # noqa: BLE001
                        continue
                # Tier 3: still nothing — just take any fitted candidate.
                if best_name is None:
                    best_name = next(iter(_fitted))

            if best_name is not None:
                fitted = _fitted.get(best_name)
                best_params = dict(fitted[1]) if fitted else {}
                score_str = f", score={best_score:.4f}" if best_score is not None else ""
                auto_rationale = (
                    f"Auto-committed '{best_name}'{score_str} after step budget "
                    f"exhausted. The agent completed evaluation but did not call "
                    f"`commit` in time."
                )
                try:
                    self.registry.call(
                        tool_name="commit", name=best_name,
                        params=best_params, rationale=auto_rationale,
                    )
                    rationale = auto_rationale
                except Exception:  # noqa: BLE001
                    pass

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
