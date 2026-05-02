"""LLM client backends for the agentic forecaster.

All clients implement the same two-method interface:

    step(system, messages, tools) → list[action]
        Drives the ReAct tool-use loop.  Returns tool_use or stop actions.

    explain(forecaster_name, forecaster_params, rationale, fingerprint,
            predictions) → dict
        One-shot call that produces a structured natural-language explanation.
        Returns::

            {
                "summary": "<one sentence>",
                "steps": [{"fh": 1, "value": 412.3, "sentence": "<why>"}, ...]
            }

Action shapes returned by ``step``::

    { "type": "tool_use", "id": str, "name": str, "input": dict }
    { "type": "stop",     "rationale": str }

Supported backends
------------------
* ``MockLLMClient``     — deterministic offline policy (no API key needed)
* ``AnthropicClient``   — Claude via ``anthropic`` SDK (prompt-cached)
* ``OpenAIClient``      — OpenAI GPT models via ``openai`` SDK
* ``GeminiClient``      — Google Gemini via native google-generativeai SDK (gemini-2.5-flash+)

All four classes have the same constructor shape so you can swap them freely::

    f = AgenticForecaster(backend="openai")   # GPT-4o
    f = AgenticForecaster(backend="gemini")   # Gemini 2.0 Flash
    f = AgenticForecaster(backend="anthropic")# Claude Sonnet
    f = AgenticForecaster(backend="mock")     # offline / CI

Message format
--------------
The ReAct loop works in **Anthropic-canonical** format (content-block lists).
OpenAI/Gemini clients translate in/out transparently so the loop never changes.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMClient(Protocol):
    def step(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return the next batch of agent actions (tool_use or stop)."""
        ...


# --------------------------------------------------------------------------- #
# Shared explain helper  (used by Mock and as fallback for all live clients)
# --------------------------------------------------------------------------- #


def _template_explain(
    forecaster_name: str,
    forecaster_params: dict[str, Any],
    rationale: str,
    fingerprint: dict[str, Any],
    predictions: list[tuple[int, float]],
) -> dict[str, Any]:
    """Rule-based explanation — no API call required."""
    sp: int | None = fingerprint.get("candidate_seasonal_period")
    slope: float = fingerprint.get("trend_slope_per_step") or 0.0
    length: int | str = fingerprint.get("length", "?")
    freq: str = fingerprint.get("frequency") or "step"

    has_trend = abs(slope) > 0.05
    trend_desc = (
        f"a {'positive' if slope > 0 else 'negative'} trend of {abs(slope):.3g}/{freq}"
        if has_trend
        else "no significant trend"
    )
    season_desc = (
        f"seasonality of period {sp} {freq}" if (sp and sp >= 2) else "no detected seasonality"
    )
    first_sentence = rationale.split(".")[0] if rationale else ""
    summary = (
        f"{forecaster_name} was chosen for a {length}-observation series "
        f"with {trend_desc} and {season_desc}. "
        + (first_sentence + "." if first_sentence else "")
    ).strip()

    steps: list[dict[str, Any]] = []
    for i, (fh_v, pred_v) in enumerate(predictions):
        if "Seasonal" in forecaster_name and sp and sp >= 2:
            pos = (i % sp) + 1
            sentence = f"Step +{fh_v}: {pred_v:.4g} — repeats position {pos}/{sp} of the seasonal cycle."
        elif "Naive" in forecaster_name:
            sentence = f"Step +{fh_v}: {pred_v:.4g} — last observed value carried forward (flat extrapolation)."
        elif "Mean" in forecaster_name:
            sentence = f"Step +{fh_v}: {pred_v:.4g} — historical mean of the training window."
        else:
            sentence = f"Step +{fh_v}: {pred_v:.4g}."
        steps.append({
            "fh": int(fh_v) if isinstance(fh_v, (int, float)) else fh_v,
            "value": round(float(pred_v), 6),
            "sentence": sentence,
        })

    return {"summary": summary, "steps": steps}


def _parse_explain_response(raw_text: str, fallback_kwargs: dict) -> dict[str, Any]:
    """Try JSON parse → regex extract → template fallback."""
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return _template_explain(**fallback_kwargs)


def _explain_system_prompt() -> str:
    return (
        "You are a time-series forecasting analyst. "
        "Explain model predictions in plain English for a non-technical audience. "
        "Be specific about *why* each value is what it is (trend, season, mean-reversion, etc.). "
        "You MUST respond with a single valid JSON object — no markdown fences, "
        "no prose outside the JSON."
    )


def _explain_user_prompt(
    forecaster_name: str,
    forecaster_params: dict,
    rationale: str,
    fingerprint: dict,
    predictions: list[tuple[int, float]],
) -> str:
    params_str = (
        ", ".join(f"{k}={v!r}" for k, v in forecaster_params.items())
        if forecaster_params
        else "defaults"
    )
    fp_lines = "\n".join(
        f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}"
        for k, v in fingerprint.items()
        if v is not None
    )
    pred_lines = "\n".join(f"  fh=+{fh_v}: {pred_v:.4g}" for fh_v, pred_v in predictions)
    step_schema = ",\n".join(
        f'    {{"fh": {fh_v}, "value": {pred_v:.4g}, "sentence": ""}}'
        for fh_v, pred_v in predictions
    )
    return (
        f"Fitted forecaster: {forecaster_name}({params_str})\n"
        f"Selection rationale: {rationale}\n\n"
        f"Series characteristics:\n{fp_lines}\n\n"
        f"Predictions:\n{pred_lines}\n\n"
        f"Fill in the sentences and return exactly this JSON:\n"
        f'{{\n  "summary": "",\n  "steps": [\n{step_schema}\n  ]\n}}'
    )


# --------------------------------------------------------------------------- #
# Anthropic → OpenAI message format translation helpers
# --------------------------------------------------------------------------- #


def _to_openai_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic-canonical messages to OpenAI chat format.

    Anthropic canonical::

        {"role": "assistant", "content": [
            {"type": "text",     "text": "..."},
            {"type": "tool_use", "id": "x", "name": "f", "input": {...}},
        ]}
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "x", "content": "..."},
        ]}

    OpenAI format::

        {"role": "assistant", "content": "...", "tool_calls": [...]}
        {"role": "tool", "tool_call_id": "x", "content": "..."}
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        # content is a list of blocks
        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            oai_msg: dict[str, Any] = {
                "role": "assistant",
                "content": " ".join(text_parts).strip() or None,
            }
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            out.append(oai_msg)

        elif role == "user":
            # May contain tool_result blocks (one per tool call) — emit as
            # separate "role: tool" messages, then a text user message if any.
            tool_results: list[dict[str, Any]] = []
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    raw = block.get("content", "")
                    if isinstance(raw, list):
                        raw = "".join(b.get("text", "") for b in raw if isinstance(b, dict))
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(raw),
                    })
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            out.extend(tool_results)
            if text_parts:
                out.append({"role": "user", "content": " ".join(text_parts).strip()})

    return out


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool schemas → OpenAI function tool format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


def _from_openai_response(response: Any) -> list[dict[str, Any]]:
    """Convert an OpenAI ChatCompletion into Anthropic-style actions."""
    actions: list[dict[str, Any]] = []
    choice = response.choices[0]
    message = choice.message

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                inp = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                inp = {}
            actions.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": inp,
            })
    else:
        actions.append({
            "type": "stop",
            "rationale": (message.content or "").strip(),
        })
    return actions


# --------------------------------------------------------------------------- #
# Anthropic backend
# --------------------------------------------------------------------------- #


@dataclass
class AnthropicClient:
    """Claude tool-use backend with prompt caching.

    Requires ``pip install anthropic`` and a valid ``ANTHROPIC_API_KEY``.
    """

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    api_key: str | None = None
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AnthropicClient requires the `anthropic` package. "
                "Install with: pip install sktime-agentic-forecaster[anthropic]"
            ) from exc
        self._client = anthropic.Anthropic(
            api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def step(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:  # pragma: no cover - exercised live
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
            messages=messages,
            tools=tools,
        )
        actions: list[dict[str, Any]] = []
        rationale_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                actions.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input or {}),
                })
            elif getattr(block, "type", None) == "text":
                rationale_parts.append(block.text)
        if response.stop_reason in ("end_turn", "stop_sequence") and not actions:
            actions.append({"type": "stop", "rationale": "\n".join(rationale_parts).strip()})
        return actions

    def explain(  # pragma: no cover - exercised live
        self,
        forecaster_name: str,
        forecaster_params: dict[str, Any],
        rationale: str,
        fingerprint: dict[str, Any],
        predictions: list[tuple[int, float]],
    ) -> dict[str, Any]:
        user = _explain_user_prompt(
            forecaster_name, forecaster_params, rationale, fingerprint, predictions
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[
                {"type": "text", "text": _explain_system_prompt(),
                 "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return _parse_explain_response(raw, dict(
            forecaster_name=forecaster_name, forecaster_params=forecaster_params,
            rationale=rationale, fingerprint=fingerprint, predictions=predictions,
        ))


# --------------------------------------------------------------------------- #
# OpenAI backend
# --------------------------------------------------------------------------- #


@dataclass
class OpenAIClient:
    """OpenAI GPT tool-use backend.

    Requires ``pip install openai`` and a valid ``OPENAI_API_KEY``.

    Parameters
    ----------
    model : str
        Any chat-completion model that supports function/tool calling.
        Default: ``"gpt-4o"``.
    base_url : str | None
        Override the API endpoint.  Used internally by ``GeminiClient`` to
        point at Google's OpenAI-compatible endpoint.
    api_key : str | None
        Override the key.  Falls back to ``OPENAI_API_KEY`` env var (or
        ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` when used via GeminiClient).
    """

    model: str = "gpt-4o"
    max_tokens: int = 2048
    api_key: str | None = None
    base_url: str | None = None
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenAIClient requires the `openai` package. "
                "Install with: pip install sktime-agentic-forecaster[openai]"
            ) from exc
        key = self.api_key or os.environ.get("OPENAI_API_KEY") or "MISSING"
        self._client = OpenAI(api_key=key, base_url=self.base_url)

    def step(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:  # pragma: no cover - exercised live
        oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
        oai_tools = _to_openai_tools(tools)
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=oai_messages,
            tools=oai_tools,
            tool_choice="auto",
        )
        return _from_openai_response(response)

    def explain(  # pragma: no cover - exercised live
        self,
        forecaster_name: str,
        forecaster_params: dict[str, Any],
        rationale: str,
        fingerprint: dict[str, Any],
        predictions: list[tuple[int, float]],
    ) -> dict[str, Any]:
        user = _explain_user_prompt(
            forecaster_name, forecaster_params, rationale, fingerprint, predictions
        )
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": _explain_system_prompt()},
                {"role": "user", "content": user},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_explain_response(raw, dict(
            forecaster_name=forecaster_name, forecaster_params=forecaster_params,
            rationale=rationale, fingerprint=fingerprint, predictions=predictions,
        ))


# --------------------------------------------------------------------------- #
# Gemini backend  (native google-generativeai SDK)
# --------------------------------------------------------------------------- #


_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength", "pattern",
    "minItems", "maxItems", "uniqueItems",
    "minProperties", "maxProperties",
    "default", "examples", "format",
    "$schema", "$id", "$ref", "$defs",
})


def _strip_gemini_schema(schema: Any) -> Any:
    """Recursively remove JSON Schema keywords that Gemini's FunctionDeclaration rejects."""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in _GEMINI_UNSUPPORTED_SCHEMA_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _strip_gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _strip_gemini_schema(v)
        elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
            out[k] = [_strip_gemini_schema(s) for s in v]
        else:
            out[k] = v
    return out


def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[Any]:
    """Convert Anthropic tool schemas → Gemini FunctionDeclaration list.

    Gemini's FunctionDeclaration only accepts a restricted OpenAPI 3.0 subset.
    Unsupported JSON Schema keywords (minimum, maximum, default, format, …)
    are stripped before the schema is handed to the SDK.
    """
    import google.generativeai.types as genai_types  # type: ignore
    decls = []
    for t in tools:
        schema = t.get("input_schema", {"type": "object", "properties": {}})
        clean_schema = _strip_gemini_schema(schema)
        decls.append(genai_types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=clean_schema,
        ))
    return [genai_types.Tool(function_declarations=decls)]


def _to_gemini_contents(
    system: str,
    messages: list[dict[str, Any]],
) -> list[Any]:
    """Convert Anthropic-canonical messages → Gemini Content list.

    Gemini doesn't have a system role — prepend it as a user turn.
    Tool results need the function name, which we look up from the
    preceding assistant turn.
    """
    import google.generativeai as genai  # type: ignore

    contents = []
    # System prompt as first user turn
    if system:
        contents.append({"role": "user", "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})

    # name lookup: tool_use_id → function name
    id_to_name: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            contents.append({"role": "user" if role == "user" else "model",
                             "parts": [{"text": content}]})
            continue

        if role == "assistant":
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "tool_use":
                    id_to_name[block["id"]] = block["name"]
                    parts.append({"function_call": {
                        "name": block["name"],
                        "args": block.get("input", {}),
                    }})
            if parts:
                contents.append({"role": "model", "parts": parts})

        elif role == "user":
            fn_parts = []
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    raw = block.get("content", "")
                    if isinstance(raw, list):
                        raw = "".join(b.get("text", "") for b in raw if isinstance(b, dict))
                    fn_name = id_to_name.get(block.get("tool_use_id", ""), "unknown")
                    fn_parts.append({"function_response": {
                        "name": fn_name,
                        "response": {"result": str(raw)},
                    }})
                elif block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])
            if fn_parts:
                contents.append({"role": "user", "parts": fn_parts})
            if text_parts:
                contents.append({"role": "user",
                                 "parts": [{"text": " ".join(text_parts)}]})

    return contents


def _proto_to_python(obj: Any) -> Any:
    """Recursively convert proto MapComposite / RepeatedComposite to plain Python dicts/lists.

    Gemini returns tool arguments as proto map objects.  A shallow ``dict()``
    call only converts the top level — nested values stay as MapComposite and
    break downstream code.  This walks the whole tree.
    """
    # proto map (behaves like a Mapping)
    if hasattr(obj, "items"):
        return {k: _proto_to_python(v) for k, v in obj.items()}
    # proto repeated / list-like
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            return [_proto_to_python(v) for v in obj]
        except TypeError:
            pass
    return obj


def _from_gemini_response(response: Any) -> list[dict[str, Any]]:
    """Convert a Gemini GenerateContentResponse → Anthropic-style actions."""
    actions: list[dict[str, Any]] = []
    try:
        parts = response.candidates[0].content.parts
    except (IndexError, AttributeError):
        return [{"type": "stop", "rationale": ""}]

    text_parts = []
    for part in parts:
        if hasattr(part, "function_call") and part.function_call.name:
            fc = part.function_call
            actions.append({
                "type": "tool_use",
                "id": f"gemini-{fc.name}-{id(fc)}",
                "name": fc.name,
                "input": _proto_to_python(fc.args) if fc.args else {},
            })
        elif hasattr(part, "text") and part.text:
            text_parts.append(part.text)

    if not actions:
        actions.append({"type": "stop", "rationale": " ".join(text_parts).strip()})
    return actions


@dataclass
class GeminiClient:
    """Google Gemini tool-use backend via the native google-generativeai SDK.

    Supports all Gemini models including ``gemini-2.5-flash``.

    Requires ``pip install google-generativeai`` and a valid ``GEMINI_API_KEY``
    (free key from https://aistudio.google.com/apikey).

    Parameters
    ----------
    model : str
        Default: ``"gemini-2.5-flash"``.
    api_key : str | None
        Falls back to ``GEMINI_API_KEY`` then ``GOOGLE_API_KEY`` env vars.
    """

    model: str = "gemini-2.5-flash"
    max_tokens: int = 2048
    api_key: str | None = None
    _genai: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "GeminiClient requires the `google-generativeai` package. "
                "Install with: pip install google-generativeai"
            ) from exc
        key = (
            self.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not key:
            raise ValueError(
                "GeminiClient requires GEMINI_API_KEY (or GOOGLE_API_KEY). "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=key)
        self._genai = genai

    def step(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:  # pragma: no cover - exercised live
        model = self._genai.GenerativeModel(
            model_name=self.model,
            tools=_to_gemini_tools(tools),
        )
        contents = _to_gemini_contents(system, messages)
        response = model.generate_content(
            contents,
            generation_config={"max_output_tokens": self.max_tokens},
        )
        return _from_gemini_response(response)

    def explain(  # pragma: no cover - exercised live
        self,
        forecaster_name: str,
        forecaster_params: dict[str, Any],
        rationale: str,
        fingerprint: dict[str, Any],
        predictions: list[tuple[int, float]],
    ) -> dict[str, Any]:
        user = _explain_user_prompt(
            forecaster_name, forecaster_params, rationale, fingerprint, predictions
        )
        model = self._genai.GenerativeModel(model_name=self.model)
        response = model.generate_content(
            f"{_explain_system_prompt()}\n\n{user}",
            generation_config={"max_output_tokens": 1024},
        )
        raw = response.text or ""
        return _parse_explain_response(raw, dict(
            forecaster_name=forecaster_name, forecaster_params=forecaster_params,
            rationale=rationale, fingerprint=fingerprint, predictions=predictions,
        ))


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #


@dataclass
class MockLLMClient:
    """Deterministic offline policy — no API key needed.

    Implements a pragmatic "pick by seasonality" policy:
      1. summarize_data
      2. list_forecasters
      3. fit_candidate (NaiveForecaster + MeanForecaster + SeasonalNaiveForecaster if sp≥2)
      4. score each
      5. commit the winner

    ``explain`` uses the template helper so tests cover the full
    fit → predict → explain path without any network call.
    """

    metric: str = "mape"
    _step: int = 0
    _seasonal_period: int | None = None
    _candidates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    _scored: dict[str, float] = field(default_factory=dict)
    _summary: dict[str, Any] = field(default_factory=dict)

    def step(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for m in messages:
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                payload = block.get("content")
                if isinstance(payload, list):
                    payload = "".join(
                        b.get("text", "") for b in payload if isinstance(b, dict)
                    )
                try:
                    parsed = json.loads(payload) if isinstance(payload, str) else payload
                except json.JSONDecodeError:
                    parsed = None
                if parsed is None:
                    continue
                if isinstance(parsed, dict) and "candidate_seasonal_period" in parsed:
                    self._summary = parsed
                    self._seasonal_period = parsed.get("candidate_seasonal_period")
                if (
                    isinstance(parsed, dict)
                    and parsed.get("ok")
                    and "metric" in parsed
                    and "value" in parsed
                ):
                    self._scored[parsed["name"]] = parsed["value"]

        self._step += 1

        if self._step == 1:
            return [self._tool("summarize_data")]
        if self._step == 2:
            return [self._tool("list_forecasters")]
        if self._step == 3:
            self._candidates = [("NaiveForecaster", {}), ("MeanForecaster", {})]
            sp = self._seasonal_period
            if sp and sp >= 2:
                self._candidates.append(("SeasonalNaiveForecaster", {"sp": int(sp)}))
            return [self._tool("fit_candidate", name=self._candidates[0][0], params=self._candidates[0][1])]

        offset = self._step - 3
        if offset < len(self._candidates):
            name, params = self._candidates[offset]
            return [self._tool("fit_candidate", name=name, params=params)]

        score_idx = self._step - 3 - len(self._candidates)
        if 0 <= score_idx < len(self._candidates):
            name, _ = self._candidates[score_idx]
            return [self._tool("score", name=name, metric=self.metric)]

        if self._scored:
            winner = min(self._scored.items(), key=lambda t: t[1])
            name = winner[0]
            params = dict(next(p for n, p in self._candidates if n == name))
            sp = self._seasonal_period
            rationale = (
                f"Picked {name} with params {params}. "
                f"Series length={self._summary.get('length')}, "
                f"candidate seasonal period={sp}. "
                f"It scored best on {self.metric} ({winner[1]:.4f}) on the holdout."
            )
            return [self._tool("commit", name=name, params=params, rationale=rationale)]

        return [{"type": "stop", "rationale": "no candidates scored"}]

    def explain(
        self,
        forecaster_name: str,
        forecaster_params: dict[str, Any],
        rationale: str,
        fingerprint: dict[str, Any],
        predictions: list[tuple[int, float]],
    ) -> dict[str, Any]:
        return _template_explain(
            forecaster_name, forecaster_params, rationale, fingerprint, predictions
        )

    @staticmethod
    def _tool(tool_name: str, **inputs) -> dict[str, Any]:
        return {
            "type": "tool_use",
            "id": f"mock-{tool_name}-{id(inputs)}",
            "name": tool_name,
            "input": inputs,
        }
