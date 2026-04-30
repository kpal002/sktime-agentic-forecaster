"""sktime-agentic-forecaster: an LLM-driven sktime forecaster.

Public API:
    AgenticForecaster   — sktime BaseForecaster-compatible class.
    MockLLMClient       — deterministic offline backend for tests.
    AnthropicClient     — Claude-backed backend (requires `anthropic`).
    ToolRegistry        — in-process registry of agent tools.
"""

from sktime_agentic.forecaster import AgenticForecaster
from sktime_agentic.llm_client import AnthropicClient, GeminiClient, MockLLMClient, OpenAIClient
from sktime_agentic.tools import ToolRegistry, load_registry_from_yaml

__all__ = [
    "AgenticForecaster",
    "AnthropicClient",
    "GeminiClient",
    "MockLLMClient",
    "OpenAIClient",
    "ToolRegistry",
    "load_registry_from_yaml",
]

__version__ = "0.0.1"
