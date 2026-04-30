"""Demo: sktime-agentic MCP server.

Current state
-------------
The MCP server (``sktime_agentic.mcp_server``) is fully implemented and can be
started with::

    sktime-agentic-mcp
    # or: python -m sktime_agentic.mcp_server

Once running, any MCP-aware client (Claude Desktop, Cursor, a custom agent) can
call the six forecasting tools directly:

    bind_data_from_json  — push a JSON-encoded series to the server
    summarize_data       — fingerprint the bound series
    list_forecasters     — list available forecasters
    inspect_forecaster   — read tags / params for a forecaster
    fit_candidate        — fit a candidate on the in-sample slice
    score                — score on the holdout window
    commit               — lock in the chosen forecaster

The ``transport='mcp'`` path in ``AgenticForecaster`` (i.e. routing the agent's
own tool calls through the MCP server rather than calling them in-process) is the
next milestone — see ``docs/design.md`` for the wiring plan.

Running this script
-------------------
This script demonstrates the *server* side: it starts the MCP server as a
subprocess, calls ``bind_data_from_json`` + ``summarize_data`` over the MCP
stdio transport using the ``mcp`` SDK, then stops the server.

Requires::

    pip install sktime-agentic-forecaster[mcp]
"""

from __future__ import annotations

import asyncio
import json
import math
import sys

import numpy as np


def _make_series(n: int = 60) -> list[float]:
    t = np.arange(n)
    return (100 + 0.5 * t + 10 * np.sin(2 * math.pi * t / 12)).tolist()


async def _run_mcp_demo() -> int:
    try:
        import mcp.client.stdio as mcp_stdio  # type: ignore
        from mcp import ClientSession  # type: ignore
        from mcp.client.stdio import StdioServerParameters  # type: ignore
    except ImportError:
        print(
            "This demo requires the `mcp` SDK.\n"
            "Install with: pip install sktime-agentic-forecaster[mcp]",
            file=sys.stderr,
        )
        return 1

    y = _make_series()
    payload = json.dumps({"y": y, "fh": list(range(1, 13))})

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "sktime_agentic.mcp_server"],
    )

    async with mcp_stdio.stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Bind the series to the server-side registry.
            bind_result = await session.call_tool(
                "bind_data_from_json",
                {"payload": payload, "holdout": 12},
            )
            print("bind_data_from_json:", bind_result.content[0].text)

            # Ask the server to fingerprint it.
            summary_result = await session.call_tool("summarize_data", {})
            print("summarize_data:", summary_result.content[0].text)

            # List available forecasters.
            list_result = await session.call_tool("list_forecasters", {})
            print("list_forecasters:", list_result.content[0].text)

    print("\nMCP server demo complete.")
    return 0


def main() -> int:
    return asyncio.run(_run_mcp_demo())


if __name__ == "__main__":
    sys.exit(main())
