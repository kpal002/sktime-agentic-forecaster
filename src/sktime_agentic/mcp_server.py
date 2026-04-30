"""FastMCP server + MCPClientRegistry for the agentic forecaster.

Server
------
Exposes the same six-tool surface used by the in-process agent over MCP stdio,
so any MCP-aware client (Claude Desktop, Cursor, custom agents) can drive sktime
forecaster selection.

Wire format: stdio. Run with::

    sktime-agentic-mcp                    # via the entry point
    python -m sktime_agentic.mcp_server   # equivalent

For Claude Desktop, add to ``~/.config/Claude/claude_desktop_config.json``::

    {
      "mcpServers": {
        "sktime-agentic": {
          "command": "sktime-agentic-mcp"
        }
      }
    }

Client — MCPClientRegistry
---------------------------
``MCPClientRegistry`` is a drop-in replacement for ``ToolRegistry`` that proxies
every tool call to the MCP server over stdio.  It is used by
``AgenticForecaster(transport='mcp')``.

Architecture
~~~~~~~~~~~~
A background thread runs ``anyio.run(_async_main)``, which opens the stdio
transport, initialises the ``ClientSession``, and keeps both alive for the
duration of ``fit()``.  Synchronous callers submit ``(tool_name, inputs, future)``
tuples to a ``queue.Queue``; the async loop drains the queue via
``anyio.to_thread.run_sync``, so the event loop is never busy-polled.

The final ``commit`` result is re-executed locally against an in-process
``ToolRegistry`` so that ``inner_forecaster_`` is a real Python object that
``predict()`` can call.  All exploration (``fit_candidate``, ``score``, …) runs on
the server.

See ``docs/design.md`` for the full design rationale.
"""

from __future__ import annotations

import concurrent.futures
import json
import queue
import sys
import threading
from typing import Any

from sktime_agentic.tools import TOOL_SCHEMAS, ToolRegistry

# Module-level singleton so the server keeps the registry across calls.
_REGISTRY = ToolRegistry()


# --------------------------------------------------------------------------- #
# FastMCP server
# --------------------------------------------------------------------------- #


def _build_app():  # pragma: no cover - requires `mcp` SDK
    """Construct the FastMCP app. Imported inside ``main()``."""
    from mcp.server.fastmcp import FastMCP  # type: ignore

    app = FastMCP("sktime-agentic")

    @app.tool()
    def summarize_data() -> dict:  # noqa: D401
        """Return a fingerprint of the currently-bound target series."""
        return _REGISTRY.summarize_data()

    @app.tool()
    def list_forecasters(tag_filter: dict | None = None) -> list[str]:
        """List sktime forecasters in the registry, optionally filtered by tags."""
        return _REGISTRY.list_forecasters(tag_filter=tag_filter)

    @app.tool()
    def inspect_forecaster(name: str) -> dict:
        """Read tags, default params, and a one-line summary for a forecaster."""
        return _REGISTRY.inspect_forecaster(name)

    @app.tool()
    def fit_candidate(name: str, params: dict | None = None) -> dict:
        """Fit a candidate forecaster on the in-sample portion of the data."""
        return _REGISTRY.fit_candidate(name=name, params=params)

    @app.tool()
    def score(name: str, metric: str = "mape") -> dict:
        """Score a previously-fitted candidate on the holdout window."""
        return _REGISTRY.score(name=name, metric=metric)

    @app.tool()
    def commit(name: str, params: dict | None = None, rationale: str = "") -> dict:
        """Lock in the chosen forecaster + params."""
        return _REGISTRY.commit(name=name, params=params or {}, rationale=rationale)

    @app.tool()
    def bind_data_from_json(payload: str, holdout: int = 12) -> dict:
        """Bind a JSON-encoded series for the next forecasting workflow.

        ``payload`` must be a JSON document ``{"y": [...], "fh": [...]}``.
        The ``y`` list is converted to a ``pd.Series`` so sktime forecasters
        receive a properly typed input.
        """
        import pandas as pd

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"bad json: {exc}"}
        y_raw = data.get("y")
        fh = data.get("fh")
        if y_raw is None:
            return {"ok": False, "error": "missing 'y'"}
        # sktime forecasters require pd.Series, not a plain Python list.
        y = pd.Series(y_raw, dtype=float)
        _REGISTRY.bind_data(y=y, X=None, fh=fh, holdout=holdout)
        return {"ok": True, "n": len(y), "holdout": holdout}

    return app


def main() -> int:  # pragma: no cover
    """Entry point referenced by ``pyproject.toml``."""
    try:
        app = _build_app()
    except ImportError:
        sys.stderr.write(
            "sktime-agentic-mcp requires the `mcp` SDK. "
            "Install with: pip install sktime-agentic-forecaster[mcp]\n"
        )
        return 1
    app.run()
    return 0


# --------------------------------------------------------------------------- #
# MCPClientRegistry
# --------------------------------------------------------------------------- #


class MCPClientRegistry:
    """Proxies ToolRegistry calls to a sktime-agentic MCP server over stdio.

    Spawns the server as a subprocess on construction, binds data via
    ``bind_data_from_json``, then proxies every ``call()`` over the MCP stdio
    transport.  The final ``commit`` is also re-executed locally so that
    ``inner_forecaster_`` is a live Python object available for ``predict()``.

    Parameters
    ----------
    server_cmd : list[str], optional
        Command to start the server.  Defaults to
        ``[sys.executable, "-m", "sktime_agentic.mcp_server"]``.
    startup_timeout : float, default 30
        Seconds to wait for the server to become ready.
    call_timeout : float, default 120
        Per-call timeout in seconds (covers slow forecasters like AutoARIMA).
    """

    # ------------------------------------------------------------------ #
    # Construction / teardown
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        server_cmd: list[str] | None = None,
        startup_timeout: float = 30.0,
        call_timeout: float = 120.0,
    ) -> None:
        try:
            import anyio  # noqa: F401 – validate dep before spawning anything
            from mcp import ClientSession  # noqa: F401
            from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "transport='mcp' requires the mcp SDK. "
                "Install with: pip install sktime-agentic-forecaster[mcp]"
            ) from exc

        self._server_cmd: list[str] = server_cmd or [
            sys.executable, "-m", "sktime_agentic.mcp_server"
        ]
        self._call_timeout = call_timeout

        # Data kept locally for the final in-process refit after commit.
        self._y: Any = None
        self._X: Any = None
        self._fh: Any = None
        self._holdout: int = 0
        self._committed: dict[str, Any] | None = None

        # Sync→async bridge: main thread puts requests here;
        # the async loop drains it.
        self._queue: queue.Queue[
            tuple[str, dict[str, Any], concurrent.futures.Future[Any]] | None
        ] = queue.Queue()

        self._ready = threading.Event()
        self._start_error: BaseException | None = None

        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mcp-registry"
        )
        self._thread.start()

        if not self._ready.wait(timeout=startup_timeout):
            if self._start_error:
                raise RuntimeError(
                    f"MCP server failed to start: {self._start_error}"
                ) from self._start_error
            raise RuntimeError(
                f"MCP server did not become ready within {startup_timeout} s"
            )
        if self._start_error:
            raise RuntimeError(
                f"MCP server failed: {self._start_error}"
            ) from self._start_error

    def close(self) -> None:
        """Shut down the background session and server subprocess."""
        try:
            self._queue.put(None)       # sentinel → _async_main exits cleanly
            self._thread.join(timeout=10)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Background thread — keeps the anyio event loop + MCP session alive
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        try:
            import anyio
            anyio.run(self._async_main)
        except BaseException as exc:  # noqa: BLE001
            self._start_error = exc
            self._ready.set()

    async def _async_main(self) -> None:
        import anyio
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=self._server_cmd[0], args=self._server_cmd[1:]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._ready.set()

                # Drain the request queue forever.
                # anyio.to_thread.run_sync runs queue.get() in a thread-pool
                # worker so the event loop is never busy-polled.
                while True:
                    item = await anyio.to_thread.run_sync(
                        self._queue.get, abandon_on_cancel=True
                    )
                    if item is None:            # shutdown sentinel
                        break
                    tool_name, inputs, fut = item
                    try:
                        raw = await session.call_tool(tool_name, inputs)
                        fut.set_result(raw)
                    except BaseException as exc:  # noqa: BLE001
                        fut.set_exception(exc)

    # ------------------------------------------------------------------ #
    # ToolRegistry interface (called by ReActLoop and AgenticForecaster)
    # ------------------------------------------------------------------ #

    def bind_data(
        self,
        y: Any,
        X: Any = None,
        fh: Any = None,
        holdout: int = 0,
    ) -> None:
        """Push the series to the server-side registry; stash locally for commit."""
        self._y = y
        self._X = X
        self._fh = fh
        self._holdout = int(holdout)
        self._committed = None

        y_list: list[float] = y.tolist() if hasattr(y, "tolist") else list(y)
        fh_list: list[int] | None = (
            [int(v) for v in fh] if fh is not None else None
        )
        payload = json.dumps({"y": y_list, "fh": fh_list})

        result = self._rpc("bind_data_from_json", {"payload": payload, "holdout": int(holdout)})
        if not result.get("ok"):
            raise RuntimeError(
                f"MCP bind_data_from_json failed: {result.get('error')}"
            )

    def schema(self) -> list[dict[str, Any]]:
        """Return the static tool schema (same as the in-process registry)."""
        return TOOL_SCHEMAS

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        """Proxy a tool call to the server; re-execute commit locally."""
        result = self._rpc(tool_name, kwargs)

        # On a successful commit, reconstruct the fitted forecaster in-process
        # so inner_forecaster_ is a real Python object for predict().
        if tool_name == "commit" and isinstance(result, dict) and result.get("ok"):
            local = ToolRegistry()
            local.bind_data(
                y=self._y, X=self._X, fh=self._fh, holdout=self._holdout
            )
            local.commit(
                name=kwargs.get("name", result.get("name", "")),
                params=dict(kwargs.get("params") or {}),
                rationale=kwargs.get("rationale", ""),
            )
            self._committed = local._committed

        return result

    # ------------------------------------------------------------------ #
    # Internal RPC helper
    # ------------------------------------------------------------------ #

    def _rpc(self, tool_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Submit a call to the async loop; block until the result arrives."""
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._queue.put((tool_name, inputs, fut))
        try:
            raw = fut.result(timeout=self._call_timeout)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(
                f"MCP tool call '{tool_name}' timed out after {self._call_timeout} s"
            ) from exc
        return self._parse(raw)

    @staticmethod
    def _parse(raw: Any) -> Any:
        """Decode a ``CallToolResult`` into a Python value.

        FastMCP serialises ``dict`` / ``bool`` / ``int`` return values as a
        single JSON text block.  For ``list[str]`` it emits *one TextContent
        per element*, each wrapped in Python repr quotes (e.g. ``'AutoARIMA'``).
        We detect the multi-block case and reassemble the list.
        """
        import ast

        if not raw.content:
            return {}

        if len(raw.content) > 1:
            # Multiple blocks → FastMCP list serialisation.
            # Each block.text is a Python repr such as "'AutoARIMA'".
            items = []
            for block in raw.content:
                try:
                    items.append(ast.literal_eval(block.text))
                except Exception:
                    items.append(block.text)
            return items

        text = raw.content[0].text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"non-JSON server response: {text!r}"}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
