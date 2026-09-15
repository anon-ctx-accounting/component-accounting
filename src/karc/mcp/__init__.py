"""K-ARC MCP server (M3 checkpoint b).

A dependency-free stdio JSON-RPC implementation of the MCP protocol (no
external SDK — R-3 zero-runtime-deps). Server name ``karc`` with two tools,
``search`` and ``get`` (integration-options §1: server name must not repeat in
the tool name so the four runtimes render clean prefixes
``mcp__karc__search`` / ``karc_search`` / ``mcp_karc_search``).

Every tool call is a first-party observation channel (data-model §4): the
server writes an ``ingest_observations`` row and a canonical ``events`` row,
and stamps the response with a ``karc_call_id`` — the tier-1 dedup key that
lets hook/transcript observations of the same call converge (§4.2).
"""

from karc.mcp import ingest, server

__all__ = ["ingest", "server"]
