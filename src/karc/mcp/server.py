"""Stdio JSON-RPC MCP server — protocol implemented directly (no SDK).

Transport: newline-delimited JSON-RPC 2.0 over stdin/stdout (the MCP stdio
convention). Methods handled: ``initialize``, ``notifications/initialized``
(notification, no reply), ``ping``, ``tools/list``, ``tools/call``. Server name
``karc``; tools ``search`` and ``get``.

Each ``tools/call`` runs its DB writes inside a ``BEGIN IMMEDIATE`` transaction
with busy retry (data-model §11 writer standard), so the server coexists with
hook writers and the CLI on one WAL database.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import unicodedata

from karc.db import connection
from karc.mcp.ingest import MCPIngestor
from karc.util import new_ulid

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "karc"
SERVER_VERSION = "0.1.0"

TOOLS = [
    {
        "name": "search",
        "description": (
            "Search K-ARC-managed project knowledge (docs, rules, skills) by "
            "keyword. Returns matching artifacts with their artifact_id — pass "
            "that id to `get` to fetch content without needing a file path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "keywords to search for"},
                "limit": {"type": "integer", "description": "max results (default 10)"},
                "type": {
                    "type": "string",
                    "description": "optional artifact_type filter (document, rule, skill, …)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get",
        "description": (
            "Fetch the full content of a K-ARC-managed artifact by its "
            "artifact_id (from `search`) or by path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "path": {"type": "string"},
            },
        },
    },
]


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _ok(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def ensure_scope(conn: sqlite3.Connection, root: str) -> str:
    root = os.path.normpath(root)
    if os.path.isdir(root):
        root = os.path.realpath(root)
    root = unicodedata.normalize("NFC", root)
    row = conn.execute("SELECT scope_id FROM scopes WHERE root_path = ?", (root,)).fetchone()
    if row:
        return row[0]
    scope_id = new_ulid()
    conn.execute(
        "INSERT INTO scopes (scope_id, root_path, display_name) VALUES (?, ?, ?)",
        (scope_id, root, os.path.basename(root) or root),
    )
    return scope_id


class MCPServer:
    def __init__(self, db_path: str, root: str, runtime: str = "mcp"):
        self.db_path = db_path
        self.root = root
        self.runtime = runtime
        self.conn = connection.connect(db_path)
        connection.migrate(self.conn, db_path=db_path)
        self._begin_immediate()
        try:
            self.scope_id = ensure_scope(self.conn, root)
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.ingestor = MCPIngestor(self.conn, db_path, self.scope_id, runtime)

    # -- write transaction with busy retry (§11) ---------------------------
    def _begin_immediate(self, retries: int = 15) -> None:
        for attempt in range(retries):
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError:
                time.sleep(0.02 + 0.01 * attempt)
        self.conn.execute("BEGIN IMMEDIATE")  # final attempt raises if still locked

    def _in_txn(self, fn):
        self._begin_immediate()
        try:
            out = fn()
            self.conn.execute("COMMIT")
            return out
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    # -- protocol ----------------------------------------------------------
    def handle_message(self, msg: dict) -> dict | None:
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        is_notification = "id" not in msg
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                }
                return _ok(msg_id, result)
            if method in ("notifications/initialized", "initialized"):
                return None  # notification
            if method == "ping":
                return _ok(msg_id, {})
            if method == "tools/list":
                return _ok(msg_id, {"tools": TOOLS})
            if method == "tools/call":
                return _ok(msg_id, self._tools_call(params))
            if is_notification:
                return None
            raise JsonRpcError(-32601, f"method not found: {method}")
        except JsonRpcError as e:
            if is_notification:
                return None
            return _err(msg_id, e.code, e.message)
        except Exception as e:  # never crash the loop on a single bad call
            if is_notification:
                return None
            return _err(msg_id, -32603, f"internal error: {e}")

    def _tools_call(self, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "search":
            query = args.get("query", "")
            limit = int(args.get("limit") or 10)
            atype = args.get("type")
            call_id, hits = self._in_txn(
                lambda: self.ingestor.search(query, limit=limit, artifact_type=atype)
            )
            structured = {
                "karc_call_id": call_id,
                "results": [h.as_dict() for h in hits],
            }
            text = json.dumps(structured, ensure_ascii=False, indent=2)
            return {
                "content": [{"type": "text", "text": text}],
                "structuredContent": structured,
                "isError": False,
            }
        if name == "get":
            ref = args.get("artifact_id") or args.get("path")
            if not ref:
                raise JsonRpcError(-32602, "get requires artifact_id or path")
            call_id, doc = self._in_txn(lambda: self.ingestor.get(ref))
            if doc is None:
                # P2 (E2-2 diagnosis): explicit not-found — no silent-empty
                # success, no provisional registration (see ingest._resolve).
                return {
                    "content": [{"type": "text", "text":
                                 f"not found: {ref!r} is not a K-ARC-managed "
                                 "artifact_id or path. Use `search` to find the "
                                 "artifact_id first."}],
                    "isError": True,
                }
            # P1 (E2-2 diagnosis): text block ONLY — no structuredContent.
            # Claude Code (≤ 2.1.212 observed) surfaces structuredContent to the
            # model INSTEAD OF the text block, so a metadata-only
            # structuredContent shadowed the document body entirely (E2-2
            # INVALID root cause). The karc_call_id echo moves into a header
            # line so transcript observers can still correlate the call.
            header = (f"[karc get] artifact_id={doc['artifact_id']} "
                      f"path={doc['path']} karc_call_id={call_id}")
            return {
                "content": [{"type": "text",
                             "text": header + "\n\n" + (doc["content"] or "")}],
                "isError": False,
            }
        raise JsonRpcError(-32602, f"unknown tool: {name}")

    # -- serve loop --------------------------------------------------------
    def serve(self, stdin=None, stdout=None) -> None:
        stdin = stdin if stdin is not None else sys.stdin
        stdout = stdout if stdout is not None else sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                stdout.write(json.dumps(_err(None, -32700, "parse error")) + "\n")
                stdout.flush()
                continue
            response = self.handle_message(msg)
            if response is not None:
                stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                stdout.flush()

    def close(self) -> None:
        self.conn.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="karc mcp serve")
    parser.add_argument("--db", default=str(connection.DEFAULT_DB_PATH))
    parser.add_argument("--root", default=os.getcwd(), help="project root (scope)")
    parser.add_argument("--runtime", default="mcp")
    args = parser.parse_args(argv)
    server = MCPServer(args.db, args.root, runtime=args.runtime)
    try:
        server.serve()
    finally:
        server.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
