"""First-party ingestion for the MCP channel (data-model §4).

Each ``search``/``get`` call:

1. issues a ``karc_call_id`` (ULID) — the tier-1 dedup key seed (§4.2), echoed
   back in the tool response so hook/transcript observers of the same call can
   converge on the same canonical event;
2. writes an ``ingest_observations`` row (source_channel='mcp', reduced
   payload — R-9: ids/paths/metadata only, never content);
3. writes canonical ``events`` (``discovered`` per search candidate,
   ``read`` per get) with ``INSERT OR IGNORE`` on the dedup_key, and links the
   observation to the event (the only UPDATE the observations trigger allows).

Path→artifact resolution reuses the ingestion identity helpers (N1–N7, §5).
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass

from karc.ingest import pipeline
from karc.snapshot import cas
from karc.util import new_ulid, sha256_hex, utc_now_iso

ADAPTER_VERSION = "karc-mcp-0.1.0"
EVENT_SCHEMA_VERSION = 1


@dataclass
class SearchHit:
    artifact_id: str
    logical_name: str
    path: str | None
    artifact_type: str
    size_tokens: int | None

    def as_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "name": self.logical_name,
            "path": self.path,
            "type": self.artifact_type,
            "size_tokens": self.size_tokens,
        }


class MCPIngestor:
    """Wraps a connection + scope for first-party event recording."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        scope_id: str,
        runtime: str = "mcp",
    ):
        self.conn = conn
        self.db_path = db_path
        self.scope_id = scope_id
        self.runtime = runtime

    # -- observation + event helpers ---------------------------------------
    def _observe(self, call_id: str, payload: dict) -> str:
        obs_id = sha256_hex(f"mcp|{call_id}".encode("utf-8"))
        self.conn.execute(
            """INSERT OR IGNORE INTO ingest_observations
                 (observation_id, source_channel, runtime, adapter_version,
                  schema_version, source_native_id, observed_payload)
               VALUES (?, 'mcp', ?, ?, ?, ?, ?)""",
            (
                obs_id,
                self.runtime,
                ADAPTER_VERSION,
                EVENT_SCHEMA_VERSION,
                call_id,
                json.dumps({**payload, "reduced": True}, ensure_ascii=False, sort_keys=True),
            ),
        )
        return obs_id

    def _emit_event(
        self,
        dedup_key: str,
        event_type: str,
        artifact_id: str,
        occurred_at: str,
        token_cost: int | None,
        obs_id: str | None,
    ) -> str:
        event_id = new_ulid()
        version_row = self.conn.execute(
            "SELECT version_id FROM versions WHERE artifact_id = ? AND invalidated_at IS NULL",
            (artifact_id,),
        ).fetchone()
        version_id = version_row[0] if version_row else None
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO events
                 (event_id, dedup_key, event_type, occurred_at, scope_id, runtime,
                  artifact_id, version_id, token_cost, confidence, source_channel,
                  schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'direct', 'mcp', ?)""",
            (
                event_id,
                dedup_key,
                event_type,
                occurred_at,
                self.scope_id,
                self.runtime,
                artifact_id,
                version_id,
                token_cost,
                EVENT_SCHEMA_VERSION,
            ),
        )
        if cur.rowcount:
            final = event_id
        else:
            final = self.conn.execute(
                "SELECT event_id FROM events WHERE dedup_key = ?", (dedup_key,)
            ).fetchone()[0]
        if obs_id is not None:
            self.conn.execute(
                "UPDATE ingest_observations SET event_id = ? WHERE observation_id = ? "
                "AND event_id IS NULL",
                (final, obs_id),
            )
        return final

    def _content_term_matches(self, path: str | None, terms: list[str]) -> int:
        """Count distinct query terms present in the artifact's on-disk content
        (read at query time, never persisted — R-9). Returns 0 if unreadable."""
        if not path or not os.path.isfile(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read().lower()
        except OSError:
            return 0
        return sum(1 for t in terms if t in body)

    def _size_tokens(self, artifact_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT v.size_tokens FROM versions v JOIN artifacts a "
            "ON a.current_version_id = v.version_id WHERE a.artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    # -- tools --------------------------------------------------------------
    def search(self, query: str, limit: int = 10, artifact_type: str | None = None):
        """Lexical/metadata registry search. Emits one `discovered` event per
        returned candidate. Archived/invalidated artifacts are excluded from
        discovery; critical-cold stays discoverable (§4.3)."""
        call_id = new_ulid()
        terms = [t for t in query.lower().split() if t]
        rows = self.conn.execute(
            "SELECT artifact_id, logical_name, canonical_path, artifact_type "
            "FROM artifacts WHERE scope_id = ? "
            "AND lifecycle_state NOT IN ('archived','invalidated')",
            (self.scope_id,),
        ).fetchall()
        scored: list[tuple[int, SearchHit]] = []
        for art_id, name, path, atype in rows:
            if artifact_type and atype != artifact_type:
                continue
            hay = f"{name} {path or ''}".lower()
            score = sum(1 for t in terms if t in hay)
            # A-9(b): content-aware — also match query terms against on-disk
            # content (read here, never persisted; R-9). A keyword knowledge
            # search that only looked at filenames returned nothing for content
            # queries, so an instructed agent would abandon the tool and fall
            # back to native reads, artificially depressing E2-2 adoption.
            if terms:
                score += 2 * self._content_term_matches(path, terms)
            if terms and score == 0:
                continue
            scored.append(
                (score, SearchHit(art_id, name, path, atype, self._size_tokens(art_id)))
            )
        scored.sort(key=lambda s: (-s[0], s[1].logical_name))
        hits = [h for _, h in scored[:limit]]

        obs_id = self._observe(
            call_id,
            {"tool": "search", "query_len": len(query), "n_results": len(hits),
             "result_ids": [h.artifact_id for h in hits]},
        )
        occurred_at = utc_now_iso()
        for h in hits:
            self._emit_event(
                f"karc:{call_id}:d:{h.artifact_id}",
                "discovered",
                h.artifact_id,
                occurred_at,
                h.size_tokens,
                obs_id,
            )
        return call_id, hits

    def get(self, ref: str):
        """Return an artifact's content. Emits a `read` event. ``ref`` is an
        artifact_id or a path (resolved via N1–N7). Archived content is served
        from the CAS; live content from the canonical path."""
        call_id = new_ulid()
        art = self._resolve(ref)
        if art is None:
            self._observe(call_id, {"tool": "get", "ref": os.path.basename(ref), "found": False})
            return call_id, None
        artifact_id, path, lifecycle, cur_hash = art
        content, content_hash = self._read_content(path, cur_hash)
        size_tokens = self._size_tokens(artifact_id)
        obs_id = self._observe(
            call_id,
            {"tool": "get", "artifact_id": artifact_id,
             "content_hash": content_hash, "length": len(content) if content else 0},
        )
        self._emit_event(
            f"karc:{call_id}", "read", artifact_id, utc_now_iso(), size_tokens, obs_id
        )
        return call_id, {
            "artifact_id": artifact_id,
            "path": path,
            "lifecycle_state": lifecycle,
            "content": content,
            "content_hash": content_hash,
            "size_tokens": size_tokens,
        }

    # -- resolution ---------------------------------------------------------
    def _resolve(self, ref: str):
        """Resolve ``ref`` (artifact_id or path) to a registered artifact —
        **lookup only, never registers** (P2, E2-2 diagnosis).

        The previous on-demand registration branch turned every unknown ref
        into a fresh provisional artifact: a typo'd artifact_id or a stale path
        yielded a silent empty-success ``get`` (dead-end retry loops), and a
        ``get`` by an already-indexed path could register a *duplicate*
        provisional twin that then polluted ``search`` results. A read tool
        must not mutate the registry; unknown refs are now a not-found."""
        row = self.conn.execute(
            "SELECT a.artifact_id, a.canonical_path, a.lifecycle_state, v.content_hash "
            "FROM artifacts a LEFT JOIN versions v ON a.current_version_id = v.version_id "
            "WHERE a.artifact_id = ?",
            (ref,),
        ).fetchone()
        if row:
            return row
        # treat ref as a path: normalize (N1–N7) and match the canonical path
        resolved, _existed, _inode = pipeline.normalize_path(ref, os.getcwd())
        return self.conn.execute(
            "SELECT a.artifact_id, a.canonical_path, a.lifecycle_state, v.content_hash "
            "FROM artifacts a LEFT JOIN versions v ON a.current_version_id = v.version_id "
            "WHERE a.canonical_path = ?",
            (resolved,),
        ).fetchone()

    def _read_content(self, path: str | None, cur_hash: str | None):
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace"), sha256_hex(raw)
        if cur_hash and cas.exists(self.conn, cur_hash):
            raw = cas.load(self.conn, self.db_path, cur_hash)
            return raw.decode("utf-8", errors="replace"), cur_hash
        return None, cur_hash
