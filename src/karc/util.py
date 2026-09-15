"""Small stdlib-only utilities: ULID generation, timestamps, hashing."""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone

# Crockford base32 alphabet (ULID spec)
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid(ts_ms: int | None = None) -> str:
    """Generate a ULID (26 chars, Crockford base32).

    48-bit millisecond timestamp + 80 bits of randomness. Stdlib only.
    """
    if ts_ms is None:
        ts_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")  # 80 bits
    value = (ts_ms << 80) | rand
    chars = []
    for _ in range(26):
        chars.append(_B32[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def utc_now_iso() -> str:
    """Current UTC time as ISO8601 with milliseconds and Z suffix.

    Matches the transcript timestamp format ("2026-07-16T15:58:21.383Z")
    so that lexicographic string ordering == chronological ordering.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file (never loads the whole file)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
