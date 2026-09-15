"""Deterministic nonce-domain primitives (experiment-design §6.2).

Everything is a pure function of (seed, key parts) via sha256 — same contract
as ``replay.corpus``. Names are invented syllable compounds with a hex suffix
so they cannot collide with real products; facts are namespaced config keys
whose *key* embeds the owning document's nonce, making cross-document answer
collisions structurally impossible; canaries are per-artifact UUID-shaped
tags for the E0-3 leakage regression check.
"""

from __future__ import annotations

import hashlib

_SYL = ["vel", "dor", "zan", "mek", "tor", "bly", "qua", "fen",
        "gri", "lus", "pra", "nim", "sor", "keb", "wix", "jal"]
_KO_SYL = ["벨", "도르", "잔", "멕", "토르", "블리", "콰", "펜",
           "그리", "루스", "프라", "님", "소르", "켑", "윅스", "잘"]

_FACT_KINDS = [
    ("retry_limit", 2, 9),
    ("timeout_ms", 500, 9900),
    ("port", 20000, 59999),
    ("batch_size", 16, 512),
    ("cache_ttl_s", 30, 3600),
    ("max_payload_kb", 64, 4096),
]

_EN_FILLER_NOUNS = ["scheduler", "gateway", "ledger", "relay", "queue",
                    "manifest", "rotor", "beacon", "shard", "planner"]
_EN_FILLER_VERBS = ["batches", "rotates", "flushes", "reconciles", "stages",
                    "drains", "mirrors", "compacts", "signs", "replays"]
_KO_FILLER_TAIL = ["이전에 모아 처리한다", "주기로 재정렬한다", "완료 후 비운다",
                   "기준으로 대사한다", "단계로 준비한다", "순서로 배출한다",
                   "방식으로 복제한다", "정책으로 압축한다", "키로 서명한다",
                   "로그로 재생한다"]


def _h(*parts) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _int(*parts) -> int:
    return int(_h(*parts)[:16], 16)


def sys_name(seed: int, key: str) -> str:
    """Invented system name, e.g. ``Veldor-3f`` (hex suffix → non-real)."""
    i, j = _int(seed, key, "a") % 16, _int(seed, key, "b") % 16
    suffix = _h(seed, key, "sfx")[:2]
    return f"{_SYL[i].capitalize()}{_SYL[j]}-{suffix}"


def sys_name_ko(seed: int, key: str) -> str:
    i, j = _int(seed, key, "a") % 16, _int(seed, key, "b") % 16
    suffix = _h(seed, key, "sfx")[:2]
    return f"{_KO_SYL[i]}{_KO_SYL[j]}-{suffix}"


def fact_for(seed: int, artifact_id: str, idx: int, value_salt: str = "") -> dict:
    """Deterministic fact owned by ``artifact_id``.

    The key embeds a doc-scoped nonce token, so the same (key, value) pair
    cannot occur in any other document by construction. ``value_salt`` lets
    superseded/conflicting counterparts share the *key* but differ in value:
    callers reuse the base doc's key and pass a different salt.
    """
    kind, lo, hi = _FACT_KINDS[_int(seed, artifact_id, idx, "kind") % len(_FACT_KINDS)]
    token = _h(seed, artifact_id, idx, "tok")[:6]
    key = f"{token}_{kind}"
    value = lo + _int(seed, artifact_id, idx, "val", value_salt) % (hi - lo + 1)
    return {"key": key, "value": value, "kind": kind}


def alt_value(seed: int, fact: dict, salt: str) -> int:
    """A different value for the same fact key (superseded v1 / conflict y)."""
    kind = fact["kind"]
    lo, hi = next((lo, hi) for k, lo, hi in _FACT_KINDS if k == kind)
    v = lo + _int(seed, fact["key"], "alt", salt) % (hi - lo + 1)
    if v == fact["value"]:  # force distinctness deterministically
        v = lo + (v - lo + 1) % (hi - lo + 1)
    return v


def canary(seed: int, artifact_id: str) -> str:
    h = _h(seed, artifact_id, "canary")
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def filler_sentence(seed: int, artifact_id: str, i: int, lang: str) -> str:
    a = _int(seed, artifact_id, i, "f1")
    b = _int(seed, artifact_id, i, "f2")
    c = _int(seed, artifact_id, i, "f3")
    if lang == "ko":
        return (f"{_KO_SYL[a % 16]}{_KO_SYL[b % 16]} 모듈은 "
                f"{_KO_SYL[c % 16]} 이벤트를 {_KO_FILLER_TAIL[(a ^ b) % 10]}.")
    return (f"The {_SYL[a % 16]}{_SYL[b % 16]} {_EN_FILLER_NOUNS[c % 10]} "
            f"{_EN_FILLER_VERBS[(a ^ c) % 10]} {_SYL[b % 16]}{_SYL[c % 16]} "
            f"records before the {_SYL[(a + c) % 16]} stage.")


def fact_line(fact: dict) -> str:
    """The canonical greppable form; also the grader regex target."""
    return f"{fact['key']} = {fact['value']}"


def answer_regex(fact: dict) -> str:
    # trailing ``(?![0-9])`` stops value ``3`` from matching ``31`` — otherwise
    # a wrong answer (or a superseded/conflict twin whose value shares a digit
    # prefix) would grade as correct.
    return rf"{fact['key']}\s*=\s*{fact['value']}(?![0-9])"
