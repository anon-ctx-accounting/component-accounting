"""FR-H1: per-run isolation + arm-specific knowledge materialization.

Every run gets a fresh working directory under a temp root. The *base* is a
fresh copy of ``fixture/stage2/repo`` (so on-demand retrieval is possible for
every arm), and the arm decides what knowledge is *injected* into the model's
context via a CLAUDE.md ``@import`` chain:

- ``closed-book`` (E0-3, top priority): NO corpus at all — an empty working
  dir, no repo copy, no CLAUDE.md. The model sees only the task prompt. This
  is the leakage-regression baseline.
- ``static-full``: fresh repo copy + CLAUDE.md importing the **entire preload
  set** (BUILD.json / manifest ``preload=true`` docs).
- ``classic`` / ``karc``: fresh repo copy + CLAUDE.md importing only the
  **budget-c\\* working set** (an explicit ``working_set`` artifact-id list
  supplied by the caller; deriving it from replay policy state is the Stage 2
  execution step, out of scope here).
- ``mcp`` (E2-2): fresh repo copy with the managed subset relocated under
  ``.karc/managed/``, a ``.mcp.json`` registering the ``karc`` server, and
  (per condition) a managed-knowledge CLAUDE.md block and/or a ``settings``
  file denying native reads of the managed path.

Injected-knowledge tokens (FR-H4, the *measured* quantity) = the sum of the
manifest ``size_tok`` of the artifacts imported via CLAUDE.md. This boundary
is tokenizer-independent and identical to the Stage-1 token accounting, so
replay's budget sweep and the LLM run measure the same units. (Documented
ambiguity: it counts injected knowledge, not the total prompt/system tokens —
those are captured separately as API usage in FR-H4.)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

MANAGED_ROOT = ".karc/managed"

# A-8: arm-neutral retrieval instruction injected into the E2-1 arms
# (static-full / classic / karc) ONLY — never the E2-2 mcp arm, where the
# channel choice is the measurement. Removes the pilot S2-003 confound where an
# agent answered from the injected context alone without searching the repo.
# Named tools are the native retrieval tools shared by all three E2-1 arms, so
# the wording stays neutral among them.
E2_1_RETRIEVAL_INSTRUCTION = (
    "If the information you need is not in the documents above, do not guess: "
    "search and read the repository's files (Read/Grep/Glob) to find and verify "
    "the answer before responding.\n"
    "필요한 정보가 위 문서에 없으면 추측하지 말고, 저장소의 파일을 검색·열람해"
    "(Read/Grep/Glob) 확인한 뒤 답하라."
)

# A-9(c): resolve the karc server binary from the running interpreter's venv so
# the headless CLI can spawn it even when ``.venv/bin`` is absent from PATH.
_KARC_BIN = str(Path(sys.executable).parent / "karc")


def _karc_command() -> str:
    return _KARC_BIN if Path(_KARC_BIN).exists() else "karc"


@dataclass
class Materialized:
    workdir: Path
    injected_knowledge_tokens: int
    injected_artifacts: list[str] = field(default_factory=list)
    injected_files: list[str] = field(default_factory=list)
    mcp_config: str | None = None
    settings: str | None = None
    managed_prefixes: tuple[str, ...] = ()
    has_corpus: bool = False
    codex_config_overrides: tuple[str, ...] = ()
    hook_mode: str = "observe"
    agents_sha256: str | None = None
    agents_bytes: int = 0


def _copy_repo(src_repo: Path, dst: Path) -> None:
    shutil.copytree(src_repo, dst, dirs_exist_ok=True)


def _import_chain(workdir: Path, rel_paths: list[str]) -> None:
    """Write a CLAUDE.md whose body is an ``@import`` chain (Claude Code loads
    ``@path`` imports into context at session start).

    Called only by the E2-1 arms (static-full / classic / karc), so the A-8
    retrieval instruction is appended here — the E2-2 mcp arm writes its own
    CLAUDE.md in ``_materialize_mcp`` and never gets this section."""
    lines = ["# Project knowledge (K-ARC managed)", ""]
    lines += [f"@{p}" for p in rel_paths]
    lines += ["", "## Retrieval", "", E2_1_RETRIEVAL_INSTRUCTION]
    (workdir / "CLAUDE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _preload_artifacts(manifest: dict) -> list[str]:
    return sorted(aid for aid, e in manifest["artifacts"].items() if e.get("preload"))


def _tokens(manifest: dict, artifact_ids: list[str]) -> int:
    arts = manifest["artifacts"]
    return sum(int(arts[a]["size_tok"]) for a in artifact_ids)


def materialize(
    arm: str,
    workdir: Path,
    *,
    fixture_root: Path,
    manifest: dict,
    working_set: list[str] | None = None,
    managed: list[str] | None = None,
    mcp_condition: str | None = None,
    db_path: str | None = None,
    runtime: str = "claude-code",
) -> Materialized:
    """Create ``workdir`` and materialize knowledge for ``arm``.

    ``working_set`` (classic/karc) and ``managed`` (mcp) are artifact-id lists.
    ``mcp_condition`` ∈ {"c0","c1","c2"} selects the E2-2 instruction/deny layer.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    src_repo = Path(fixture_root) / "repo"
    arts = manifest["artifacts"]

    if runtime == "codex":
        return _materialize_codex(
            arm, workdir, src_repo=src_repo, manifest=manifest,
            working_set=working_set, managed=managed,
            mcp_condition=mcp_condition or "c0", db_path=db_path,
        )
    if runtime != "claude-code":
        raise ValueError(f"unsupported benchmark runtime {runtime!r}")

    if arm == "closed-book":
        # No corpus, no CLAUDE.md — the model gets only the task prompt.
        return Materialized(workdir=workdir, injected_knowledge_tokens=0, has_corpus=False)

    _copy_repo(src_repo, workdir)

    if arm == "static-full":
        ids = _preload_artifacts(manifest)
        rels = [arts[a]["path"] for a in ids]
        _import_chain(workdir, rels)
        return Materialized(workdir=workdir, injected_knowledge_tokens=_tokens(manifest, ids),
                            injected_artifacts=ids, injected_files=rels, has_corpus=True)

    if arm in ("classic", "karc"):
        if working_set is None:
            raise ValueError(f"arm {arm!r} requires an explicit working_set")
        ids = [a for a in working_set if a in arts]
        rels = [arts[a]["path"] for a in ids]
        _import_chain(workdir, rels)
        return Materialized(workdir=workdir, injected_knowledge_tokens=_tokens(manifest, ids),
                            injected_artifacts=ids, injected_files=rels, has_corpus=True)

    if arm == "mcp":
        return _materialize_mcp(workdir, manifest, managed or [], mcp_condition or "c0", db_path)

    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------
# Codex semantic-parity materialization
# --------------------------------------------------------------------------
CODEX_E2_1_RETRIEVAL_INSTRUCTION = (
    "If the information you need is not in the documents below, do not guess. "
    "Search and read the repository files with the available local tools, verify "
    "the answer, and then respond.\n"
    "아래 문서에 필요한 정보가 없으면 추측하지 말고, 사용 가능한 로컬 도구로 "
    "저장소 파일을 검색·열람해 확인한 뒤 답하라."
)

CODEX_MCP_INSTRUCTION = (
    "Use the `karc` MCP server's `search` and `get` tools to find and read "
    "K-ARC-managed project knowledge before answering."
)


def _init_git(workdir: Path) -> None:
    proc = subprocess.run(
        ["git", "init", "--quiet"], cwd=workdir,
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git init failed: {(proc.stderr or proc.stdout)[:300]}")


def _toml_value(value) -> str:
    # JSON strings/arrays are also valid TOML scalar/array syntax for the
    # values used here and avoid hand-written escaping of absolute paths.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _codex_agents(workdir: Path, manifest: dict, ids: list[str], *,
                  instruction: str) -> tuple[str, int]:
    import hashlib

    arts = manifest["artifacts"]
    ordered = sorted((a for a in ids if a in arts), key=lambda a: arts[a]["path"])
    parts = ["# K-ARC benchmark context", "", instruction, "", "## Injected knowledge", ""]
    for aid in ordered:
        entry = arts[aid]
        body = (workdir / entry["path"]).read_text(encoding="utf-8")
        parts.append(
            f"<!-- artifact:start id={aid} path={entry['path']} sha256={entry['sha256']} -->"
        )
        parts.append(body.rstrip("\n"))
        parts.append(f"<!-- artifact:end id={aid} -->")
        parts.append("")
    text = "\n".join(parts).rstrip() + "\n"
    path = workdir / "AGENTS.md"
    path.write_text(text, encoding="utf-8")
    raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _codex_mcp_overrides(workdir: Path, index_db: Path) -> tuple[str, ...]:
    command = _karc_command()
    args = ["mcp", "serve", "--root", str(workdir), "--db", str(index_db),
            "--runtime", "codex"]
    return (
        f"mcp_servers.karc.command={_toml_value(command)}",
        f"mcp_servers.karc.args={_toml_value(args)}",
        "mcp_servers.karc.required=true",
        "mcp_servers.karc.startup_timeout_sec=10",
        "mcp_servers.karc.tool_timeout_sec=60",
    )


def _materialize_codex(arm: str, workdir: Path, *, src_repo: Path,
                       manifest: dict, working_set: list[str] | None,
                       managed: list[str] | None, mcp_condition: str,
                       db_path: str | None) -> Materialized:
    arts = manifest["artifacts"]
    if arm == "closed-book":
        _init_git(workdir)
        return Materialized(
            workdir=workdir, injected_knowledge_tokens=0, has_corpus=False,
            hook_mode="deny-all",
        )

    if arm == "closed-context":
        if working_set is None:
            raise ValueError("closed-context requires an explicit working_set")
        ids = [a for a in working_set if a in arts]
        for aid in ids:
            source = src_repo / arts[aid]["path"]
            target = workdir / arts[aid]["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        _init_git(workdir)
        sha, nbytes = _codex_agents(
            workdir, manifest, ids,
            instruction=(
                "Use only the injected document below. Do not search for or infer "
                "any later version; return the value stated in this document."
            ),
        )
        return Materialized(
            workdir=workdir,
            injected_knowledge_tokens=_tokens(manifest, ids),
            injected_artifacts=sorted(ids, key=lambda a: arts[a]["path"]),
            injected_files=[arts[a]["path"] for a in sorted(ids, key=lambda a: arts[a]["path"])],
            has_corpus=False,
            codex_config_overrides=(f"project_doc_max_bytes={nbytes + 4096}",),
            hook_mode="observe", agents_sha256=sha, agents_bytes=nbytes,
        )

    _copy_repo(src_repo, workdir)
    _init_git(workdir)

    if arm in ("static-full", "classic", "karc", "search-only",
               "rag-bm25", "rag-embed", "recency"):
        if arm == "static-full":
            ids = _preload_artifacts(manifest)
        elif arm == "search-only":
            ids = []
        else:
            if working_set is None:
                raise ValueError(f"arm {arm!r} requires an explicit working_set")
            ids = [a for a in working_set if a in arts]
        sha, nbytes = _codex_agents(
            workdir, manifest, ids, instruction=CODEX_E2_1_RETRIEVAL_INSTRUCTION
        )
        return Materialized(
            workdir=workdir,
            injected_knowledge_tokens=_tokens(manifest, ids),
            injected_artifacts=sorted(ids, key=lambda a: arts[a]["path"]),
            injected_files=[arts[a]["path"] for a in sorted(ids, key=lambda a: arts[a]["path"])],
            has_corpus=True,
            codex_config_overrides=(f"project_doc_max_bytes={nbytes + 4096}",),
            hook_mode="observe", agents_sha256=sha, agents_bytes=nbytes,
        )

    if arm != "mcp":
        raise ValueError(f"unknown arm {arm!r}")

    managed_ids = [a for a in (managed or []) if a in arts]
    managed_dir = workdir / MANAGED_ROOT
    for aid in managed_ids:
        old = workdir / arts[aid]["path"]
        new = managed_dir / arts[aid]["path"]
        new.parent.mkdir(parents=True, exist_ok=True)
        if old.exists():
            shutil.move(str(old), str(new))

    from karc.bench import mcp_index
    index_db = workdir / ".karc" / "index.db"
    mcp_index.build_index(
        workdir, manifest, managed_ids, managed_root=MANAGED_ROOT, db_path=index_db
    )
    sha = None
    nbytes = 0
    overrides = list(_codex_mcp_overrides(workdir, index_db))
    if mcp_condition in ("c1", "c2"):
        sha, nbytes = _codex_agents(
            workdir, manifest, [], instruction=CODEX_MCP_INSTRUCTION
        )
        overrides.append(f"project_doc_max_bytes={nbytes + 4096}")
    if mcp_condition == "c2":
        marker = workdir / ".karc" / "deny-native"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("c2\n", encoding="utf-8")

    return Materialized(
        workdir=workdir, injected_knowledge_tokens=0,
        injected_artifacts=[], injected_files=[], has_corpus=True,
        managed_prefixes=(MANAGED_ROOT + "/",),
        codex_config_overrides=tuple(overrides),
        hook_mode="deny-managed" if mcp_condition == "c2" else "observe",
        agents_sha256=sha, agents_bytes=nbytes,
    )


def _materialize_mcp(workdir: Path, manifest: dict, managed: list[str],
                     condition: str, db_path: str | None) -> Materialized:
    """E2-2 materialization. Relocate the managed subset under ``.karc/managed``
    so FR-H7 can attribute native reads to the managed path, register the karc
    MCP server, and layer the condition-specific instruction/deny."""
    arts = manifest["artifacts"]
    managed_ids = [a for a in managed if a in arts]
    managed_dir = workdir / MANAGED_ROOT
    for aid in managed_ids:
        old = workdir / arts[aid]["path"]
        new = managed_dir / arts[aid]["path"]
        new.parent.mkdir(parents=True, exist_ok=True)
        if old.exists():
            shutil.move(str(old), str(new))

    # A-9(a): index the relocated managed docs into a per-run DB scoped to this
    # workdir so the server's search/get actually return them. Sibling of (not
    # inside) .karc/managed/, so the C2 read-deny does not touch it.
    from karc.bench import mcp_index
    index_db = workdir / ".karc" / "index.db"
    mcp_index.build_index(workdir, manifest, managed_ids,
                          managed_root=MANAGED_ROOT, db_path=index_db)

    mcp_cfg = {
        "mcpServers": {
            "karc": {
                "command": _karc_command(),
                "args": ["mcp", "serve", "--root", str(workdir),
                         "--db", str(index_db)],
            }
        }
    }
    mcp_path = workdir / ".mcp.json"
    mcp_path.write_text(json.dumps(mcp_cfg, indent=2), encoding="utf-8")

    settings_path: str | None = None
    if condition in ("c1", "c2"):
        block = ("# K-ARC managed knowledge\n\n"
                 "K-ARC 관리 지식은 `karc` MCP tool(`search`/`get`)로 검색·열람하라. "
                 "관리 경로(`.karc/managed/`)의 파일을 직접 열지 말 것.\n")
        (workdir / "CLAUDE.md").write_text(block, encoding="utf-8")
    if condition == "c2":
        settings = {"permissions": {"deny": [
            f"Read({MANAGED_ROOT}/**)", f"Grep({MANAGED_ROOT}/**)"]}}
        sp = workdir / ".claude" / "settings.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        settings_path = str(sp)

    return Materialized(
        workdir=workdir,
        injected_knowledge_tokens=0,  # MCP arm injects nothing; retrieval is on-demand
        mcp_config=str(mcp_path),
        settings=settings_path,
        managed_prefixes=(MANAGED_ROOT + "/",),
        has_corpus=True,
    )
