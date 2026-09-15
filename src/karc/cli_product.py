"""Product-surface CLI commands (UX §7 spec): init, doctor, guard, status,
review, why, pin, unpin, archive, restore, list, log.

Wired into ``karc.cli``. Query commands accept ``--json``. Emoji badges fall
back to ASCII when stdout is not a TTY or ``NO_COLOR`` is set (§7).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from karc import runtimes, scan
from karc.db import connection
from karc.live import queue as live_queue
from karc.live import state as live_state
from karc.mcp.server import ensure_scope
from karc.snapshot import fsck, store
from karc.util import utc_now_iso

# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _ascii_mode() -> bool:
    return bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def _icon(kind: str) -> str:
    emoji = {"protect": "🛡", "lifecycle": "🗄", "undo": "↩", "warn": "⚠", "ok": "✔", "no": "✘"}
    ascii_ = {"protect": "[pin]", "lifecycle": "[arc]", "undo": "<<", "warn": "!", "ok": "ok", "no": "x"}
    return (ascii_ if _ascii_mode() else emoji)[kind]


ACTION_ICON = {
    "pin": "protect", "archive": "lifecycle", "cold_transition": "lifecycle",
    "split": "lifecycle", "invalidate_or_correct": "warn",
}


def _print_json(obj) -> int:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# shared open / scope resolution
# ---------------------------------------------------------------------------

def _open(args):
    conn = connection.connect(args.db)
    connection.migrate(conn, db_path=args.db)
    return conn


def _resolve_scope(conn, args) -> str | None:
    if getattr(args, "scope", None):
        return args.scope
    cwd = os.path.realpath(os.getcwd())
    rows = conn.execute("SELECT scope_id, root_path FROM scopes").fetchall()
    best = None
    for scope_id, root in sorted(rows, key=lambda r: len(r[1]), reverse=True):
        if cwd == root or cwd.startswith(root.rstrip("/") + "/") or root.startswith(cwd):
            best = scope_id
            break
    if best is None and len(rows) == 1:
        best = rows[0][0]
    return best


def _txn(conn, fn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        out = fn()
        conn.execute("COMMIT")
        return out
    except BaseException:
        conn.execute("ROLLBACK")
        raise


# ===========================================================================
# init  (UX §2.1 five-stage flow)
# ===========================================================================

def cmd_init(args) -> int:
    root = os.path.realpath(args.root or os.getcwd())
    valid_runtimes = {"claude-code", "codex", "hermes", "opencode"}
    if args.runtime and args.runtime not in valid_runtimes:
        print(f"unsupported --runtime {args.runtime!r}; choose one of "
              f"{', '.join(sorted(valid_runtimes))}", file=sys.stderr)
        return 2
    apply = args.yes and not args.dry_run
    mode = "APPLY" if apply else "DRY-RUN (no files changed; re-run with --yes to apply)"
    print(f"K-ARC init  [{mode}]")
    print("-" * 60)

    # [1/5] detect runtimes
    infos = runtimes.detect_runtimes(home=args.home)
    detected = {ri.key for ri in infos if ri.found}
    selected = {args.runtime} if args.runtime else detected
    print("[1/5] Detecting agent runtimes...")
    for ri in infos:
        if ri.key not in selected:
            continue
        mark = _icon("ok") if ri.found else _icon("no")
        detail = f"({ri.version}, config: {ri.config_path})" if ri.found else "(not found)"
        print(f"  {mark} {ri.name:14s} {detail}")

    # [2/5] scan artifacts + preload seed
    print("\n[2/5] Scanning project knowledge artifacts...")
    conn = None
    if apply:
        conn = _open(args)  # dry-run must not even create the DB file
        scope_id = _txn(conn, lambda: ensure_scope(conn, root))
        sres = _txn(conn, lambda: scan.scan_and_register(conn, args.db, scope_id, root))
    else:
        sres = scan.scan_counts(root)
    bt = ", ".join(f"{k} {v}" for k, v in sorted(sres.by_type.items()))
    print(f"  {sres.total} artifacts ({bt})")
    print(f"  {len(sres.preload_files)} preload-seed file(s) (CLAUDE.md/AGENTS.md + @imports)")
    print("  -> preload seed imported at F=0 (preloading alone never counts).")

    # [3/5] MCP registration
    print("\n[3/5] Register K-ARC MCP server (`karc`: search/get)?")
    if "claude-code" in selected:
        cc_plan = runtimes.plan_claude_mcp(root, args.db)
        print(f"  Claude Code : {cc_plan['path']}"
              + (" (already configured)" if cc_plan["already_configured"] else ""))
        if apply:
            r = runtimes.apply_claude_mcp(root, args.db)
            print(f"  -> Claude Code .mcp.json "
                  f"{'written' if r['changed'] else r.get('reason','')}")
    if "codex" in selected:
        cx_plan = runtimes.plan_codex_mcp(root, args.db)
        hk_plan = runtimes.plan_codex_hooks(root, args.db)
        print(f"  Codex CLI   : {cx_plan['path']}"
              + (f" (CONFLICT: {cx_plan['reason']})" if cx_plan.get("conflict")
                 else " (already configured)" if cx_plan["already_configured"] else ""))
        print(f"  Codex hooks : {hk_plan['path']}"
              + (f" (CONFLICT: {hk_plan['reason']})" if hk_plan.get("conflict")
                 else " (already configured)" if hk_plan["already_configured"] else ""))
        if apply:
            for label, result in (
                ("Codex config", runtimes.apply_codex_mcp(root, args.db)),
                ("Codex hooks", runtimes.apply_codex_hooks(root, args.db)),
            ):
                print(f"  -> {label}: "
                      f"{'written' if result['changed'] else result.get('reason','')}")
    if "hermes" in selected:
        print("  Hermes      : ~/.hermes/config.yaml "
              "(add snippet manually — no stdlib YAML writer)")
        if apply:
            print("  -> Hermes snippet:")
            for line in runtimes.hermes_mcp_snippet(root, args.db).splitlines():
                print(f"       {line}")
    print("  (Same tools surface as mcp__karc__search / karc_search / mcp_karc_search"
          " per runtime — not a misconfiguration.)")

    # [4/5] managed AGENTS.md block (runtime-neutral)
    print("\n[4/5] Install managed AGENTS.md block (diff preview)?")
    ab = runtimes.plan_agents_block(root)
    print(f"  --- {ab['agents_path']} ---")
    for line in ab["block"].splitlines():
        print(f"  + {line}")
    if "claude-code" in selected and not ab["claude_has_import"]:
        print(f"  --- {ab['claude_path']} ---\n  + @AGENTS.md")
    hermes_found = "hermes" in selected and any(
        r.found and r.key == "hermes" for r in infos)
    if ab["hermes_md_present"] and hermes_found:
        print(f"  {_icon('warn')} Hermes loads only the FIRST match of "
              ".hermes.md > AGENTS.md > CLAUDE.md, and .hermes.md exists here.")
        print("     Hermes will NOT see the AGENTS.md block "
              "(re-run with --append-hermes to append it to .hermes.md).")
    if apply:
        res = runtimes.apply_agents_block(
            root, append_to_hermes=args.append_hermes,
            bridge_claude="claude-code" in selected,
        )
        for ch in res["changes"]:
            print(f"  -> {ch['path']}: {'written' if ch['changed'] else ch.get('reason','')}")
        if "claude-code" in selected:
            print(f"  {_icon('warn')} Claude Code shows a one-time approval dialog for the "
                  "@AGENTS.md import on your next session — expected, not an error.")
        if "codex" in selected:
            print(f"  {_icon('warn')} Review and trust the project hook definition in Codex "
                  "before relying on telemetry.")

    # [5/5] guard is deferred
    print("\n[5/5] Restrict native file reads to K-ARC paths? (guard)")
    print("  Deferred — recommended only after K-ARC has run reliably a while.")
    print("  Enable later with: karc guard enable")

    print("-" * 60)
    print("Setup complete." if apply else "Dry-run complete — no files were changed.")
    if not apply:
        print("Re-run with --yes to apply (and --append-hermes if you use Hermes here).")
    if conn is not None:
        conn.close()
    return 0


# ===========================================================================
# doctor
# ===========================================================================

def cmd_doctor(args) -> int:
    conn = _open(args)
    infos = runtimes.detect_runtimes(home=args.home)
    root = os.path.realpath(os.getcwd())
    if getattr(args, "root", None):
        root = os.path.realpath(args.root)
    cc_plan = runtimes.plan_claude_mcp(root, args.db)
    scope_id = _resolve_scope(conn, args)
    coverage = None
    health = {}
    if scope_id:
        ls = live_state.compute(conn, args.db, scope_id, budget=args.budget)
        coverage = ls.coverage
        rep = fsck.run(conn, args.db, limit=args.fsck_limit)
        health = {
            "cas_checked": rep.cas_checked,
            "cas_corrupt": len(rep.cas_corrupt),
            "missing_archive_content": len(rep.missing_archive_content),
            "duplicate_paths": len(rep.dup_active_paths),
            "orphan_blobs": len(rep.orphan_blobs),
            "stale_artifacts": _count_stale(conn, scope_id),
            "orphan_artifacts": _count_orphans(conn, scope_id),
        }
    codex = None
    if getattr(args, "runtime", None) in (None, "codex"):
        probe = runtimes.probe_codex()
        cx_plan = runtimes.plan_codex_mcp(root, args.db)
        hk_plan = runtimes.plan_codex_hooks(root, args.db)
        agents = runtimes.plan_agents_block(root)
        codex_events = conn.execute(
            "SELECT COUNT(*) FROM ingest_observations WHERE runtime='codex'"
        ).fetchone()[0]
        if probe.get("status") != "ok":
            status = "unsupported"
        elif cx_plan.get("conflict") or hk_plan.get("conflict"):
            status = "conflict"
        elif not cx_plan.get("already_configured") or not agents["agents_has_block"]:
            status = "degraded-floor"
        elif not hk_plan.get("already_configured") or codex_events == 0:
            status = "degraded-observation"
        else:
            status = "ok"
        codex = {
            "status": status, "capabilities": probe,
            "mcp_configured": bool(cx_plan.get("already_configured")),
            "mcp_conflict": cx_plan.get("reason") if cx_plan.get("conflict") else None,
            "hooks_configured": bool(hk_plan.get("already_configured")),
            "hooks_conflict": hk_plan.get("reason") if hk_plan.get("conflict") else None,
            "agents_managed_block": agents["agents_has_block"],
            "recent_observations": codex_events,
            "trust": "user-review-required-for-new-or-changed-project-hooks",
        }
    if args.json:
        return _print_json({
            "runtimes": [ri.as_dict() for ri in infos
                         if not getattr(args, "runtime", None)
                         or ri.key == args.runtime],
            "mcp_claude_configured": cc_plan["already_configured"],
            "codex": codex,
            "scope_id": scope_id,
            "telemetry_coverage": coverage,
            "knowledge_health": health,
        })
    print("K-ARC doctor")
    print("Runtimes:")
    for ri in infos:
        if getattr(args, "runtime", None) and ri.key != args.runtime:
            continue
        print(f"  {_icon('ok') if ri.found else _icon('no')} {ri.name:14s} "
              f"{ri.version or '(not found)'}")
    print(f"MCP (Claude Code .mcp.json): "
          f"{'registered' if cc_plan['already_configured'] else 'NOT registered'}")
    if codex is not None:
        print(f"Codex: {codex['status']} (MCP="
              f"{'ok' if codex['mcp_configured'] else 'missing'}, hooks="
              f"{'ok' if codex['hooks_configured'] else 'missing'}, "
              f"observations={codex['recent_observations']})")
    if scope_id is None:
        print("No K-ARC scope for this directory yet — run `karc init`.")
        conn.close()
        return 0
    cov = f"{coverage:.0%}" if coverage is not None else "n/a"
    print(f"Telemetry coverage: {cov}")
    print("Knowledge health:")
    for k, v in health.items():
        print(f"  {k:24s}: {v}")
    conn.close()
    return 0


def _count_stale(conn, scope_id) -> int:
    return conn.execute(
        """SELECT COUNT(*) FROM artifacts a JOIN versions v ON a.current_version_id=v.version_id
             WHERE a.scope_id=? AND (v.superseded_by_version_id IS NOT NULL
                   OR (v.valid_to IS NOT NULL AND v.valid_to < ?))""",
        (scope_id, utc_now_iso()),
    ).fetchone()[0]


def _count_orphans(conn, scope_id) -> int:
    # artifacts whose canonical file no longer exists on disk
    n = 0
    for (p,) in conn.execute(
        "SELECT canonical_path FROM artifacts WHERE scope_id=? AND canonical_path IS NOT NULL",
        (scope_id,),
    ).fetchall():
        if not os.path.isfile(p):
            n += 1
    return n


# ===========================================================================
# guard  (opt-in native-read restriction; K-ARC records intent + snippet)
# ===========================================================================

def cmd_guard(args) -> int:
    flag = Path(args.db).parent / "guard.json"
    if args.guard_command == "enable":
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(json.dumps({"enabled": True, "at": utc_now_iso()}) + "\n")
        print(f"{_icon('ok')} guard enabled (recorded in {flag}).")
        print("Add a native-read deny for K-ARC-managed paths in your runtime:")
        print("  Claude Code: permissions.deny  (managed settings)")
        print("  Codex CLI:   permissions.<name>.filesystem.<glob> = \"deny\"")
        print("  OpenCode:    permission.read.<glob> = \"deny\"")
        return 0
    if args.guard_command == "disable":
        if flag.exists():
            flag.write_text(json.dumps({"enabled": False, "at": utc_now_iso()}) + "\n")
        print(f"{_icon('ok')} guard disabled.")
        return 0
    print("usage: karc guard enable|disable", file=sys.stderr)
    return 2


# ===========================================================================
# status
# ===========================================================================

def cmd_status(args) -> int:
    conn = _open(args)
    scope_id = _resolve_scope(conn, args)
    if scope_id is None:
        if args.json:
            return _print_json({"scope": None, "state": "no_scope"})
        print("K-ARC — no scope for this directory yet.")
        print("  Run `karc init` to register this project's knowledge.")
        conn.close()
        return 0
    ls = live_queue.refresh(conn, args.db, scope_id, budget=args.budget)
    conn.commit() if conn.in_transaction else None
    n_pinned = conn.execute(
        "SELECT COUNT(*) FROM artifacts WHERE scope_id=? AND pinned=1", (scope_id,)
    ).fetchone()[0]
    n_artifacts = conn.execute(
        "SELECT COUNT(*) FROM artifacts WHERE scope_id=?", (scope_id,)
    ).fetchone()[0]
    reason = ls.empty_queue_reason()
    if args.json:
        out = {
            "scope": scope_id,
            "artifacts": n_artifacts,
            "pinned": n_pinned,
            "queue": len(ls.active),
            "queue_empty_reason": reason,
            "coverage": ls.coverage,
            "coverage_frozen": ls.coverage_frozen,
            "in_observation_window": ls.in_observation_window,
            "days_observed": round(ls.days_observed, 1),
            "tasks_observed": ls.n_tasks,
            "recommendations": [r.as_dict() for r in ls.active],
        }
        conn.close()
        return _print_json(out)

    print(f"K-ARC — {n_artifacts} artifacts · pinned {n_pinned} · "
          f"day {ls.days_observed:.0f} of ~14 · tasks {ls.n_tasks} of 50")
    cov = f"{ls.coverage:.0%}" if ls.coverage is not None else "n/a"
    print(f"Telemetry coverage: {cov}"
          + ("  (FROZEN — recommendations paused)" if ls.coverage_frozen else ""))
    if ls.active:
        print(f"\nReview queue: {len(ls.active)} pending")
        for r in ls.active:
            print(f"  {_icon(ACTION_ICON.get(r.action, 'warn'))} {r.action:22s} {r.name}")
        print("  -> karc review")
    else:
        print("\nReview queue: empty")
        if reason == "coverage_frozen":
            print(f"  {_icon('warn')} PAUSED: telemetry coverage {cov} below threshold — "
                  "demote/archive recs frozen. An empty queue does NOT mean all is well.")
            print("  -> karc doctor")
        elif reason == "observation_window":
            print("  Why empty: still in the initial observation window. Archive/split "
                  "recs need usage history; low usage alone is never evidence for removal.")
        else:
            print("  No pending recommendations (healthy).")
    # unprotected critical-pattern warning
    unprotected = [r for r in ls.active if r.action == "pin"]
    if n_pinned == 0 and unprotected:
        print(f"\n{_icon('warn')} 0 artifacts pinned. {len(unprotected)} file(s) match "
              "critical patterns. Review: karc list --suggest-pin")
    conn.close()
    return 0


# ===========================================================================
# review
# ===========================================================================

def cmd_review(args) -> int:
    conn = _open(args)
    scope_id = _resolve_scope(conn, args)
    if scope_id is None:
        print("No scope for this directory — run `karc init`.", file=sys.stderr)
        conn.close()
        return 2

    if args.review_command in ("approve", "reject"):
        rc = 0
        for rec_id in args.rec_ids:
            try:
                res = _txn(conn, lambda rid=rec_id: live_queue.decide(
                    conn, args.db, rid, "accept" if args.review_command == "approve" else "reject",
                    confirm_token=args.confirm,
                ))
                print(f"{_icon('ok')} {res['decision']}: {res['action']} ({rec_id})")
            except store.MutationError as e:
                print(f"{_icon('no')} {rec_id}: {e}", file=sys.stderr)
                rc = 1
        conn.close()
        return rc

    ls = live_queue.refresh(conn, args.db, scope_id, budget=args.budget)
    conn.commit() if conn.in_transaction else None
    rows = live_queue.pending_rows(conn, scope_id)
    if args.list or not sys.stdin.isatty():
        if args.json:
            conn.close()
            return _print_json(rows)
        if not rows:
            print("Queue empty.")
        for r in rows:
            print(f"  [{r['rec_id'][:8]}] {_icon(ACTION_ICON.get(r['action'],'warn'))} "
                  f"{r['action']:22s} {r['name'] or r['artifact_id']}")
            print(f"        {r['rationale']}  (risk: {r['risk']})")
        if rows:
            print("\nApprove: karc review approve <rec_id>   Reject: karc review reject <rec_id>")
        conn.close()
        return 0
    # interactive loop (TTY)
    return _review_interactive(conn, args, scope_id, rows)


def _review_interactive(conn, args, scope_id, rows) -> int:
    total = len(rows)
    accepted = rejected = skipped = 0
    for i, r in enumerate(rows, 1):
        icon = _icon(ACTION_ICON.get(r["action"], "warn"))
        print(f"\n[{i}/{total}] {icon} {r['action']}  {r['name'] or r['artifact_id']}")
        print(f"  {r['rationale']}  (risk: {r['risk']})")
        choice = input("  [a]ccept [r]eject [s]kip [q]uit > ").strip().lower()
        if choice == "q":
            break
        if choice == "a":
            try:
                _txn(conn, lambda: live_queue.decide(conn, args.db, r["rec_id"], "accept"))
                accepted += 1
                print(f"  {_icon('ok')} accepted.")
            except store.MutationError as e:
                print(f"  {_icon('no')} {e}")
        elif choice == "r":
            _txn(conn, lambda: live_queue.decide(conn, args.db, r["rec_id"], "reject"))
            rejected += 1
        else:
            skipped += 1
    print(f"\nSession: {accepted} accepted, {rejected} rejected, {skipped} skipped.")
    conn.close()
    return 0


# ===========================================================================
# why
# ===========================================================================

def cmd_why(args) -> int:
    conn = _open(args)
    scope_id = _resolve_scope(conn, args)
    if scope_id is None:
        print("No scope for this directory.", file=sys.stderr)
        conn.close()
        return 2
    art = store.resolve_artifact(conn, args.artifact, scope_id)
    if art is None:
        print(f"artifact not found: {args.artifact}", file=sys.stderr)
        conn.close()
        return 1
    ls = live_state.compute(conn, args.db, scope_id, budget=args.budget)
    rep = live_state.artifact_report(ls, conn, art.artifact_id)
    if args.json:
        conn.close()
        return _print_json(rep)
    print(f"{rep['name']}  [{rep.get('list') or 'untracked'}]")
    print(f"  lifecycle: {art.lifecycle_state}  size: {rep['size_tokens']} tok  "
          f"pinned: {rep['pinned']}  critical: {rep['critical']}")
    if "f_score" in rep:
        print("  recent use   :", rep["f_score"], f"(last task: {rep['last_ref_task']})")
        print(f"  validation   : validated {rep['n_val']} / corrected {rep['n_corr']} "
              f"-> q={rep['q']}")
        print(f"  validity     : {rep['validity']}")
        if rep.get("blocked_reason"):
            print(f"  blocked      : {rep['blocked_reason']}")
    if rep.get("recent_transitions"):
        print("  recent transitions:")
        for t in rep["recent_transitions"]:
            print(f"    {t['from']} -> {t['to']}  {t['rule_id']}")
    conn.close()
    return 0


# ===========================================================================
# pin / unpin
# ===========================================================================

def cmd_pin(args) -> int:
    conn = _open(args)
    scope_id = _resolve_scope(conn, args)
    matched = store.match_artifacts(conn, args.pattern, scope_id)
    if not matched:
        print(f"no artifacts match {args.pattern!r}", file=sys.stderr)
        conn.close()
        return 1
    def _do():
        for a in matched:
            store.pin(conn, a.artifact_id, critical=args.critical)
        return None
    _txn(conn, _do)
    print(f"{_icon('protect')} pinned {len(matched)} artifact(s):")
    for a in matched:
        print(f"  {a.logical_name}")
    conn.close()
    return 0


def cmd_unpin(args) -> int:
    conn = _open(args)
    scope_id = _resolve_scope(conn, args)
    matched = store.match_artifacts(conn, args.pattern, scope_id)
    if not matched:
        print(f"no artifacts match {args.pattern!r}", file=sys.stderr)
        conn.close()
        return 1
    rc = 0
    for a in matched:
        try:
            _txn(conn, lambda a=a: store.unpin(conn, a.artifact_id, confirm_token=args.confirm))
            print(f"{_icon('ok')} unpinned {a.logical_name}")
        except store.UnpinConfirmationRequired as e:
            print(f"{_icon('warn')} {a.logical_name}: {e}", file=sys.stderr)
            rc = 1
    conn.close()
    return rc


# ===========================================================================
# archive / restore
# ===========================================================================

def cmd_archive(args) -> int:
    conn = _open(args)
    scope_id = _resolve_scope(conn, args)
    art = store.resolve_artifact(conn, args.artifact, scope_id)
    if art is None:
        print(f"artifact not found: {args.artifact}", file=sys.stderr)
        conn.close()
        return 1
    res = _txn(conn, lambda: store.archive(conn, args.db, art.artifact_id))
    print(f"{_icon('lifecycle')} archived {art.logical_name}"
          + (f" (snapshot saved, restorable)" if res.detail.get("restorable") else ""))
    print(f"  restore with: karc restore {art.logical_name}")
    conn.close()
    return 0


def cmd_restore(args) -> int:
    conn = _open(args)
    scope_id = _resolve_scope(conn, args)
    art = store.resolve_artifact(conn, args.artifact, scope_id)
    if art is None:
        print(f"artifact not found: {args.artifact}", file=sys.stderr)
        conn.close()
        return 1
    deps = store.dependents(conn, art.artifact_id)
    if deps:
        names = ", ".join(d["src_name"] for d in deps)
        print(f"{len(deps)} dependent reference(s) will be reconnected: {names}")
    res = _txn(conn, lambda: store.restore(conn, args.db, art.artifact_id))
    print(f"{_icon('ok')} restored {art.logical_name}"
          + (" (content rewritten from snapshot)" if res.detail.get("wrote_file") else ""))
    conn.close()
    return 0


# ===========================================================================
# list
# ===========================================================================

def cmd_list(args) -> int:
    conn = _open(args)
    scope_id = _resolve_scope(conn, args)
    if scope_id is None:
        print("No scope for this directory.", file=sys.stderr)
        conn.close()
        return 2
    ls = live_state.compute(conn, args.db, scope_id, budget=args.budget)
    policy = ls.policy
    rows = []
    for art_id, m in ls.registry.items():
        lst = policy.list_of(art_id) if hasattr(policy, "list_of") else None
        if art_id in getattr(policy, "pinned_ids", set()):
            lst = "PINNED"
        cold = art_id in getattr(policy, "cold_candidates", set())
        art = store.get_artifact(conn, art_id)
        rows.append({
            "artifact_id": art_id, "name": m.path or art_id, "list": lst,
            "tier": lst, "pinned": bool(getattr(m, "pinned", False)),
            "cold": cold, "size_tokens": m.size_tok,
            "lifecycle": art.lifecycle_state if art else None,
            "suggest_pin": (live_state._pin_pattern_hit(m.path or "") is not None
                            and not getattr(m, "pinned", False)),
            "unmonitored": lst is None and not cold,
        })
    if args.pinned:
        rows = [r for r in rows if r["pinned"]]
    if args.cold:
        rows = [r for r in rows if r["cold"] or r["lifecycle"] in ("cold", "archived")]
    if args.tier:
        rows = [r for r in rows if r["tier"] == args.tier.upper()]
    if args.suggest_pin:
        rows = [r for r in rows if r["suggest_pin"]]
    if args.unmonitored:
        rows = [r for r in rows if r["unmonitored"]]
    rows.sort(key=lambda r: (r["tier"] or "~", r["name"]))
    if args.json:
        conn.close()
        return _print_json(rows)
    if not rows:
        print("(no matching artifacts)")
    for r in rows:
        badge = _icon("protect") if r["pinned"] else (" " if _ascii_mode() else "  ")
        tag = r["tier"] or ("COLD" if r["cold"] else "-")
        print(f"  {badge} {tag:8s} {r['name']}  ({r['size_tokens']} tok)")
    conn.close()
    return 0


# ===========================================================================
# log
# ===========================================================================

def cmd_log(args) -> int:
    conn = _open(args)
    scope_id = _resolve_scope(conn, args)
    if args.regret:
        rows = conn.execute(
            "SELECT at, action, object_id, details FROM audit_log "
            "WHERE action='restore' AND rollback_of IS NOT NULL ORDER BY at DESC LIMIT ?",
            (args.limit,),
        ).fetchall()
        if args.json:
            conn.close()
            return _print_json([{"at": r[0], "action": r[1], "object_id": r[2]} for r in rows])
        print("Archive regret (restores of archived artifacts):")
        for at, action, obj, details in rows:
            print(f"  {at}  restore {obj}")
        if not rows:
            print("  none")
        conn.close()
        return 0
    q = "SELECT at, actor_type, action, object_type, object_id FROM audit_log"
    params: list = []
    if args.since:
        q += " WHERE at >= ?"
        params.append(args.since)
    q += " ORDER BY at DESC LIMIT ?"
    params.append(args.limit)
    rows = conn.execute(q, params).fetchall()
    if args.json:
        conn.close()
        return _print_json([
            {"at": r[0], "actor_type": r[1], "action": r[2],
             "object_type": r[3], "object_id": r[4]} for r in rows
        ])
    if not rows:
        print("(no audit entries)")
    for at, actor_type, action, obj_type, obj in rows:
        print(f"  {at}  {actor_type:6s} {action:20s} {obj_type}:{obj}")
    conn.close()
    return 0


# ---------------------------------------------------------------------------
# argparse registration
# ---------------------------------------------------------------------------

def _db(p):
    p.add_argument("--db", default=str(connection.DEFAULT_DB_PATH))


def _scope(p):
    p.add_argument("--scope", default=None, help="scope_id (default: cwd match)")


def _budget(p):
    p.add_argument("--budget", default=live_state.DEFAULT_BUDGET,
                   help="token budget: absolute ('8000') or corpus share ('25pct')")


def register(sub) -> None:
    p = sub.add_parser("init", help="detect runtimes, scan, register MCP + AGENTS.md")
    _db(p)
    p.add_argument("--root", default=None, help="project root (default: cwd)")
    p.add_argument("--home", default=None, help="HOME override (testing)")
    p.add_argument("--runtime", default=None, help="limit to one runtime")
    p.add_argument("--dry-run", action="store_true", help="show plan, change nothing")
    p.add_argument("--yes", action="store_true", help="apply changes")
    p.add_argument("--append-hermes", action="store_true",
                   help="append the managed block to an existing .hermes.md")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("doctor", help="cross-runtime + coverage + knowledge health")
    _db(p); _scope(p); _budget(p)
    p.add_argument("--home", default=None)
    p.add_argument("--root", default=None, help="project root (default: cwd)")
    p.add_argument("--runtime", default=None, help="limit diagnosis to one runtime")
    p.add_argument("--json", action="store_true")
    p.add_argument("--fsck-limit", type=int, default=None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("guard", help="opt-in native-read restriction")
    _db(p)
    p.add_argument("guard_command", choices=["enable", "disable"])
    p.set_defaults(func=cmd_guard)

    p = sub.add_parser("status", help="queue + gates + tier usage")
    _db(p); _scope(p); _budget(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("review", help="approval loop (5 recommendation types)")
    _db(p); _scope(p); _budget(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--list", action="store_true", help="print queue, don't prompt")
    p.add_argument("--confirm", default=None, help="confirmation token (critical unpin)")
    rsub = p.add_subparsers(dest="review_command")
    for verb in ("approve", "reject"):
        rp = rsub.add_parser(verb)
        _db(rp); _scope(rp); _budget(rp)
        rp.add_argument("rec_ids", nargs="+")
        rp.add_argument("--confirm", default=None)
        rp.add_argument("--json", action="store_true")
        rp.add_argument("--list", action="store_true")
        rp.set_defaults(func=cmd_review, review_command=verb)
    p.set_defaults(func=cmd_review, review_command=None)

    p = sub.add_parser("why", help="6-component breakdown + transitions")
    _db(p); _scope(p); _budget(p)
    p.add_argument("artifact")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_why)

    p = sub.add_parser("pin", help="protect artifact(s) (path or glob)")
    _db(p); _scope(p)
    p.add_argument("pattern")
    p.add_argument("--critical", action="store_true", help="also tag as critical")
    p.set_defaults(func=cmd_pin)

    p = sub.add_parser("unpin", help="remove protection (critical needs filename)")
    _db(p); _scope(p)
    p.add_argument("pattern")
    p.add_argument("--confirm", default=None, help="filename confirmation for critical")
    p.set_defaults(func=cmd_unpin)

    p = sub.add_parser("archive", help="archive an artifact (snapshot, restorable)")
    _db(p); _scope(p)
    p.add_argument("artifact")
    p.set_defaults(func=cmd_archive)

    p = sub.add_parser("restore", help="restore an archived artifact")
    _db(p); _scope(p)
    p.add_argument("artifact")
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("list", help="working-set view")
    _db(p); _scope(p); _budget(p)
    p.add_argument("--tier", default=None, help="T1|T2|B1|B2|OVERSIZE|PINNED")
    p.add_argument("--pinned", action="store_true")
    p.add_argument("--cold", action="store_true")
    p.add_argument("--unmonitored", action="store_true")
    p.add_argument("--suggest-pin", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("log", help="audit log / transitions / regret")
    _db(p); _scope(p)
    p.add_argument("--since", default=None, help="ISO8601 lower bound")
    p.add_argument("--regret", action="store_true", help="archive regret (restores)")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_log)
