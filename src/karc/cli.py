"""karc CLI — M0: db migrate / ingest claude-code / analyze.
M1: dev replay (정책 replay runner, FR-R1~R9).
M3: product surface — mcp serve, hook, fsck, and the UX §7 commands
(init/doctor/guard/status/review/why/pin/unpin/archive/restore/list/log),
registered via ``cli_product.register``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from karc.analysis import locality, preload
from karc.db import connection
from karc.ingest import pipeline


def _add_db_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--db",
        default=str(connection.DEFAULT_DB_PATH),
        help="SQLite DB path (default: .karc/karc.db)",
    )


def cmd_db_migrate(args: argparse.Namespace) -> int:
    conn = connection.connect(args.db)
    applied = connection.migrate(conn, db_path=args.db)
    if applied:
        for version, name in applied:
            print(f"applied migration {version:03d} {name}")
    else:
        print("no pending migrations")
    ver = conn.execute("PRAGMA user_version").fetchone()[0]
    print(f"schema user_version = {ver}")
    return 0


def cmd_ingest_claude_code(args: argparse.Namespace) -> int:
    conn = connection.connect(args.db)
    connection.migrate(conn, db_path=args.db)
    stats = pipeline.ingest_projects_root(conn, args.projects_dir)
    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_ingest_codex(args: argparse.Namespace) -> int:
    from karc.ingest import codex as codex_ingest

    version = args.codex_version
    if version is None:
        import subprocess
        proc = subprocess.run(["codex", "--version"], capture_output=True,
                              text=True, timeout=15)
        version = (proc.stdout or proc.stderr).strip().splitlines()[-1]
    conn = None
    if not args.dry_run:
        conn = connection.connect(args.db)
        connection.migrate(conn, db_path=args.db)
        conn.execute("BEGIN IMMEDIATE")
    try:
        stats = codex_ingest.ingest_sessions_dir(
            conn, args.sessions_dir, runtime_version=version,
            schema_family=args.schema_family, since=args.since,
            dry_run=args.dry_run,
        )
        if conn is not None:
            conn.execute("COMMIT")
    except BaseException:
        if conn is not None:
            conn.execute("ROLLBACK")
        raise
    finally:
        if conn is not None:
            conn.close()
    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_analyze_locality(args: argparse.Namespace) -> int:
    conn = connection.connect(args.db)
    result = locality.analyze_all(conn)
    out = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out + "\n", encoding="utf-8")
        print(f"written: {args.out}")
    else:
        print(out)
    return 0


def cmd_analyze_preload_waste(args: argparse.Namespace) -> int:
    conn = connection.connect(args.db)
    result = preload.analyze_all(conn, active_cutoff=args.active_cutoff)
    session_rows = result.pop("_session_rows")
    item_rows = result.pop("_item_rows")
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "preload_items.jsonl", "w", encoding="utf-8") as f:
            for row in item_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(out_dir / "session_summary.jsonl", "w", encoding="utf-8") as f:
            for row in session_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (out_dir / "aggregate.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"written: {out_dir}/preload_items.jsonl ({len(item_rows)} items), "
              f"session_summary.jsonl ({len(session_rows)} sessions), aggregate.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parse_multi(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_seeds(value: str) -> list[int]:
    """'0-4' → [0..4]; '1,3,7' → [1,3,7]; '5' → [5]."""
    seeds: list[int] = []
    for part in _parse_multi(value):
        if "-" in part and not part.startswith("-"):
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _parse_config_overrides(pairs: list[str] | None, config_json: str | None) -> dict:
    out: dict = {}
    if config_json:
        out.update(json.loads(Path(config_json).read_text(encoding="utf-8")))
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--config expects key=value, got {pair!r}")
        k, v = pair.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def cmd_dev_replay(args: argparse.Namespace) -> int:
    from karc.replay import runner as replay_runner
    from karc.replay import trace as replay_trace

    overrides = _parse_config_overrides(args.config, args.config_json)

    if args.real_trace:
        if args.list_scopes:
            for s in replay_trace.list_scopes(args.db):
                print(json.dumps(s, ensure_ascii=False))
            return 0
        if not args.scope:
            print("--real-trace requires --scope (use --list-scopes)", file=sys.stderr)
            return 2
        rc = 0
        for policy in _parse_multi(args.policy):
            for budget in _parse_multi(args.budget):
                res = replay_trace.replay_trace(
                    args.db,
                    args.scope,
                    policy_name=policy,
                    budget=budget,
                    config_overrides=overrides,
                    debug_invariants=not args.no_invariants,
                )
                print(json.dumps(res.summary, ensure_ascii=False))
                if not res.ok:
                    rc = 1
        return rc

    specs = [
        replay_runner.RunSpec(
            policy=policy,
            family=family,
            seed=seed,
            budget=budget,
            alpha_obs=args.alpha_obs,
            drop_rate=args.drop_rate,
            shuffle=args.shuffle,
            task_id_missing_rate=args.task_id_missing,
            inject_outcomes=not args.no_outcomes,
            debug_invariants=not args.no_invariants,
            config_overrides=overrides,
        )
        for family in _parse_multi(args.workload)
        for seed in _parse_seeds(args.seeds)
        for budget in _parse_multi(args.budget)
        for policy in _parse_multi(args.policy)
    ]
    results = replay_runner.run_batch(specs, out_dir=args.out_dir)
    rc = 0
    for res in results:
        line = dict(res.summary)
        if args.compact:  # one-line summary for eyeballing batches
            line = {
                "run_id": res.summary["run_id"],
                "coverage": res.summary["metrics"]["coverage_mean"],
                "stale": res.summary["metrics"]["stale_exposure_mean"],
                "tw_hit": res.summary["metrics"]["token_weighted_hit_rate"],
                "ghost_hits": res.summary["metrics"]["ghost_hits"],
                "crit_auto_cold": res.summary["metrics"]["critical_auto_cold"],
                "violations": res.summary["invariant_violations"],
                "state_hash": res.summary["state_hash"][:12],
            }
        print(json.dumps(line, ensure_ascii=False))
        if not res.ok:
            rc = 1
            print(
                f"INVARIANT VIOLATION in {res.summary['run_id']}: "
                f"{res.violation['rule']}: {res.violation['detail']}",
                file=sys.stderr,
            )
    if args.out_dir:
        print(f"written: {args.out_dir}/runs.jsonl (+ per-run task JSONL)", file=sys.stderr)
    return rc


def cmd_dev_benchmark(args: argparse.Namespace) -> int:
    """LLM benchmark harness (FR-H1~H9). Default arm = closed-book (E0-3).

    Real ``claude -p`` runs happen ONLY with ``--execute``; without it the
    command materializes nothing and just reports the schedule size + CLI
    preflight (so wiring can be inspected without any API call)."""
    from karc.bench.driver import ClaudeCliDriver
    from karc.bench.harness import CLOSED_BOOK_ARM, Arm, BenchSpec, Harness

    fixture = Path(args.fixture)
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    tasks = json.loads((fixture / "tasks.json").read_text(encoding="utf-8"))
    if args.main_only:
        tasks = [t for t in tasks if t.get("tier") == "main"]
    if args.limit:
        tasks = tasks[: args.limit]

    mode_arms = {
        "closed-book": [CLOSED_BOOK_ARM],
        "static-full": [Arm("static-full", "static-full", max_turns=args.max_turns)],
    }
    arms = mode_arms.get(args.arm)
    if arms is None:
        print(f"--arm {args.arm!r}: only closed-book/static-full are CLI-wired "
              "(working-set arms need a derived working set — Stage 2 step)",
              file=sys.stderr)
        return 2

    spec = BenchSpec(
        experiment_id=args.experiment, arms=arms, tasks=tasks, reps=args.reps,
        seed=args.seed, model=args.model, fixture_root=fixture,
        concurrency=args.concurrency, timeout_s=args.timeout,
    )
    driver = ClaudeCliDriver()
    preflight = driver.preflight()
    print(json.dumps({"schedule_units": len(tasks) * len(arms) * args.reps,
                      "arms": [a.name for a in arms], "n_tasks": len(tasks),
                      "reps": args.reps, "model": args.model,
                      "cli_preflight": preflight}, ensure_ascii=False, indent=2))
    if not args.execute:
        print("dry run (no --execute): no LLM calls made", file=sys.stderr)
        return 0
    if not preflight.get("available"):
        print("claude CLI not available; aborting --execute", file=sys.stderr)
        return 2
    harness = Harness(spec, driver, manifest)
    outcomes = harness.run(out_dir=args.out_dir, preflight=preflight)
    print(json.dumps({"runs": len(outcomes),
                      "passed": sum(1 for o in outcomes if o.passed),
                      "leakage": sum(1 for o in outcomes if o.leakage)},
                     ensure_ascii=False))
    return 0


def cmd_mcp_serve(args: argparse.Namespace) -> int:
    from karc.mcp.server import MCPServer

    server = MCPServer(args.db, args.root, runtime=args.runtime)
    try:
        server.serve()
    finally:
        server.close()
    return 0


def cmd_hook(args: argparse.Namespace) -> int:
    if args.runtime == "codex":
        from karc.hooks import codex
        return codex.main(["--db", args.db])
    from karc.hooks import adapter
    return adapter.main(["--db", args.db])


def cmd_fsck(args: argparse.Namespace) -> int:
    from karc.snapshot import fsck

    conn = connection.connect(args.db)
    connection.migrate(conn, db_path=args.db)
    report = fsck.run(conn, args.db, limit=args.limit)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    from karc import cli_product

    parser = argparse.ArgumentParser(prog="karc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_db = sub.add_parser("db", help="database management")
    db_sub = p_db.add_subparsers(dest="db_command", required=True)
    p_migrate = db_sub.add_parser("migrate", help="apply pending migrations")
    _add_db_arg(p_migrate)
    p_migrate.set_defaults(func=cmd_db_migrate)

    p_ingest = sub.add_parser("ingest", help="ingest runtime telemetry")
    ing_sub = p_ingest.add_subparsers(dest="ingest_command", required=True)
    p_cc = ing_sub.add_parser("claude-code", help="batch-parse Claude Code transcripts")
    _add_db_arg(p_cc)
    p_cc.add_argument(
        "--projects-dir",
        default=os.path.expanduser("~/.claude/projects"),
        help="Claude Code projects root (default: ~/.claude/projects)",
    )
    p_cc.set_defaults(func=cmd_ingest_claude_code)

    p_cx = ing_sub.add_parser("codex", help="batch-parse recognized Codex JSONL")
    _add_db_arg(p_cx)
    p_cx.add_argument("--sessions-dir", required=True,
                      help="directory containing captured Codex JSONL files")
    p_cx.add_argument("--since", default=None, help="ISO8601 mtime lower bound")
    p_cx.add_argument("--dry-run", action="store_true", help="parse without DB writes")
    p_cx.add_argument("--codex-version", default=None,
                      help="captured codex --version (default: probe current CLI)")
    p_cx.add_argument("--schema-family", default="codex-exec-jsonl-0.144.x-v1")
    p_cx.set_defaults(func=cmd_ingest_codex)

    p_an = sub.add_parser("analyze", help="analysis commands")
    an_sub = p_an.add_subparsers(dest="analyze_command", required=True)
    p_loc = an_sub.add_parser("locality", help="E0-1 locality metrics")
    _add_db_arg(p_loc)
    p_loc.add_argument("--out", default=None, help="write JSON to this path")
    p_loc.set_defaults(func=cmd_analyze_locality)

    p_pre = an_sub.add_parser("preload-waste", help="E0-2 preload waste metrics")
    _add_db_arg(p_pre)
    p_pre.add_argument("--out-dir", default=None, help="write raw JSONL/JSON outputs here")
    p_pre.add_argument(
        "--active-cutoff",
        default=None,
        help="ISO8601 UTC — exclude sessions whose transcript files were "
             "modified at/after this instant (right-censored/active sessions)",
    )
    p_pre.set_defaults(func=cmd_analyze_preload_waste)

    p_dev = sub.add_parser("dev", help="development/experiment tooling (M1)")
    dev_sub = p_dev.add_subparsers(dest="dev_command", required=True)
    p_rp = dev_sub.add_parser(
        "replay",
        help="trace replay runner (FR-R1~R9); 다중 run 배치는 콤마 목록/seed 범위",
    )
    p_rp.add_argument(
        "--policy",
        default="karc",
        help="comma list: karc,karc-no-outcome,karc-no-validity-gate,"
        "karc-preload-as-hit,arc,static,fifo,lru,lfu",
    )
    p_rp.add_argument(
        "--workload", default="f-a", help="comma list of families: f-a,f-b,f-c"
    )
    p_rp.add_argument(
        "--seeds", "--seed", dest="seeds", default="0",
        help="seed list/range: '0', '0-19', '1,3,7'",
    )
    p_rp.add_argument(
        "--budget", default="10pct",
        help="comma list; absolute tokens ('8000') or corpus share ('10pct')",
    )
    p_rp.add_argument("--alpha-obs", type=float, default=1.0,
                      help="outcome 관측률 α_obs (E1-5)")
    p_rp.add_argument("--drop-rate", type=float, default=0.0,
                      help="이벤트 drop률 0~0.5 (V9)")
    p_rp.add_argument("--shuffle", action="store_true", help="task 내 순서 섞기")
    p_rp.add_argument("--task-id-missing", type=float, default=0.0,
                      help="task_id 누락률 (Q3)")
    p_rp.add_argument("--no-outcomes", action="store_true",
                      help="outcome 주입기 비활성화")
    p_rp.add_argument("--no-invariants", action="store_true",
                      help="invariant assertion 비활성화 (성능용; 기본 on)")
    p_rp.add_argument("--config", action="append",
                      help="PolicyConfig override key=value (반복 가능)")
    p_rp.add_argument("--config-json", default=None,
                      help="PolicyConfig override JSON 파일")
    p_rp.add_argument("--out-dir", default=None,
                      help="runs.jsonl + per-run task JSONL 출력 디렉터리")
    p_rp.add_argument("--compact", action="store_true",
                      help="stdout에 요약 필드만 출력")
    p_rp.add_argument("--real-trace", action="store_true",
                      help="합성 workload 대신 M0 DB 이벤트 fold (FR-R8)")
    p_rp.add_argument("--db", default=str(connection.DEFAULT_DB_PATH),
                      help="--real-trace: SQLite DB 경로")
    p_rp.add_argument("--scope", default=None, help="--real-trace: scope_id")
    p_rp.add_argument("--list-scopes", action="store_true",
                      help="--real-trace: scope 목록 출력")
    p_rp.set_defaults(func=cmd_dev_replay)

    p_bm = dev_sub.add_parser(
        "benchmark",
        help="LLM benchmark harness (FR-H1~H9); default arm closed-book (E0-3)",
    )
    p_bm.add_argument("--experiment", default="E0-3", help="experiment id (report/run-id prefix)")
    p_bm.add_argument("--arm", default="closed-book",
                      choices=["closed-book", "static-full"],
                      help="knowledge-supply arm (CLI-wired subset)")
    p_bm.add_argument("--fixture", default="fixture/stage2", help="fixture root")
    p_bm.add_argument("--model", default="claude-sonnet-4-5",
                      help="pinned dated model ID (A-2/R-17; verified per run)")
    p_bm.add_argument("--reps", type=int, default=3, help="repetitions per task (E0-3: 3)")
    p_bm.add_argument("--seed", type=int, default=0, help="schedule shuffle seed")
    p_bm.add_argument("--concurrency", type=int, default=1, help="parallel workers N")
    p_bm.add_argument("--max-turns", type=int, default=30, help="agent max turns (non-closed-book)")
    p_bm.add_argument("--timeout", type=int, default=300, help="per-run driver timeout (s)")
    p_bm.add_argument("--limit", type=int, default=None, help="first N tasks only")
    p_bm.add_argument("--main-only", action="store_true", help="tier=main tasks only")
    p_bm.add_argument("--out-dir", default=None, help="R-16 output dir (config.yaml + raw/)")
    p_bm.add_argument("--execute", action="store_true",
                      help="actually invoke claude -p (default: dry run, no API calls)")
    p_bm.set_defaults(func=cmd_dev_benchmark)

    # ---- M3 product surface ----
    p_mcp = sub.add_parser("mcp", help="MCP server")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_command", required=True)
    p_serve = mcp_sub.add_parser("serve", help="stdio JSON-RPC karc MCP server")
    _add_db_arg(p_serve)
    p_serve.add_argument("--root", default=os.getcwd(), help="project root (scope)")
    p_serve.add_argument("--runtime", default="mcp")
    p_serve.set_defaults(func=cmd_mcp_serve)

    p_hook = sub.add_parser("hook", help="runtime hook adapter (reads stdin)")
    _add_db_arg(p_hook)
    p_hook.add_argument("--runtime", choices=["claude-code", "codex"],
                        default="claude-code")
    p_hook.set_defaults(func=cmd_hook)

    p_fsck = sub.add_parser("fsck", help="snapshot store / registry integrity check")
    _add_db_arg(p_fsck)
    p_fsck.add_argument("--limit", type=int, default=None, help="CAS objects to re-hash")
    p_fsck.set_defaults(func=cmd_fsck)

    cli_product.register(sub)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
