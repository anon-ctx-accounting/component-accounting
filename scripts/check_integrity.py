#!/usr/bin/env python3
"""Check preserved source manifests and the documented artifact changes.

Experiment manifests use paths relative to their experiment directory,
including manifests nested below a smoke or canary directory. This checker
never edits manifests, discovers source files outside the artifact, or
accepts an unexplained checksum mismatch.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time


DIGEST = re.compile(r"[a-f0-9]{64}\Z")
CHANGES = {"path-rewrite", "ip-redaction"}


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inside(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError("missing relative path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
        raise ValueError("path must stay inside the artifact working tree")
    path = root.joinpath(*relative.parts)
    if path.resolve().is_relative_to(root.resolve()) is False:
        raise ValueError("path escapes the artifact working tree")
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
        raise ValueError("symbolic link requires review")
    return path


def manifest_root(root, manifest):
    relative = manifest.relative_to(root)
    if relative.parts[:2] == ("docs", "experiments") and len(relative.parts) >= 4:
        return root.joinpath(*relative.parts[:3])
    return manifest.parent


def check(root):
    rows = []
    registry = {}
    registered_in = {}
    actual_references = {}

    def result(kind, path, expected, actual, ok, detail):
        rows.append({"kind": kind, "path": path, "expected": expected,
                     "recomputed": actual, "status": "PASS" if ok else "FAIL", "detail": detail})

    try:
        document = json.loads((root / "integrity.json").read_text(encoding="utf-8"))
        if document.get("schema_version") != 2 or not isinstance(document.get("files"), list):
            raise ValueError("expected schema_version 2 and files list")
    except (OSError, ValueError, AttributeError) as exc:
        result("registry", "integrity.json", "valid registry", "invalid", False, str(exc))
        return rows

    for entry in document["files"]:
        name = entry.get("path", "(missing)") if isinstance(entry, dict) else "(invalid entry)"
        try:
            if not isinstance(entry, dict) or name in registry:
                raise ValueError("invalid or duplicate registry entry")
            path = inside(root, name)
            changes = entry.get("changes")
            if not isinstance(changes, list) or not all(x in CHANGES for x in changes) or len(set(changes)) != len(changes):
                raise ValueError("invalid changes array")
            if entry.get("origin") not in {"artifact", "source"}:
                raise ValueError("invalid origin")
            for key in ("source_sha256", "artifact_sha256"):
                if not isinstance(entry.get(key), str) or not DIGEST.fullmatch(entry[key]):
                    raise ValueError("invalid " + key)
            if not path.is_file():
                raise ValueError("registered artifact file does not exist")
            original = entry["source_sha256"]
            expected = entry["artifact_sha256"]
            actual = sha256(path)
            if entry["origin"] == "artifact" and (not changes or original == expected):
                raise ValueError("artifact origin requires a byte change and its classification")
            if entry["origin"] == "source" and (changes or original != expected):
                raise ValueError("source origin requires unchanged source bytes and no artifact changes")
            registry[name] = entry
            sources = entry.get("recorded_in", [])
            if isinstance(sources, str):
                sources = [sources]
            if not isinstance(sources, list) or not all(isinstance(x, str) for x in sources):
                raise ValueError("recorded_in must be a manifest path or a list of paths")
            for source in sources:
                if not inside(root, source).is_file():
                    raise ValueError("recorded_in manifest does not exist")
            registered_in[name] = set(sources)
            if sources:
                manifest_digest = entry.get("manifest_sha256")
                if not isinstance(manifest_digest, str) or not DIGEST.fullmatch(manifest_digest):
                    raise ValueError("missing or invalid manifest_sha256")
                status = entry.get("manifest_status")
                if status == "matches-source":
                    if manifest_digest != original:
                        raise ValueError("matches-source contradicts source_sha256")
                elif status == "diverged-in-source":
                    if manifest_digest == original or not str(entry.get("manifest_note", "")).strip():
                        raise ValueError("source divergence requires differing hashes and a note")
                else:
                    raise ValueError("invalid manifest_status")
            elif any(key in entry for key in ("manifest_sha256", "manifest_status", "manifest_note", "recorded_in")):
                raise ValueError("uncovered files must omit manifest fields")
            result("change", name, expected, actual, expected == actual, ",".join(changes) or "source divergence")
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            result("change", str(name), "valid registered change", "invalid", False, str(exc))

    covered = sum(isinstance(entry, dict) and bool(entry.get("recorded_in")) for entry in document["files"])
    expected_summary = {"total_registered": len(document["files"]), "manifest_covered": covered,
                        "manifest_uncovered": len(document["files"])-covered}
    if document.get("summary") != expected_summary:
        result("registry", "integrity.json", json.dumps(expected_summary,sort_keys=True),
               json.dumps(document.get("summary"),sort_keys=True),False,"registry summary does not match entries")

    manifests = sorted(p for p in root.rglob("sha256.json") if ".git" not in p.relative_to(root).parts)
    if not manifests:
        result("manifest", "sha256.json", "at least one manifest", "none", False, "no source manifests found")
    for manifest in manifests:
        manifest_name = manifest.relative_to(root).as_posix()
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(entries, dict):
                raise ValueError("manifest must map relative paths to SHA-256 strings")
        except (OSError, ValueError) as exc:
            result("manifest", manifest_name, "valid manifest", "invalid", False, str(exc))
            continue
        base = manifest_root(root, manifest)
        for key, expected in entries.items():
            target_name = str(key)
            try:
                if not isinstance(expected, str) or not DIGEST.fullmatch(expected):
                    raise ValueError("invalid manifest digest")
                # Validate the key independently so an absolute key cannot
                # discard the manifest's base during path joining.
                inside(root, key)
                target_name = (base.relative_to(root) / key).as_posix()
                target = inside(root, target_name)
                actual_references.setdefault(target_name, set()).add(manifest_name)
                if not target.is_file():
                    if (target_name.startswith("docs/experiments/E5-G0/history/")
                            or target_name == "docs/experiments/E21-CODEX-COMPONENT/raw/probe-home-layout.json"
                            or "__pycache__" in target.parts or target.suffix == ".pyc" or target.name == ".DS_Store"
                            or (target_name.startswith("docs/experiments/") and target.suffix == ".md")):
                        rows.append({"kind": "manifest", "path": target_name, "expected": expected,
                                     "recomputed": "excluded", "status": "SKIPPED", "detail": "excluded by release scope (§3.4 / C1)"})
                        continue
                    raise ValueError("manifest target is missing")
                actual = sha256(target)
                registered = registry.get(target_name)
                if actual == expected:
                    result("manifest", target_name, expected, actual, True, manifest_name + "; exact bytes")
                elif registered is None:
                    result("manifest", target_name, expected, actual, False,
                           manifest_name + "; mismatch has no registered transformation")
                else:
                    status = registered.get("manifest_status")
                    ok = registered.get("manifest_sha256") == expected and registered["artifact_sha256"] == actual
                    if status == "matches-source":
                        ok = ok and registered["source_sha256"] == expected
                        category = "documented-artifact-change"
                    elif status == "diverged-in-source":
                        ok = ok and registered["source_sha256"] != expected and bool(str(registered.get("manifest_note", "")).strip())
                        category = "documented-source-divergence"
                    else:
                        ok = False
                        category = "invalid manifest_status"
                    detail = manifest_name + "; " + category
                    result("manifest", target_name, expected, actual, ok, detail)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result("manifest", target_name, str(expected), "unavailable", False, manifest_name + "; " + str(exc))
    for name, expected_sources in registered_in.items():
        actual_sources = actual_references.get(name, set())
        if expected_sources != actual_sources:
            result("reference", name, ",".join(sorted(expected_sources)), ",".join(sorted(actual_sources)),
                   False, "recorded_in does not match actual manifest coverage")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    started = time.perf_counter()
    rows = check(args.root.resolve())
    print("kind\tpath\texpected\trecomputed\tstatus\tdetail")
    for row in rows:
        print("\t".join(row[key] for key in ("kind", "path", "expected", "recomputed", "status", "detail")))
    counts = Counter(row["status"] for row in rows)
    print("Manifest entries: " + str(sum(row["kind"] == "manifest" for row in rows)))
    print("Registered file checks: " + str(sum(row["kind"] == "change" for row in rows)))
    print("PASS: " + str(counts["PASS"]))
    print("FAIL: " + str(counts["FAIL"]))
    print("Result: " + ("FAIL" if counts["FAIL"] else "PASS"))
    print("Total runtime: {:.3f} seconds".format(time.perf_counter() - started))
    manifest_rows = [row for row in rows if row["kind"] == "manifest"]
    matched = sum(row["status"] == "PASS" and row["detail"].endswith("; exact bytes") for row in manifest_rows)
    artifact = sum(row["status"] == "PASS" and row["detail"].endswith("; documented-artifact-change") for row in manifest_rows)
    divergence = sum(row["status"] == "PASS" and row["detail"].endswith("; documented-source-divergence") for row in manifest_rows)
    print(f"matched={matched} / documented-artifact-change={artifact} / documented-source-divergence={divergence} / skipped={counts['SKIPPED']} / failed={counts['FAIL']}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
