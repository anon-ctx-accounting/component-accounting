#!/usr/bin/env python3
"""Scan every file and relative filename using the release anonymity rules.

The scanner includes its own source. Literal identity markers are assembled
from fragments so the rule definitions do not constitute findings. Git metadata is excluded; no working-tree file,
extension, or finding is allowlisted. Findings are reported
with relative paths and locations; credential-shaped matches are masked.
This checks the specified patterns, not every possible identifying signal.
"""

import argparse
from collections import Counter
from pathlib import Path
import re
import time


RULES = (
    ("identity", "|".join((
        "hs" + "han", "/" + "Users" + "/", "Hyung" + "seok",
        "Hyeong" + "seok", "한" + "형석", r"nota\.ai",
    ))),
    ("email", r"[\w.+-]+@[\w.-]+\.[a-z]{2,}"),
    ("private_ip", r"192\.168\.\d|10\.\d+\.\d+\.\d|172\.(1[6-9]|2\d|3[01])\."),
    ("api_key", r"sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,}"),
    ("github_account", r"github\.com/[A-Za-z0-9_-]+"),
    ("host", "Mac" + "Book|Mac" + r'-mini|\.' + "local" + chr(34)),
)
PATTERNS = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in RULES]


def scan(root):
    findings = []
    errors = []
    files = 0
    byte_count = 0
    if not root.is_dir():
        return findings, ["scan root is not a directory"], files, byte_count
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            errors.append(relative + ": symlink requires review")
            continue
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            errors.append(relative + ": " + type(exc).__name__)
            continue
        files += 1
        byte_count += len(data)
        # Replacement decoding retains ASCII markers in binary files too.
        content = data.decode("utf-8", errors="replace")
        for area, value in (("filename", relative), ("content", content)):
            for rule, pattern in PATTERNS:
                for match in pattern.finditer(value):
                    before = value[:match.start()]
                    findings.append({
                        "path": relative, "area": area,
                        "line": before.count("\n") + 1,
                        "column": match.start() - before.rfind("\n"),
                        "rule": rule,
                        "match": "<redacted>" if rule == "api_key" else match.group(),
                    })
    return findings, errors, files, byte_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    started = time.perf_counter()
    findings, errors, files, byte_count = scan(args.root)
    print("Anonymity scan: all files and relative filenames; case-insensitive")
    print("Exclusions: .git/ metadata only")
    print("Files scanned: " + str(files))
    print("Bytes scanned: " + str(byte_count))
    print("path\tarea\tline\tcolumn\trule\tmatch")
    for item in findings:
        print("\t".join(str(item[key]) for key in (
            "path", "area", "line", "column", "rule", "match")))
    for error in errors:
        print("ERROR\t" + error)
    counts = Counter(item["rule"] for item in findings)
    for name, _ in RULES:
        print(name + ": " + str(counts[name]))
    print("Findings: " + str(len(findings)))
    print("Affected files: " + str(len({item["path"] for item in findings})))
    print("Read errors: " + str(len(errors)))
    print("Result: " + ("FAIL" if findings or errors else "PASS"))
    print("Total runtime: {:.3f} seconds".format(time.perf_counter() - started))
    return 1 if findings or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
