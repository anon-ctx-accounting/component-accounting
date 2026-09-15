#!/usr/bin/env python3
"""Check the anonymous single-commit release; source archives contain no Git metadata."""
import argparse
from pathlib import Path
import re
import subprocess

RELEASE_DATE = '2026-09-15 00:00:00 +0000'
EMAIL = re.compile(r'^[a-z0-9-]+@users\.noreply\.github\.com$')


def git(root, *args):
    result = subprocess.run(
        ['git', '--no-optional-locks', '-c', 'log.showSignature=false', '-C', str(root), *args],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError('Git query failed: ' + args[0])
    return result.stdout.strip()


def check(root, require_git=False):
    root = Path(root).resolve()
    if not (root / '.git').exists():
        return [dict(check='repository', status='FAIL' if require_git else 'SKIPPED',
                     detail='No Git metadata; source archives omit it.')]
    checks = []

    def add(name, passed, detail):
        checks.append(dict(check=name, status='PASS' if passed else 'FAIL', detail=detail))

    try:
        add('repository', Path(git(root, 'rev-parse', '--show-toplevel')).resolve() == root,
            'Git must describe this directory.')
        count = int(git(root, 'rev-list', '--count', '--all'))
        add('commit_count', count == 1, 'Exactly one reachable commit is required; found ' + str(count) + '.')
        fields = git(root, 'log', '-1', '--format=%an%x00%ae%x00%cn%x00%ce%x00%ai%x00%ci').split('\x00')
        if len(fields) != 6:
            raise RuntimeError('Unexpected commit metadata format.')
        author, author_email, committer, committer_email, author_date, committer_date = fields
        add('author', author == 'Anonymous' and bool(EMAIL.fullmatch(author_email)),
            'Anonymous name and a lowercase GitHub noreply address are required.')
        add('committer', committer == 'Anonymous' and bool(EMAIL.fullmatch(committer_email)),
            'Anonymous name and a lowercase GitHub noreply address are required.')
        add('dates', author_date == committer_date == RELEASE_DATE,
            'Author and committer dates must both be ' + RELEASE_DATE + '.')
        refs = git(root, 'for-each-ref', '--format=%(refname)').splitlines()
        local = [r for r in refs if r.startswith('refs/heads/')]
        tags = [r for r in refs if r.startswith('refs/tags/')]
        # A clone may also have tracking copies of main and the remote HEAD.
        other = [r for r in refs if r not in local + tags
                 and not re.fullmatch(r'refs/remotes/[^/]+/(main|HEAD)', r)]
        add('branches_and_tags', local == ['refs/heads/main'] and not tags and not other,
            'Only main is permitted, with no tags or other branch names.')
        add('checkout', git(root, 'symbolic-ref', '--short', 'HEAD') == 'main',
            'The checked-out branch must be main.')
    except (OSError, RuntimeError, ValueError) as exc:
        checks.append(dict(check='git_query', status='FAIL', detail=str(exc)))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--require-git', action='store_true',
                        help='Fail when metadata is absent; used for the release gate.')
    args = parser.parse_args()
    rows = check(args.root, args.require_git)
    for row in rows:
        print('{check}\t{status}\t{detail}'.format(**row))
    failed = any(row['status'] == 'FAIL' for row in rows)
    result = 'FAIL' if failed else 'SKIPPED' if all(r['status'] == 'SKIPPED' for r in rows) else 'PASS'
    print('Git metadata: ' + result)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
