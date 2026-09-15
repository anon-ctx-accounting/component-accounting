"""Exercise valid releases, metadata violations, and archives without Git."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_metadata', ROOT/'scripts/check_git_metadata.py')
METADATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METADATA)


class GitMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
        address = 'anonymous' + '@' + 'users.noreply.github.com'
        self.env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_AUTHOR_NAME='Anonymous', GIT_COMMITTER_NAME='Anonymous',
                        GIT_AUTHOR_EMAIL=address, GIT_COMMITTER_EMAIL=address,
                        GIT_AUTHOR_DATE='2026-09-15T00:00:00 +0000',
                        GIT_COMMITTER_DATE='2026-09-15T00:00:00 +0000', TZ='UTC')

    def git(self, *args, extra_env=None):
        result = subprocess.run(
            ['git', '-c', 'core.hooksPath=' + os.devnull, '-c', 'commit.gpgsign=false',
             '-C', str(self.root), *args], env={**self.env, **(extra_env or {})},
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def release(self):
        self.git('init', '-b', 'main')
        self.git('commit', '--allow-empty', '-m', 'Artifact for double-blind review')

    def test_anonymous_release_passes(self):
        self.release()
        self.assertTrue(all(r['status'] == 'PASS' for r in METADATA.check(self.root, True)))

    def test_archive_is_skipped_unless_git_is_required(self):
        self.assertEqual(METADATA.check(self.root)[0]['status'], 'SKIPPED')
        self.assertEqual(METADATA.check(self.root, True)[0]['status'], 'FAIL')

    def test_rejects_identity_and_date_changes(self):
        self.release()
        cases = [
            ('author', {'GIT_AUTHOR_NAME': 'Other'}),
            ('committer', {'GIT_COMMITTER_NAME': 'Other'}),
            ('author', {'GIT_AUTHOR_EMAIL': 'anonymous' + '@' + 'example.invalid'}),
            ('committer', {'GIT_COMMITTER_EMAIL': 'anonymous' + '@' + 'example.invalid'}),
            ('dates', {'GIT_AUTHOR_DATE': '2026-09-15T09:00:00 +0900'}),
            ('dates', {'GIT_COMMITTER_DATE': '2026-09-16T00:00:00 +0000'}),
        ]
        for name, env in cases:
            with self.subTest(env=env):
                self.git('commit', '--amend', '--allow-empty', '--no-edit', '--reset-author', extra_env=env)
                checks = {r['check']:r['status'] for r in METADATA.check(self.root, True)}
                self.assertEqual(checks[name], 'FAIL')

    def test_rejects_extra_history_branches_and_tags(self):
        self.release()
        self.git('branch', 'extra')
        checks = {r['check']:r['status'] for r in METADATA.check(self.root, True)}
        self.assertEqual(checks['branches_and_tags'], 'FAIL')
        self.git('branch', '-D', 'extra')
        self.git('tag', 'extra')
        checks = {r['check']:r['status'] for r in METADATA.check(self.root, True)}
        self.assertEqual(checks['branches_and_tags'], 'FAIL')
        self.git('tag', '-d', 'extra')
        self.git('commit', '--allow-empty', '-m', 'Second commit')
        checks = {r['check']:r['status'] for r in METADATA.check(self.root, True)}
        self.assertEqual(checks['commit_count'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
