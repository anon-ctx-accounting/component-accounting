"""Verify nested manifest roots, documented changes and explicit exclusions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_integrity_test',ROOT/'scripts/check_integrity.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.experiment = self.root/'docs/experiments/EXAMPLE'
        (self.experiment/'raw').mkdir(parents=True)
        (self.experiment/'smoke/raw').mkdir(parents=True)
        self.path = 'docs/experiments/EXAMPLE/raw/value.json'
        self.manifest = 'docs/experiments/EXAMPLE/smoke/raw/sha256.json'
        self.original = b'{"value": 12}\n'
        (self.root/self.path).write_bytes(self.original)
        self.write_manifest({'raw/value.json':digest(self.original)})
        self.write_registry([])

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self,entries):
        (self.root/self.manifest).write_text(json.dumps(entries))

    def write_registry(self,entries):
        covered=sum(bool(e.get('recorded_in')) for e in entries)
        summary=dict(total_registered=len(entries),manifest_covered=covered,manifest_uncovered=len(entries)-covered)
        (self.root/'integrity.json').write_text(json.dumps(dict(schema_version=2,summary=summary,files=entries)))

    def failures(self):
        return [row for row in CHECK.check(self.root) if row['status']=='FAIL']

    def changed_entry(self,content):
        (self.root/self.path).write_bytes(content)
        return dict(path=self.path,changes=['path-rewrite'],origin='artifact',
                    source_sha256=digest(self.original),artifact_sha256=digest(content),
                    manifest_sha256=digest(self.original),recorded_in=self.manifest,
                    manifest_status='matches-source')

    def test_current_release_integrity(self):
        rows = CHECK.check(ROOT)
        self.assertFalse([r for r in rows if r['status']=='FAIL'])
        self.assertEqual(sum(r['status']=='SKIPPED' for r in rows),14)

    def test_nested_manifest_resolves_from_experiment_root(self):
        self.assertEqual(self.failures(),[])

    def test_unregistered_change_fails_and_complete_chain_passes(self):
        entry = self.changed_entry(b'{"value": 13}\n')
        self.assertTrue(self.failures())
        self.write_registry([entry])
        self.assertEqual(self.failures(),[])
        entry['source_sha256'] = digest(b'other input')
        self.write_registry([entry])
        self.assertTrue(self.failures())

    def test_source_divergence_needs_differing_hash_and_explanation(self):
        entry = self.changed_entry(b'{"value": 13}\n')
        entry.update(origin='source',changes=[],source_sha256=entry['artifact_sha256'],
                     manifest_status='diverged-in-source')
        self.write_registry([entry])
        self.assertTrue(self.failures())
        entry['manifest_note'] = 'The source record already contained this change.'
        self.write_registry([entry])
        self.assertEqual(self.failures(),[])

    def test_uncovered_registry_entry_is_still_checked(self):
        path = self.root/'extra.json'
        path.write_bytes(b'changed')
        entry = dict(path='extra.json',changes=['path-rewrite'],origin='artifact',
                     source_sha256=digest(b'original'),artifact_sha256=digest(b'changed'))
        self.write_registry([entry])
        self.assertEqual(self.failures(),[])
        path.unlink()
        self.assertTrue(self.failures())

    def test_excluded_markdown_is_skipped_but_registered_missing_file_fails(self):
        self.write_manifest({'report.md':digest(self.original),'raw/value.json':digest(self.original)})
        rows = CHECK.check(self.root)
        self.assertEqual(sum(r['status']=='SKIPPED' for r in rows),1)
        entry = dict(path='docs/experiments/EXAMPLE/report.md',changes=['path-rewrite'],origin='artifact',
                     source_sha256=digest(self.original),artifact_sha256=digest(b'changed'))
        self.write_registry([entry])
        self.assertTrue(self.failures())

    def test_missing_data_is_not_skipped(self):
        (self.root/self.path).unlink()
        self.assertTrue(self.failures())


if __name__ == '__main__':
    unittest.main()
