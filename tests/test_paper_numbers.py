"""Check the printed-measurement catalog and the reduced release boundary."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_verify_catalog', ROOT/'scripts/verify.py')
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class PaperNumbersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT/'paper-numbers.json').read_text())

    def test_sources_are_present_and_inside_artifact(self):
        self.assertEqual(self.catalog['schema_version'], 2)
        self.assertRegex(self.catalog['paper_source_sha256'],r'^[a-f0-9]{64}$')
        self.assertRegex(self.catalog['generated_on'],r'^\d{4}-\d{2}-\d{2}$')
        self.assertTrue(self.catalog['items'])
        for item in self.catalog['items']:
            with self.subTest(id=item['id'], quantity=item['quantity']):
                self.assertTrue({'id','quantity','expected','source','tolerance','method','printed_as','paper_locations'} <= item.keys())
                self.assertTrue(item['printed_as'])
                self.assertTrue(item['paper_locations'])
                self.assertGreaterEqual(item['tolerance'], 0)
                self.assertTrue(VERIFY.inside(ROOT,item['source']).is_file())
                for source in item['inputs']:
                    self.assertTrue(VERIFY.inside(ROOT,source).is_file())
        for name, config in self.catalog['configurations'].items():
            with self.subTest(configuration=name):
                self.assertTrue(VERIFY.inside(ROOT,config['analysis']).is_file())
                if name != 'BASE3':
                    self.assertTrue(VERIFY.inside(ROOT,config['turns']).is_file())
                    self.assertEqual(len(config['sessions']),12)
        for name in ('E10-P2-H16','E11-BASE3'):
            self.assertTrue((ROOT/'docs/experiments'/name/'run/raw/turns.jsonl').is_file())

    def test_every_catalog_item_is_processed(self):
        # Numerical disagreement must remain a FAIL in test_verify's CLI gate.
        # This test checks handling, not whether expected values agree.
        rows = VERIFY.Verifier(ROOT,self.catalog).run()
        self.assertEqual(len(rows),len(self.catalog['items']))
        for item,row in zip(self.catalog['items'],rows):
            self.assertEqual((row['id'],row['quantity']),(item['id'],item['quantity']))
            self.assertIsNotNone(row['diff'],row)
        self.assertEqual(len(self.catalog['main_and_auxiliary']),15)
        self.assertEqual(len(set(self.catalog['main_and_auxiliary'])),15)

    def test_experiments_follow_release_scope(self):
        experiments = ROOT/'docs/experiments'
        directories = sorted(p for p in experiments.iterdir() if p.is_dir())
        self.assertEqual(len(directories),18)
        selected = set().union(*(set(i['experiments']) for i in self.catalog['items']))
        self.assertTrue(selected <= {p.name for p in directories})
        self.assertTrue({'E15-CODEX-WRITE','E16-METERED-WRITE','E20-CODEX-METERED','E10-P2-XR','E10-P2-H16'} <= {p.name for p in directories})
        encoded=json.dumps(self.catalog)
        for name in ('E5-G0','E5-G1','E8-D5'):
            self.assertFalse((experiments/name).exists())
            self.assertNotIn(name,encoded)
        for name in ('ci_endpoints','convention_note'):
            self.assertNotIn(name,encoded)
        self.assertFalse({'L36','L80'} & {i['id'] for i in self.catalog['items']})
        for directory in directories:
            self.assertEqual(sorted(p.relative_to(directory).as_posix() for p in directory.rglob('*.md')),
                             ['README.md'])
        for name in ('claims-ledger.csv','docs/translation-fingerprints.json',
                     'scripts/check_translation.py','tests/test_ledger.py','tests/test_translation.py'):
            self.assertFalse((ROOT/name).exists(),name)
        self.assertFalse((ROOT/'docs/paper2/tex').exists())

    def test_paths_cannot_escape(self):
        for path in ('../private.json','/private.json','.git/config'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                VERIFY.inside(ROOT,path)


if __name__ == '__main__':
    unittest.main()
