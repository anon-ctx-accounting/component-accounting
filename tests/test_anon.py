"""Scan the actual release and exercise detection without embedding identifiers."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_anon_test',ROOT/'scripts/anon_scan.py')
SCAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCAN)


class AnonymityTests(unittest.TestCase):
    def test_release_has_zero_findings(self):
        findings,errors,files,_ = SCAN.scan(ROOT)
        self.assertGreater(files,0)
        self.assertEqual(errors,[])
        self.assertEqual(findings,[])

    def test_rules_scan_content_and_filenames_and_mask_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            address = 'reader'+'@'+'example'+'.org'
            token = 'sk-'+'a'*24
            (root/'sample.txt').write_text(address+'\n'+'192.'+'168.1.4'+'\n'+token)
            (root/('Mac'+'Book.txt')).write_text('test')
            (root/'.git').mkdir()
            (root/'.git'/'config').write_text(address)
            findings,errors,files,_ = SCAN.scan(root)
            self.assertEqual(errors,[])
            self.assertEqual(files,2)
            self.assertEqual({x['rule'] for x in findings},{'email','private_ip','api_key','host'})
            self.assertTrue(any(x['area']=='filename' for x in findings))
            self.assertEqual([x['match'] for x in findings if x['rule']=='api_key'],['<redacted>'])


if __name__ == '__main__':
    unittest.main()
