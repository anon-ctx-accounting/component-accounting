"""Regression checks for pairing, cumulative usage and verifier exit status."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_verify_test',ROOT/'scripts/verify.py')
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class VerifyTests(unittest.TestCase):
    def test_verify_cli_passes(self):
        result = subprocess.run([sys.executable,'-B',str(ROOT/'scripts/verify.py')],
                                cwd=ROOT,capture_output=True,text=True,timeout=300)
        failures = '\n'.join(line for line in result.stdout.splitlines() if '\tFAIL' in line)
        self.assertEqual(result.returncode,0,failures+'\n'+result.stderr)
        self.assertIn('unprocessed: 0',result.stdout)

    def test_ratio_is_ratio_of_sums_and_preserves_pairs(self):
        self.assertEqual(VERIFY.paired_ratio([1,9],[1,3])['point'],2.5)
        self.assertEqual(VERIFY.paired_ratio([20,2],[10,1]),
                         dict(point=2.0,ci95_lo=2.0,ci95_hi=2.0))
        self.assertIs(VERIFY.resample_matrix(12),VERIFY.resample_matrix(12))
        self.assertEqual(len(VERIFY.resample_matrix(12)),10000)
        self.assertEqual(VERIFY.interval(list(range(10000))),(250,9749))

    def test_full_document_budget_differs_from_utilization_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            (root/'manifest.json').write_text(json.dumps(dict(artifacts={'a':dict(size_tok=82),'b':dict(size_tok=101)})))
            (root/'build.json').write_text(json.dumps(dict(resident_ramp=[
                dict(budget_tokens=3512,resident_tokens=3369),
                dict(budget_tokens=3512,resident_tokens=3451),
                dict(budget_tokens=3512,resident_tokens=3432)])))
            verifier=VERIFY.Verifier(root,{})
            self.assertEqual(verifier.expression(dict(resident_cap='build.json',fixture_manifest='manifest.json')),2)

    def test_cumulative_difference_agrees_with_independent_calls(self):
        previous = dict(input_tokens=40,cached_input_tokens=10,cache_write_input_tokens=5,
                        output_tokens=2,reasoning_output_tokens=1)
        row = dict(usage_raw=dict(input_tokens=100,cached_input_tokens=30,cache_write_input_tokens=20,
                                 output_tokens=4,reasoning_output_tokens=2),
                   calls=[dict(fresh=25,write=15,cached=20,output=2,reasoning=1,input_tokens=60,total_tokens=62)])
        comp,current = VERIFY.codex_components(row,previous)
        self.assertEqual((comp['gross'],comp['uncached'],comp['output']),(60,25,2))
        self.assertEqual(current,row['usage_raw'])
        self.assertEqual(comp['reasoning'],1)
        tampered = copy.deepcopy(row)
        tampered['calls'][0]['fresh'] += 1
        with self.assertRaisesRegex(ValueError,'stdout difference/session transcript mismatch'):
            VERIFY.codex_components(tampered,previous)
        tampered = copy.deepcopy(row)
        tampered['calls'][0]['reasoning'] = 0
        with self.assertRaisesRegex(ValueError,'mismatch: reasoning'):
            VERIFY.codex_components(tampered,previous)
        with self.assertRaisesRegex(ValueError,'nonmonotone'):
            VERIFY.codex_components(row,{**previous,'input_tokens':101})

    def test_reasoning_is_already_in_output_and_removed_once(self):
        card = dict(input=2,read=.2,output=12,write=2.5)
        comp = dict(uncached=10,cache_read=20,creation_1h=0,creation_5m=4,output=8,reasoning=3)
        self.assertAlmostEqual(VERIFY.price(comp,card),130/1e6)
        self.assertAlmostEqual(VERIFY.price_without_reasoning(comp,3,card),94/1e6)
        for invalid in (-1,9):
            with self.subTest(reasoning=invalid), self.assertRaisesRegex(ValueError,'subset of output'):
                VERIFY.price_without_reasoning(comp,invalid,card)

    def test_pi_reasoning_comes_from_retained_turn_ledger(self):
        row = dict(calls=[dict(input=2,cacheWrite=3,cacheRead=5,output=4,totalTokens=14)],
                   usage_turn=dict(input=2,cacheWrite=3,cacheRead=5,output=4,reasoning=2),gross_tokens=10)
        comp = VERIFY.pi_components(row)
        self.assertEqual((comp['gross'],comp['output'],comp['reasoning']),(10,4,2))
        row['usage_turn']['reasoning'] = 5
        with self.assertRaisesRegex(ValueError,'reasoning/output inclusion'):
            VERIFY.pi_components(row)

    def test_changed_measurement_and_unknown_method_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root/'value.json').write_text('{"tokens": 12}')
            item = dict(id='TEST',quantity='tokens',source='value.json',method='json',
                        selector=['tokens'],expected=11,tolerance=0)
            verifier = VERIFY.Verifier(root,dict(items=[item]))
            self.assertEqual(verifier.run()[0]['status'],'FAIL')
            item['method'] = 'unsupported'
            row = verifier.run()[0]
            self.assertEqual(row['status'],'FAIL')
            self.assertIn('unhandled method',row['recomputed'])

    def test_mismatch_exit_code_is_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root/'value.json').write_text('{"tokens": 12}')
            item = dict(id='TEST',quantity='tokens',source='value.json',method='json',
                        selector=['tokens'],expected=11,tolerance=0)
            (root/'paper-numbers.json').write_text(json.dumps(dict(items=[item])))
            result = subprocess.run([sys.executable,'-B',str(ROOT/'scripts/verify.py'),'--root',str(root)],
                                    capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,1)
            self.assertIn('FAIL: 1',result.stdout)


if __name__ == '__main__':
    unittest.main()
