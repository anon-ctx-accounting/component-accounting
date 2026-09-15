#!/usr/bin/env python3
"""Offline numerical verification using only the Python standard library.

Printed measurements are rebuilt from retained turns, calls and fixture records.
Declared rates and experimental dimensions are inputs, not catalog assertions.
No experiment runner is imported; no input file is modified.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import time

SEED = 1313
RESAMPLES = 10000
ARMS = ('karc-full', 'rag-bm25')
COMPONENTS = ('uncached', 'creation', 'creation_1h', 'creation_5m', 'cache_read', 'output', 'gross')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, relative):
    require(isinstance(relative, str) and relative, 'missing source path')
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts and '.git' not in path.parts,
            'source path must remain inside the artifact')
    result = root / path
    require(result.resolve().is_relative_to(root.resolve()), 'source escapes artifact')
    require(result.is_file(), 'missing source: ' + relative)
    return result


def read_json(root, relative):
    return json.loads(inside(root, relative).read_text(encoding='utf-8'))


def read_jsonl(root, relative):
    with inside(root, relative).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(value, keys):
    for key in keys:
        value = value[key]
    return value


@lru_cache(maxsize=None)
def resample_matrix(n):
    require(n > 0, 'empty sample')
    rng = random.Random(SEED)
    return tuple(tuple(rng.randrange(n) for _ in range(n)) for _ in range(RESAMPLES))


def interval(values):
    require(len(values) == RESAMPLES, 'missing bootstrap draws')
    ordered = sorted(values)
    # E13's declared convention uses zero-based indices 250 and 9749.
    return ordered[250], ordered[9749]


def paired_ratio(numerator, denominator):
    require(len(numerator) == len(denominator) > 0, 'unpaired sample')
    require(all(x > 0 for x in denominator), 'nonpositive paired denominator')
    draws = [sum(numerator[i] for i in row) / sum(denominator[i] for i in row)
             for row in resample_matrix(len(numerator))]
    lo, hi = interval(draws)
    return dict(point=sum(numerator)/sum(denominator), ci95_lo=lo, ci95_hi=hi)


def paired_mean(values):
    draws = [sum(values[i] for i in row)/len(row) for row in resample_matrix(len(values))]
    lo, hi = interval(draws)
    return dict(point=sum(values)/len(values), ci95_lo=lo, ci95_hi=hi)


def price(comp, card):
    return (comp['uncached']*card['input'] + comp['cache_read']*card['read']
            + comp['output']*card['output']
            + comp['creation_1h']*card.get('write_1h', card.get('write', 0))
            + comp['creation_5m']*card.get('write_5m', card.get('write', 0)))/1e6


def price_without_reasoning(comp, reasoning, card):
    require(0 <= reasoning <= comp['output'], 'reasoning must be a subset of output')
    return price({**comp, 'output': comp['output']-reasoning}, card)


def pi_components(row):
    calls = row['calls']
    require(bool(calls), 'missing pi call records')
    comp = dict.fromkeys(COMPONENTS, 0)
    fields = dict(uncached='input', creation='cacheWrite', cache_read='cacheRead', output='output')
    for call in calls:
        values = {key:int(call[field]) for key, field in fields.items()}
        values['creation_1h'] = int(call.get('cacheWrite1h') or 0)
        values['creation_5m'] = values['creation'] - values['creation_1h']
        values['gross'] = values['uncached'] + values['creation'] + values['cache_read']
        require(all(v >= 0 for v in values.values()), 'negative pi component')
        require(call['totalTokens'] == values['gross']+values['output'], 'pi totality identity failed')
        for key in COMPONENTS:
            comp[key] += values[key]
    for key, field in fields.items():
        require(comp[key] == row['usage_turn'][field], 'pi call/turn mismatch: '+key)
    require(comp['gross'] == row['gross_tokens'], 'pi gross mismatch')
    # Normalized calls omit this field; the retained turn ledger records it.
    comp['reasoning'] = int(row['usage_turn']['reasoning'])
    require(0 <= comp['reasoning'] <= comp['output'], 'pi reasoning/output inclusion failed')
    return comp


def codex_components(row, previous):
    raw = row['usage_raw']
    fields = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens',
              'reasoning_output_tokens')
    current = {key:int(raw[key]) for key in fields}
    delta = {key:current[key]-previous.get(key, 0) for key in fields}
    require(all(x >= 0 for x in delta.values()), 'nonmonotone Codex cumulative counter')
    fresh = delta['input_tokens']-delta['cached_input_tokens']-delta['cache_write_input_tokens']
    require(fresh >= 0, 'Codex input inclusion failed')
    comp = dict(uncached=fresh, creation=delta['cache_write_input_tokens'], creation_1h=0,
                creation_5m=delta['cache_write_input_tokens'], cache_read=delta['cached_input_tokens'],
                output=delta['output_tokens'], gross=delta['input_tokens'],
                reasoning=delta['reasoning_output_tokens'])
    require(comp['reasoning'] <= comp['output'], 'Codex reasoning/output inclusion failed')
    call_keys = dict(uncached='fresh', creation='write', cache_read='cached', output='output',
                     gross='input_tokens', reasoning='reasoning')
    require(bool(row['calls']), 'missing Codex second-route calls')
    for key, field in call_keys.items():
        require(comp[key] == sum(int(call[field]) for call in row['calls']),
                'Codex stdout difference/session transcript mismatch: '+key)
    for call in row['calls']:
        require(call['input_tokens'] == call['fresh']+call['write']+call['cached'],
                'Codex call inclusion failed')
        require(call['total_tokens'] == call['input_tokens']+call['output'], 'Codex call totality failed')
        require(0 <= call['reasoning'] <= call['output'], 'Codex call reasoning/output inclusion failed')
    return comp, current


def claude_components(row):
    fields = dict(uncached='fresh_input_tokens_provider', creation='cache_creation_input_tokens_provider',
                  creation_1h='cache_creation_1h_input_tokens_provider',
                  creation_5m='cache_creation_5m_input_tokens_provider',
                  cache_read='cache_read_input_tokens_provider', output='output_tokens', gross='gross_input_tokens')
    comp = {key:int(row[field]) for key, field in fields.items()}
    require(all(x >= 0 for x in comp.values()), 'negative Claude component')
    require(comp['creation'] == comp['creation_1h']+comp['creation_5m'], 'Claude write bucket mismatch')
    require(comp['gross'] == comp['uncached']+comp['creation']+comp['cache_read'], 'Claude gross inclusion failed')
    return comp


def component_profile(root, config):
    rows = read_jsonl(root, config['turns'])
    units = defaultdict(list)
    for row in rows:
        require(row['arm'] in ARMS, 'unexpected component arm')
        units[row['arm'], row['session_id']].append(row)
    sessions = sorted({sid for arm, sid in units})
    require(len(sessions) == 12 and len(units) == 24, 'expected 12 paired sessions')
    require(sessions == config['sessions'], 'measurement window differs')
    totals = {arm:dict.fromkeys(COMPONENTS, 0) for arm in ARMS}
    per = {arm:{} for arm in ARMS}
    grades = {arm:dict(passed=0, position_1_passed=0, later_passed=0) for arm in ARMS}
    reuse = dict.fromkeys(ARMS, 0)
    reuse_per_session = {arm:{} for arm in ARMS}
    calls = dict.fromkeys(ARMS, 0)
    # Claude records used here do not separately expose reasoning usage.
    reasoning = dict.fromkeys(ARMS, 0) if config['format'] in ('pi', 'codex') else None
    positions = {arm:{p:dict.fromkeys(COMPONENTS,0) for p in range(1,9)} for arm in ARMS}
    for (arm, sid), unit in sorted(units.items()):
        unit.sort(key=lambda row:row['position'])
        require([r['position'] for r in unit] == list(range(1, 9)), 'missing or duplicate turn positions')
        acc = dict.fromkeys(COMPONENTS, 0)
        previous = {}
        reuse_per_session[arm][sid] = 0
        for row in unit:
            if config['format'] == 'pi':
                comp = pi_components(row)
            elif config['format'] == 'codex':
                comp, previous = codex_components(row, previous)
            elif config['format'] == 'claude':
                comp = claude_components(row)
            else:
                raise ValueError('unknown component format')
            if reasoning is not None:
                reasoning[arm] += comp['reasoning']
            for key in COMPONENTS:
                acc[key] += comp[key]
                positions[arm][row['position']][key] += comp[key]
            calls[arm] += 1+row['tool_calls'] if config['format']=='claude' else len(row['calls'])
            passed = row.get('passed', row.get('grade_incidental', {}).get('passed', False))
            grades[arm]['passed'] += int(passed)
            grades[arm]['position_1_passed' if row['position']==1 else 'later_passed'] += int(passed)
            if row['position'] > 1:
                reuse[arm] += int(row['reuse'])
                reuse_per_session[arm][sid] += int(row['reuse'])
        per[arm][sid] = acc
        for key in COMPONENTS:
            totals[arm][key] += acc[key]
    delta = {key:totals[ARMS[0]][key]-totals[ARMS[1]][key] for key in COMPONENTS}
    card = config['rates']
    gross = [[per[arm][sid]['gross'] for sid in sessions] for arm in ARMS]
    costs = [[price(per[arm][sid], card) for sid in sessions] for arm in ARMS]
    gross_ratio, priced_ratio = paired_ratio(*gross), paired_ratio(*costs)
    mu_r, mu_o = card['read']/card['input'], card['output']/card['input']
    require(delta['creation'] != 0, 'undefined write-price threshold')
    threshold = -(delta['uncached']+mu_r*delta['cache_read']+mu_o*delta['output'])/delta['creation']
    mu_w = config['observed_mu_w']
    without_reasoning = None
    if reasoning is not None:
        without_reasoning = (price_without_reasoning(totals[ARMS[0]], reasoning[ARMS[0]], card)
                             / price_without_reasoning(totals[ARMS[1]], reasoning[ARMS[1]], card))
    return dict(gross_ratio=gross_ratio, priced_ratio=priced_ratio, delta=delta, totals=totals,
                reasoning_totals=reasoning, priced_without_reasoning_ratio=without_reasoning,
                mu_w_star=threshold, distance_actual_minus_threshold=mu_w-threshold,
                actual_over_threshold=mu_w/threshold if threshold else None,
                cost_per_gross_ratio=priced_ratio['point']/gross_ratio['point'],
                gross_over_uncached={a:totals[a]['gross']/totals[a]['uncached'] for a in ARMS},
                gross_percent_change=100*(gross_ratio['point']-1),
                cost_percent_reduction=100*(1-priced_ratio['point']),
                calls=calls, call_difference=calls[ARMS[0]]-calls[ARMS[1]],
                calls_per_turn={a:calls[a]/(len(rows)//2) for a in ARMS},
                gross_per_call={a:totals[a]['gross']/calls[a] for a in ARMS},
                cache_read_percent={a:100*totals[a]['cache_read']/totals[a]['gross'] for a in ARMS},
                later_creation_mean={a:sum(positions[a][p]['creation'] for p in range(2,9))/(len(sessions)*7) for a in ARMS},
                last_over_first_gross={a:positions[a][8]['gross']/positions[a][1]['gross'] for a in ARMS},
                reuse_fraction={a:reuse[a]/(len(sessions)*7) for a in ARMS},
                reuse_session_min=min(reuse_per_session[ARMS[0]].values()),
                reuse_session_max=max(reuse_per_session[ARMS[0]].values()),
                threshold_without_output=-(delta['uncached']+mu_r*delta['cache_read'])/delta['creation'],
                creation_difference_ci=paired_mean([per[ARMS[0]][s]['creation']-per[ARMS[1]][s]['creation'] for s in sessions]),
                read_difference_ci=paired_mean([per[ARMS[0]][s]['cache_read']-per[ARMS[1]][s]['cache_read'] for s in sessions]),
                priced_difference=paired_mean([a-b for a,b in zip(*costs)]),
                gross_difference=paired_mean([a-b for a,b in zip(*gross)]),
                grades=grades, reuse=reuse, sessions=len(sessions), turns=len(rows),
                per_session=per)


def base3_profile(root):
    rows = []
    for experiment in ('E10-P2-H16', 'E11-BASE3'):
        rows.extend(read_jsonl(root, 'docs/experiments/'+experiment+'/run/raw/turns.jsonl'))
    units = defaultdict(list)
    for row in rows:
        units[row['arm'], row['session_id']].append(row)
    per = defaultdict(dict)
    totals = defaultdict(lambda:dict(gross=0, fresh=0, cache_read=0, output=0))
    fields = dict(gross='api_input_tokens_with_cache', fresh='api_input_tokens_no_cache',
                  cache_read='api_cache_read_tokens', output='api_output_tokens')
    for (arm, sid), unit in sorted(units.items()):
        unit.sort(key=lambda row:row['position'])
        require([x['position'] for x in unit] == list(range(1,17)), 'incomplete BASE3 session')
        threads = {row['native_session_sha256'] for row in unit}
        require(None not in threads, 'missing BASE3 thread identity')
        persistent = len(threads) == 1
        require(persistent == (arm in ('karc-full', 'rag-bm25', 'full-history')), 'BASE3 thread classification changed')
        acc = dict.fromkeys(fields, 0)
        previous = dict.fromkeys(fields, 0)
        for row in unit:
            for key, field in fields.items():
                value = row[field]-previous[key] if persistent else row[field]
                require(value >= 0, 'negative BASE3 difference')
                acc[key] += value
                previous[key] = row[field]
        require(acc['gross'] == acc['fresh']+acc['cache_read'], 'BASE3 closure failed')
        per[arm][sid] = acc['gross']
        for key in fields:
            totals[arm][key] += acc[key]
    sessions = sorted(per['full-history'])
    require(len(sessions)==12 and all(sorted(x)==sessions for x in per.values()), 'BASE3 pairing failed')
    ratios = {}
    for left in per:
        for right in per:
            if left != right:
                ratios[left+'/'+right] = dict(point=sum(per[left].values())/sum(per[right].values()))
    comparisons=[ratios[a+'/'+b]['point'] for a in ARMS
                 for b in ('stateless-rag','sliding-window-compaction')]
    return dict(totals=dict(totals),ratios=ratios,
                primary_over_rebuilt_min=min(comparisons),primary_over_rebuilt_max=max(comparisons))


def evidence_profile(root, name):
    if name == 'fixture':
        artifacts=read_json(root,'fixture/e4-v2/manifest.json')['artifacts']
        tasks=read_json(root,'fixture/e4-v2/tasks.json')
        return dict(documents=len(artifacts),tokens=sum(a['size_tok'] for a in artifacts.values()),
                    supersessions=sum(e.get('event_type')=='validity_changed' and e.get('new_validity')=='STALE'
                                      for task in tasks for e in task['pre_events']),tasks=len(tasks))
    if name == 'pi-gate':
        records=[read_json(root,'docs/experiments/E17B-PI-OPENAI-GATE/raw/smoke-'+tag+'.json')
                 for tag in ('pilot','short','long')]
        calls=[c for record in records for row in record['rows'] for c in row['api_calls']]
        traffic=[c for c in calls if c['cacheRead']+c['cacheWrite']>0]
        exclusive=sum(c['totalTokens']==c['input']+c['cacheWrite']+c['cacheRead']+c['output'] for c in traffic)
        inclusive_fail=sum(c['totalTokens']!=c['input']+c['output'] for c in traffic)
        return dict(discriminating=len(traffic),exclusive_matches=exclusive,
                    inclusive_failures=inclusive_fail,inclusive_matches=len(traffic)-inclusive_fail)
    if name == 'steady':
        ramp=read_json(root,'docs/experiments/E19-STEADY-STATE/raw/prepare-A.json')['resident_ramp']
        rows={r['session_id']:r for r in ramp}
        last,previous=rows['S08']['resident_tokens'],rows['S07']['resident_tokens']
        window=[rows[f'S{i:02d}']['resident_tokens'] for i in range(9,21)]
        return dict(relative_change_percent=100*abs(last-previous)/previous,
                    utilization_percent=100*last/rows['S08']['budget_tokens'],
                    minimum_injection=min(window),maximum_injection=max(window))
    if name == 'warm-probe':
        directories=('docs/experiments/E10-P2-XR/canary/raw',
                     'docs/experiments/E24-BASE-RECAP/run/raw',
                     'docs/experiments/E24-BASE-RECAP/probe-warm-s01/raw')
        heads=[];costs=[]
        for directory in directories:
            rows=read_jsonl(root,directory+'/turns.jsonl')
            chosen=[r for r in rows if r['arm']=='karc-full' and r['session_id']=='S01']
            require(len(chosen)==8,'warm-probe session incomplete')
            heads.append(next(r for r in chosen if r['position']==1))
            attempts=read_jsonl(root,directory+'/attempts.jsonl')
            costs.append(sum(r.get('harness_reported_cost_usd',r['rate_card_equivalent_usd']) for r in attempts
                             if r['arm']=='karc-full' and r['session_id']=='S01'))
        gross=[r['gross_input_tokens'] for r in heads]
        return dict(gross=gross,observations=len(heads),spread_percent=100*(max(gross)-min(gross))/min(gross),
                    shifted_creation=heads[1]['cache_creation_input_tokens_provider']-heads[0]['cache_creation_input_tokens_provider'],
                    session_cost_min=min(costs),session_cost_max=max(costs),session_cost_factor=max(costs)/min(costs))
    if name == 'history':
        p=read_json(root,'docs/experiments/E18-PI-REVERSAL/raw/provenance.json')
        reruns=next(r for r in p['deviations_from_the_work_order'] if 'attempt_1' in r)
        discarded=sum('discarded' in reruns[k] or 'See raw/discarded-' in reruns[k]
                      for k in ('attempt_1','attempt_2'))
        incident=read_json(root,'docs/experiments/E28-BUDGET-SWEEP/raw/o20/h9-incident.json')
        units={r['unit'] for r in incident['wave_abort_records']}
        return dict(discarded_pi_openai_attempts=discarded,remeasured_budget_units=len(units))
    raise ValueError('unknown evidence profile: '+name)


def amount_residual(root,config):
    require(config['format'] != 'codex', 'Codex records no harness amount')
    if config['format']=='claude':
        rows=read_jsonl(root,config['attempts'])
        residuals=[abs(price(claude_components(row),config['rates'])-row['harness_reported_cost_usd']) for row in rows]
    else:
        rows=read_jsonl(root,config['turns']);card=config['harness_rates']
        residuals=[]
        for row in rows:
            for c in row['calls']:
                one_hour=c.get('cacheWrite1h') or 0
                comp=dict(uncached=c['input'],cache_read=c['cacheRead'],output=c['output'],
                          creation_1h=one_hour,creation_5m=c['cacheWrite']-one_hour)
                residuals.append(abs(price(comp,card)-c['pi_cost_total']))
    require(bool(residuals),'missing harness amount records')
    return max(residuals)


def amount_matches(root,config):
    return config['format'] != 'codex' and amount_residual(root,config)<=1e-12


def difference(expected, actual):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return 0.0 if type(expected) is type(actual) and expected == actual else math.inf
    if isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int,float)) or not math.isfinite(actual):
            return math.inf
        return abs(expected-actual)
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected)!=set(actual):
            return math.inf
        return max((difference(v,actual[k]) for k,v in expected.items()), default=0.0)
    if isinstance(expected, list):
        if not isinstance(actual,list) or len(expected)!=len(actual):
            return math.inf
        return max((difference(a,b) for a,b in zip(expected,actual)),default=0.0)
    return math.inf


class Verifier:
    def __init__(self, root, catalog):
        self.root, self.catalog = root, catalog
        self.profiles = {}

    def profile(self, name):
        if name not in self.profiles:
            if name == 'BASE3':
                self.profiles[name] = base3_profile(self.root)
            elif name in self.catalog.get('evidence_inputs',{}):
                self.profiles[name] = evidence_profile(self.root,name)
            else:
                self.profiles[name] = component_profile(self.root, self.catalog['configurations'][name])
        return self.profiles[name]

    def expression(self,node):
        if isinstance(node,(int,float,bool)):
            return node
        if 'profile' in node:
            return select(self.profile(node['profile']),node['selector'])
        if 'amount_matches' in node:
            return int(amount_matches(self.root,self.catalog['configurations'][node['amount_matches']]))
        if 'amount_residual' in node:
            return amount_residual(self.root,self.catalog['configurations'][node['amount_residual']])
        if 'resident_cap' in node:
            records=read_json(self.root,node['resident_cap'])[node.get('ramp','resident_ramp')]
            artifacts=read_json(self.root,node['fixture_manifest'])['artifacts']
            smallest=min(a['size_tok'] for a in artifacts.values())
            # Whole-document residency is full when no fixture document fits
            # in the remaining budget. This is distinct from the 95% steady
            # warm-up criterion: a 10% growth snapshot can exceed 95% while
            # still having room for another document.
            return sum(r['budget_tokens']-r['resident_tokens']<smallest for r in records)
        values=[self.expression(x) for x in node['args']]
        op=node['op']
        if op=='sum':return sum(values)
        if op=='min':return min(values)
        if op=='max':return max(values)
        if op=='subtract':return values[0]-values[1]
        if op=='divide':return values[0]/values[1]
        if op=='multiply':return math.prod(values)
        if op=='less':return int(values[0]<values[1])
        if op=='greater':return int(values[0]>values[1])
        if op=='abs':return abs(values[0])
        if op=='consistent':
            require(values and all(x==values[0] for x in values),'quantities described as equal differ')
            return values[0]
        raise ValueError('unknown expression operation: '+str(op))

    def evaluate(self, item):
        inside(self.root, item['source'])
        if item['method'] == 'component':
            return select(self.profile(item['configuration']), item['selector'])
        if item['method'] == 'json':
            return select(read_json(self.root, item['source']), item['selector'])
        if item['method'] == 'derived':
            return self.expression(item['expression'])
        raise ValueError('unhandled method: '+str(item.get('method')))

    def run(self, experiment=None):
        rows=[]
        for item in self.catalog['items']:
            if experiment and experiment not in item.get('experiments', []):
                continue
            try:
                actual=self.evaluate(item)
                diff=difference(item['expected'],actual)
                ok=diff <= item['tolerance']
                rows.append(dict(id=item['id'], quantity=item['quantity'], expected=item['expected'],
                                 recomputed=actual, diff=diff, status='PASS' if ok else 'FAIL', method=item['method']))
            except (ValueError,KeyError,TypeError,OSError,ZeroDivisionError) as exc:
                rows.append(dict(id=item['id'],quantity=item['quantity'],expected=item['expected'],
                                 recomputed=str(exc),diff=None,status='FAIL',method=item.get('method')))
        require(bool(rows), 'no numerical checks selected')
        return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--experiment',help='limit to one original experiment identifier')
    parser.add_argument('--json-output',type=Path,help='optional complete result file')
    args=parser.parse_args()
    started=time.perf_counter()
    try:
        catalog=read_json(args.root,'paper-numbers.json')
        rows=Verifier(args.root,catalog).run(args.experiment)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print('FAIL: '+str(exc))
        return 1
    print('id\tquantity\tmethod\texpected\trecomputed\tdiff\tstatus')
    for row in rows:
        print('\t'.join(json.dumps(row[k],ensure_ascii=True,separators=(',',':')) if not isinstance(row[k],str)
                        else row[k] for k in ('id','quantity','method','expected','recomputed','diff','status')))
    failed=sum(row['status']=='FAIL' for row in rows)
    print(f'Processed: {len(rows)}; unprocessed: 0; PASS: {len(rows)-failed}; FAIL: {failed}')
    if args.json_output:
        args.json_output.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    print('Total runtime: {:.3f} seconds'.format(time.perf_counter()-started))
    return int(bool(failed))


if __name__=='__main__':
    raise SystemExit(main())
