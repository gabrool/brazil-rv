"""Measure the corrected parser on previously unresolved cached FCA sources only."""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import time

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round5_cvm_fca import load_fca

PROJECT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    run = json.loads((PROJECT/'docs/v2_economic_data_scaling_run.json').read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='fca_fallback_audit')
    args = parser.parse_args()
    output = Path(run['root'])/args.output
    output.mkdir(exist_ok=False)
    accepted = json.loads((PROJECT/'docs/v2_data_inputs.json').read_text())
    family = bound_json(accepted['financial_family'])
    original = bound_json(family['original_family'])
    source = {'path': str(Path(family['original_family']['path']).parent/'fca_identity_documents.json'), 'sha256': original['files']['fca_identity_documents.json']['sha256']}
    documents = [d for d in bound_json(source) if d.get('metadata_source') == 'exact_fca_html_generic_equity_only']
    outcomes, changes, failures, receipts = [], [], [], []
    for i, d in enumerate(documents):
        assert d['receipt'] <= '2024-12-30' and d['reference'] <= '2024-12-30'
        original_source = d['original_fca_source']
        record = {'path': original_source['manifest_path'], 'sha256': original_source['manifest_sha256']}
        try:
            manifest = bound_json(record)
            parsed = load_fca({**d, **{k: date.fromisoformat(d[k]) for k in ('receipt', 'reference')}}, Path(record['path']).parent)
            before = [(s['ticker'], s['class'], s['start'], s['end']) for s in d.get('securities', [])]
            after = [(s['ticker'], s['class'], s['start'], s['end']) for s in parsed.get('securities', [])]
            changed = before != after or d.get('sector_code') != parsed.get('sector_code')
            outcomes.append({'id': d['id'], 'cnpj': d['cnpj'], 'cvm_code': d['cvm_code'], 'changed': changed, 'source': record, 'result_type': parsed['metadata_source'], 'tickers': [s['ticker'] for s in parsed.get('securities', [])]})
            receipts.extend(manifest['sources'])
            if changed:
                changes.append({**d, **parsed, 'sector_label': parsed.get('sector'), 'amended_source': record})
        except (ValueError, KeyError, OSError) as error:
            failures.append({'id': d['id'], 'source': record, 'error': str(error)})
        if (i+1) % 200 == 0:
            write_json_atomic(output/'progress.json', {'done': i+1, 'total': len(documents), 'changes': len(changes), 'failures': len(failures)})
    write_json_atomic(output/'candidate_documents.json', changes)
    write_json_atomic(output/'source_receipts.json', list({s['path']: s for s in receipts}.values()))
    report = {'source_documents': source, 'checked': len(documents), 'parsed': len(outcomes), 'changed_documents': len(changes), 'changed_issuers': len({(x['cnpj'][:8], x['cvm_code']) for x in changes}), 'result_types': dict(Counter(x['result_type'] for x in outcomes)), 'failures': failures, 'outcomes': outcomes, 'candidate_documents': binding(output/'candidate_documents.json'), 'source_receipts': binding(output/'source_receipts.json'), 'seconds': time.perf_counter()-start, 'reproducer': binding(Path(__file__)), 'status': 'bounded_cached_source_audit_candidates_only_no_model_or_full_identity_changes', 'limits': ['Only the 2820 old HTML fallback records are reconsidered; no other passed census repeated and no downloads.', 'Derived global identity, ambiguity, causal availability and dependent-family propagation must be checked before any new store admission. The separately accepted C&A overlay uses its newer source receipts where available.', 'Flat layouts beyond the source-supported shares/stock-exchange path remain explicit parser/source gaps, not blanket security exclusions.']}
    write_json_atomic(output/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('outcomes','failures')}, indent=2), flush=True)
    print('failures',len(failures),flush=True)


if __name__ == '__main__':
    main()
