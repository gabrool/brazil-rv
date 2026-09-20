"""Retry only C&A's missing original FCA packages in a new source root."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
from pathlib import Path
import time

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round5_cvm_fca import recover_fca

PROJECT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    run = json.loads((PROJECT / 'docs/v2_economic_data_scaling_run.json').read_text())
    root = Path(run['root']) / 'cvm_fca_missing_probe'
    accepted = json.loads((PROJECT / 'docs/v2_data_inputs.json').read_text())
    family = bound_json(accepted['financial_family'])
    original = bound_json(family['original_family'])
    path = Path(family['original_family']['path']).parent / 'fca_identity_documents.json'
    documents = bound_json({'path': str(path), 'sha256': original['files'][path.name]['sha256']})
    documents = [d for d in documents if d['cnpj'][:8] == '45242914' and d['cvm_code'] == '024848' and d['metadata_source'] != 'original_fca_xml']
    assert len(documents) == 10

    def recover(d):
        d = {**d, **{k: date.fromisoformat(d[k]) for k in ('reference', 'receipt')}}
        manifest = recover_fca(d, root / d['id'])
        outcome = {'id': d['id'], 'status': manifest['status'], 'metadata_source': (manifest.get('metadata') or {}).get('metadata_source'), 'failures': manifest['failures']}
        if (root / d['id'] / 'manifest.json').exists():
            outcome['manifest'] = binding(root / d['id'] / 'manifest.json')
        print(json.dumps(outcome), flush=True)
        return outcome

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(recover, documents))
    write_json_atomic(root / 'recovery.json', {'original_documents': binding(path), 'outcomes': results, 'seconds': time.perf_counter() - start, 'reproducer': binding(Path(__file__)), 'scope': 'Ten existing C&A generic/missing-XML FCA records only; immutable source archives unchanged.'})


if __name__ == '__main__':
    main()
