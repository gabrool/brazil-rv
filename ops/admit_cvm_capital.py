"""Quantify exact-filing capital recovery without mutating accepted coordinates."""

import copy
from datetime import date
import json
from pathlib import Path
import pickle
import time

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from audit_cvm_sources import capital_from_html
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round5_cvm import fundamental_features, rad_rows, valuation_market

PROJECT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    run = json.loads((PROJECT / 'docs/v2_economic_data_scaling_run.json').read_text())
    root = Path(run['root'])
    output = root / 'cvm_capital_admission'
    output.mkdir(exist_ok=False)
    accepted = json.loads((PROJECT / 'docs/v2_data_inputs.json').read_text())
    family = bound_json(accepted['financial_family'])
    original = bound_json(family['original_family'])
    assert sha256_file(Path(family['extraction']['cache']['path'])) == family['extraction']['cache']['sha256']
    documents, _ = pickle.loads(Path(family['extraction']['cache']['path']).read_bytes())
    documents = [d for d in documents if d['cnpj'][:8] == '45242914' and d['cvm_code'] == '024848']
    assert documents and all(str(d['receipt']) <= '2024-12-30' for d in documents)
    updated = copy.deepcopy(documents)
    sources = []
    for d in updated:
        if d['id'] not in ('96365', '114352'):
            continue
        manifest_path = root / 'cvm_capital_missing_probe' / d['id'] / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert all(str(d[k]) == str(v) for k, v in manifest['document'].items())
        for source in manifest['sources']:
            path = manifest_path.parent / source['file']
            assert sha256_file(path) == source['sha256']
            if source['role'] == 'capital':
                capital = capital_from_html(path.read_bytes(), d)
                assert capital == manifest['capital']
        assert d.get('shares') is None
        d['shares'] = capital['shares']
        sources.append({'id': d['id'], 'source_manifest': binding(manifest_path), 'reference': str(d['reference']), 'original_receipt_date': str(d['receipt']), 'capital': capital})
    assert len(sources) == 2
    identity_binding = family['source_identity']
    assert sha256_file(Path(identity_binding['path'])) == identity_binding['sha256']
    identity = pl.read_parquet(identity_binding['path']).filter((pl.col('cnpj').str.slice(0, 8) == '45242914') & (pl.col('cvm_code') == '024848'))
    parent = Path(family['parent']['root'])
    bound_json({'path': str(parent / 'manifest.json'), 'sha256': family['parent']['manifest_sha256']})
    sessions = np.load(parent / 'date_index.npy').astype(object).tolist()
    market = valuation_market(parent)
    events = bound_json(family['capital_changes'])
    events = [e for e in events if e['cnpj'][:8] == '45242914' and e['cvm_code'] == '024848']
    for event in events:
        event['effective'] = date.fromisoformat(event['effective'])
    ids = {d['id'] for d in documents}
    rad = [r for r in rad_rows(Path(original['source_root'])) if r['id'] in ids]
    before, _ = fundamental_features(documents, rad, identity, sessions, market, events)
    after, _ = fundamental_features(updated, rad, identity, sessions, market, events)
    before, after = before.sort('date', 'isin'), after.sort('date', 'isin')
    assert sha256_file(Path(family['data']['path'])) == family['data']['sha256']
    sealed = pl.read_parquet(family['data']['path']).join(identity.select('date', 'isin'), on=['date', 'isin'], how='semi').sort('date', 'isin')
    assert_frame_equal(before, sealed, check_exact=True)
    changes = []
    affected = np.zeros(len(before), dtype=bool)
    for column in before.columns:
        equal = before[column].eq_missing(after[column]).to_numpy()
        mask = ~equal
        affected |= mask
        if mask.any():
            rows = before.filter(pl.Series(mask))
            changes.append({'column': column, 'cells': int(mask.sum()), 'first': str(rows['date'].min()), 'last': str(rows['date'].max()), 'old_null': rows[column].null_count(), 'new_null': after.filter(pl.Series(mask))[column].null_count()})
    assert_frame_equal(before.select('date', 'isin'), after.select('date', 'isin'))
    for d in updated:
        if d['id'] in ('96365', '114352'):
            entry = next(x for x in sources if x['id'] == d['id'])
            entry['available_session'] = d['available_index']
            entry['first_available_decision'] = str(sessions[d['available_index']])
    first_known = min(x['first_available_decision'] for x in sources)
    assert_frame_equal(before.filter(pl.col('date') < date.fromisoformat(first_known)), after.filter(pl.col('date') < date.fromisoformat(first_known)), check_exact=True)
    before.write_parquet(output / 'before.parquet')
    after.write_parquet(output / 'candidate.parquet')
    report = {'original_family': accepted['financial_family'], 'extracted_documents': family['extraction']['cache'], 'source_identity': identity_binding, 'sources': sources, 'issuer_rows': len(before), 'changed_rows': int(affected.sum()), 'changed_columns': changes, 'sealed_baseline_exact': True, 'date_and_isin_axes_exact': True, 'before_earliest_receipt_exact': True, 'unchanged_other_issuers': 'Overlay contains only existing exact C&A date/ISIN rows; all other original rows are reused without regeneration.', 'before': binding(output / 'before.parquet'), 'candidate': binding(output / 'candidate.parquet'), 'reproducer': binding(Path(__file__)), 'seconds': time.perf_counter() - start, 'status': 'source_recovered_and_changed_family_overlay_quantified_not_yet_admitted_to_full_store_or_old_fits', 'limits': ['No full model store or static policy cache is changed. Integrate this explicit overlay with identity/wealth/label repairs in the separately accepted derived contract.', 'Capital source dates are original own-version receipts; the 2019 reference filing was received in August2020 and is never backdated.', 'Existing financial calculations are held fixed for paired attribution; this is not an independent numerical audit of every metric or an alpha result.']}
    write_json_atomic(output / 'manifest.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
