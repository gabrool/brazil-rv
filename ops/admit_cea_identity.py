"""Source-bound C&A identity and financial overlays; no accepted-store mutation."""

import copy
from datetime import date
import json
from pathlib import Path
import pickle
import re
import time

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from audit_cvm_sources import capital_from_html
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round5_cvm import build_identity, fundamental_features, rad_rows, valuation_market
from brazil_rv.v2.round5_cvm_fca import load_fca

PROJECT = Path(__file__).resolve().parents[1]
ISIN = 'BRCEABACNOR1'


def issuer(d):
    return d['cnpj'][:8] == '45242914' and d['cvm_code'] == '024848'


def dated(d):
    d = copy.deepcopy(d)
    for k in ('reference', 'receipt'):
        d[k] = date.fromisoformat(d[k])
    for security in d.get('securities', []):
        for k in ('start', 'end'):
            security[k] = date.fromisoformat(security[k])
    return d


def main():
    start = time.perf_counter()
    pointer = json.loads((PROJECT / 'docs/v2_economic_data_scaling_run.json').read_text())
    root = Path(pointer['root'])
    output = root / 'cea_identity_admission' / 'completed'
    output.mkdir(exist_ok=False)
    accepted = json.loads((PROJECT / 'docs/v2_data_inputs.json').read_text())
    family = bound_json(accepted['financial_family'])
    original = bound_json(family['original_family'])
    original_docs = {'path': str(Path(family['original_family']['path']).parent / 'fca_identity_documents.json'), 'sha256': original['files']['fca_identity_documents.json']['sha256']}
    all_docs = bound_json(original_docs)
    before_docs = [d for d in all_docs if issuer(d)]
    after_docs, sources = [], []
    for d in before_docs:
        recovered = root / 'cvm_fca_missing_probe' / d['id']
        if not (recovered / 'manifest.json').exists():
            after_docs.append(d)
            continue
        metadata = load_fca(dated(d), recovered)
        after_docs.append({**d, **metadata, 'sector_label': metadata.get('sector')})
        source = binding(recovered / 'manifest.json')
        sources.append({'id': d['id'], 'source': source, 'original_receipt': d['receipt'], 'available_index': d['available_index'], 'metadata_source': metadata['metadata_source'], 'decoding': metadata.get('xml_decoding'), 'explicit_tickers': [s['ticker'] for s in metadata['securities']]})
    assert len(after_docs) == len(before_docs) == 11
    # New numeric sector observations do not invent an earlier global mapping.
    for d in after_docs:
        if d.get('sector_code'):
            assert any(x.get('sector_code') == d['sector_code'] and x.get('sector_label') == d['sector_label'] and x['available_index'] <= d['available_index'] for x in all_docs)
    parent = Path(family['parent']['root'])
    parent_manifest = bound_json({'path': str(parent / 'manifest.json'), 'sha256': family['parent']['manifest_sha256']})
    sessions = np.load(parent / 'date_index.npy').astype('datetime64[D]').tolist()
    assert sessions[-1] <= date(2024, 12, 31)
    observations, quote_sources = [], []
    for source in parent_manifest['sources']:
        path = Path(source['path'])
        match = re.fullmatch(r'equities_daily_(\d{4})\.parquet', path.name)
        if not match or not 2009 <= int(match[1]) <= 2024:
            continue
        assert sha256_file(path) == source['sha256']
        observations.append(pl.scan_parquet(path).filter(pl.col('isin') == ISIN).select('trade_date', 'ticker', 'isin', 'issuer_short_name', 'security_spec_base').collect())
        quote_sources.append(source)
    observations = pl.concat(observations).unique().sort('trade_date')
    before = build_identity([dated(d) for d in before_docs], observations, sessions, [ISIN])
    after = build_identity([dated(d) for d in after_docs], observations, sessions, [ISIN])
    assert sha256_file(Path(family['source_identity']['path'])) == family['source_identity']['sha256']
    sealed = pl.read_parquet(family['source_identity']['path']).filter(pl.col('isin') == ISIN)
    assert_frame_equal(before, sealed, check_exact=True)
    assert_frame_equal(before, after.join(before.select('date', 'isin'), on=['date', 'isin'], how='semi'), check_exact=True)
    # Future-document deletion must not alter any earlier derived identity row.
    cutoff = next(d['available_index'] for d in after_docs if d['id'] == '112279')
    past = build_identity([dated(d) for d in after_docs if d['available_index'] < cutoff], observations, sessions[:cutoff], [ISIN])
    assert_frame_equal(past, after.filter(pl.col('date') < sessions[cutoff]), check_exact=True)
    for label, rows in [('before', before), ('candidate', after)]:
        rows.write_parquet(output / f'identity_{label}.parquet')
    write_json_atomic(output / 'fca_documents.json', after_docs)
    cache = family['extraction']['cache']
    assert sha256_file(Path(cache['path'])) == cache['sha256']
    financial, _ = pickle.loads(Path(cache['path']).read_bytes())
    financial = [d for d in financial if issuer(d)]
    updated = copy.deepcopy(financial)
    capital_sources = []
    for d in updated:
        if d['id'] not in ('96365', '114352'):
            continue
        path = root / 'cvm_capital_missing_probe' / d['id'] / 'manifest.json'
        manifest = json.loads(path.read_text())
        assert all(str(d[k]) == v for k, v in manifest['document'].items())
        for s in manifest['sources']:
            payload = path.parent / s['file']
            assert sha256_file(payload) == s['sha256']
            if s['role'] == 'capital':
                capital = capital_from_html(payload.read_bytes(), d)
        assert capital == manifest['capital'] and d.get('shares') is None
        d['shares'] = capital['shares']
        capital_sources.append({'id': d['id'], 'source': binding(path), 'capital': capital})
    events = [e for e in bound_json(family['capital_changes']) if issuer(e)]
    for e in events:
        e['effective'] = date.fromisoformat(e['effective'])
    ids = {d['id'] for d in financial}
    rad = [r for r in rad_rows(Path(original['source_root'])) if r['id'] in ids]
    market = valuation_market(parent)
    f_before, _ = fundamental_features(copy.deepcopy(financial), rad, before, sessions, market, events)
    f_identity, _ = fundamental_features(copy.deepcopy(financial), rad, after, sessions, market, events)
    f_after, _ = fundamental_features(updated, rad, after, sessions, market, events)
    assert sha256_file(Path(family['data']['path'])) == family['data']['sha256']
    f_sealed = pl.read_parquet(family['data']['path']).filter(pl.col('isin') == ISIN)
    assert_frame_equal(f_before, f_sealed, check_exact=True)
    assert_frame_equal(f_before, f_after.join(before.select('date', 'isin'), on=['date', 'isin'], how='semi'), check_exact=True)
    store = Path(accepted['store']['root'])
    isins = np.load(store / 'isin_index.npy').tolist()
    assert np.array_equal(np.load(store / 'date_index.npy'), np.array(sessions, dtype='datetime64[D]'))
    active = np.load(store / 'active.npy', mmap_mode='r')[:, isins.index(ISIN)]
    active_dates = [d for d, ok in zip(sessions, active, strict=True) if ok]
    added = after.join(before.select('date', 'isin'), on=['date', 'isin'], how='anti')
    details = []
    for col in f_after.columns[2:]:
        old = f_before[col]
        full = f_after[col]
        capital_changed = ~f_identity[col].eq_missing(full)
        active_rows = f_after.filter(pl.col('date').is_in(active_dates))
        details.append({'column': col, 'before_nonnull': len(old)-old.null_count(), 'candidate_nonnull': len(full)-full.null_count(), 'active_candidate_nonnull': len(active_rows)-active_rows[col].null_count(), 'capital_only_changed': int(capital_changed.sum())})
    for label, rows in [('before', f_before), ('identity_only', f_identity), ('candidate', f_after)]:
        rows.write_parquet(output / f'fundamentals_{label}.parquet')
    report = {'schema': 'CEA_SOURCE_IDENTITY_ADMISSION_V1', 'accepted_financial_family': accepted['financial_family'], 'source_documents': original_docs, 'sources': sources, 'capital_sources': capital_sources, 'quote_sources': quote_sources, 'source_receipt_dates_preserved': True, 'identity_before_rows': len(before), 'identity_candidate_rows': len(after), 'identity_added_rows': len(added), 'identity_added_active_rows': added.filter(pl.col('date').is_in(active_dates)).height, 'identity_added_first': str(added['date'].min()), 'identity_added_last': str(added['date'].max()), 'no_existing_rows_lost_or_changed': True, 'sealed_identity_and_financial_baseline_exact': True, 'future_document_removal_prior_rows_exact': True, 'financial_columns': details, 'artifacts': {p.stem: binding(p) for p in output.iterdir() if p.suffix in ('.json', '.parquet')}, 'code': {str(p.resolve().relative_to(PROJECT)): binding(p) for p in [Path(__file__), PROJECT/'research/src/brazil_rv/v2/round5_cvm.py', PROJECT/'research/src/brazil_rv/v2/round5_cvm_fca.py']}, 'seconds': time.perf_counter()-start, 'status': 'explicit_derived_identity_financial_overlay_accepted_for_next_store_build_not_applied_to_old_fits', 'limits': ['Full store and other identity-dependent auxiliary families require propagation and separate acceptance; no old checkpoint or policy cache changes.', 'Financial formulas held fixed for attribution; original capital/receipt audit is separate from full financial numerator and wealth/label audit.', 'Source ZIP requests that returned service errors are preserved. Exact HTML ticker is accepted without adopting its modern enum labels.', 'Only this issuer overlay is admitted; global re-extraction impact remains to be audited before a full store rebuild.']}
    write_json_atomic(output / 'manifest.json', report)
    print(json.dumps({k: v for k, v in report.items() if k in ('identity_before_rows', 'identity_candidate_rows', 'identity_added_rows', 'identity_added_active_rows', 'identity_added_first', 'identity_added_last', 'seconds', 'financial_columns')}, indent=2))


if __name__ == '__main__':
    main()
