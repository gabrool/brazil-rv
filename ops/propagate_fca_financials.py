"""Propagate a bound identity amendment into financial and event observations."""

import argparse
import copy
from datetime import date
import json
from pathlib import Path
import pickle
import time

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2 import round5_cvm as cvm
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json, source_families

PROJECT = Path(__file__).resolve().parents[1]


def compare(before, after, active_keys):
    keys = ['date', 'isin']
    joined = before.join(after, on=keys, how='full', coalesce=True, suffix='_after')
    joined = joined.join(active_keys, on=keys, how='inner')
    return {c: {
        'valid_before': int(joined[c].is_not_null().sum()),
        'valid_after': int(joined[c+'_after'].is_not_null().sum()),
        'gained': int((joined[c].is_null() & joined[c+'_after'].is_not_null()).sum()),
        'lost': int((joined[c].is_not_null() & joined[c+'_after'].is_null()).sum()),
        'changed_shared': int((joined[c].is_not_null() & joined[c+'_after'].is_not_null() & ~joined[c].eq_missing(joined[c+'_after'])).sum()),
    } for c in before.columns if c not in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    def progress(stage, **fields):
        write_json_atomic(args.output/'progress.json', {'stage': stage, 'seconds': time.perf_counter()-start, **fields})
        print(json.dumps({'stage': stage, **fields}), flush=True)
    run = json.loads((PROJECT/'docs/v2_economic_data_scaling_run.json').read_text(encoding='utf8'))
    pointer = json.loads((PROJECT/'docs/v2_data_inputs.json').read_text(encoding='utf8'))
    family = bound_json(pointer['financial_family'])
    original = bound_json(family['original_family'])
    admission = bound_json(binding(args.identity))
    source = admission['artifacts']['identity']
    assert sha256_file(Path(source['path'])) == source['sha256']
    identity = pl.read_parquet(source['path'])
    assert sha256_file(Path(family['source_identity']['path'])) == family['source_identity']['sha256']
    before_identity = pl.read_parquet(family['source_identity']['path'])
    store = Path(pointer['store']['root'])
    sessions = np.load(store/'date_index.npy').astype('datetime64[D]').tolist()
    isins = np.load(store/'isin_index.npy').tolist()
    assert sessions[-1] <= date(2024,12,31)
    active = np.load(store/'active.npy', mmap_mode='r')
    ti, ni = np.where(active)
    active_keys = pl.DataFrame({'date':[sessions[i] for i in ti], 'isin':[isins[i] for i in ni]})
    cache = family['extraction']['cache']
    assert sha256_file(Path(cache['path'])) == cache['sha256']
    documents, _ = pickle.loads(Path(cache['path']).read_bytes())
    cached_roots = {d['cnpj'][:8] for d in documents}
    missing = {c for c in identity['cnpj'].unique() if c[:8] not in cached_roots}
    root = Path(original['source_root'])
    progress('new_issuer_documents', missing_issuer_roots=len(missing))
    extra, evidence = cvm.load_financial_documents(root, missing, repair_root=Path(cache['path']).parent)
    with (args.output/'new_issuer_documents.pkl').open('wb') as handle:
        pickle.dump((extra, evidence), handle)
    write_json_atomic(args.output/'new_issuer_sources.json', evidence)
    documents.extend(extra)
    # Two previously reviewed own-version capital tables; no date backfill.
    capital = bound_json(run['cvm_capital_admission'])
    by_id = {d['id']: d for d in documents}
    for row in capital['sources']:
        d = by_id[row['id']]
        bound_json(row['source_manifest'])
        assert d.get('shares') is None
        d['shares'] = row['capital']['shares']
    rad = cvm.rad_rows(root)
    changes = bound_json(family['capital_changes'])
    for change in changes:
        change['effective'] = date.fromisoformat(change['effective'])
    progress('financial_propagation', new_documents=len(extra), new_accounts=sum(bool(d.get('accounts')) for d in extra), new_capital=sum(d.get('shares') is not None for d in extra))
    financial, audit = cvm.fundamental_features(copy.deepcopy(documents), rad, identity, sessions, cvm.valuation_market(Path(family['parent']['root'])), changes)
    financial.write_parquet(args.output/'fundamentals.parquet')
    # Existing issuers outside the identity delta must retain every old value/age.
    keys = ['date','isin']
    altered = identity.join(before_identity, on=keys, how='full', coalesce=True, suffix='_before').filter(pl.any_horizontal([~pl.col(c).eq_missing(pl.col(c+'_before')) for c in ['cnpj','cvm_code','class','preferred_class','sector','sector_label']]))
    affected_isins = set(altered['isin'])
    old_financial = pl.read_parquet(family['data']['path'])
    affected_roots = set(identity.filter(pl.col('isin').is_in(affected_isins))['cnpj']) | set(before_identity.filter(pl.col('isin').is_in(affected_isins))['cnpj'])
    unaffected = set(identity.filter(~pl.col('cnpj').is_in(affected_roots))['isin'])
    assert_frame_equal(old_financial.filter(pl.col('isin').is_in(unaffected)).sort(keys), financial.filter(pl.col('isin').is_in(unaffected)).sort(keys), check_exact=True)
    progress('event_propagation', financial_rows=len(financial), unaffected_names_exact=len(unaffected))
    calendar = json.loads((root/'calendar_2025_announced.json').read_text(encoding='utf8'))
    assert sha256_file(Path(calendar['source']['path'])) == calendar['source']['sha256']
    for key in ('available_date','base_through','through'):
        calendar[key] = date.fromisoformat(calendar[key])
    # This is the already bound announced trading calendar, not market outcomes.
    calendar['full_sessions'] = sessions + [date.fromisoformat(d) for d in calendar['sessions']]
    codes = set(identity['cvm_code'])
    events = [r for r in rad if r['cvm_code'] in codes and r['group'] != 'cadastre']
    lags = cvm.header_only_filing_lags(events,[d for d in cvm.filing_headers(root,'itr')+cvm.filing_headers(root,'dfp') if d['cvm_code'] in codes])
    event_state = cvm.event_features([*events,*lags],sessions,calendar)
    event_frame = identity.select('date','isin','cvm_code').join(event_state,on=['date','cvm_code'],how='left').drop('cvm_code')
    event_frame.write_parquet(args.output/'events.parquet')
    families = source_families(bound_json({'path':str(store/'manifest.json'),'sha256':pointer['store']['manifest_sha256']}))
    old_events = pl.read_parquet(families['events']['data']['path'])
    assert_frame_equal(old_events.filter(pl.col('isin').is_in(unaffected)).sort(keys),event_frame.filter(pl.col('isin').is_in(unaffected)).sort(keys),check_exact=True)
    report = {'schema':'FCA_FINANCIAL_EVENT_PROPAGATION_V1','identity_admission':binding(args.identity),'accepted_financial_family':pointer['financial_family'],'accepted_event_family':families['events']['source_manifest'],'prior_document_cache':cache,'new_issuer_cnpjs':sorted(missing),'new_documents':len(extra),'new_documents_with_accounts':sum(bool(d.get('accounts')) for d in extra),'new_documents_with_capital':sum(d.get('shares') is not None for d in extra),'financial_active_changes':compare(old_financial,financial,active_keys),'event_active_changes':compare(old_events,event_frame,active_keys),'unaffected_names_exact':len(unaffected),'financial_audit':audit,'artifacts':{p.stem:binding(p) for p in args.output.iterdir() if p.suffix in ('.parquet','.pkl','.json') and p.name!='progress.json'},'source_code':{p.name:binding(p) for p in (Path(__file__),Path(cvm.__file__))},'seconds':time.perf_counter()-start,'limits':['Derived financial/event observations only; no accepted store, fitted coordinates, forecasts or model scores changed.','Financial formulas are held fixed, not yet an independent numerator/TTM/denominator audit.','New issuer annual accounts retain their own version and unavailable originals/capital remain missing; downstream recovery may add support.','Other identity-dependent families, universe/wealth/labels and original source anomalies remain required before full store acceptance.']}
    write_json_atomic(args.output/'manifest.json',report)
    progress('complete',seconds=report['seconds'])


if __name__ == '__main__':
    main()
