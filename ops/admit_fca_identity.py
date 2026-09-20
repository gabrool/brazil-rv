"""Full-axis causal FCA reconstruction from accepted sources and explicit repairs."""

import argparse
import ast
import bisect
import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2 import round5_cvm
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def dated(documents):
    rows = copy.deepcopy(documents)
    for d in rows:
        for k in ('reference', 'receipt'):
            d[k] = date.fromisoformat(d[k])
        for s in d.get('securities', []):
            for k in ('start', 'end'):
                s[k] = date.fromisoformat(s[k])
    return rows


def apply_amendments(documents, amendments, sessions):
    """Apply reviewed literal-source repairs, never an automatic ticker guess."""
    result = copy.deepcopy(documents)
    by_id = {d['id']: d for d in result}
    for amendment in amendments:
        d = by_id[amendment['id']]
        if any(d[k] != amendment[k] for k in ('cnpj', 'cvm_code')) or d['securities'] != amendment['before']:
            raise ValueError('FCA amendment differs from its exact source identity/content')
        for evidence in amendment['evidence']:
            if sha256_file(Path(evidence['path'])) != evidence['sha256']:
                raise ValueError('FCA correction source bytes changed')
        d['securities'] = copy.deepcopy(amendment['after'])
        known = bisect.bisect_left(sessions, date.fromisoformat(amendment['known_date']))
        for security in d['securities']:
            security['source_identity_known_index'] = max(d['available_index'], known)
        d['source_identity_amendment'] = amendment
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='fca_identity_admission')
    parser.add_argument('--amendments', type=Path)
    parser.add_argument('--baseline-proof', type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    run = json.loads((PROJECT/'docs/v2_economic_data_scaling_run.json').read_text(encoding='utf8'))
    output = Path(run['root'])/args.output
    output.mkdir(parents=True, exist_ok=False)
    def progress(stage, **details):
        write_json_atomic(output/'progress.json', {'stage': stage, 'seconds': time.perf_counter()-start, **details})
        print(json.dumps({'stage': stage, **details}), flush=True)
    accepted = json.loads((PROJECT/'docs/v2_data_inputs.json').read_text(encoding='utf8'))
    family = bound_json(accepted['financial_family'])
    original = bound_json(family['original_family'])
    docs_source = {'path':str(Path(family['original_family']['path']).parent/'fca_identity_documents.json'), 'sha256':original['files']['fca_identity_documents.json']['sha256']}
    documents = bound_json(docs_source)
    fallback = bound_json(run['fca_fallback_audit'])
    cea = bound_json(run['cea_identity_admission'])
    replacements = {d['id']:d for d in bound_json(fallback['candidate_documents'])}
    replacements.update({d['id']:d for d in bound_json(cea['artifacts']['fca_documents'])})
    updated = []
    for d in documents:
        replacement = replacements.get(d['id'], d)
        for k in ('id','cnpj','cvm_code','kind','reference','version','receipt','available_index'):
            assert d[k] == replacement[k], (d['id'], k)
        updated.append(replacement)
    assert set(replacements).issubset({d['id'] for d in documents})
    parent = Path(family['parent']['root'])
    bound_json({'path':str(parent/'manifest.json'),'sha256':family['parent']['manifest_sha256']})
    sessions, isins, observations = round5_cvm.store_axes_and_identity_observations(parent)
    if args.amendments:
        updated = apply_amendments(updated, json.loads(args.amendments.read_text(encoding='utf8'))['amendments'], sessions)
    write_json_atomic(output/'fca_documents.json', updated)
    assert sessions[-1] <= date(2024,12,31) and observations['trade_date'].max() <= date(2024,12,31)
    progress('original_baseline', source_documents=len(documents), replacement_documents=len(replacements), observations=len(observations))
    # Execute only the original pure identity function to reproduce sealed rows.
    # This readout provenance does not load/alter any old fit or worktree.
    old_source = subprocess.check_output(['git','show','43250b0:research/src/brazil_rv/v2/round5_cvm.py'],cwd=PROJECT)
    old_ast = ast.parse(old_source)
    node = next(n for n in old_ast.body if isinstance(n,ast.FunctionDef) and n.name=='build_identity')
    namespace = dict(vars(round5_cvm))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'43250b0:build_identity','exec'),namespace)
    assert sha256_file(Path(family['source_identity']['path'])) == family['source_identity']['sha256']
    sealed = pl.read_parquet(family['source_identity']['path']).sort('date','isin')
    if args.baseline_proof:
        proof = json.loads(args.baseline_proof.read_text(encoding='utf8'))
        assert proof['source_documents'] == docs_source and proof['baseline'] == family['source_identity']
        assert proof['old_function_source_sha256'] == hashlib.sha256(old_source).hexdigest()
        assert proof['sealed_baseline_exact_rows'] == len(sealed)
        old = sealed
    else:
        old = namespace['build_identity'](dated(documents),observations,sessions,isins).sort('date','isin')
        assert_frame_equal(old,sealed,check_exact=True)
    progress('amended_identity', sealed_rows_exact=len(old))
    try:
        candidate = round5_cvm.build_identity(dated(updated),observations,sessions,isins).sort('date','isin')
    except ValueError as error:
        trace = error.__traceback__
        while trace and trace.tb_frame.f_code.co_name != 'build_identity':
            trace = trace.tb_next
        local = trace.tb_frame.f_locals if trace else {}
        record = {'error':str(error),'date':str(local.get('current_date')),'isin':local.get('isin'),'current_document':local.get('document'),'current_security':local.get('security'),'prior_mapping':local.get('mapped',{}).get(local.get('isin')),'sealed_baseline_exact_rows':len(old)}
        write_json_atomic(output/'identity_conflict.json',record)
        progress('source_conflict', error=str(error))
        return
    candidate.write_parquet(output/'identity.parquet')
    keys = ['date','isin']
    added = candidate.join(old.select(keys),on=keys,how='anti')
    removed = old.join(candidate.select(keys),on=keys,how='anti')
    both = candidate.join(old,on=keys,suffix='_before')
    columns = [c for c in old.columns if c not in keys]
    changed = both.filter(pl.any_horizontal([~pl.col(c).eq_missing(pl.col(c+'_before')) for c in columns]))
    for name,frame in [('added',added),('removed',removed),('changed',changed)]:
        frame.write_parquet(output/f'{name}.parquet')
    store = Path(accepted['store']['root'])
    active = np.load(store/'active.npy',mmap_mode='r')
    assert np.array_equal(np.load(store/'date_index.npy'),np.array(sessions,dtype='datetime64[D]'))
    assert np.load(store/'isin_index.npy').tolist() == isins
    date_index={d:i for i,d in enumerate(sessions)}
    isin_index={s:i for i,s in enumerate(isins)}
    def active_count(frame):
        return sum(bool(active[date_index[d],isin_index[s]]) for d,s in frame.select(keys).iter_rows())
    counts={name:{'rows':len(frame),'active':active_count(frame),'names':frame['isin'].n_unique()} for name,frame in [('added',added),('removed',removed),('changed',changed)]}
    progress('causal_prefix', **counts)
    cutoff=next(i for i,d in enumerate(sessions) if d>=date(2021,1,4))
    prefix=round5_cvm.build_identity(dated([d for d in updated if d['available_index']<cutoff]),observations.filter(pl.col('trade_date')<sessions[cutoff]),sessions[:cutoff],isins).sort('date','isin')
    assert_frame_equal(prefix,candidate.filter(pl.col('date')<sessions[cutoff]),check_exact=True)
    report={'schema':'FCA_GLOBAL_IDENTITY_ADMISSION_V1','source_documents':docs_source,'amendments':{'fallback':run['fca_fallback_audit'],'cea':run['cea_identity_admission']},'parent':family['parent'],'baseline':family['source_identity'],'old_function_source_commit':'43250b0','old_function_source_sha256':hashlib.sha256(old_source).hexdigest(),'sealed_baseline_exact_rows':len(old),'candidate_rows':len(candidate),'differences':counts,'changed_fields':{c:int((~both[c].eq_missing(both[c+'_before'])).sum()) for c in columns},'future_source_and_observation_deletion_prefix_exact':True,'prefix_end':str(sessions[cutoff-1]),'artifacts':{p.stem:binding(p) for p in output.iterdir() if p.suffix in ('.parquet','.json')},'reproducer':binding(Path(__file__)),'implementation':binding(Path(round5_cvm.__file__)),'seconds':time.perf_counter()-start,'status':'full_identity_candidate_reconstructed_differences_require_source_review_before_dependent_families','limits':['No store, universe, quotes, labels, auxiliary tensors, fits or profitability changed.','Metadata changes include provenance; core issuer/class/sector changes must be distinguished and reviewed.']}
    report['source_corrections'] = binding(args.amendments) if args.amendments else None
    report['reused_baseline_proof'] = binding(args.baseline_proof) if args.baseline_proof else None
    report['artifacts'].pop('progress', None)
    write_json_atomic(output/'manifest.json',report)
    progress('complete', candidate_rows=len(candidate), **counts)


if __name__=='__main__':
    main()
