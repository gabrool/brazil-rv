"""Independent own-filing capital and original receipt audit; no model scoring."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, time, timedelta
from decimal import Decimal
import hashlib
import html as html_text
import io
import json
from pathlib import Path
import pickle
import re
import time as timer
import unicodedata
from urllib.parse import parse_qs, urlsplit
import zipfile

from lxml import html
import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import bound_json, binding

PROJECT = Path(__file__).resolve().parents[1]


def plain(value):
    value = unicodedata.normalize('NFKD', value)
    return ' '.join(''.join(c for c in value if not unicodedata.combining(c)).lower().split())


def number(value, scale):
    return float(Decimal(value.replace('.', '').replace(',', '.')) * scale) if value else None


def outstanding(paid, treasury):
    answer = {}
    for key in ('ON', 'PN'):
        q, t = paid.get(key), treasury.get(key)
        if q == 0 and t is None:
            t = 0
        if q is None or t is None or not 0 <= t <= q:
            return None
        answer[key] = q - t
    return answer


def capital_from_html(payload, document):
    tree = html.fromstring(payload.decode('utf-8'))
    rows = [[c.text_content().strip() for c in row.xpath('./th|./td')]
            for row in tree.xpath('//table//tr')]
    assert len(rows) == 9, document['id']
    unit = plain(rows[0][0])
    assert unit in ('numero de acoes (unidade)', 'numero de acoes (mil)'), unit
    scale = 1000 if unit.endswith('(mil)') else 1
    assert datetime.strptime(rows[0][1], '%d/%m/%Y').date().isoformat() == str(document['reference'])
    quantities = {}
    for offset, label, key in [(1, 'do capital integralizado', 'paid_in'), (5, 'em tesouraria', 'treasury')]:
        assert plain(rows[offset][0]) == label
        values = {}
        for row, expected, code in zip(rows[offset + 1:offset + 4], ('ordinarias', 'preferenciais', 'total'), ('ON', 'PN', 'total'), strict=True):
            assert plain(row[0]) == expected
            values[code] = number(row[1], scale)
        quantities[key] = values
    return {'reference': str(document['reference']), 'quantity_scale': scale, **quantities,
            'shares': outstanding(quantities['paid_in'], quantities['treasury'])}


def main():
    start = timer.perf_counter()
    run = json.loads((PROJECT / 'docs/v2_economic_data_scaling_run.json').read_text())
    output = Path(run['root']) / 'cvm_source_audit'
    output.mkdir(exist_ok=False)
    accepted = json.loads((PROJECT / 'docs/v2_data_inputs.json').read_text())
    family = bound_json(accepted['financial_family'])
    original = bound_json(family['original_family'])
    cache = family['extraction']['cache']
    assert sha256_file(Path(cache['path'])) == cache['sha256']
    documents, evidence = pickle.loads(Path(cache['path']).read_bytes())
    by_id = {d['id']: d for d in documents}
    assert len(by_id) == len(documents)
    assert all(str(d['receipt']) <= '2024-12-30' and str(d['reference']) <= '2024-12-30' for d in documents)
    store = Path(accepted['store']['root'])
    bound_json({'path': str(store / 'manifest.json'), 'sha256': accepted['store']['manifest_sha256']})
    dates = np.load(store / 'date_index.npy')
    cutoffs = np.array([datetime.combine(d, time(15, 45)) for d in dates.astype(object)], dtype='datetime64[us]')
    sources = {}

    def read_source(path, expected):
        path = Path(path)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        assert digest == expected, str(path)
        sources[str(path)] = digest
        return payload

    # Source annual headers, not financial CSV values or later-period accounts.
    annual = bound_json(original['source_manifests']['annual_manifest.json'])
    headers = {}
    for source in annual['files']:
        if source['kind'] not in ('itr', 'dfp'):
            continue
        path = Path(source['path'])
        payload = read_source(path, source['sha256'])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            rows = csv.DictReader(io.StringIO(archive.read(path.stem + '.csv').decode('latin1')), delimiter=';')
            for row in rows:
                if row['DT_RECEB'] > '2024-12-30':
                    continue
                identifier = re.sub(r'\D', '', row['ID_DOC'])
                if identifier not in by_id:
                    continue
                expected = {'id': identifier, 'cnpj': re.sub(r'\D', '', row['CNPJ_CIA']),
                            'cvm_code': row['CD_CVM'].zfill(6), 'reference': row['DT_REFER'],
                            'version': int(row['VERSAO']), 'receipt': row['DT_RECEB'], 'kind': source['kind'].upper()}
                assert all(str(by_id[identifier][k]) == str(v) for k, v in expected.items()), identifier
                assert identifier not in headers or headers[identifier] == expected
                headers[identifier] = expected
    assert set(headers) == set(by_id)
    write_json_atomic(output / 'progress.json', {'stage': 'headers', 'verified_documents': len(headers)})

    # Exact public receipt minutes, keeping same-ID duplicates only when identical.
    rad = bound_json(original['source_manifests']['rad_manifest.json'])
    receipts = {}
    repeated = 0
    for source in rad['files']:
        if source['group'] != 'structured':
            continue
        data = json.loads(read_source(source['path'], source['sha256']))['d']['dados']
        for raw in data.split('$&&*'):
            identifier = re.search(r'NumeroSequencialDocumento=(\d+)', raw)
            if identifier is None:
                identifier = re.search(r"OpenDownloadDocumentos\('([0-9]+)'", raw)
            if identifier is None or identifier[1] not in by_id:
                continue
            fields = [html_text.unescape(re.sub('<[^>]+>', '', f)).strip() for f in raw.split('$&')]
            stamp = re.search(r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})', fields[6])
            instant = datetime.strptime(' '.join(stamp.groups()), '%d/%m/%Y %H:%M')
            if instant.date().isoformat() > '2024-12-30':
                continue
            document = by_id[identifier[1]]
            reference = re.search(r'\d{2}/\d{2}/\d{4}', fields[5])[0]
            assert re.sub(r'\D', '', fields[0]).zfill(6) == document['cvm_code']
            assert datetime.strptime(reference, '%d/%m/%Y').date() == document['reference']
            if fields[8].isdigit():
                assert int(fields[8]) == document['version']
            if identifier[1] in receipts:
                assert receipts[identifier[1]] == instant
                repeated += 1
            receipts[identifier[1]] = instant
    clocks = {}
    for identifier, d in by_id.items():
        # A minute is an interval: receipt at 15:45 cannot precede the cutoff.
        known = receipts[identifier] + timedelta(minutes=1) if identifier in receipts else datetime.combine(d['receipt'] + timedelta(days=1), time())
        clocks[identifier] = int(np.searchsorted(cutoffs, np.datetime64(known), side='left'))

    # Verify the actual financial producer's statement-age field on each identity.
    identity_path = Path(family['original_family']['path']).parent / 'identity.parquet'
    read_source(identity_path, original['files']['identity.parquet']['sha256'])
    identity = pl.read_parquet(identity_path).select('date', 'isin', 'cnpj', 'cvm_code')
    read_source(family['data']['path'], family['data']['sha256'])
    features = pl.read_parquet(family['data']['path']).select('date', 'isin', 'statement_age_sessions')
    joined = features.join(identity, on=['date', 'isin'], how='left', validate='1:1')
    assert joined['cnpj'].null_count() == 0
    by_issuer = defaultdict(list)
    for d in documents:
        by_issuer[(d['cnpj'][:8], d['cvm_code'])].append(clocks[d['id']])
    age_cells = 0
    for key, frame in joined.with_columns(pl.col('cnpj').str.slice(0, 8)).partition_by(['cnpj', 'cvm_code'], as_dict=True).items():
        known = np.array(sorted(set(by_issuer[key])), dtype=int)
        indices = np.searchsorted(dates, frame['date'].to_numpy())
        prior = np.searchsorted(known, indices, side='right') - 1
        expected = np.where(prior >= 0, indices - known[np.maximum(prior, 0)], np.nan) if len(known) else np.full(len(frame), np.nan)
        actual = frame['statement_age_sessions'].to_numpy()
        np.testing.assert_equal(actual, expected, err_msg=str(key))
        age_cells += len(frame)
    write_json_atomic(output / 'progress.json', {'stage': 'receipts', 'exact_receipts': len(receipts), 'statement_age_cells': age_cells})

    # Independent capital HTML arithmetic before own-note corrections. Record
    # all reviewed overrides separately; a later note never repairs another ID.
    overrides = {}
    html_counts = Counter()
    final_counts = {}
    manifest_records = list({(x['document_id'], x['manifest_path']): x for x in evidence['capital_sources']}.values())
    for count, source in enumerate(manifest_records, 1):
        path = Path(source['manifest_path'])
        manifest = json.loads(read_source(path, source['manifest_sha256']))
        identifier = source['document_id']
        document = by_id[identifier]
        if 'disposition' in source:
            item = next(x for x in manifest['documents'] if str(x['document']['id']) == identifier)
            assert all(str(document[k]) == str(v) for k, v in item['document'].items())
            for receipt in item['evidence']:
                read_source(receipt['path'], receipt['sha256'])
            value = outstanding(item['paid_in_shares'], item['treasury_shares']) if item['disposition'] == 'reconciled' else None
            overrides[identifier] = value
            continue
        assert all(str(document[k]) == str(v) for k, v in manifest['document'].items())
        for item in manifest['sources']:
            payload = read_source(path.parent / item['file'], item['sha256'])
            if item['role'] in ('viewer', 'group'):
                tree = html.fromstring(payload.decode('utf-8'))
                form = {x.get('name'): x.get('value') for x in tree.xpath('//input')}
                assert form['hdnNumeroSequencialDocumento'] == identifier
                assert form['hdnCodigoCvm'] == document['cvm_code']
            if item['role'] == 'capital':
                query = parse_qs(urlsplit(item['url']).query)
                assert query['NumeroSequencialDocumento'] == [identifier]
                assert query['Versao'] == [str(document['version'])]
                assert query['DataReferencia'] == [str(document['reference'])]
                parsed = capital_from_html(payload, document)
                assert parsed == manifest['capital'], identifier
                final_counts[identifier] = parsed['shares']
                html_counts[str(parsed['quantity_scale'])] += 1
        if count % 1000 == 0:
            write_json_atomic(output / 'progress.json', {'stage': 'capital', 'manifests_completed': count, 'total': len(manifest_records)})
    final_counts.update(overrides)
    for identifier, value in final_counts.items():
        assert value == by_id[identifier].get('shares'), identifier
    write_json_atomic(output / 'source_receipts.json', [{'path': p, 'sha256': s} for p, s in sorted(sources.items())])
    report = {
        'accepted_financial_family': accepted['financial_family'], 'extracted_documents': cache,
        'source_manifests': {k: v for k, v in original['source_manifests'].items() if k != 'calendar_2025_announced.json'},
        'documents': len(documents), 'exact_receipts': len(receipts), 'date_only_receipts': len(documents) - len(receipts),
        'identical_repeat_receipts': repeated, 'statement_age_cells': age_cells,
        'capital_html_units': dict(html_counts), 'capital_note_dispositions': len(overrides),
        'final_share_count_documents_checked': len(final_counts), 'original_zip_or_no_capital_documents': len(documents) - len(final_counts),
        'source_files_verified': len(sources), 'source_receipts': binding(output / 'source_receipts.json'),
        'mismatches': 0, 'elapsed_seconds': timer.perf_counter() - start, 'changed_sources_or_model_arrays': 0,
        'reproducer': binding(Path(__file__)),
        'limits': [
            'Own-note dispositions retain their previous exact-source human review; this audit verifies identity, hashes and corrected arithmetic, not a new independent interpretation of every PDF note.',
            'Original ZIP capital layouts and financial account values/TTM formulas are separate numerical boundaries, not certified by capital HTML and receipt agreement.',
            'Own-version headers and viewer parameters bind requested versions; preservation fidelity of the current CVM viewer converter cannot be proved from one retrieved archive.',
            'Statement-age equality does not prove every derived metric endpoint/denominator; full issuer valuation, wealth/labels and identity propagation remain.',
            'No 2025/2026 financial account payloads, source modifications, neural forward or model profitability.',
        ],
    }
    write_json_atomic(output / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
