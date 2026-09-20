"""Check the capital-source cases outside the completed HTML audit."""

import io
import json
from pathlib import Path
import pickle
import re
import time
from xml.etree import ElementTree as ET
import zipfile

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import bound_json, binding

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    run = json.loads((PROJECT / 'docs/v2_economic_data_scaling_run.json').read_text())
    output = Path(run['root']) / 'cvm_source_audit/zip_capital.json'
    if output.exists():
        raise FileExistsError(output)
    accepted = json.loads((PROJECT / 'docs/v2_data_inputs.json').read_text())
    family = bound_json(accepted['financial_family'])
    cache = family['extraction']['cache']
    assert sha256_file(Path(cache['path'])) == cache['sha256']
    documents, evidence = pickle.loads(Path(cache['path']).read_bytes())
    covered = {x['document_id'] for x in evidence['capital_sources']}
    outcomes, unavailable = [], []
    for d in documents:
        if d['id'] in covered:
            continue
        assert str(d['receipt']) <= '2024-12-30' and str(d['reference']) <= '2024-12-30'
        if d.get('shares') is None:
            unavailable.append({'id': d['id'], 'reference': str(d['reference']), 'reason': 'No accepted capital source; not replaced by another period or zero shares.'})
            continue
        sources = [x for x in evidence['original_sources'] if x['document_id'] == d['id'] and Path(x['manifest_path']).parent.parent.name == 'original_zips']
        assert len(sources) == 1
        source = sources[0]
        manifest = bound_json({'path': source['manifest_path'], 'sha256': source['manifest_sha256']})
        path = Path(source['manifest_path']).parent / 'source.zip'
        assert sha256_file(path) == manifest['sha256']
        with zipfile.ZipFile(path) as archive:
            envelope_name = next(n for n in archive.namelist() if n.startswith('FormularioDemonstracaoFinanceira') and n.endswith('.xml'))
            envelope = ET.fromstring(archive.read(envelope_name))
            assert envelope.findtext('NumeroSequencialDocumento') == d['id']
            assert int(envelope.findtext('NumeroVersaoDocumento')) == d['version']
            assert envelope.findtext('DataReferenciaDocumento')[:10] == str(d['reference'])
            assert re.sub(r'\D', '', envelope.findtext('CompanhiaAberta/CodigoCvm')).zfill(6) == d['cvm_code']
            scale = {'1': 1, '2': 1000}[envelope.findtext('CodigoEscalaQuantidade')]
            nested = [n for n in archive.namelist() if n.lower().endswith(('.itr', '.dfp'))]
            paid, treasury = {}, {}
            if nested:
                assert len(nested) == 1
                with zipfile.ZipFile(io.BytesIO(archive.read(nested[0]))) as inner:
                    original = ET.fromstring(inner.read('Documento.xml'))
                    assert re.sub(r'\D', '', original.findtext('CompanhiaAberta/NumeroCnpjCompanhiaAberta')) == d['cnpj']
                    periods = {n.findtext('NumeroIdentificacaoPeriodo') for n in ET.fromstring(inner.read('PeriodoDemonstracaoFinanceira.xml')) if n.findtext('DataFimPeriodo')[:10] == str(d['reference'])}
                    nodes = [n for n in ET.fromstring(inner.read('ComposicaoCapitalSocialDemonstracaoFinanceiraNegocios.xml')) if n.findtext('PeriodoDemonstracaoFinanceira/NumeroIdentificacaoPeriodo') in periods]
                    assert len(nodes) == 1
                    for key, label in [('ON', 'Ordinaria'), ('PN', 'Preferencial')]:
                        paid[key] = float(nodes[0].findtext(f'QuantidadeAcao{label}CapitalIntegralizado'))
                        raw = nodes[0].findtext(f'QuantidadeAcao{label}Tesouraria')
                        treasury[key] = float(raw) if raw else None
                layout = 'relational'
            else:
                payloads = [n for n in archive.namelist() if re.fullmatch(r'\d+DFP\d{2}-\d{2}-\d{4}v\d+\.xml', n)]
                assert len(payloads) == 1
                tree = ET.fromstring(archive.read(payloads[0]))
                assert tree.tag == 'XmlDemonstracoesFinanceiras'
                assert re.sub(r'\D', '', tree.findtext('DadosEmpresa/CnpjEmpresa')) == d['cnpj']
                assert tree.findtext('DadosDFP/EscalaQtdAcoes') == envelope.findtext('CodigoEscalaQuantidade')
                capital = tree.find('DadosDFP/Formulario/DadosEmpresa/ComposicaoCapital')
                for key, label in [('ON', 'Ordinarias'), ('PN', 'Preferenciais')]:
                    raw = capital.findtext(f'CaptalIntegralizado/{label}')
                    paid[key] = float(raw.replace('.', '').replace(',', '.'))
                    raw = capital.findtext(f'Tesouraria/{label}')
                    treasury[key] = float(raw.replace('.', '').replace(',', '.')) if raw else None
                layout = 'flat_DFP'
            shares = {}
            for key in ('ON', 'PN'):
                if paid[key] == 0 and treasury[key] is None:
                    treasury[key] = 0
                assert treasury[key] is not None and 0 <= treasury[key] <= paid[key]
                shares[key] = (paid[key] - treasury[key]) * scale
            assert shares == d['shares'], d['id']
            outcomes.append({'id': d['id'], 'reference': str(d['reference']), 'version': d['version'], 'layout': layout, 'quantity_scale': scale, 'shares': shares, 'source': binding(path), 'manifest': source})
    report = {'cache': cache, 'checked': len(outcomes), 'unavailable': unavailable, 'documents': outcomes, 'mismatches': 0, 'elapsed_seconds': time.perf_counter() - started, 'reproducer': binding(Path(__file__)), 'scope': 'Remaining positive capital counts only; no financial account or full denominator certification.', 'changed_arrays': 0}
    write_json_atomic(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'documents'}, indent=2))


if __name__ == '__main__':
    main()
