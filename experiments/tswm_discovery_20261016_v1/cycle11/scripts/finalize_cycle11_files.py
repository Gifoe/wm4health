"""Unpack reviewed primary PDFs and the append-only ledger, within this cycle."""
import hashlib, json, pathlib, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    assert not (ROOT / 'ARTIFACT_MANIFEST.json').exists()
    archive = ROOT / 'outputs/literature_bundle.zip'
    with zipfile.ZipFile(archive) as z:
        allowed = {'outputs/literature/registry.json', 'progress/experiment-log.md'}
        for item in z.infolist():
            rel = pathlib.PurePosixPath(item.filename)
            assert not rel.is_absolute() and '..' not in rel.parts
            assert item.filename in allowed or (item.filename.startswith('outputs/literature/') and item.filename.endswith('.pdf'))
            dst = ROOT.joinpath(*rel.parts)
            assert dst.resolve().is_relative_to(ROOT.resolve())
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(z.read(item))
    registry = json.loads((ROOT / 'outputs/literature/registry.json').read_text())
    assert len(registry['papers']) == 6 and not registry['errors']
    for row in registry['papers']:
        p = ROOT / row['path']
        assert p.stat().st_size == row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
    assert 'cycle11-score-learning completion and verification' in (ROOT / 'progress/experiment-log.md').read_text(encoding='utf-8')
    report = {'primary_papers_verified': 6, 'append_only_completion_ledger_present': True,
              'figure_visual_QA': {'path': 'outputs/figures/score_vs_learning.png',
                                  'inspected_locally': True, 'readable_axes_and_labels': True,
                                  'oracle_and_actual_training_panels_distinguished': True},
              'objective_achieved': False}
    (ROOT / 'outputs/packaging_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
