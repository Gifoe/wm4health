"""Create a remote-only, closed-evidence text/SVG publication snapshot."""
import datetime, hashlib, json, math, os, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / 'outputs'

def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))

def main():
    assert read(OUT / 'postprocess54_v3.json')['status'] == 'COMPLETE_PRIMARY12_AND_FOUR_MATCHED_EXISTING_EM_OPTION_CONTROLS'
    assert read(OUT / 'watch_result_figures54_v1.json')['status'] == 'COMPLETE_RENDER_PENDING_ACTUAL_IMAGE_INSPECTION'
    risk = read(OUT / 'causal_risk_interpretation_v1/summary.json')
    assert risk['status'] == 'COMPLETE_READ_ONLY_CAUSAL_RISK_INTERPRETATION_OF_ALL16_FROZEN_FITS'
    gaussian_path = OUT / 'gaussian_belief_audit_v1/summary.json'
    gaussian = read(gaussian_path)
    assert gaussian['status'] == 'COMPLETE_ALL16_FIXED_HORIZON_GAUSSIAN_BELIEF_AUDIT'
    assert len(gaussian['records']) == 96 and gaussian['additional_optimizer_updates'] == 0
    assert gaussian['model_or_hyperparameter_selection'] is False
    for rel, value in read(OUT / 'GAUSSIAN_BELIEF_REGISTRATION54_V1.json')['files_sha256'].items():
        assert sha((ROOT / rel).read_bytes()) == value
    caveat = read(OUT / 'gaussian_readout_caveat_v1.json')
    assert caveat['status'] == 'EXACT_ORACLE_POINT_READOUT_COUNTEREXAMPLE_CONFIRMED'
    figure_manifest = OUT / 'result_figures_v1/manifest.json'
    qa = read(OUT / 'RESULT_FIGURES_ACTUAL_QA54_V1.json')
    assert qa['status'] == 'ACTUAL_ALL16_RESULT_IMAGES_INSPECTED'
    assert qa['figure_manifest_sha256'] == sha(figure_manifest.read_bytes())
    assert qa['local_result_files_written'] == 0
    summary = read(OUT / 'summary54_with_em_option_v1.json')
    assert summary['total_completed_fits'] == 16 and summary['all16_models_and_readouts_frozen_before_test'] is True
    for path in [OUT / 'SELECTION_SEAL54_V1.json', OUT / 'em_option_learning_v1/SELECTION_SEAL54_EM_TRUE_V1.json']:
        for row in read(path)['fits']:assert sha(pathlib.Path(row['checkpoint']).read_bytes()) == row['checkpoint_sha256']
    for path in [OUT / 'READOUT_SEAL54_V1.json', OUT / 'em_option_learning_v1/evaluation/READOUT_SEAL_EM_TRUE54_V1.json']:
        for row in read(path)['readouts']:assert sha(pathlib.Path(row['file']).read_bytes()) == row['sha256']
    for rel, value in read(OUT / 'RESULT_FIGURES_REGISTRATION54_V1.json')['files_sha256'].items():
        assert sha((ROOT / rel).read_bytes()) == value
    for row in read(figure_manifest)['outputs']:
        assert sha((figure_manifest.parent / row['file']).read_bytes()) == row['sha256']

    dest = OUT / 'closed_publication_snapshot_v3'
    assert not dest.exists(), 'Preserve the closed snapshot; version a changed export procedure'
    dest.mkdir()
    prior = {}
    for directory in OUT.iterdir():
        manifest = directory / 'manifest.json'
        if directory.is_dir() and 'publication_snapshot' in directory.name and directory != dest and manifest.exists():
            data = read(manifest)
            for row in data.get('files', []):prior[row['path']] = row['sha256']

    candidates = {}
    def add(p, name=None):candidates[name or p.relative_to(ROOT).as_posix()] = p
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ['.md', '.json', '.ps1', '.py']:add(p)
    for folder in [ROOT / 'scripts', ROOT / 'visual-composer']:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ['.py', '.md', '.json', '.ps1'] and '__pycache__' not in p.parts:add(p)
    # Prune immutable publication copies instead of recursively republishing their contents.
    for folder, directories, files in os.walk(OUT):
        directories[:] = [d for d in directories if 'publication_snapshot' not in d and d not in ['__pycache__', 'visual_qualification_v1'] and not d.startswith('.venv')]
        for name in files:
            p = pathlib.Path(folder) / name
            if p.suffix not in ['.json', '.jsonl', '.csv', '.log', '.md', '.svg']:continue
            if 'PUBLICATION_RECEIPT' in name:continue
            add(p)
    # Dummy fixtures stay visibly separate; retain only the execution/hash receipt.
    add(OUT / 'visual_qualification_v1/manifest.json')
    index = '# C54 closed evidence index\n\n'
    index += 'All12 primary fits and4 initialization-matched existing-source-option controls completed. All16 models and diagnostic readouts were frozen before the first heldout score. This is an adapted native-component linear-recipe audit, with the previously qualified ordinary EM covariance correction. It is not a full original-paper benchmark reproduction.\n\n'
    index += '- [Complete experiment tables](experimental_results.md)\n- [Read-only causal risk analysis](CAUSAL_RISK_REPORT54_V1.md)\n- [Fixed-horizon Gaussian belief analysis](GAUSSIAN_BELIEF_REPORT54_V1.md)\n- [Required interpretation guard for the Gaussian readout](GAUSSIAN_READOUT_CAVEAT54_V1.md)\n- [Scientific disposition and next discriminator](C54_SCIENTIFIC_DECISION_V1.md)\n- [All96 Gaussian score rows](outputs/gaussian_belief_audit_v1/gaussian_scores.csv)\n- [All-fit recovery and dynamics figure](outputs/result_figures_v1/recovery_and_dynamics.svg)\n- [Raw and normalized causal error figure](outputs/result_figures_v1/causal_risk.svg)\n- [Both initialization-matched source-option comparisons](outputs/result_figures_v1/paired_source_option.svg)\n- [Exact plotted values](outputs/result_figures_v1/plotted_values.json)\n- [Actual-data image QA receipt](outputs/RESULT_FIGURES_ACTUAL_QA54_V1.json)\n- [Source-option protocol](EM_OPTION_LEARNING54_V1.md)\n- [Numerical verification](outputs/verification54.json)\n\n'
    index += 'One data draw per task; initialization repetitions are not independent domains. Full-sequence smoothing is retrospective. Forecasts use only the first50 observations. State-supervised maps are training-only diagnostics. The conditional-mean error is separated from irreducible realized oracle noise and the finite-sample cross term. Sequence-bootstrap intervals are pointwise and conditional on fixed fits. No new method, full-distribution calibration result or final research direction is accepted automatically.\n\n'
    index += 'Raw signals, model weights, native arrays, and PNG previews remain on authorized server storage. Editable SVG and compact reproducibility evidence are included here. No result files are synchronized to local disk. Overall research objective remains ACTIVE_UNACHIEVED; accepted final directions:0.\n'
    p = dest / 'C54_CLOSED_EVIDENCE_INDEX_V1.md';p.write_text(index, encoding='utf-8');add(p, p.name)
    rows = []
    for public, p in sorted(candidates.items()):
        original = p.read_bytes(); content = original.decode('utf-8-sig'); data = content.encode('utf-8')
        if prior.get(public) == sha(data):continue
        assert len(data) <= 1500000, (public, len(data))
        assert not any(t in public.split('/') for t in ['Lib', 'site-packages', '__pycache__', '.venv_dcl'])
        target = dest / 'files' / public;target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes(data)
        rows.append({'path': public, 'bytes': len(data), 'sha256': sha(data), 'original_sha256': sha(original), 'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest(), 'parts': math.ceil(len(content) / 3500), 'server_snapshot_file': str(target), 'publication_transform': 'Remove UTF8 BOM' if original.startswith(b'\xef\xbb\xbf') else None})
    manifest = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'C54_ALL16_CLOSED_SCORING_WITH_GAUSSIAN_READOUT_INTERPRETATION_GUARD_AND_IMAGE_QA', 'files': rows, 'file_count': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'total_content_parts': sum(r['parts'] for r in rows), 'accepted_final_directions': 0, 'goal_status': 'active', 'local_result_files_written': 0, 'exclusions': 'Virtual environments, upstream package bodies, model weights, raw signal/NPZ arrays, PNG previews, private transport helpers and prior publication copies.'}
    p = dest / 'manifest.json';p.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'file_count': len(rows), 'bytes': manifest['bytes'], 'parts': manifest['total_content_parts'], 'manifest_sha256': sha(p.read_bytes())}), flush=True)

if __name__ == '__main__':main()
