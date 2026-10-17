"""Register immutable figure procedure after executed dummy-only visual QA."""
import datetime, hashlib, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / 'outputs'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
assert not (OUT / 'evaluation54.json').exists(), 'Register before formal heldout scores'
dummy = OUT / 'visual_qualification_v1'
proof = read(dummy / 'manifest.json')
assert proof['status'] == 'DUMMY_LAYOUT_EXECUTED_NO_FORMAL_RESULTS'
assert proof['formal_data_reads'] == proof['additional_optimizer_updates'] == 0
assert proof['script_sha256'] == sha(ROOT / 'scripts/plot_results54_v1.py')
assert len(proof['outputs']) == 6
for r in proof['outputs']:assert sha(dummy / r['file']) == r['sha256']
values = read(dummy / 'plotted_values.json')
assert values['dummy_only'] is True and values['point_count'] == 264
files = ['scripts/plot_results54_v1.py', 'scripts/watch_result_figures54_v1.py', 'visual-composer/visual-contract.md', 'visual-composer/qa-ledger.md']
record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'DUMMY_RENDER_AND_VISUAL_LAYOUT_QA_COMPLETE_FORMAL_RENDER_WAITS_FOR_EXISTING_SCORING', 'heldout_opened_at_registration': False, 'files_sha256': {p: sha(ROOT / p) for p in files}, 'qualification_manifest_sha256': sha(dummy / 'manifest.json'), 'dummy_images': 3, 'dummy_plotted_values': 264, 'additional_optimizer_updates': 0, 'model_or_hyperparameter_selection': False, 'formal_visual_QA': 'Pending after actual-data rendering', 'local_result_files_written': 0}
p = OUT / 'RESULT_FIGURES_REGISTRATION54_V1.json'
with p.open('x', encoding='utf-8') as f:json.dump(record, f, indent=2)
print(json.dumps({'status': record['status'], 'registration_sha256': sha(p)}), flush=True)
