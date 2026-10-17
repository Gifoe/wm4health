"""Clean-directory native subset reproduction, without historical repair receipts."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root', type=Path, required=True, help='New isolated run directory; do not reuse the original experiment directory')
parser.add_argument('--storage-root', type=Path, required=True, help='Dedicated new data-cache/checkpoint/artifact subdirectory')
parser.add_argument('--upstream', type=Path, required=True, help='Original TSWM source matching the published 125-file pin')
parser.add_argument('--data-root', type=Path, required=True, help='Parent containing greenhouse3_original_v1 with both original experiment folders')
parser.add_argument('--execute', action='store_true', help='After staging, execute qualifications, codec, all 40 fits and locked evaluation')
parser.add_argument('--resume', action='store_true', help='Reuse only verified complete steps in this same clean run; partial fit folders cause a stop')
args = parser.parse_args()

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

run = args.run_root.resolve(); storage = args.storage_root.resolve(); upstream = args.upstream.resolve(); data = args.data_root.resolve()
assert run != HERE and run not in HERE.parents and HERE not in run.parents, 'Use a separate clean run directory'
assert upstream.is_dir() and data.is_dir()
assert storage != data and storage not in data.parents and data not in storage.parents, 'Raw inputs and fresh generated storage must be separate'
source_pin = HERE / 'outputs/native_probe_v2/source_manifest.json'
source_entries = json.loads(source_pin.read_text(encoding='utf-8'))
assert len(source_entries) == 125
for item in source_entries:
    relative = Path(item['path']); assert not relative.is_absolute() and '..' not in relative.parts
    assert sha(upstream / relative) == item['sha256'], item['path']
data_pin = HERE / 'required_data_manifest_v1.json'
required_data = json.loads(data_pin.read_text(encoding='utf-8'))
assert len(required_data['files']) == 26
for item in required_data['files']:
    path = data / 'greenhouse3_original_v1' / item['path']
    assert path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['path']

R = run / 'cycle53'; scripts = R / 'scripts'; stage_path = R / 'outputs/FRESH_STAGE.json'
source_text = (HERE / 'scripts/train_native53_v7.py').read_text(encoding='utf-8')
tree = ast.parse(source_text); removed = []
class RemoveHistoricalChecks(ast.NodeTransformer):
    def visit_If(self, node):
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == 'tag'
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value in
            ['PatchTST__ae_concat__seed0', 'PatchTST__obs__seed3']):
            removed.append(test.comparators[0].value); return None
        return self.generic_visit(node)
clean_tree = RemoveHistoricalChecks().visit(tree)
assert sorted(removed) == ['PatchTST__ae_concat__seed0', 'PatchTST__obs__seed3']
clean_text = ast.unparse(ast.fix_missing_locations(clean_tree)) + '\n'
assert ast.dump(ast.parse(clean_text), include_attributes=False) == ast.dump(clean_tree, include_attributes=False)

copy_names = ['native53.py', 'microbatch53.py', 'replay53.py', 'qualify_microbatch53_v2.py',
    'qualify_analysis53.py', 'score_native53_v8.py', 'verify_native53.py', 'analyze_native53.py',
    'render_native53.py', 'summarize_mechanisms53.py', 'audit_scaling_timing53.py']
payload = {'scripts/' + name: (HERE / 'scripts' / name).read_bytes() for name in copy_names}
payload['scripts/train_clean53.py'] = clean_text.encode('utf-8')
probe = (HERE / 'scripts/probe_native53_v2.py').read_text(encoding='utf-8')
old = "source=P/'upstream/TSWM-Official-Repo'"; assert probe.count(old) == 1
payload['scripts/probe_clean53.py'] = probe.replace(old, "source=pathlib.Path(os.environ['TSWM_UPSTREAM_ROOT'])").encode('utf-8')
concat = (HERE / 'scripts/qualify_concat53.py').read_text(encoding='utf-8')
old = "checkpoint=CKPT/'CrossLinear__ae_concat__seed0.pt'"; assert concat.count(old) == 1
payload['scripts/qualify_concat_clean53.py'] = concat.replace(old, "checkpoint=CKPT/'runtime_microbatch_v7/CrossLinear__ae_concat__seed0.pt'").encode('utf-8')
for name in ['PROTOCOL.md', 'EVALUATION_SUPPLEMENT.md', 'ANALYSIS_SUPPLEMENT.md']:
    payload[name] = (HERE / name).read_bytes()
payload['scripts/check_input_cache_clean53.py'] = "from native53 import *\nimport pandas as pd\nfrom dataset_utils.greenhouse.dataset import _load_team_df\nout=R/'outputs/input_cache_qualification_v1.json'\nassert not out.exists()\ncompared=[]\nfor experiment in ['experiment_1','experiment_2']:\n    root=pathlib.Path(STORAGE['data_root'])/'greenhouse3_original_v1'/experiment/'TimeSeries'\n    for team in sorted(p for p in root.iterdir() if p.is_dir() and (p/'GreenhouseClimate.xlsx').exists()):\n        cache=root/'_cache'/(team.name+'.parquet')\n        if cache.exists():\n            actual=pd.read_parquet(cache)\n            expected=_load_team_df(team,root/'Weather.xlsx')\n            pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=True)\n            compared.append({'experiment':experiment,'team':team.name,'rows':len(actual),'cache_sha256':sha(cache)})\ndump(out,{'utc':now(),'status':'PASS_EXISTING_NATIVE_CACHE_EQUALS_PINNED_RAW_INPUTS','existing_caches_checked':compared,'raw_inputs_checked_against_published_pin_by_runner':True})\nprint(json.dumps({'status':'PASS_EXISTING_NATIVE_CACHE_EQUALS_PINNED_RAW_INPUTS','existing_caches':len(compared)}))\n".encode('utf-8')
expected_payload = {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()}
config = {'data_root': str(data), 'cache_root': str(storage / 'caches'),
    'checkpoint_root': str(storage / 'checkpoints'), 'artifact_root': str(storage / 'artifacts')}

if stage_path.exists():
    assert args.resume, 'Existing clean stage requires --resume'
    previous = json.loads(stage_path.read_text(encoding='utf-8'))
    assert previous['launcher_sha256'] == sha(__file__)
    assert previous['payload_sha256'] == expected_payload and previous['storage_config'] == config and previous['upstream'] == str(upstream)
    for name, expected in expected_payload.items(): assert sha(R / name) == expected
    assert json.loads((run / 'STORAGE_LOCATION_V1.json').read_text()) == config
else:
    assert not run.exists() or not any(run.iterdir()), 'Run directory must be new and empty'
    assert not storage.exists() or not any(storage.iterdir()), 'Storage directory must be new and empty'
    for name, value in payload.items():
        path = R / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
    storage.mkdir(parents=True, exist_ok=True)
    dump(run / 'STORAGE_LOCATION_V1.json', config)
    dump(stage_path, {'utc': now(), 'status': 'STAGED_NOT_YET_EXECUTED', 'payload_sha256': expected_payload,
        'source_pin_sha256': sha(source_pin), 'required_data_pin_sha256': sha(data_pin),
        'storage_config': config, 'upstream': str(upstream), 'removed_historical_only_initialization_checks': removed,
        'history_receipts_fabricated': False, 'launcher_sha256': sha(__file__), 'training_code_ast_otherwise_unchanged': True,
        'interface': 'supervisor_v7 and SELECTION_SEAL_V6 are compatibility schemas; writer is this fresh runner, not the historical recovery process'})
if not args.execute:
    print(json.dumps({'status': 'STAGED_NOT_EXECUTED', 'run_root': str(run), 'fresh_qualification_and_training_performed': False})); sys.exit(0)

env = os.environ.copy(); env['TSWM_UPSTREAM_ROOT'] = str(upstream)
env['TSWM_STORAGE_CONFIG'] = str(run / 'STORAGE_LOCATION_V1.json'); env['WANDB_MODE'] = 'disabled'
env['WANDB_DISABLED'] = 'true'
art = storage / 'artifacts/cycle53/native_v1'; ckpt = storage / 'checkpoints/cycle53/native_v1'
overall = R / 'outputs/FRESH_RUN_STATE.json'

if overall.exists():
    previous_process=json.loads(overall.read_text()).get('pid')
    live=False
    if previous_process:
        if os.name=='nt':
            import ctypes
            from ctypes import wintypes
            kernel=ctypes.windll.kernel32
            kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];kernel.GetExitCodeProcess.restype=wintypes.BOOL
            kernel.CloseHandle.argtypes=[wintypes.HANDLE]
            handle=kernel.OpenProcess(0x1000,False,int(previous_process))
            if handle:
                code=wintypes.DWORD()
                try:
                    assert kernel.GetExitCodeProcess(handle,ctypes.byref(code));live=code.value==259
                finally:kernel.CloseHandle(handle)
        else:
            try:os.kill(int(previous_process),0);live=True
            except ProcessLookupError:pass
        assert not live,'Prior runner PID is live; inspect instead of launching concurrently'
def state(status, **extra): dump(overall, {'utc': now(), 'status': status, 'pid': os.getpid(), **extra})
def child(name, *arguments):
    log = R / 'outputs' / (name + ('__' + '_'.join(arguments) if arguments else '') + '.log')
    assert not log.exists(), 'Existing partial step/log requires inspection, not automatic rerun'
    with log.open('wb') as stream:
        result = subprocess.run([sys.executable, '-B', '-u', str(scripts / name), *arguments], cwd=R, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        state('FAILED_CHILD_NO_AUTOMATIC_RETRY', script=name, arguments=list(arguments), exit_code=result.returncode, log=str(log))
        raise RuntimeError(f'Child failed; inspect {log}')
def qualified(script, receipt, expected_status):
    path = R / receipt
    if not path.exists(): child(script)
    assert json.loads(path.read_text())['status'] == expected_status
def completed_fit(tag):
    folder = art / 'runtime_microbatch_v7' / tag
    if not folder.exists(): return None
    assert (folder / 'state.json').exists() and (folder / 'summary.json').exists(), 'Partial fit retained; no automatic overwrite'
    terminal = json.loads((folder / 'state.json').read_text()); fit = json.loads((folder / 'summary.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['tag'] == tag
    assert sha(fit['checkpoint']) == fit['checkpoint_sha256']
    return fit, folder / 'summary.json'

state('QUALIFYING_FRESH_SOURCE_AND_DATA')
qualified('check_input_cache_clean53.py','outputs/input_cache_qualification_v1.json','PASS_EXISTING_NATIVE_CACHE_EQUALS_PINNED_RAW_INPUTS')
qualified('probe_clean53.py', 'outputs/native_probe_v2/probe.json', 'QUALIFIED_NATIVE_SOURCE_DATA_AND_FORWARD')
qualified('qualify_analysis53.py', 'outputs/analysis_qualification_v1/qualification.json', 'PASS_PAIRED_RATIO_NULL_DIRECTION_AND_CALENDAR_IDENTITIES')
if completed_fit('shared_codec') is None: child('train_clean53.py', '--job', 'codec')
qualified('qualify_microbatch53_v2.py', 'outputs/microbatch_qualification_v2/qualification.json', 'PASS_NATIVE_EFFECTIVE_BATCH_MICROBATCHING')
finished = []; selections = []; training_state = R / 'outputs/supervisor_v7.json'
for seed in range(5):
    for model in ['CrossLinear', 'PatchTST']:
        for arm in ['obs', 'ae_concat', 'ae_gate', 'ae_gate_soft']:
            tag = f'{model}__{arm}__seed{seed}'; state('RUNNING_FRESH_NATIVE_GRID', active=tag, completed=len(finished))
            dump(training_state, {'utc': now(), 'status': 'RUNNING', 'pid': os.getpid(), 'writer': 'fresh_reproduce53.py', 'execution_kind': 'fresh_public_reproduction', 'completed': finished, 'active': {'tag': tag}})
            current = completed_fit(tag)
            if current is None:
                child('train_clean53.py', '--job', 'fit', '--model', model, '--arm', arm, '--seed', str(seed))
                current = completed_fit(tag)
            fit, path = current
            finished.append({'tag': tag, 'summary_path': str(path)})
            selections.append({'tag': tag, 'checkpoint': fit['checkpoint'], 'sha256': fit['checkpoint_sha256'], 'best_epoch': fit['best_epoch'], 'summary_path': str(path)})
            if tag == 'CrossLinear__ae_concat__seed0':
                qualified('qualify_concat_clean53.py', 'outputs/concat_qualification_v1/qualification.json', 'PASS_NATIVE_CONCAT_TRAINING_AND_CHECKPOINT_FORMAT')
assert len(selections) == 40
seal = R / 'outputs/SELECTION_SEAL_V6.json'
if not seal.exists(): dump(seal, {'utc': now(), 'writer': 'fresh_reproduce53.py', 'execution_kind': 'fresh_public_reproduction', 'protocol_sha256': sha(R / 'PROTOCOL.md'), 'selected_checkpoints': selections, 'second_batch_evaluation_opened': False, 'runtime_repair': 'Qualified effective-batch128/microbatch64 for PatchTST AE concat only; iid dropout draw ordering differs from full-batch forward'})
else:
    old = json.loads(seal.read_text()); assert old['selected_checkpoints'] == selections
dump(training_state, {'utc': now(), 'status': 'ALL_40_NATIVE_FITS_COMPLETE_TRANSFER_UNOPENED', 'pid': os.getpid(), 'writer': 'fresh_reproduce53.py', 'execution_kind': 'fresh_public_reproduction', 'completed': finished})
state('LOCKED_SCORING_AND_ANALYSIS')
for name, receipt in [('score_native53_v8.py', art / 'evaluation_v1/COMPLETE.json'),
    ('verify_native53.py', art / 'evaluation_v1/verification.json'),
    ('analyze_native53.py', art / 'analysis_v1/manifest.json'),
    ('render_native53.py', art / 'analysis_v1/figures/manifest.json'),
    ('summarize_mechanisms53.py', art / 'mechanism_summary_v1/manifest.json'),
    ('audit_scaling_timing53.py', art / 'data_diagnostics_v1/manifest.json')]:
    if not receipt.exists(): child(name)
    assert receipt.is_file()
state('FRESH_NATIVE_SUBSET_COMPLETE_NO_NOVEL_RESEARCH_CLAIM', selected_models=40)
print(json.dumps({'status': 'FRESH_NATIVE_SUBSET_COMPLETE_NO_NOVEL_RESEARCH_CLAIM', 'run_root': str(run)}))
