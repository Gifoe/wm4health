"""Revalidate reused checkpoint identities and record current official repository pins."""
import pathlib,json,csv,hashlib,datetime,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        while b:=f.read(1048576):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
old=pathlib.Path('D:/baseline_reproduction_seed2021_v1');rows=list(csv.DictReader((old/'results/all_baselines.csv').open(encoding='utf-8-sig')))
verify=read(old/'results/metric_verification.json');assert verify['all_passed'] and verify['complete'] and verify['verified_configs']==24
kept=[]
for row in rows:
    if row['model'] not in ['WorldTS','LatentTSF']:continue
    assert sha(row['checkpoint_path'])==row['checkpoint_sha256']
    checked=next(x for x in verify['results'] if x['run_id']==row['run_id']);assert checked['passed'] and checked['all_test_origins_verified'] and checked['train_only_scaler_verified']
    kept.append({k:row[k] for k in ['run_id','model','dataset','horizon','seed','source_commit','checkpoint_sha256','predictions_sha256','reproduced_mse','reproduced_mae','status']})
assert sum(x['model']=='WorldTS' for x in kept)==4 and sum(x['model']=='LatentTSF' for x in kept)==16
p=ROOT.parent/'cycle53';selection=read(p/'outputs/SELECTION_SEAL_V6.json');items=selection['selected_checkpoints'];assert len(items)==40
for row in items:assert sha(row['checkpoint'])==row['sha256']
source_manifest=read(p/'outputs/native_probe_v2/source_manifest.json');src=ROOT.parent/'upstream/TSWM-Official-Repo'
for row in source_manifest:assert sha(src/row['path'])==row['sha256']
pins={'decisionintelligence/WorldTS':'269426ea6217d7152a244da0e209d922acd46d4f','Muyiiiii/LatentTSF':'7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2','HaochenZhang717/TSWM-Official-Repo':'2e5e48540ff0da48e8192163969a6c58df1d5a3e'}
current=[]
for repo,pin in pins.items():
    try:
        request=urllib.request.Request('https://api.github.com/repos/'+repo+'/commits/main',headers={'User-Agent':'native-baseline-failure-audit'})
        with urllib.request.urlopen(request,timeout=25) as f:o=json.load(f)
        current.append(dict(repository=repo,execution_pin=pin,current_main=o['sha'],same_version=o['sha']==pin,commit_date=o['commit']['committer']['date']))
    except Exception as e:current.append(dict(repository=repo,execution_pin=pin,metadata_error=type(e).__name__))
obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='REUSED_BASELINE_CHECKPOINTS_AND_SOURCE_REVALIDATED',old_ledger_sha256=sha(old/'results/all_baselines.csv'),old_metric_verification_sha256=sha(old/'results/metric_verification.json'),reused_twenty_cells=kept,tswm_selected_checkpoint_count=len(items),tswm_source_files_rechecked=len(source_manifest),tswm_selection_seal_sha256=sha(p/'outputs/SELECTION_SEAL_V6.json'),tswm_checkpoint_identities=[{k:x[k] for k in ['tag','sha256'] if k in x} for x in items],official_heads=current,
         scope='Current hash revalidation and existing metric-verification inspection; not a fresh rerun of all historical forecasts or the full three-paper published benchmarks',new_method=False)
with (OUT/'EXISTING_BASELINE_AUDIT_V1.json').open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2)
print(json.dumps({k:obj[k] for k in ['status','tswm_selected_checkpoint_count','tswm_source_files_rechecked','official_heads']}),flush=True)
