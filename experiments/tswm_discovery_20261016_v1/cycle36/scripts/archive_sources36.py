"""Archive selected public provenance loaders and KITE tree, verify Git object identities."""
import pathlib,json,urllib.request,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];META=ROOT/'public_metadata';OUT=ROOT/'source_audit'
OUT.mkdir(exist_ok=True);receipts=[]
def get(url,limit=3*1024*1024):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TSWM-source-audit/1.0'}),timeout=30) as r:b=r.read(limit+1)
    assert len(b)<=limit;return b
def save(relative,data,url,blob=None):
    if blob:assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==blob,relative
    f=OUT/relative;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
    receipts.append({'path':relative,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob':blob})
repo='OpenSTEF/openstef';rev=json.loads((META/'openstef_commit.json').read_text(encoding='utf-8'))['sha']
tree=json.loads((META/'openstef_tree.json').read_text(encoding='utf-8'))['tree'];lookup={x['path']:x for x in tree}
paths=['examples/benchmarks/liander2024/README.md',
 'examples/benchmarks/liander2024/run_chronos2_benchmark.py',
 'examples/benchmarks/liander2024/run_xgboost_gblinear_benchmark.py',
 'packages/openstef-beam/src/openstef_beam/benchmarking/benchmarks/liander2024.py',
 'packages/openstef-core/src/openstef_core/datasets/versioned_timeseries_dataset.py',
 'packages/openstef-core/src/openstef_core/datasets/timeseries_dataset.py',
 'packages/openstef-models/src/openstef_models/transforms/time_domain/versioned_lags_adder.py']
for path in paths:
    url=f'https://raw.githubusercontent.com/{repo}/{rev}/{path}'
    save('openstef/'+path,get(url),url,lookup[path]['sha'])
hf=json.loads((META/'openstef_hf_info.json').read_text(encoding='utf-8'));revhf=hf['sha'];lookup={x['rfilename']:x for x in hf['siblings']}
for path in ['README.md','liander2024_targets.yaml']:
    url=f'https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark/resolve/{revhf}/{path}'
    save('hf_openstef/'+path,get(url),url,lookup[path]['blobId'])
url='https://api.github.com/repos/decisionintelligence/KITE/commits/main';commit=json.loads(get(url));save('KITE/commit.json',json.dumps(commit,indent=2).encode(),url)
revkite=commit['sha'];url=f'https://api.github.com/repos/decisionintelligence/KITE/git/trees/{revkite}?recursive=1'
tree=json.loads(get(url));save('KITE/tree.json',json.dumps(tree,indent=2).encode(),url)
hits=[x['path'] for x in tree['tree'] if x['path'].endswith('.py') and any(y in x['path'].lower() for y in ['/kite/','/kite.py'])]
print(json.dumps({'openstef':rev,'hf':revhf,'kite':revkite,'kite_source_paths':hits}),flush=True)
for path in ['README.md','requirements.txt']+hits:
    item=next(x for x in tree['tree'] if x['path']==path);assert item['size']<300000
    url=f'https://raw.githubusercontent.com/decisionintelligence/KITE/{revkite}/{path}'
    save('KITE/'+path,get(url),url,item['sha'])
(OUT/'manifest.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files':receipts,'source_only':True,'no_code_executed':True,'no_weights_or_signals_downloaded':True},indent=2),encoding='utf-8')
print(json.dumps({'source_files_verified':len(receipts),'bytes':sum(x['bytes'] for x in receipts)}))
