"""Bounded public metadata/source discovery, no signal/weight downloads."""
import pathlib,json,hashlib,datetime,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'public_metadata';OUT.mkdir(exist_ok=True)
receipts=[]
def get(name,url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TSWM-public-metadata-audit/1.0'}),timeout=30) as response:
        content=response.read(6*1024*1024+1);assert len(content)<6*1024*1024
    (OUT/name).write_bytes(content)
    receipts.append({'path':name,'url':url,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
    return json.loads(content)
hf=get('openstef_hf_info.json','https://huggingface.co/api/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark?blobs=true')
print(json.dumps({'hf_sha':hf.get('sha'),'files':len(hf.get('siblings',[])),
 'examples':[{'name':x['rfilename'],'size':x.get('size'),'lfs':x.get('lfs')} for x in hf.get('siblings',[]) if x['rfilename'].endswith(('.yaml','.md','.py','.json')) or 'weather_forecasts_versioned' in x['rfilename']][:20]}),flush=True)
for owner,repo in [('OpenSTEF','openstef'),('OpenSTEF','openstef-dbc'),('OpenSTEF','openstef-reference')]:
    try:
        info=get(repo+'_info.json',f'https://api.github.com/repos/{owner}/{repo}')
        branch=info['default_branch'];commit=get(repo+'_commit.json',f'https://api.github.com/repos/{owner}/{repo}/commits/{branch}')['sha']
        tree=get(repo+'_tree.json',f'https://api.github.com/repos/{owner}/{repo}/git/trees/{commit}?recursive=1')
        hits=[x['path'] for x in tree['tree'] if any(k in x['path'].lower() for k in ['benchmark','liander','forecast_version','openmeteo','open_meteo','weather'])]
        print(json.dumps({'repo':owner+'/'+repo,'commit':commit,'paths':hits[:100]}),flush=True)
    except urllib.error.HTTPError as e:print(json.dumps({'repo':owner+'/'+repo,'http_error':str(e)}),flush=True)
(OUT/'receipts.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':receipts},indent=2),encoding='utf-8')
