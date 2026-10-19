"""Task-local verified wheels, frozen weights and two small public parquets."""
import pathlib,json,hashlib,urllib.request,zipfile,sys,time,datetime
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/asset_setup_v1';ASSET=ROOT/'assets';VENDOR=ROOT/'vendor'
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()
def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TSWM-public-baseline-qualification/1.0'}),timeout=40)
def small(url):
    with request(url) as r:assert r.status==200;return r.read()
def hf(path):
    errors=[]
    for host in ['https://hf-mirror.com','https://huggingface.co']:
        try:return small(host+path),host+path
        except Exception as e:errors.append(str(e))
    raise RuntimeError(errors)
def download(url,path,expected,size):
    if path.exists():
        assert path.stat().st_size==size and sha(path)==expected
        return 'reused_verified'
    part=path.with_name(path.name+'.part');assert not part.exists()
    path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();n=0
    with request(url) as r,part.open('xb') as f:
        assert r.status==200
        while b:=r.read(4*1024*1024):f.write(b);h.update(b);n+=len(b)
    assert n==size and h.hexdigest()==expected,(path,n,h.hexdigest())
    part.replace(path);return 'downloaded_verified'
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);start=time.perf_counter();receipts=[]
    supported=set(sys_tags())
    for name,version in [('transformers','4.40.1'),('tokenizers','0.19.1'),('huggingface-hub','0.23.5'),('regex','2024.5.15')]:
        meta=json.loads(small(f'https://pypi.org/pypi/{name}/{version}/json'))
        choices=[f for f in meta['urls'] if f['filename'].endswith('.whl') and parse_wheel_filename(f['filename'])[3]&supported]
        assert choices,(name,version)
        item=sorted(choices,key=lambda f:f['filename'])[0];path=ASSET/'wheels'/item['filename']
        action=download(item['url'],path,item['digests']['sha256'],item['size'])
        VENDOR.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as z:
            for ent in z.infolist():
                dest=(VENDOR/ent.filename).resolve();assert dest.is_relative_to(VENDOR.resolve())
                if not ent.is_dir() and dest.exists():assert dest.read_bytes()==z.read(ent)
            z.extractall(VENDOR)
        receipts.append({'kind':'pypi_wheel','package':name,'version':version,'path':str(path.relative_to(ROOT)),
            'url':item['url'],'sha256':item['digests']['sha256'],'bytes':item['size'],'action':action})
        print(json.dumps({'package':name,'version':version,'status':action}),flush=True)
    expected='414435b508391f92afadd2aaeec418c806776aeccbce12e638d73a139ca5ca78'
    path=ASSET/'Sundial/model.safetensors';url='https://hf-mirror.com/thuml/sundial-base-128m/resolve/3212e42564493f520593e5414af4367fc4b49226/model.safetensors'
    action=download(url,path,expected,513341448)
    receipts.append({'kind':'model_weight','path':str(path.relative_to(ROOT)),'url':url,'sha256':expected,'bytes':513341448,'action':action})
    print(json.dumps({'weights':action,'bytes':513341448}),flush=True)
    revision='f71c0fff4cf81283a2c43e7f3a73aa4f9826aef8'
    data,metaurl=hf('/api/datasets/autogluon/fev_datasets/revision/'+revision+'?blobs=true')
    meta=json.loads(data);assert meta['sha']==revision
    (OUT/'fev_metadata.json').write_bytes(data)
    for relative in ['solar_with_weather/15T/train-00000-of-00001.parquet','jena_weather/10T/train-00000-of-00001.parquet']:
        item=next(r for r in meta['siblings'] if r['rfilename']==relative);expected=item['lfs']['sha256']
        url='https://hf-mirror.com/datasets/autogluon/fev_datasets/resolve/'+revision+'/'+relative
        path=ASSET/'fev'/relative;action=download(url,path,expected,item['size'])
        receipts.append({'kind':'public_parquet','path':str(path.relative_to(ROOT)),'url':url,'sha256':expected,'bytes':item['size'],'action':action})
        print(json.dumps({'dataset':relative,'status':action,'bytes':item['size']}),flush=True)
    result={'status':'complete_verified','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'receipts':receipts,'fev_metadata_url':metaurl,'fev_metadata_sha256':sha(OUT/'fev_metadata.json'),
        'script_sha256':sha(__file__),'plan_sha256':sha(ROOT/'QUALIFICATION_PLAN.md'),'seconds':time.perf_counter()-start,
        'global_environment_unchanged':True,'no_model_execution_or_training':True}
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'status':result['status'],'seconds':result['seconds']}),flush=True)
if __name__=='__main__':main()
