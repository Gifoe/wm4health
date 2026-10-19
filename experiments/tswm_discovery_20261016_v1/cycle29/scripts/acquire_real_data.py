"""Public author-source benchmark acquisition only. No model score, fit, or selection."""
import runtime_profile
import hashlib,io,json,pathlib,time,urllib.request,zipfile
from scipy.io import loadmat
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/real_data'
SOURCES=[
 dict(name='cascaded_tanks',urls=['https://data.4tu.nl/file/d4810b78-6cdd-48fe-8950-9bd601e5f47f/3b697e42-01a4-4979-a370-813a456c36f5','https://drive.usercontent.google.com/download?id=1HnQf_gu0g_UlggoBqy2s34l9YJiFdN01&export=download&confirm=t'],member='dataBenchmark.mat',page='https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks'),
 dict(name='silverbox',urls=['https://drive.usercontent.google.com/download?id=17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy&export=download&confirm=t','https://drive.google.com/uc?export=download&id=17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy'],member='SNLS80mV.mat',page='https://www.nonlinearbenchmark.org/benchmarks/silverbox'),
 dict(name='wiener_hammerstein',urls=['https://www.ee.kth.se/~hjalmars/ifac_tc11_benchmarks/2009_wienerhammerstein/WienerHammerBenchMark.mat','http://www.ee.kth.se/~hjalmars/ifac_tc11_benchmarks/2009_wienerhammerstein/WienerHammerBenchMark.mat'],member='WienerHammerBenchMark.mat',page='https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein'),
]
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True);summary=[];start=time.monotonic()
    loader=ROOT.parent/'cycle28/upstream/nonlinear_benchmarks/nonlinear_benchmarks/benchmarks.py'
    for source in SOURCES:
        folder=OUT/source['name'];folder.mkdir(exist_ok=True);receipt=folder/'acquisition.json'
        if receipt.exists():
            r=json.loads(receipt.read_text());assert sha((folder/source['member']).read_bytes())==r['member_sha256'];summary.append(r);print(json.dumps(dict(name=source['name'],status='already_verified')),flush=True);continue
        attempts=[];raw=None;chosen=None;final=None
        for url in source['urls']:
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'ReproducibleResearchAudit/1.0'})
                with urllib.request.urlopen(request,timeout=30) as response:
                    content=response.read(64*1024*1024+1);final=response.geturl();content_type=response.headers.get('Content-Type','')
                assert len(content)<=64*1024*1024,'unexpected large response'
                if source['name']=='wiener_hammerstein':
                    schema=loadmat(io.BytesIO(content));assert {'uBenchMark','yBenchMark','fs'}<=set(schema)
                else:
                    assert zipfile.is_zipfile(io.BytesIO(content)),f'not a public ZIP,content_type={content_type},bytes={len(content)}'
                raw=content;chosen=url;break
            except Exception as exc:attempts.append(dict(url=url,error=type(exc).__name__+': '+str(exc)[:220]));print(json.dumps(dict(name=source['name'],attempt=attempts[-1])),flush=True)
        if raw is None:
            r=dict(name=source['name'],status='acquisition_unavailable',attempts=attempts,source_page=source['page']);(folder/'unavailable.json').write_text(json.dumps(r,indent=2),encoding='utf-8');summary.append(r);continue
        if source['name']=='wiener_hammerstein':member=raw;archive_member=source['member']
        else:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                names=[n for n in archive.namelist() if pathlib.PurePosixPath(n.replace('\\','/')).name.lower()==source['member'].lower()];assert len(names)==1,names
                archive_member=names[0];assert archive.getinfo(archive_member).file_size<64*1024*1024;member=archive.read(archive_member)
            (folder/'original.zip').write_bytes(raw)
        (folder/source['member']).write_bytes(member);mat=loadmat(io.BytesIO(member));schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in mat.items() if not k.startswith('__')}
        r=dict(name=source['name'],status='downloaded_and_schema_checked',source_page=source['page'],url=chosen,final_url=final,archive_bytes=len(raw),archive_sha256=sha(raw),archive_member=archive_member,member=source['member'],member_bytes=len(member),member_sha256=sha(member),loader_version='nonlinear_benchmarks1.0.1',loader_sha256=sha(loader.read_bytes()),schema=schema,failed_attempts=attempts,no_model_performance=True)
        receipt.write_text(json.dumps(r,indent=2),encoding='utf-8');summary.append(r);print(json.dumps(r),flush=True)
    result=dict(sources=summary,elapsed_s=time.monotonic()-start,script_sha256=sha(pathlib.Path(__file__).read_bytes()),scope='author-source public acquisition/schema only; no new model inference or test selection')
    (OUT/'acquisition_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(phase='acquisition_complete',statuses={r['name']:r['status'] for r in summary},elapsed_s=time.monotonic()-start)),flush=True)
if __name__=='__main__':main()
