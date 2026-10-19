"""Eight verified HTTP byte ranges, immutable prefix reuse, full publisher hash."""
import runtime_profile
import concurrent.futures,hashlib,json,pathlib,requests,threading,time,tarfile,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/external/explore_context';OUT=BASE/'parallel_attempt2'
EXPECTED=2210754075;URL='https://keeper.mpdl.mpg.de/f/988ed8adf071486c80dc/?dl=1';CHUNK=16*1024*1024
assert not OUT.exists();OUT.mkdir(parents=True);assert shutil.disk_usage(ROOT).free>8*1024**3
assert json.loads((BASE/'range_support.json').read_text())['content_range']==f'bytes 0-0/{EXPECTED}'
old=json.loads((ROOT/'outputs/jobs/corrected_native_context_archive_download/status.json').read_text());assert old['status']=='failed'
prefix=BASE/'experiments_corrected.tgz.part';prefix_size=prefix.stat().st_size
assert 0<prefix_size<EXPECTED
start=time.perf_counter();local=threading.local();lock=threading.Lock();done=[];last=0
def fetch(pair):
    global last
    a,b=pair;target=OUT/f'bytes_{a:010d}_{b:010d}.bin'
    if not hasattr(local,'session'):
        local.session=requests.Session();local.session.trust_env=False
    for attempt in range(3):
        try:
            with local.session.get(URL,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(20,50)) as r:
                assert r.status_code==206 and r.headers['Content-Range']==f'bytes {a}-{b}/{EXPECTED}'
                size=0;h=hashlib.sha256()
                with target.open('wb') as f:
                    for part in r.iter_content(1024*1024):
                        f.write(part);h.update(part);size+=len(part);assert size<=b-a+1
                assert size==b-a+1
            row=dict(start=a,end=b,bytes=size,path=target.name,sha256=h.hexdigest(),attempt=attempt+1)
            with lock:
                done.append(row)
                if time.perf_counter()-last>15:
                    p=dict(bytes=prefix_size+sum(x['bytes'] for x in done),expected=EXPECTED,completed_ranges=len(done),elapsed_seconds=time.perf_counter()-start)
                    temp=OUT/'download_progress.tmp';temp.write_text(json.dumps(p),encoding='utf-8');temp.replace(OUT/'download_progress.json');print(json.dumps(p),flush=True);last=time.perf_counter()
            return row
        except Exception:
            if attempt==2:raise
ranges=[(a,min(EXPECTED-1,a+CHUNK-1)) for a in range(prefix_size,EXPECTED,CHUNK)]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(fetch,ranges))
assert prefix.stat().st_size==prefix_size
final=OUT/'experiments_corrected.tgz';sha1=hashlib.sha1();sha256=hashlib.sha256();total=0
with final.open('xb') as f:
    for source in [prefix]+[OUT/r['path'] for r in sorted(rows,key=lambda x:x['start'])]:
        with source.open('rb') as src:
            while b:=src.read(4*1024*1024):f.write(b);sha1.update(b);sha256.update(b);total+=len(b)
assert total==EXPECTED and sha1.hexdigest()=='f24b54d4c817ac8d32fb80d1f0c8b08ededa4933',(total,sha1.hexdigest())
members=[]
with tarfile.open(final,'r:gz') as tf:
    for m in tf:
        assert len(members)<100000
        members.append(dict(name=m.name,size=m.size,type=m.type.decode('ascii',errors='replace'),isfile=m.isfile(),islink=m.islnk() or m.issym()))
(OUT/'archive_members.json').write_text(json.dumps(members,indent=2),encoding='utf-8')
receipt=dict(url=URL,archive=final.name,bytes=total,sha1=sha1.hexdigest(),sha256=sha256.hexdigest(),published_checksum_match=True,tls_verification=True,elapsed_seconds=time.perf_counter()-start,members=len(members),extracted_files=0,reused_prefix_bytes=prefix_size,range_workers=8,ranges=rows,original_failed_job_preserved=True)
(OUT/'download_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in receipt.items() if k!='ranges'}),flush=True)
