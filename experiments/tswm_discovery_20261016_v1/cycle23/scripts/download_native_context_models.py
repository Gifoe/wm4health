"""Verified public pretrained archive, server only; list metadata, no extraction."""
import runtime_profile
import hashlib,json,pathlib,shutil,tarfile,time,requests
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/external/explore_context'
EXPECTED=2210754075;EXPECTED_SHA1='f24b54d4c817ac8d32fb80d1f0c8b08ededa4933'
URL='https://keeper.mpdl.mpg.de/f/988ed8adf071486c80dc/?dl=1'
assert not OUT.exists();OUT.mkdir(parents=True)
assert shutil.disk_usage(ROOT).free>8*1024**3
part=OUT/'experiments_corrected.tgz.part';final=OUT/'experiments_corrected.tgz'
s=requests.Session();s.trust_env=False
start=time.perf_counter();sha1=hashlib.sha1();sha256=hashlib.sha256();size=0;last=0
with s.get(URL,stream=True,timeout=(20,60)) as r:
    r.raise_for_status();assert int(r.headers['Content-Length'])==EXPECTED
    with part.open('xb') as f:
        for b in r.iter_content(4*1024*1024):
            if not b:continue
            f.write(b);sha1.update(b);sha256.update(b);size+=len(b);assert size<=EXPECTED
            if time.perf_counter()-last>15:
                progress=dict(bytes=size,expected=EXPECTED,elapsed_seconds=time.perf_counter()-start)
                (OUT/'download_progress.json').write_text(json.dumps(progress),encoding='utf-8');print(json.dumps(progress),flush=True);last=time.perf_counter()
assert size==EXPECTED and sha1.hexdigest()==EXPECTED_SHA1,(size,sha1.hexdigest())
part.replace(final)
members=[]
with tarfile.open(final,'r:gz') as tf:
    for m in tf:
        assert len(members)<100000
        members.append(dict(name=m.name,size=m.size,type=m.type.decode('ascii',errors='replace'),isfile=m.isfile(),islink=m.islnk() or m.issym()))
(OUT/'archive_members.json').write_text(json.dumps(members,indent=2),encoding='utf-8')
receipt=dict(url=URL,archive=final.name,bytes=size,sha1=sha1.hexdigest(),sha256=sha256.hexdigest(),published_checksum_match=True,tls_verification=True,elapsed_seconds=time.perf_counter()-start,members=len(members),declared_member_bytes=sum(m['size'] for m in members),extracted_files=0)
(OUT/'download_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
