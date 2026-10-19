"""Unpack verified legacy pure Python dependency into this cycle only."""
import hashlib,io,json,pathlib,requests,tarfile
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runtime_dependencies';OUT.mkdir(exist_ok=True)
s=requests.Session();s.trust_env=False
r=s.get('https://pypi.org/pypi/gym/0.17.2/json',timeout=30);r.raise_for_status();meta=r.json()
f=next(x for x in meta['urls'] if x['packagetype']=='sdist');b=s.get(f['url'],timeout=30);b.raise_for_status();content=b.content
assert hashlib.sha256(content).hexdigest()==f['digests']['sha256']
rows=[]
with tarfile.open(fileobj=io.BytesIO(content),mode='r:gz') as tf:
    for m in tf:
        p=pathlib.PurePosixPath(m.name)
        if not m.isfile() or len(p.parts)<2 or p.parts[1]!='gym':continue
        relative=pathlib.PurePosixPath(*p.parts[1:]);assert '..' not in relative.parts
        target=(OUT/pathlib.Path(*relative.parts)).resolve();assert target.is_relative_to(OUT.resolve())
        data=tf.extractfile(m).read();target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
        rows.append(dict(path=relative.as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
receipt=dict(package='gym',version='0.17.2',source=f['url'],source_sha256=f['digests']['sha256'],files=rows,environment_mutated=False)
(ROOT/'outputs/legacy_gym_provenance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(dict(package='gym',version='0.17.2',files=len(rows),archive_bytes=len(content))))
