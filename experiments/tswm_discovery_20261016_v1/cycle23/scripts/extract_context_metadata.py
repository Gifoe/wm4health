import hashlib,json,pathlib,tarfile,time
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/external/explore_context/parallel_attempt2';OUT=ROOT/'outputs/native_context_metadata'
assert not OUT.exists();OUT.mkdir(parents=True)
receipt=json.loads((BASE/'download_receipt.json').read_text());assert receipt['published_checksum_match']
members=json.loads((BASE/'archive_members.json').read_text())
targets={m['name'] for m in members if m['isfile'] and pathlib.PurePosixPath(m['name']).name in ['config.json','run.json','info.json'] and m['size']<1024*1024}
rows=[]
with tarfile.open(BASE/receipt['archive'],'r:gz') as tf:
    for m in tf:
        if m.name not in targets:continue
        p=pathlib.PurePosixPath(m.name);assert not p.is_absolute() and '..' not in p.parts and m.isfile()
        target=(OUT/'files'/pathlib.Path(*p.parts)).resolve();assert target.is_relative_to(OUT.resolve())
        b=tf.extractfile(m).read();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
        rows.append(dict(member=m.name,path=target.relative_to(OUT).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),content=json.loads(b)))
weights=[m for m in members if m['isfile'] and ('checkpoint' in m['name'].lower() or m['name'].endswith(('.pkl','.pt')))]
result=dict(archive_sha256=receipt['sha256'],metadata=rows,weights=weights,extracted_weight_files=0)
(OUT/'inventory.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(metadata=len(rows),weight_members=len(weights),weight_bytes=sum(x['size'] for x in weights),first_weight_paths=[m['name'] for m in weights[:8]])))
