import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    p=(ROOT/entry['path']).resolve();assert p.is_relative_to(ROOT.resolve())
    assert p.stat().st_size==entry['bytes'],entry['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],entry['path']
print('VERIFIED_COMPACT_ARTIFACT_MANIFEST',len(manifest['files']),'files')
