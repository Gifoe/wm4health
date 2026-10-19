"""Unpack the private stage with a bounded path whitelist, then check provenance."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    with zipfile.ZipFile(ROOT/'data/source_data_bundle.zip') as z:
        assert len(z.infolist())<50
        for i in z.infolist():
            r=pathlib.PurePosixPath(i.filename);assert not r.is_absolute() and '..' not in r.parts
            assert r.parts[0] in ['sources','data'] or i.filename in ['outputs/data_registry.json','outputs/timeflies_tree.json','outputs/timeflies_commit.json','outputs/prodyn_data_tree.json']
            dest=ROOT.joinpath(*r.parts);assert dest.resolve().is_relative_to(ROOT.resolve());assert i.file_size<10_000_000
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(i))
    registry=json.loads((ROOT/'outputs/data_registry.json').read_text(encoding='utf-8'))
    for row in registry['datasets']:
        p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    pin=json.loads((ROOT/'sources/timeflies/SOURCE_PIN.json').read_text(encoding='utf-8'))
    assert not pin['AGENTS_files'] and not pin['credential_pattern_flag_files']
    for row in pin['files']:
        assert hashlib.sha256((ROOT/'sources/timeflies'/row['path']).read_bytes()).hexdigest()==row['sha256']
    r={'public_series_verified':len(registry['datasets']),'native_timeflies_training_code_present':False,'source_revision':pin['revision'],'verified':True}
    (ROOT/'outputs/stage_verification.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))

if __name__=='__main__':main()
