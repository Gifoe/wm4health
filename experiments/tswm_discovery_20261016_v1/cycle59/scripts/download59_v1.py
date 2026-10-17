"""Fetch only the pinned official tutorial; verify bytes and Git blobs before use."""
from common59 import *
import urllib.request,time
def main():
    selected=read(ROOT/'source_selection59_v1.json');rows=[]
    assert selected['commit']=='ab4913ca63ebda1f8eab429c64d3e93f96204713'
    assert sum(x['bytes'] for x in selected['files'])<12000000
    for row in selected['files']:
        relative=row['path'].removeprefix('tutorials/tutorial-2/')
        dest=data_root()/pathlib.Path(relative).name if row['kind']=='data' else source_root()/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        url=f"https://raw.githubusercontent.com/{selected['repository']}/{selected['commit']}/{row['path']}"
        if not dest.exists():
            partial=dest.with_name(dest.name+'.download')
            assert not partial.exists(),('Preserve interrupted download for inspection',str(partial))
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'tswm-research-provenance-audit'}),timeout=30) as response,partial.open('xb') as out:
                count=0
                while True:
                    chunk=response.read(262144)
                    if not chunk:break
                    count+=len(chunk);assert count<=row['bytes'],('Unexpected object size',relative)
                    out.write(chunk)
            raw=partial.read_bytes()
            assert len(raw)==row['bytes']
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob_sha1'],relative
            partial.rename(dest)
        raw=dest.read_bytes()
        assert len(raw)==row['bytes']
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob_sha1'],relative
        rows.append({**row,'url':url,'server_path':str(dest),'sha256':hashlib.sha256(raw).hexdigest()})
        print(json.dumps({'status':'PINNED_OBJECT_VERIFIED','path':row['path'],'bytes':len(raw)}),flush=True)
    obj={'utc':utc(),'status':'ALL_38_PINNED_SOURCE_AND_DATA_OBJECTS_VERIFIED','source_commit':selected['commit'],'source_repo':selected['repository'],'selection_sha256':sha(ROOT/'source_selection59_v1.json'),'protocol_sha256':sha(ROOT/'PROTOCOL59_V1.md'),'files':rows,'local_result_files_written':0,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    assert len(rows)==38
    dump(OUT/'DOWNLOAD59_V1.json',obj)
if __name__=='__main__':main()
