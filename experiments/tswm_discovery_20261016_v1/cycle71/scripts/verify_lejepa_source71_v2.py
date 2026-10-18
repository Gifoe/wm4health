"""Verify exact public-source bytes supplied through an authorized transport."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
selection=read(ROOT/'SOURCE_SELECTION71_LEJEPA_V1.json');dest=PROJECT/'upstream'/'lejepa_identifiability_de7503f1';rows=[]
for row in selection['files']:
 path=dest/row['path'];assert path.resolve().is_relative_to(dest.resolve());raw=path.read_bytes()
 assert len(raw)==row['bytes'] and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob_sha1']
 rows.append(dict(**row,sha256=hashlib.sha256(raw).hexdigest(),server_path=str(path)))
assert b'MIT License' in (dest/'LICENSE').read_bytes()
obj=dict(utc=utc(),status='AUTHOR_SOURCE_PINNED_NOT_EXECUTED',repository=selection['repository'],revision=selection['revision'],tree=selection['tree'],source_root=str(dest),files=rows,bytes=sum(r['bytes'] for r in rows),license='MIT',transport='Approved GitHub connector to RAM to pinned SFTP; no local result files',previous_direct_attempt=dict(script='scripts/acquire_lejepa_source71_v1.py',exit_code=1,error='ConnectionRefusedError WinError10061',diagnosis='Both public GitHub hostnames resolved to loopback; proxy enumeration was empty',network_configuration_changed=False),new_training_fits=0,new_performance_results=0,data_downloaded=False,shared_environment_modified=False,accepted_final_directions=0)
dump(OUT/'SOURCE_AUDIT71_LEJEPA_V2.json',obj);print(json.dumps({k:obj[k] for k in ['status','revision','bytes','new_training_fits','new_performance_results']}),flush=True)
