import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
import numpy as np
a=read(OUT/'ACQUISITION71_V1.json');rows=[]
for old in a['failed_relay_files_preserved']:
 p=pathlib.Path(old['path']);new=pathlib.Path(a['data_root'])/p.name;b=p.read_bytes();c=new.read_bytes();r=dict(original_preserved=old,direct_file_sha256=sha(new),old_bytes=len(b),new_bytes=len(c),header_bytes_equal=b[:128]==c[:128],byte_positions_different=int(np.count_nonzero(np.frombuffer(b,dtype=np.uint8)!=np.frombuffer(c,dtype=np.uint8))) if len(b)==len(c) else None)
 try:
  x=np.load(p,allow_pickle=False);y=np.load(new,allow_pickle=False);r.update(shape_equal=x.shape==y.shape,dtype_equal=x.dtype==y.dtype,numeric_entries_unequal=int(np.count_nonzero(x!=y)) if x.shape==y.shape else None,old_finite=bool(np.isfinite(x).all()),new_finite=bool(np.isfinite(y).all()))
 except Exception as ex:r['parse_error']=str(ex)
 rows.append(r)
dump(OUT/'TRANSFER_AUDIT71_V1.json',dict(utc=utc(),status='FAILED_RELAY_PRESERVED_DIRECT_COPY_SELECTED',rows=rows,cause='UNESTABLISHED',data_changed_for_scientific_gain=False));print('TRANSFER_AUDIT_SAVED')
