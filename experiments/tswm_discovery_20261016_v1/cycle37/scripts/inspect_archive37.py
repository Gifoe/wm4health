"""Inspect nested official data members; no arbitrary archive extraction."""
import pathlib,zipfile,io,json,hashlib
R=pathlib.Path(__file__).resolve().parents[1];D=R/'data_intake';dst=D/'archive_inventory.json'
assert not dst.exists(),'Preserve inventory'
raw=D/'GEFCom2014_original.zip'
receipt=json.loads((D/'download_receipts.json').read_text(encoding='utf-8'))[-1]
assert receipt['success'] and hashlib.sha256(raw.read_bytes()).hexdigest()==receipt['sha256']
result={'raw_sha256':receipt['sha256'],'outer_members':[],'selected_nested':[]}
with zipfile.ZipFile(raw) as outer:
    result['outer_members']=outer.namelist()
    for name in outer.namelist():
        if not name.lower().endswith('.zip') or not any(k in name for k in ['-W_','-S_']):continue
        data=outer.read(name)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            members=[{'name':i.filename,'bytes':i.file_size,'crc32':i.CRC} for i in z.infolist()]
            task15=[i.filename for i in z.infolist() if ('task 15' in i.filename.lower() or 'task15' in i.filename.lower())]
            result['selected_nested'].append({'outer_member':name,'sha256':hashlib.sha256(data).hexdigest(),
                                              'bytes':len(data),'members':members,'task15_members':task15})
dst.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'outer':result['outer_members'],'nested':[{'outer_member':x['outer_member'],'task15':x['task15_members']} for x in result['selected_nested']]}))
