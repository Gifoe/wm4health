import pathlib,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';m=json.loads((OUT/'DATA_RELAY_PROVISIONAL67_V1.json').read_text());assert len(m['files'])==45
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));base=(pathlib.Path(cfg['data_root'])/'cycle67'/'cgmacros_v1').resolve();assert base.drive.upper()=='G:'
for row in m['files']+m['metadata']+[m['author_demo']]:
 p=pathlib.Path(row['path']).resolve();assert p.drive.upper()=='G:' and p.is_relative_to(pathlib.Path(cfg['data_root']).resolve()) or p.is_relative_to(pathlib.Path(cfg['artifact_root']).resolve())
 b=p.read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],p.name
m['status']='45_PUBLIC_NUMERIC_PARTICIPANT_FILES_VERIFIED';m['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['independent_server_SHA256_verified']=True
with (OUT/'DATA_SOURCE67_V2.json').open('x',encoding='utf-8') as f:json.dump(m,f,indent=2)
print(json.dumps(dict(status=m['status'],files=45,network_bytes=m['network_bytes'],columns=m['files'][0]['columns'],local_result_files_written=0)))
