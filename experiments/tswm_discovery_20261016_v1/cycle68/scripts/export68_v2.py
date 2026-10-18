import pathlib,json,hashlib,math,datetime,os,base64
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=read(OUT/'CLOSURE68_V1.json');assert c['status']=='CLOSED_CONTROLLED_OPERATOR_COMPARISON_WITH_INDEPENDENT_REPLAY'
for n,h in c['reports'].items():assert sha(ROOT/n)==h
for n,h in c['artifacts'].items():assert sha(OUT/n)==h
for n,h in c['sources'].items():assert sha(ROOT/n)==h
patterns=json.loads(os.environ['C68_PRIVATE_EXPORT_PATTERNS_JSON']);dest=OUT/'publication_v2';dest.mkdir(exist_ok=False);rows=[]
files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.json','.ps1']]+list((ROOT/'scripts').glob('*.py'))+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.log'] and not p.name.startswith('PUBLICATION_RECEIPT')]
for p in sorted(set(files)):
 raw=p.read_bytes();rel=p.relative_to(ROOT).as_posix();original_rel=rel
 for needle in patterns:assert needle.encode('utf-8') not in raw,(rel,'private export pattern')
 try:text=raw.decode('utf-8-sig');encoding='UTF-8'
 except UnicodeDecodeError:text=base64.b64encode(raw).decode('ascii');rel+='.base64.txt';encoding='Base64 of exact original bytes; original log remains server-side'
 b=text.encode('utf-8');target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b);rows.append(dict(path=rel,original_path=original_rel,original_sha256=hashlib.sha256(raw).hexdigest(),encoding=encoding,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),parts=math.ceil(len(text)/3500),server_snapshot_file=str(target)))
obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='C68_CONTROLLED_OPERATOR_EVIDENCE_CLOSED',files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),closure_sha256=sha(OUT/'CLOSURE68_V1.json'),forecast_records=480,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',upstream_source_raw_data_or_weight_files_included=False,local_result_files_written=0)
with (dest/'manifest.json').open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2)
print(json.dumps(dict(status='PUBLICATION_READY',files=len(rows),bytes=obj['bytes'],manifest_sha256=sha(dest/'manifest.json'))))
