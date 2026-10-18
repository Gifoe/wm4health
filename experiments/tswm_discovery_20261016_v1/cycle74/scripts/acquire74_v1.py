"""Direct strict-HTTPS public archives on server; extract TRAIN MAT members only."""
import sys,pathlib,urllib.request,urllib.parse,zipfile,time,ast
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common74_v1 import *
def main():
 started=time.perf_counter();receipt=read(PROJECT/'cycle69'/'outputs'/'DATA_SOURCE69_V2.json');src=next(r for r in receipt['sources'] if r['path']=='official_loader/nonlinear_benchmarks/benchmarks.py');assert sha(src['server_path'])=='cf6749f3e1c34df5686ee4a1e2ec0909ad5c04fd8efefd64c18674cc2f61ee0b'
 tree=ast.parse(pathlib.Path(src['server_path']).read_bytes());fn={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['EMPS','F16']};assert len(fn)==2
 requested=[('EMPS','https://drive.usercontent.google.com/download?id=1zwoXYa9-3f8NQ0ohzmjpF7UxbNgRTHkS&export=download',1949929),('F16','https://data.4tu.nl/file/b6dc643b-ecc6-437c-8a8a-1681650ec3fe/5414dfdc-6e8d-4208-be6e-fa553de9866f',148455295)]
 assert '1zwoXYa9-3f8NQ0ohzmjpF7UxbNgRTHkS' in ast.unparse(fn['EMPS']) and requested[1][1] in ast.unparse(fn['F16'])
 root=storage('data');raw=root/'raw';raw.mkdir(exist_ok=True);datasets=[];fail=[]
 for name,url,expected in requested:
  p=raw/(name+'.zip');partial=raw/(name+'.zip.acquire74_v1.part')
  try:
   assert not p.exists() and not partial.exists(),'Existing acquisition: inspect instead of overwrite'
   req=urllib.request.Request(url,headers={'User-Agent':'reproducible-research-source-audit/1.0'})
   with urllib.request.urlopen(req,timeout=45) as response:
    assert urllib.parse.urlparse(response.geturl()).scheme=='https';headers={k:response.headers.get(k) for k in ['Content-Length','Content-Type','ETag','Last-Modified']};total=0;tick=0
    with partial.open('xb') as f:
     while True:
      b=response.read(1048576)
      if not b:break
      if total==0:assert b[:2]==b'PK','Response is not a ZIP; preserve failure'
      f.write(b);total+=len(b);assert total<=300000000
      if total-tick>=8388608:tick=total;state('acquisition74_v1_progress.json',dict(dataset=name,bytes=total,status='PUBLIC_HTTPS_DOWNLOAD_RUNNING'));print(json.dumps(dict(dataset=name,downloaded_bytes=total)),flush=True)
   assert total>1000
   if headers['Content-Length'] is not None:assert total==int(headers['Content-Length'])
   partial.rename(p);selected=[];inventory=[]
   with zipfile.ZipFile(p) as z:
    assert z.testzip() is None,'Archive CRC failure'
    for item in z.infolist():
     if item.is_dir():continue
     namepath=pathlib.PurePosixPath(item.filename);assert '..' not in namepath.parts
     chosen=(name=='EMPS' and namepath.name=='DATA_EMPS.mat') or (name=='F16' and 'BenchmarkData' in namepath.parts and namepath.suffix=='.mat' and 'Validation' not in namepath.name and 'SpecialOddMSine' not in namepath.name)
     if namepath.suffix.lower()=='.mat':inventory.append(dict(name=item.filename,bytes=item.file_size,crc32=format(item.CRC,'08x'),selected_training=chosen))
     if not chosen:continue
     target=root/name/'training_members'/namepath.name;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();data=z.read(item.filename)
     with target.open('xb') as f:f.write(data)
     selected.append(dict(member=item.filename,path=str(target),bytes=len(data),sha256=sha(target),crc32=format(item.CRC,'08x')))
   assert selected and (name!='EMPS' or len(selected)==1)
   row=dict(name=name,source_url=url,archive_path=str(p),archive_bytes=total,archive_sha256=sha(p),legacy_loader_expected_bytes=expected,legacy_size_matches=total==expected,publisher_checksum_verified=False,zip_crcs_verified=True,headers=headers,training_members=selected,mat_inventory=inventory,official_test_numeric_values_parsed=False)
   datasets.append(row);dump(OUT/('DATA_'+name+'74_V1.json'),row);print(json.dumps(dict(dataset=name,status='ARCHIVE_AND_SELECTED_TRAIN_MEMBERS_VERIFIED',archive_bytes=total,training_members=len(selected))),flush=True)
  except Exception as ex:
   row=dict(name=name,url=url,error_type=type(ex).__name__,error=str(ex),partial_path=str(partial),partial_exists=partial.exists());fail.append(row);dump(OUT/('FAILURE_'+name+'74_V1.json'),row);print(json.dumps(dict(dataset=name,status='ACQUISITION_FAILED_PRESERVED',error_type=type(ex).__name__,error=str(ex)[:200])),flush=True)
 obj=dict(utc=utc(),status='ALL_REQUESTED_ARCHIVES_AND_TRAIN_MEMBERS_READY' if not fail else 'PARTIAL_ACQUISITION_REQUIRES_INSPECTION',datasets=datasets,failures=fail,official_loader_revision=src['revision'],official_loader_sha256=src['sha256'],function_ast_sha256={k:hashlib.sha256(ast.dump(v,include_attributes=False).encode()).hexdigest() for k,v in fn.items()},protocol_sha256=sha(ROOT/'DATA_PROTOCOL74_V1.md'),download_tls_verification_disabled=False,local_signal_model_or_result_files_written=0,official_test_numeric_values_parsed=False,new_model_fits=0,wall_seconds=time.perf_counter()-started)
 dump(OUT/'ACQUISITION74_V1.json',obj);print(json.dumps(dict(status=obj['status'],datasets=[d['name'] for d in datasets],failures=len(fail),wall_seconds=obj['wall_seconds'])),flush=True)
if __name__=='__main__':main()
