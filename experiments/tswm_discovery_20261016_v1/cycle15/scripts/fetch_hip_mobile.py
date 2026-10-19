"""Fetch one official numeric dataset, verify Git blob and record array schema."""
import hashlib,json,pathlib,requests,datetime,base64,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
REV='f4adef0ce5ed4b37c9847973df320f72965bd330'
REL='data/MobileRobot/sin2/ts_002_50x2000_w_grad.npz'
SIZE=30162192
BLOB='0a099cb2bfb515828cb705cf43f7acaa66693634'
def main():
 folder=ROOT/'data/hip_mobile';folder.mkdir(parents=True,exist_ok=True)
 path=folder/pathlib.PurePosixPath(REL).name
 url='https://raw.githubusercontent.com/ALRhub/HiP-RSSM/'+REV+'/'+REL
 via_api='--api' in sys.argv
 request_url='https://api.github.com/repos/ALRhub/HiP-RSSM/git/blobs/'+BLOB if via_api else url
 if not path.exists():
  session=requests.Session();session.trust_env=False
  temporary=path.with_name(path.name+'.download')
  with session.get(request_url,stream=True,timeout=(8,30),headers={'Accept':'application/vnd.github.raw+json'}) as response:
   response.raise_for_status();count=0
   content=[]
   for chunk in response.iter_content(1024*1024):
    count+=len(chunk);assert count<=SIZE*2;content.append(chunk)
  data=b''.join(content)
  if data[:1]==b'{':
   blob=json.loads(data);assert blob['encoding']=='base64';data=base64.b64decode(blob['content'])
  assert len(data)==SIZE
  temporary.write_bytes(data);temporary.replace(path)
 data=path.read_bytes();assert len(data)==SIZE
 assert hashlib.sha1(b'blob '+str(SIZE).encode()+b'\0'+data).hexdigest()==BLOB
 rows=[]
 with np.load(path,allow_pickle=False) as archive:
  for key in archive.files:
   array=archive[key];assert not array.dtype.hasobject
   row={'key':key,'shape':list(array.shape),'dtype':str(array.dtype)}
   if np.issubdtype(array.dtype,np.number):row['finite_fraction']=float(np.isfinite(array).mean())
   rows.append(row)
 receipt={'repository':'https://github.com/ALRhub/HiP-RSSM','revision':REV,'source_path':REL,'url':url,'path':path.relative_to(ROOT).as_posix(),
          'request_url':request_url,'bytes':SIZE,'git_blob_sha1':BLOB,'sha256':hashlib.sha256(data).hexdigest(),'arrays':rows,'allow_pickle':False,
          'scope':'Original HiP-RSSM numeric source, not the unavailable MTS3 processed pickle; downstream protocol must be declared',
          'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 out=ROOT/'outputs/hip_mobile_data_registry.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
