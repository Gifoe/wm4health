"""One pinned public checkpoint, bounded resumable transfer, SHA256 required."""
import runtime_profile
import argparse,datetime,hashlib,json,pathlib,time
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO='thummd/do-over-time-pfn';REV='bec6618c8b25f3a5d8cf608d4c597978d5a694ac'
NAME='s9ho_all_causal/do_over_time_pfn_best.pt';SHA='214db2fab16fb5fa17e68244abc0df3b7b0154b9fb2d2ce16b2be3f4d5d93618';SIZE=295027515
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--local-fetch',action='store_true');args=parser.parse_args()
 out=ROOT/'outputs';out.mkdir(exist_ok=True);dst=ROOT/'data/do_over_time_pfn_best.pt';dst.parent.mkdir(exist_ok=True)
 errors=[]
 if dst.exists():assert dst.stat().st_size==SIZE and sha(dst)==SHA;print('Already verified');return
 for base in ['https://hf-mirror.com','https://huggingface.co']:
  session=requests.Session();session.trust_env=False
  try:
   r=session.get(base+'/api/models/'+REPO+'/revision/'+REV,params={'blobs':'true'},timeout=(5,15));r.raise_for_status();meta=r.json();assert meta['sha']==REV
   row=next(x for x in meta['siblings'] if x['rfilename']==NAME);assert row['lfs']['sha256']==SHA and row['size']==SIZE
   (out/'dotime_checkpoint_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
   tmp=dst.with_suffix('.pt.partial');offset=tmp.stat().st_size if tmp.exists() else 0
   headers={'Range':f'bytes={offset}-'} if offset else {}
   with session.get(base+'/'+REPO+'/resolve/'+REV+'/'+NAME,headers=headers,stream=True,timeout=(5,20)) as response:
    response.raise_for_status()
    if offset:assert response.status_code==206,'Range not honored; never concatenate full response'
    with tmp.open('ab') as target:
     count=offset;last=time.monotonic()
     for chunk in response.iter_content(1048576):
      if chunk:target.write(chunk);count+=len(chunk)
      if time.monotonic()-last>10:print('bytes',count,'of',SIZE,flush=True);last=time.monotonic()
   assert tmp.stat().st_size==SIZE and sha(tmp)==SHA
   tmp.replace(dst);record={'repository':REPO,'revision':REV,'file':NAME,'bytes':SIZE,'sha256':SHA,'endpoint':base,'earlier_errors':errors,'verified':True,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
   (out/'dotime_checkpoint_download.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record));return
  except (requests.RequestException,OSError) as error:
   errors.append({'endpoint':base,'error':repr(error)});print('endpoint failed',base,type(error).__name__,flush=True)
 (out/'dotime_checkpoint_download_attempts.json').write_text(json.dumps(errors,indent=2),encoding='utf-8');raise RuntimeError('Both TLS-verified endpoints failed; use authenticated relay without restarting an already-live fetch.')
if __name__=='__main__':main()
