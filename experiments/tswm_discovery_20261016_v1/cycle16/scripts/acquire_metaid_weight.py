"""Acquire one explicitly identified official release asset; never deserialize it."""
import hashlib,json,pathlib,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
NAME='ckpt_430_fine-tuned_3100000_lower_noise.pt';SIZE=90193824
URL='https://github.com/PuraFilippo/sysid_benchmarks/releases/download/v0.1/'+NAME
def main():
 out=ROOT/'outputs/external/metaid';out.mkdir(parents=True,exist_ok=True)
 p=out/NAME;temporary=p.with_suffix('.partial');assert not p.exists() and not temporary.exists()
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 start=time.perf_counter();h=hashlib.sha256();count=0;last=0
 with opener.open(URL,timeout=40) as response,temporary.open('xb') as f:
  while True:
   block=response.read(1048576)
   if not block:break
   f.write(block);h.update(block);count+=len(block)
   if count-last>=20*1048576:
    print(json.dumps({'asset':NAME,'downloaded_bytes':count,'expected_bytes':SIZE}),flush=True);last=count
 assert count==SIZE,(count,SIZE);temporary.replace(p)
 receipt={'repository':'https://github.com/PuraFilippo/sysid_benchmarks','release':'v0.1','asset_id':237784764,
          'asset_name':NAME,'url':URL,'bytes':count,'sha256':h.hexdigest(),'publisher_digest_available':False,
          'elapsed_seconds':time.perf_counter()-start,'deserialized':False,'purpose':'Native pretrained probabilistic meta-system-ID baseline; no retraining or original metric claim'}
 (out/'acquisition.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
