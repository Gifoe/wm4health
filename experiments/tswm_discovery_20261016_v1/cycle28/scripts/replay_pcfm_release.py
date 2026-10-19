"""New-process numeric replay of every recorded native/example/screen inference."""
import runtime_profile
import hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/pcfm_native'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.monotonic();result=json.loads((OUT/'results.json').read_text());calls=json.loads((OUT/'calls.json').read_text())
 weight=ROOT/'outputs/external/PCFM/model.safetensors';assert sha(weight)==result['weights_sha256']
 sys.path.insert(0,str(ROOT/'upstream/PCFM'));from model import PCFMPredictor
 model=PCFMPredictor.from_pretrained(str(weight),device='cuda');maximum=0.
 for row in calls:
  path=OUT/row['path'];assert sha(path)==row['sha256'];data=dict(np.load(path));kwargs={k:(data[v[6:]] if isinstance(v,str) and v.startswith('array:') else v) for k,v in row['kwargs'].items()}
  actual=getattr(model,row['method'])(data['input'],*row.get('positional_tail',[]),**kwargs).cpu().numpy();reference=data['output'];error=float(np.max(np.abs(actual-reference)))
  maximum=max(maximum,error);assert np.allclose(actual,reference,atol=2e-5,rtol=2e-5),(row['index'],error)
 receipt=dict(status='passed',calls_replayed=len(calls),maximum_abs_error=maximum,runtime_s=time.monotonic()-start,results_sha256=sha(OUT/'results.json'),script_sha256=sha(pathlib.Path(__file__)),
  scope='All28 native numerical calls replayed with same unmodified release in a new process;not independent reimplementation of network')
 (OUT/'neural_replay.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
