"""Independent released-data/physical-reference and output score reconstruction. No model import."""
import runtime_profile
import csv,hashlib,json,pathlib,re,time
import numpy as np
from scipy.signal import lfilter
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/pcfm_native'
C=4096;H=128;LAG=48
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def physical(u,y0):
 delayed=np.r_[-np.ones(LAG),u[:-LAG]];setpoint=20+30*(delayed+1)
 return lfilter([.05],[1,-.95],setpoint,zi=[.95*y0])[0]
def seeded_window():
 r=np.random.RandomState(7);u=2*(np.cumsum(r.rand(20000)<.02)%2)-1
 y=physical(u,20)+r.normal(0,.5,len(u));v=(u+1)/2
 for i in range(20000-C-H):
  if v[i+C-64:i+C].mean()<.1 and .2<v[i:i+C].mean()<.8:return np.stack([u[i:i+C+H],y[i:i+C+H]])
 return np.stack([u[:C+H],y[:C+H]])
def main():
 started=time.monotonic();result=json.loads((OUT/'results.json').read_text());calls=json.loads((OUT/'calls.json').read_text())
 assert result['status']=='completed' and sha(ROOT/'PCFM_EXECUTION_PLAN.md')==result['plan_sha256']
 assert sha(ROOT/'outputs/source/pcfm_manifest.json')==result['source_manifest_sha256']
 manifest=json.loads((ROOT/'outputs/source/pcfm_manifest.json').read_text())
 for r in manifest['files']:assert sha(ROOT/'upstream/PCFM'/r['path'])==r['sha256']
 native=seeded_window();data={};max_quantile=0.;physical_error=0.;counts=dict(source_hashes=len(manifest['files']),array_hashes=0,seeded_input_values=0,physical_values=0,finite_output_values=0)
 for call in calls:
  p=OUT/call['path'];assert sha(p)==call['sha256'];d=dict(np.load(p));data[call['index']]=d;counts['array_hashes']+=1
  pred=d['output'];assert np.isfinite(pred).all();counts['finite_output_values']+=pred.size
  if pred.ndim==3:
   violation=float(max(0,-np.diff(pred,axis=-1).min()));max_quantile=max(max_quantile,violation);assert violation<=1e-5
  if call['method']=='what_if' and call['caller']=='demo1_whatif':
   x=d['input'];assert np.array_equal(x[0,:C],native[0,:C]);assert np.allclose(x[1],native[1],atol=1e-11,rtol=1e-12);counts['seeded_input_values']+=2*C+H
   true=physical(x[0],x[1,0]);v=x[1,0];loop=[]
   for t in range(len(x[0])):
    u=x[0,t-LAG] if t>=LAG else -1.;v=.95*v+.05*(20+30*(u+1));loop.append(v)
   err=float(np.max(np.abs(true-loop)));physical_error=max(physical_error,err);assert err<1e-10;counts['physical_values']+=len(true)
 native_rows=[]
 for call,name in zip(calls[:3],['ON','OFF','PULSE32']):
  assert call['caller']=='demo1_whatif';d=data[call['index']];truth=physical(d['input'][0],d['input'][1,0])[C:];pred=d['output'][1,C:]
  native_rows.append(dict(plan=name,mae=float(np.mean(np.abs(pred-truth))),predicted_end=float(pred[-1]),physical_end=float(truth[-1])))
 on=data[0];off=data[1];effect_p=on['output'][1,C:]-off['output'][1,C:];effect_t=(physical(on['input'][0],on['input'][1,0])-physical(off['input'][0],off['input'][1,0]))[C:]
 effect_mae=float(np.mean(np.abs(effect_p-effect_t)))
 stdout=(ROOT/'outputs/jobs/pcfm_native_v1/stdout.log').read_text(encoding='utf-8')
 printed=[float(x) for x in re.findall(r'\| MAE\s+([\d.]+)',stdout)];assert len(printed)==3
 for printed_value,row in zip(printed,native_rows):assert abs(printed_value-row['mae'])<=.00051
 m=re.search(r'ON-minus-OFF\):\s+([\d.]+)',stdout);assert m and abs(float(m[1])-effect_mae)<=.00051
 screen=[]
 for entry in result['screen']:
  d=data[entry['call_index']];mu=d['output'][1,C:,4];truth=physical(d['input'][0],d['input'][1,0])[C:]
  base=next(e for e in result['screen'] if e['glue']==entry['glue'] and e['plan']=='OFF');b=data[base['call_index']];bmu=b['output'][1,C:,4];bt=physical(b['input'][0],b['input'][1,0])[C:]
  prefix=min(H,(int(entry['plan'][5:]) if entry['plan'].startswith('DELAY') else 0)+LAG) if entry['plan']!='OFF' else H
  assert np.max(np.abs(truth[:prefix]-bt[:prefix]))==0.
  diff=mu[:prefix]-bmu[:prefix]
  screen.append(dict(plan=entry['plan'],glue=entry['glue'],mae=float(np.mean(np.abs(mu-truth))),prefix_steps=prefix,
   forbidden_prefix_max_abs=float(np.max(np.abs(diff))),forbidden_prefix_rms=float(np.sqrt(np.mean(diff**2))),
   prefix_rms_over_context_std=float(np.sqrt(np.mean(diff**2))/np.std(d['input'][1,:C])),first_forecast_anchor_abs=float(abs(mu[0]-d['input'][1,C-1]))))
 with (OUT/'contract_rows.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(screen[0]));w.writeheader();w.writerows(screen)
 verification=dict(status='passed',results_sha256=sha(OUT/'results.json'),verifier_sha256=sha(pathlib.Path(__file__)),counts=counts,
  native_demo1=native_rows,native_on_off_effect_mae=effect_mae,physical_implementations_max_abs=physical_error,quantile_order_max_violation=max_quantile,
  screen=screen,elapsed_s=time.monotonic()-started,
  scope='Independent source/array hashes,vectorized seeded native input,physical lfilter/recurrence and scores. Not a neural-weight or full five-demo numerical replay')
 (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8');print(json.dumps(verification),flush=True)
if __name__=='__main__':main()
