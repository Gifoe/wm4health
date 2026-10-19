"""Verify matched prefixes and native sources; summarize completed fixed runs."""
import csv,hashlib,json,pathlib,platform
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/'outputs';pilot=out/'noise_objective_pilot';attribution=out/'noise_attribution'
 assert json.loads((pilot/'status.json').read_text())['state']=='complete'
 assert json.loads((attribution/'status.json').read_text())['state']=='complete'
 registry=json.loads((out/'source_registry.json').read_text());checked=0
 for repo in registry:
  for f in repo['files']:
   folder='motion-jepa' if repo['repository']=='mkarmann/motion-jepa' else 'vjepa'
   assert digest(ROOT/'sources'/folder/f['path'])==f['sha256'],f['path'];checked+=1
 prefix=[]
 for variant in ['lewm_native_weight','lewm_matched_static','motion']:
  a=torch.load(pilot/f'independent_nuisance_{variant}_seed41'/'model.pt',map_location='cpu',weights_only=True)
  b=torch.load(attribution/f'{variant}_random'/'model_step2000.pt',map_location='cpu',weights_only=True)
  assert a.keys()==b.keys()
  error=max((a[k]-b[k]).abs().max().item() for k in a)
  assert error==0,(variant,error)
  prefix.append({'variant':variant,'tensors':len(a),'maximum_absolute_difference':error,'exact_match':True})
 rows=json.loads((attribution/'results.json').read_text())
 with (attribution/'summary.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['run','step','physical_probe_nmse','signal_subspace_overlap','parameters']);w.writeheader()
  for row in rows:
   for s in row['snapshots']:w.writerow({k:row.get(k,s.get(k)) for k in w.fieldnames})
 matched={r['run']:r for r in rows}
 pilotrows=json.loads((pilot/'results.json').read_text())
 val={(r['regime'],r['variant']):r['physical_probe_nmse'] for r in pilotrows}
 noise_gap=val['independent_nuisance','motion']-val['independent_nuisance','lewm_matched_static']
 clean_gap=val['clean','motion']-val['clean','lewm_matched_static']
 gate=noise_gap>.10 and clean_gap<.05
 assert not gate
 report={'source_files_verified':checked,'matched_2000_step_prefixes':prefix,'pilot_noise_gap':noise_gap,'pilot_clean_gap':clean_gap,'preregistered_noise_specific_gate_passed':gate,'final_nmse':{r['run']:r['snapshots'][-1]['physical_probe_nmse'] for r in rows},'conclusion':'Pilot structural attribution rejected. Ordinary budget fixes default LeWM; training-only CCA initialization fixes matched-static LeWM. Motion remains initialization-sensitive in these adapters, without meeting pilot acceptance rule or establishing a new method gap.','scope':'Single initialization and fixed data cohort, exploratory development probes. Not native visual benchmark, independent seed confirmation, or publication result.','environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}}
 (attribution/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 try:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
  colors=['#1b6ca8','#e59a35','#ae3f47','#528b68','#845aa5']
  for row,color in zip(rows,colors):
   ax.plot([s['step'] for s in row['snapshots']],[s['physical_probe_nmse'] for s in row['snapshots']],marker='o',label=row['run'].replace('_',' '),color=color)
  ax.set(xlabel='Fixed training step',ylabel='Held-out physical-state probe NMSE',title='Noise objective attribution: one initialization, one data cohort',yscale='log')
  ax.grid(alpha=.2);ax.legend(fontsize=8);fig.savefig(attribution/'attribution.png',dpi=170);plt.close(fig)
 except ImportError:
  report['figure']='matplotlib unavailable; numeric CSV complete'
  (attribution/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
