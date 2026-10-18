from common65 import *
f=read(OUT/'FITS65_V1.json') if (OUT/'FITS65_V1.json').exists() else read(OUT/'fit_registry65_v1.json');o=read(OUT/'ORACLE_DSM65_V1.json');rows=[]
for fit in f['fits']:
 if fit['mechanism']!='linear' or not fit['model'].startswith('CSDI'):continue
 for cp in fit['checkpoints']:
  for view in ['block','prefix']:
   truth=next(r for r in o['rows'] if r['key']==fit['key'] and r['view']==view)['realized_Bayes_validation_DSM'];rows.append(dict(key=fit['key'],model=fit['model'],step=cp['step'],view=view,learned_DSM=cp['validation'][view],Bayes_DSM=truth,excess=cp['validation'][view]-truth,selected=cp['sha256']==fit['selected_checkpoint']['sha256']))
print(json.dumps(dict(closed_fits=len(f['fits']),linear_validation_rows=rows)))
