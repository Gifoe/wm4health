"""Prediction invariance to unavailable future values, selected real checkpoints."""
from common40_v2 import *
S=R/'outputs/screen_v1';O=R/'outputs';assert (S/'COMPLETE.json').exists()
rows=json.loads((S/'results.json').read_text());checks=[]
with torch.no_grad():
 for row in rows:
  model=make_fnp() if row['family']=='flownp' else ARMixture().cuda()
  saved=torch.load(S/row['key']/'selected.pt',map_location=DEVICE,weights_only=False);model.load_state_dict(saved['model'],strict=True);model.eval()
  seq,c=windows(data(row['data']),[origins(row['data'],'val')[0]])
  for r in REVEALS:
   corrupt=seq.clone();corrupt[:,HISTORY+r:]=37.+torch.arange(HORIZON-r,device=DEVICE)[None,:,None]
   def predict(x):
    if row['family']=='flownp':return fnp_sample(model,x,c,r,4,8)
    return model.sample_cached(x,c,r,model.context(x,c,r),4)
   seed(20264077+r);a=predict(seq);seed(20264077+r);b=predict(corrupt)
   err=float((a-b).abs().max());assert err==0,(row['key'],r,err)
   checks.append({'key':row['key'],'r':r,'max_prediction_change_after_unavailable_future_corruption':err})
out={'status':'PASS','checks':checks,'test_scored':False,'interpretation':'Checks current prediction path for leakage, not statistical independence or causal identification.'}
(O/'information_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({'status':'PASS','checks':len(checks)}))
