from native87_v1 import *
from evaluate87_v2 import NativeField
import numpy as np
def main():
 res=read(O/'EVALUATION_RESULTS87_V3.json');row=next(r for r in res['rows'] if r['model']=='native_frozen' and r['cohort']=='retained');context=[];initial=[]
 for fam in ['duffing','stuart_landau']:
  with np.load(RAW/'evaluation_v2'/f'{fam}_retained_v1.npz') as z:context.append(z['context']);initial.append(z['query_initial'])
 context=np.concatenate(context)
 with np.load(row['array_path']) as z:pred=z['prediction']
 ix=np.unique([0,int(abs(pred[:128]).max(axis=(1,2)).argmax()),128,128+int(abs(pred[128:]).max(axis=(1,2)).argmax())]);m=load_native();m.eval()
 for p in m.parameters():p.requires_grad_(False)
 full=NativeField(m,context);subset=NativeField(m,context[ix]);rows=[]
 for step in [0,50,100,150,200]:
  q=torch.as_tensor(pred[:,step],device='cuda',dtype=torch.float32);all_y=full(q).cpu().numpy()[ix];some_y=subset(q[ix]).cpu().numpy();again=subset(q[ix]).cpu().numpy();r=dict(step=step,subset_vs_original_batch_max_abs=float(abs(all_y-some_y).max()),subset_repeat_max_abs=float(abs(some_y-again).max()),max_abs_field=float(abs(all_y).max()));rows.append(r);log(dict(stage='native_batch_precision',**r))
 write(O/'BATCH_PRECISION87_V1.json',dict(utc=utc(),status='NATIVE_BATCH_SHAPE_AND_REPEAT_PRECISION_DIAGNOSTIC',indices=ix.tolist(),rows=rows,native_model_unchanged=True))
if __name__=='__main__':main()
