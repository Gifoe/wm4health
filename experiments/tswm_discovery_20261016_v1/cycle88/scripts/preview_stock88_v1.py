from common88_v1 import *
import numpy as np,torch
from evaluate_plan88_v1 import predict
from plan_models88_v1 import make_model

def main():
    fit=read(O/'PLAN_FIT88_V1_stock.json');assert sha(fit['selected_path'])==fit['selected_sha256']
    row=next(r for r in read(O/'PLAN_DATA88_V1.json')['files'] if r['split']=='val');assert sha(row['path'])==row['sha256']
    with np.load(row['path']) as z:obs=z['observations'][:,48:145];act=z['actions'][:,48:144];family=z['family']
    model=make_model('stock');model.load_state_dict(torch.load(fit['selected_path'],map_location='cpu',weights_only=False)['model']);model.eval();base,_=predict(model,obs,act);repeat,_=predict(model,obs,act);assert np.array_equal(base,repeat)
    rows=[]
    for cut in [80,88]:
        for sign in [-1,1]:
            edited=act.copy();edited[:,cut:,0]=sign*.8;p,_=predict(model,obs,edited);early=cut-64
            for f in [0,1]:
                delta=(p-base)[family==f,:early]
                rows.append(dict(family=f,cut=cut,sign=sign,early_max=float(abs(delta).max()),early_rms=float(np.sqrt(np.mean(delta.astype(np.float64)**2)))))
    write(O/'STOCK_PAIRED_PREVIEW88_V1.json',dict(utc=utc(),status='TRAINED_STOCK_ONLY_PREVIEW_NO_CONTROL_COMPARISON_YET',selected_step=fit['selected']['step'],rows=rows,other_arms_protocol_unchanged=True,checkpoint_selection_uses_edited_plans=False))
    log(dict(stage='trained_stock_prefix_preview',rows=rows))

if __name__=='__main__':main()
