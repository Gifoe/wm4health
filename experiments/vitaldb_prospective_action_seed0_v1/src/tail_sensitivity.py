"""Exploratory sensitivity to transient high pump rates using TRAIN-only dose caps."""
import json
import numpy as np
import pandas as pd
from core import ROOT,Store
from evaluate import abs_error,scalar_error,bootstrap

def main():
    out=ROOT/'outputs';store=Store();ref=np.load(out/'test_reference.npz')
    subject=ref['subject'];future=ref['future_action'];cases=ref['case'];anchors=ref['t']
    current=np.array([store.cases[int(cid)]['action'][t] for cid,t in zip(cases,anchors)])
    y=ref['target'];mask=ref['mask'];donor=ref['donor']
    errors={}
    for name in ['true','hold','wrong','direct_history','direct_future']:
        filename=f'prediction_{name}.npz'
        p=np.load(out/filename)['prediction']
        errors[name]=scalar_error(abs_error(p,y,mask),0)
    rows=[];caps={}
    for quantile in [.99,.995]:
        limit=[]
        for j in range(2):
            a=np.concatenate([store.cases[c]['action'][:,j] for c in store.splits['train']])
            a=a[np.isfinite(a)&(a>0)];limit.append(float(np.quantile(a,quantile)))
        limit=np.array(limit)
        clean=(future<=limit).all((1,2))&(current<=limit).all(1)
        caps[str(quantile)]={'maximum_ml_per_10s':limit.tolist(),
                            'maximum_ml_per_h':(limit*360).tolist(),
                            'n_test_clean_windows':int(clean.sum())}
        for subset,base in [('overall',np.ones(len(subject),bool)),('Q4',ref['quartile']==4),
                            ('large_transition',ref['large_label']>0),
                            ('upcoming_large_transition',ref['upcoming_label']>0)]:
            for metric,name in [('FAV','hold'),('MFAV','wrong'),('Direct future value','direct_history')]:
                keep=clean&base&((donor>=0) if name=='wrong' else True)
                delta=errors[name]-(errors['direct_future'] if metric=='Direct future value' else errors['true'])
                v,lo,hi,n=bootstrap(delta[keep],subject[keep])
                rows.append({'train_positive_rate_quantile_cap':quantile,'subset':subset,'metric':metric,
                             'difference_mae':v,'ci_low':lo,'ci_high':hi,
                             'n_windows':int(keep.sum()),'n_patients':n})
    pd.DataFrame(rows).to_csv(out/'pump_rate_tail_sensitivity.csv',index=False)
    (out/'pump_rate_tail_protocol.json').write_text(json.dumps(caps,indent=2))
    print('TAIL_SENSITIVITY_COMPLETE',flush=True)

if __name__=='__main__':main()
