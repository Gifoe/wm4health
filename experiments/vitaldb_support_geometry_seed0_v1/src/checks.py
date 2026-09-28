"""Focused leakage check for the reused factual prospective-input pathway."""
import numpy as np
import torch
from common import *

def main():
    seed_all(0);store=Store();sets={s:Windows(store,s) for s in ['train','val','test']}
    subjects={s:set(np.load(ARRAYS/f'{s}_subject.npy')) for s in sets}
    assert not any(subjects[a]&subjects[b] for a,b in [('train','val'),('train','test'),('val','test')])
    result={'subject_splits_disjoint':True,'case_counts':{s:len(store.splits[s]) for s in sets},
            'window_counts':{s:len(sets[s]) for s in sets}}
    ds=sets['test'];cid,t=ds.indices[0];c=store.cases[int(cid)];before=ds[0]
    ce=c['ce'].copy();future=c['sn'][t+1:t+F+1].copy()
    try:
        c['ce'][:]=1e6;c['sn'][t+1:t+F+1]=1e6
        after=ds[0]
        for key in ['state','action','future_action','demo','current']:
            assert np.array_equal(before[key],after[key]),key
    finally:c['ce']=ce;c['sn'][t+1:t+F+1]=future
    model=model_for(0,store,'cpu')
    b={k:torch.as_tensor(v)[None] for k,v in before.items() if k!='index'}
    with torch.no_grad():
        p,z,steps=model(b,'true',return_steps=True)
        pert=dict(b);pert['target']=b['target']+9999;pert['target_mask']=b['target_mask']*0
        p2,z2,steps2=model(pert,'true',return_steps=True)
        assert torch.equal(p,p2) and torch.equal(z,z2) and torch.equal(steps,steps2)
    result.update({'ce_and_future_physiology_mutation_does_not_change_model_inputs':True,
                   'target_mutation_does_not_change_predictions_or_latents':True,
                   'geometry_feature_allowlist':['history_state','history_action','demographics','future_action','predicted_latent'],
                   'geometry_excludes':['future_BIS','future_MAP','TCI_CE'],
                   'seed0_parameters':sum(p.numel() for p in model.parameters())})
    jwrite('preflight_checks.json',result)
    print('CHECKS_PASS',result,flush=True)

if __name__=='__main__':main()
