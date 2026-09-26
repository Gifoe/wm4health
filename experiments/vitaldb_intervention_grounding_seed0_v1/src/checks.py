"""Focused scientific integrity checks, not performance evidence."""
import json
import hashlib
import datetime
import numpy as np
import torch
from core import *
from prepare import resample
from torch.utils.data import DataLoader

def main():
    freeze={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'stage':'before_forecast_training','configuration':CFG,
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in list((ROOT/'src').glob('*.py'))+[ROOT/'config.yaml']}}
    freeze_path=ROOT/'outputs/protocol_freeze.json'
    if not freeze_path.exists(): freeze_path.write_text(json.dumps(freeze,indent=2))
    seed_all(); store=Store(); ds=Windows(store,'train')
    splits=json.loads((ROOT/'outputs/split_subjectids.json').read_text())
    assert not(set(splits['train'])&set(splits['val']))
    assert not(set(splits['test'])&(set(splits['val'])|set(splits['train'])))
    cid,t=ds.indices[0]; c=store.cases[int(cid)]; before=ds[0]
    orig_ce=c['ce'].copy(); c['ce'][:]=999
    after=ds[0]
    for k in ('state','action','demo','current'): assert np.array_equal(before[k],after[k])
    c['ce']=orig_ce
    orig_s=c['sn'].copy(); orig_a=c['an'].copy()
    c['sn'][t+1:]=777; c['an'][t+1:]=888
    after=ds[0]
    for k in ('state','action','demo','current'): assert np.array_equal(before[k],after[k])
    c['sn']=orig_s; c['an']=orig_a
    a=np.array([[5,10],[15,20],[200,30]],float)
    values,_=resample(a,np.array([0,10,20,90,190,200]),0,100)
    assert np.isnan(values[0]) and values[1]==10 and values[2]==20
    assert np.isnan(values[3]) and np.isnan(values[4]) and values[5]==30
    b=next(iter(DataLoader(ds,batch_size=4)))
    shifted=perturb_actions(b,store,shift=30)['action']
    assert torch.equal(shifted[:,3:],b['action'][:,:-3])
    assert torch.equal(shifted[:,:3],b['action'][:,:1].expand(-1,3,-1))
    shifted=perturb_actions(b,store,shift=-30)['action']
    assert torch.equal(shifted[:,:-3],b['action'][:,3:])
    zero=perturb_actions(b,store,scale=0)['action']
    mean=torch.tensor(store.norm['action_mean']); std=torch.tensor(store.norm['action_std'])
    reconstructed=torch.expm1(zero[:,:,:2]*std+mean)/10
    valid=zero[:,:,2:].bool(); assert reconstructed[valid].abs().max()<1e-6
    for name in MODEL_NAMES:
        model=model_for(name).eval(); y,z=model(b)
        assert y.shape==(4,30,2) and z.shape==(4,64)
        assert torch.isfinite(y).all()
        loss=((y-b['target'])**2*b['target_mask']).mean(); loss.backward()
        if name=='State-only':
            changed=dict(b); changed['action']=b['action']+100
            assert torch.equal(model(changed)[0],model(b)[0])
    result={'patient_split_disjoint':True,'ce_mutation_no_input_effect':True,
            'future_state_and_action_no_input_effect':True,'no_backward_fill_or_long_gap_interpolation':True,
            'shift_sign_padding_no_wrap':True,'zero_means_physical_zero':True,
            'models_shapes_and_backward':True,'state_model_action_independent':True,
            'feature_allowlist':['past_BIS_HR_MAP','past_drug_administration','demographics','availability_masks'],
            'forbidden_features':['CE','CP','CT','future_states','future_actions','caseid','subjectid']}
    (ROOT/'outputs/integrity_checks.json').write_text(json.dumps(result,indent=2))
    print('INTEGRITY_CHECKS_PASS',flush=True)

if __name__=='__main__': main()
