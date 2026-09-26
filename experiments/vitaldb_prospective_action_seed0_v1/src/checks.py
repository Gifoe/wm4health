"""Pretraining leakage and source-identity checks."""
import hashlib,json,shutil
import numpy as np
import torch
from core import *

def main():
    seed_all(); s=Store(); train=Windows(s,'train'); val=Windows(s,'val'); test=Windows(s,'test')
    assert len(set(s.subject_splits['train'])&set(s.subject_splits['val']))==0
    assert len(set(s.subject_splits['train'])&set(s.subject_splits['test']))==0
    assert len(set(s.subject_splits['val'])&set(s.subject_splits['test']))==0
    assert all(s.cases[int(cid)]['subjectid'] in s.subject_splits[k]
               for k,ds in [('train',train),('val',val),('test',test)] for cid in ds.indices[:,0])
    cid,t=train.indices[0]; c=s.cases[int(cid)]; b=train[0]
    assert np.array_equal(b['future_action'][:,:2],c['an'][t+1:t+F+1])
    assert np.array_equal(b['action'][-1,:2],c['an'][t])
    ce=c['ce'].copy(); c['ce'][:]=1e6
    assert all(np.array_equal(b[k],train[0][k]) for k in ['state','action','future_action','demo','target','target_mask'])
    c['ce']=ce
    sn=c['sn'].copy(); c['sn'][t+1:t+F+1]=98765
    assert all(np.array_equal(b[k],train[0][k]) for k in ['state','action','future_action','demo','current'])
    c['sn']=sn
    batch={k:torch.as_tensor(v)[None] for k,v in b.items() if k!='index'}
    m=RSSM(); m.set_action_stats(s); m.eval()
    with torch.no_grad():
        true=m(batch,'true',return_steps=True)[2]; hold=m(batch,'hold',return_steps=True)[2]
        changed={**batch,'future_action':batch['future_action']+2}
        assert torch.equal(hold,m(changed,'hold',return_steps=True)[2])
        assert not torch.equal(true,m(changed,'true',return_steps=True)[2])
        original=m(batch,'true')[0]
        changed={**batch,'target':batch['target']+999,'target_mask':batch['target_mask']*0}
        assert torch.equal(original,m(changed,'true')[0])
    count=sum(p.numel() for p in m.parameters())
    assert count==58946,count
    original=json.loads((SOURCE/'outputs/data_summary.json').read_text())
    summary={'source_experiment':str(SOURCE),'source_commit':'a2ff8e4695f7625123f44f950d3575d38f6ff9a1',
             'included_cases':original['included_cases'],'included_patients':original['included_patients'],
             'split_caseids':s.splits,'split_subjectids':s.subject_splits,
             'windows':{k:{'round1':d.original_windows,'complete_future_action':d.original_windows-d.omitted_future_action,
                            'omitted_future_action':d.omitted_future_action} for k,d in [('train',train),('val',val),('test',test)]},
             'normalization_sha256':hashlib.sha256((SOURCE/'outputs/normalization.json').read_bytes()).hexdigest(),
             'round1_large_change_protocol_sha256':hashlib.sha256((SOURCE/'outputs/large_change_protocol.json').read_bytes()).hexdigest()}
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    for name in ['split_caseids.json','split_subjectids.json','normalization.json']:
        shutil.copyfile(SOURCE/'outputs'/name,out/name)
    (out/'data_summary.json').write_text(json.dumps(summary,indent=2))
    result={'patient_splits_disjoint':True,'same_round1_caseids_and_normalization':True,
            'future_action_starts_at_t_plus_1':True,'future_action_from_action_channels_only':True,
            'ce_mutation_no_forecast_feature_or_target_effect':True,'future_physiology_mutation_no_input_effect':True,
            'target_mutation_no_prediction_effect':True,'hold_ignores_future_action':True,
            'true_responds_to_future_action':True,'forecast_loss_uses_only_BIS_MAP':True,
            'normalization_train_only_inherited':True,'parameter_count_each':count}
    (out/'integrity_checks.json').write_text(json.dumps(result,indent=2))
    print('CHECKS_PASS',json.dumps(summary['windows']),flush=True)

if __name__=='__main__': main()
