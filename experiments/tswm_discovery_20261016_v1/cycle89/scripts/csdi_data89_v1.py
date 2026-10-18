from csdi_core89_v1 import *
import time

def main():
    started=time.perf_counter()
    assert not (O/'CSDI_DATA89_V1.json').exists()
    dest=Path(STORAGE['data_root'])/'cycle89/csdi_v1'
    dest.mkdir(parents=True,exist_ok=True)
    arrays={}
    for split,seed,n in [('train',20268910,4096),('cal',20268911,128)]:
        rng=np.random.default_rng(seed)
        arrays[split]=[]
        for family in [0,1]:
            a=draw_actions(rng,n);world=rng.choice(9,size=n,p=PRIOR);s=simulate(a,world,family)
            arrays[split].append(dict(states=s,actions=a,world=world,family=np.full(n,family)))
    states=np.concatenate([d['states'] for d in arrays['train']])
    center=states.mean(axis=(0,1));scale=np.maximum(states.std(axis=(0,1)),.05)
    rows=[]
    for split in ['train','cal']:
        p=dest/(split+'.npz')
        data={key:np.concatenate([d[key] for d in arrays[split]]) for key in ['states','actions','world','family']}
        data['X']=np.concatenate([features(d['states'],d['actions'],int(d['family'][0]),center,scale) for d in arrays[split]])
        with p.open('xb') as f:np.savez_compressed(f,**data,center=center,scale=scale)
        rows.append(dict(split=split,path=str(p),sha256=sha(p),shape=list(data['X'].shape)))
    rng=np.random.default_rng(20268912);banks=[];acts=[];fams=[];xs=[]
    for family in [0,1]:
        a=draw_actions(rng,32)
        bank=np.stack([simulate(a,w,family) for w in range(9)],axis=1)
        placeholder=np.zeros((len(a),LENGTH,2))
        banks.append(bank);acts.append(a);fams.extend([family]*len(a));xs.append(features(placeholder,a,family,center,scale))
    p=dest/'eval_queries.npz'
    with p.open('xb') as f:np.savez_compressed(f,bank=np.concatenate(banks),actions=np.concatenate(acts),family=np.array(fams),X=np.concatenate(xs),center=center,scale=scale,prior=PRIOR,k_values=KS,group=np.tile(np.repeat(np.arange(8),4),2))
    rows.append(dict(split='eval_queries',path=str(p),sha256=sha(p),queries=64,groups_per_family=8,plans_per_group=4))
    receipt=dict(utc=utc(),files=rows,center=center.tolist(),scale=scale.tolist(),protocol_sha256=sha(C/'CSDI_COUPLING_PROTOCOL89_V1.md'),simulator_sha256=sha(ROOT/'cycle88/scripts/plan_data88_v1.py'),elapsed_seconds=time.perf_counter()-started,eval_used_for_fit_or_selection=False)
    write(O/'CSDI_DATA89_V1.json',receipt)
    log(dict(stage='csdi_data_created',**{k:v for k,v in receipt.items() if k!='utc'}))

if __name__=='__main__':main()
