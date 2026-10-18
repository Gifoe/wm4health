from critic_stress89_v1 import *

def ipf(log_joint,K):
    z=log_joint.reshape((9,)*K).copy();z-=logsumexp(z);started=time.perf_counter()
    for sweep in range(1,10001):
        for axis in range(K):
            others=tuple(a for a in range(K) if a!=axis)
            marginal=logsumexp(z,axis=others)
            shape=[1]*K;shape[axis]=9
            z+=(np.log(PI)-marginal).reshape(shape)
        marginals=np.stack([np.exp(logsumexp(z,axis=tuple(a for a in range(K) if a!=axis))) for axis in range(K)])
        error=float(abs(marginals-PI).max())
        if error<=1e-10:break
    w=np.exp(z.reshape(-1));assert abs(w.sum()-1)<1e-10
    return w,marginals,dict(sweeps=sweep,max_abs_marginal_error=error,converged=error<=1e-10,elapsed_seconds=time.perf_counter()-started)

def main():
    started=time.perf_counter();assert not (O/'MARGINAL_CONTROL89_V1.json').exists()
    original=read(O/'CRITIC_STRESS89_V1.json');rows=[];replays=[]
    for saved in original['models']:
        assert sha(saved['checkpoint_path'])==saved['checkpoint_sha256']
        cp=torch.load(saved['checkpoint_path'],map_location='cpu',weights_only=False)
        model=CountMLP() if saved['arm'].startswith('count') else GaussianProduct()
        model.load_state_dict(cp['model']);model.eval();value=val(model,saved['policy'])
        replay_error=abs(value['bce']-saved['selected_population']['bce']);assert replay_error<1e-12
        assert cp['step']==min(saved['trace'],key=lambda t:t['bce'])['step']
        replays.append(dict(arm=saved['arm'],seed=saved['seed'],population_bce_max_abs=replay_error,selected_step=cp['step']))
        anchor={}
        for K in [2,3,4]:
            p=POPS[K,'independent'];ids=torch.as_tensor(p['ids'])
            with torch.no_grad():score=torch.cat([model(ids[b:b+1024]) for b in range(0,len(ids),1024)]).numpy().astype(np.float64)
            base=np.log(p['proposal'])+score
            w,marginals,info=ipf(base,K)
            tv=.5*abs(marginals-PI).sum(1);anchor[K]=marginals[0]
            raw=next(r for r in saved['coupling'] if r['K']==K and r['gamma']==1.)
            row=dict(arm=saved['arm'],seed=saved['seed'],K=K,raw_coherence=raw['coherence'],projected_coherence=float(w[p['same']].sum()),raw_anchor_tv=raw['anchor_tv'],projected_max_branch_tv=float(tv.max()),anchor_marginal=marginals[0].tolist(),**info)
            rows.append(row);log(dict(stage='ordinary_ipf_control',**row))
        for row in rows[-3:]:row['projected_anchor_tv_K2_to_K4']=float(.5*abs(anchor[2]-anchor[4]).sum())
    receipt=dict(utc=utc(),status='ORDINARY_MARGINAL_CONSTRAINED_CONTROL_COMPLETE',rows=rows,checkpoint_replay=replays,all_converged=all(r['converged'] for r in rows),original_manifest_sha256=sha(O/'CRITIC_STRESS89_V1.json'),protocol_sha256=sha(C/'CSDI_COUPLING_PROTOCOL89_V1.md'),elapsed_seconds=time.perf_counter()-started,method_is_established_IPF=True,native_OneWorld_reproduction=False,new_world_model_fits=0,new_method_claim=False,accepted_research_directions=0)
    write(O/'MARGINAL_CONTROL89_V1.json',receipt)
    log(dict(stage='ordinary_ipf_closed',rows=len(rows),all_converged=receipt['all_converged']))

if __name__=='__main__':main()
