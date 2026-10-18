from common89_v1 import *
import importlib.util,numpy as np,time
from scipy.optimize import minimize_scalar

def load_c88_dynamics():
    folder=ROOT/'cycle88/scripts';sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('c88_fixed_simulator',folder/'plan_data88_v1.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.advance

advance=load_c88_dynamics()

def queries(seed,n):
    rng=np.random.default_rng(seed);roots=np.zeros((n,3));roots[:,:2]=rng.uniform(-.5,.5,(n,2))
    actions=np.zeros((n,96,2));actions[:,:,0]=np.repeat(rng.uniform(-1,1,(n,12)),8,axis=1)
    return roots,actions

def simulate(roots,actions,k,family):
    parameters=np.tile(np.array([k,.35,1.,.5,3.]),(len(roots),1));state=roots.copy();out=[]
    for t in range(actions.shape[1]):state=advance(state,actions[:,t],parameters,family);out.append(state[:,:2].copy())
    return np.stack(out,axis=1)

def rms(a,b):return float(np.sqrt(np.mean((a-b)**2)))

def mixture_fit(bank0,bank1,target):
    delta=bank0-bank1;weight=float(np.clip(np.sum(delta*(target-bank1))/np.sum(delta**2),0,1))
    return weight,weight*bank0+(1-weight)*bank1

def main():
    started=time.perf_counter();cal=queries(2026891,8);ev=queries(2026892,32);rows=[];arrays={}
    zero_root=np.zeros((2,3));zero_act=np.zeros((2,32,2))
    for family in [0,1]:
        z0=simulate(zero_root,zero_act,.6,family);z1=simulate(zero_root,zero_act,1.3,family);assert np.array_equal(z0,z1) and (z0==0).all()
        c0,c1=[simulate(*cal,k,family) for k in [.6,1.3]];e0,e1=[simulate(*ev,k,family) for k in [.6,1.3]]
        cy=.5*(c0+c1);ey=.5*(e0+e1)
        grid=np.linspace(.4,1.6,201);objectives=np.array([np.mean((simulate(*cal,k,family)-cy)**2) for k in grid]);idx=int(objectives.argmin());lo=grid[max(idx-1,0)];hi=grid[min(idx+1,len(grid)-1)]
        opt=minimize_scalar(lambda k:np.mean((simulate(*cal,k,family)-cy)**2),bounds=(lo,hi),method='bounded',options=dict(xatol=1e-10));k=float(opt.x)
        single_cal=simulate(*cal,k,family);single_ev=simulate(*ev,k,family);weight,mix_cal=mixture_fit(c0,c1,cy);mix_ev=weight*e0+(1-weight)*e1
        assert abs(weight-.5)<=1e-12 and rms(mix_ev,ey)<=1e-10
        cswitch=np.where(cal[1][:,0,0,None,None]>=0,c0,c1);eswitch=np.where(ev[1][:,0,0,None,None]>=0,e0,e1)
        w_bad,bad_cal=mixture_fit(c0,c1,cswitch);bad_ev=w_bad*e0+(1-w_bad)*e1
        row=dict(family=family,posterior_weights=[.5,.5],identical_context_max_abs=float(abs(z0-z1).max()),best_single_k=k,single_world_calibration_rms=rms(single_cal,cy),single_world_heldout_rms=rms(single_ev,ey),single_world_relative_heldout_rms=rms(single_ev,ey)/float(np.sqrt(np.mean(ey**2))),mixture_weight_from_calibration=weight,mixture_calibration_rms=rms(mix_cal,cy),mixture_heldout_rms=rms(mix_ev,ey),query_switch_global_weight=w_bad,query_switch_calibration_rms=rms(bad_cal,cswitch),query_switch_heldout_rms=rms(bad_ev,eswitch),calibration_queries=8,heldout_queries=32)
        assert row['single_world_heldout_rms']>1e-6 and row['query_switch_heldout_rms']>1e-6
        rows.append(row);arrays.update({f'f{family}_oracle':ey,f'f{family}_single':single_ev,f'f{family}_mixture':mix_ev,f'f{family}_query_switch':eswitch,f'f{family}_global_fit_to_switch':bad_ev,f'f{family}_bank0':e0,f'f{family}_bank1':e1})
    raw=Path(STORAGE['artifact_root'])/'cycle89';raw.mkdir(parents=True,exist_ok=True);path=raw/'oracle_belief_qualification89_v1.npz'
    with path.open('xb') as f:np.savez_compressed(f,**arrays,calibration_roots=cal[0],calibration_actions=cal[1],heldout_roots=ev[0],heldout_actions=ev[1])
    receipt=dict(utc=utc(),status='SINGLE_WORLD_MEMBERSHIP_NOT_VALID_FOR_POSTERIOR_MEAN',rows=rows,qualification_only=True,learned_models_executed=0,new_scientific_training_fits=0,proof_of_novelty=False,full_joint_distribution_coherence_certified=False,payload_path=str(path),payload_sha256=sha(path),simulator_script_sha256=sha(ROOT/'cycle88/scripts/plan_data88_v1.py'),protocol_sha256=sha(C/'RESEARCH_BRIEF89_V1.md'),elapsed_seconds=time.perf_counter()-started)
    write(O/'ORACLE_BELIEF_QUALIFICATION89_V1.json',receipt);log(dict(stage='oracle_belief_qualification_closed',rows=rows,elapsed_seconds=receipt['elapsed_seconds'],novel_direction_accepted=False))

if __name__=='__main__':main()
