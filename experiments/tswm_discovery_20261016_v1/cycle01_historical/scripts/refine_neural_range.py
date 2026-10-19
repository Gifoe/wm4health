"""Fresh-cohort falsification: angle discretization versus fitted-kernel error.

No new training or selection. Reuse the three frozen kernels. A dense grid is
still an inner sensitivity range, not an all-SCM confidence certificate.
"""
import hashlib,json,pathlib,time
import numpy as np
import torch
from probe_neural_coupling import GaussianSSM,factual,truth,roll
ROOT=pathlib.Path(__file__).resolve().parents[1]

def ranges(fn,f,eps,grid):
    # Batch all angles to avoid hundreds of small GPU launches.
    N=len(f);K=len(grid);x=f[:,0].repeat(K,1)
    e=eps.repeat(K,1,1);ang=torch.tensor(grid,device='cuda').repeat_interleave(N)[:,None]
    a=torch.ones((K*N,1),device='cuda')
    for t in range(e.shape[1]):
        c=torch.cos(ang);s=torch.sin(ang);ee=e[:,t]
        rot=torch.cat([c*ee[:,:1]-s*ee[:,1:],s*ee[:,:1]+c*ee[:,1:]],1)
        x=fn(x,a)+.1*rot
    delta=x[:,0].reshape(K,N)-f[:,-1,0]
    return delta.min(0).values,delta.max(0).values

def main():
    p=ROOT/'outputs/neural_range_refinement_plan.json';plan=json.loads(p.read_text())
    records=[];start=time.perf_counter()
    f,e=factual(plan['fresh_episodes'],plan['horizon'],plan['fresh_seed'])
    coarse=json.loads((ROOT/'outputs/neural_coupling_plan.json').read_text())['allowed_global_angle_grid']
    # Include the declared true angle as a diagnostic only, NEVER as a deployed
    # bound's privileged input. This separates grid error from kernel error.
    dense=np.linspace(-np.pi,np.pi,plan['dense_angles'],endpoint=False).tolist()
    augmented=sorted(set(dense+[plan['true_angle']]))
    with torch.inference_mode():
      target=roll(truth,f[:,0],e,plan['true_angle'])-f[:,-1,0]
      oracle_lo,oracle_hi=ranges(truth,f,e,augmented)
      assert bool(((target>=oracle_lo-1e-6)&(target<=oracle_hi+1e-6)).all())
      for seed in plan['frozen_training_seeds']:
        ck=ROOT/f'outputs/neural_coupling/seed{seed}/best.pt'
        model=GaussianSSM().cuda();model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state']);model.eval()
        x=f[:,:-1].reshape(-1,2);a=torch.zeros((len(x),1),device='cuda')
        abduct=((f[:,1:].reshape(-1,2)-model(x,a))/.1).reshape(len(f),-1,2)
        for name,grid in [('coarse',coarse),('dense',dense),('dense_plus_true_angle_ORACLE_DIAGNOSTIC',augmented)]:
            lo,hi=ranges(model,f,abduct,grid)
            row={'seed':seed,'method':name,'n_angles':len(grid),'n_episodes':len(f),
                 'true_cf_coverage':float(((target>=lo)&(target<=hi)).float().mean()),
                 'contains_oracle_family_range':float(((lo<=oracle_lo)&(hi>=oracle_hi)).float().mean()),
                 'mean_width':float((hi-lo).mean()),
                 'mean_endpoint_error_vs_oracle':float(((lo-oracle_lo).abs()+(hi-oracle_hi).abs()).mean()/2),
                 'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}
            records.append(row);print(json.dumps(row),flush=True)
    result={'plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'records':records,
            'oracle_true_cf_coverage':1.,'oracle_family_mean_width':float((oracle_hi-oracle_lo).mean()),
            'warning':'All grids give only sensitivity inner ranges. Including true angle is oracle falsification, not an implementable method.',
            'runtime_seconds':time.perf_counter()-start}
    (ROOT/'outputs/neural_range_refinement.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
