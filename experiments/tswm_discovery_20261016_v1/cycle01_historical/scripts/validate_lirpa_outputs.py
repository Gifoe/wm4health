"""Cross-check generic-verifier graph and its saved bounds against witnesses."""
import json,pathlib
import numpy as np
import torch
from probe_lirpa import RetrospectiveGraph
from confidence_tube import family,mean,simulate
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    records=[];max_forward_error=0.;rg=np.random.default_rng(501016)
    for p in sorted((ROOT/'outputs/lirpa').glob('rep*_q*_H*.json')):
        if p.name.endswith('.partial.json'):continue
        r=json.loads(p.read_text());rep,query,H=r['rep'],r['query'],r['horizon']
        z=np.load(ROOT/f'outputs/feasible_witnesses/rep{rep}.npz');w,P=z['w_hat'],z['P'];f=z['factual'][query:query+1,:H+1]
        lam,U=np.linalg.eigh(P);S=(U*np.sqrt(lam))@U.T;model=RetrospectiveGraph(w,P,f[0]).eval()
        # Nonzero parameter inputs validate the ellipsoid-to-graph mapping.
        for _ in range(5):
            g=rg.normal(size=(6,2));g*=r['confidence_radius']*.8/np.linalg.norm(g);theta=rg.uniform(-np.pi,np.pi)
            ww=w+S@g;residual=f[:,1:]-mean(ww,f[:,:-1],0.)
            target=simulate(ww,f[:,0],residual,1.,theta)[0,-1,0]-f[0,-1,0]
            with torch.no_grad():actual=float(model(torch.tensor(g.reshape(1,12)),torch.tensor([[theta]])).item())
            err=abs(actual-target);max_forward_error=max(max_forward_error,err);assert err<1e-10
        inner_lo=[];inner_hi=[]
        for side in [0,1]:
            ww=w+S@z['whitened_parameter_witnesses'][side,query]
            lo,hi=family(ww,f);inner_lo.append(float(lo[0]));inner_hi.append(float(hi[0]))
        for rr in r['results']:
            assert rr['lo']<=min(inner_lo)+1e-8 and rr['hi']>=max(inner_hi)-1e-8,(p.name,rr)
        records.append({'file':p.name,'witness_outer_check':'PASS','nonzero_parameter_forward_check':'PASS'})
    result={'max_forward_absolute_error':max_forward_error,'records':records,'n_completed_workers_checked':len(records),
            'interpretation':'Numerical cross-checks, not a formal proof including floating-point rounding.'}
    (ROOT/'outputs/lirpa_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
