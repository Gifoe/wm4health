import runtime_profile
import json,time,math,pathlib
import numpy as np
import torch
from belief_core import *
OUT=ROOT/'outputs/ordinary_filter_controls'
def grad_map(model,beta,u):
    beta=beta.detach().float().requires_grad_(True)
    y=model(torch.zeros(len(beta),2,device=DEVICE),torch.full((len(beta),1),float(u),device=DEVICE),beta[:,None])
    h=y.mean;J=[]
    for j in range(2):J.append(torch.autograd.grad(h[:,j].sum(),beta,retain_graph=j==0)[0])
    return h.detach().double(),torch.stack(J,-1).detach().double(),y.variance[0].detach().double()
def objective(model,beta,u,loc,scale,y):
    with torch.no_grad():
        mu,var=decode(model,beta,u)
        return .5*((beta-loc)/scale)**2+.5*(((y-mu.double())**2)/var).sum(-1)
def aggregate(sc):return summary_scores(sc)
def main():
    start=time.perf_counter();assert not OUT.exists();OUT.mkdir(parents=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original=ROOT/'outputs/native_belief';r=json.loads((original/'results.json').read_text());z,w=normal_grid();cells=[];files=[]
    for record in r['checkpoints']:
        f=PREV/'outputs/native_context_weights'/record['selected_path'];assert sha(f)==record['checkpoint_sha256'];env,model,encoder=load(record['config'],f)
        for p in model.parameters():p.requires_grad_(False)
        for c in [x for x in r['cells'] if x['kind']==record['kind'] and x['checkpoint_seed']==record['seed']]:
            with np.load(original/c['base_file']) as b:base={k:torch.tensor(b[k],device=DEVICE) for k in b.files}
            with np.load(original/c['raw_file']) as q:raw={k:torch.tensor(q[k],device=DEVICE) for k in ['real_probe','real_post_loc','probe_grid_mean','probe_variance','real_reweighted_weights']}
            loc=base['loc'].double();scale=base['scale'].double();y=raw['real_probe'];probe_u=c['probe_u'];query_u=0. if c['kind']==1 else 1.5
            torch.cuda.synchronize();t=time.perf_counter();h,J,R=grad_map(model,loc,probe_u)
            v=1/(1/scale**2+(J*J/R).sum(-1));m=loc+v*(J*(y-h)/R).sum(-1)
            torch.cuda.synchronize();ekf_seconds=time.perf_counter()-t
            torch.cuda.synchronize();t=time.perf_counter();beta=raw['real_post_loc'].double().clone();stalls=torch.zeros(64,device=DEVICE,dtype=torch.long)
            initial_cost=objective(model,beta,probe_u,loc,scale,y)
            for iteration in range(32):
                h,J,R=grad_map(model,beta,probe_u);g=(beta-loc)/scale**2+(J*(h-y)/R).sum(-1)
                precision=1/scale**2+(J*J/R).sum(-1);step=(g/precision).clamp(-3*scale,3*scale)
                old_cost=objective(model,beta,probe_u,loc,scale,y);accepted=torch.zeros(64,device=DEVICE,dtype=torch.bool);nextbeta=beta.clone()
                for half in range(8):
                    proposal=beta-step*(.5**half);newcost=objective(model,proposal,probe_u,loc,scale,y)
                    take=(~accepted)&(newcost<=old_cost+1e-9);nextbeta[take]=proposal[take];accepted|=take
                stalls+=(~accepted).long();beta=nextbeta
            h,J,R=grad_map(model,beta,probe_u);postvar=1/(1/scale**2+(J*J/R).sum(-1))
            final_cost=objective(model,beta,probe_u,loc,scale,y)
            gradient=(beta-loc)/scale**2+(J*(h-y)/R).sum(-1)
            torch.cuda.synchronize();map_seconds=time.perf_counter()-t
            grid_cost=.5*((base['grid_beta']-loc[:,None])/scale[:,None])**2+.5*((y[:,None,:]-raw['probe_grid_mean'].double())**2/raw['probe_variance']).sum(-1)
            gap=(final_cost-grid_cost.min(-1).values).clamp_min(0)
            results={};saved=dict(ekf_loc=m,ekf_scale=torch.sqrt(v),map_loc=beta,map_scale=torch.sqrt(postvar),map_initial_cost=initial_cost,map_final_cost=final_cost,map_grid_cost_gap=gap,map_normalized_gradient=gradient*torch.sqrt(postvar),map_stalls=stalls)
            with torch.no_grad():
                for name,mean,var in [('ekf',m,v),('map_ggn_laplace',beta,postvar)]:
                    mu,pv,_=normal_components(model,mean,torch.sqrt(var),query_u,z);sc=score(mu,pv,w,base['actual_query'])
                    zr,wr=normal_grid(16385);mur,pvr,_=normal_components(model,mean[:8],torch.sqrt(var[:8]),query_u,zr);scr=score(mur,pvr,wr,base['actual_query'][:8])
                    zt,wt=normal_grid(16385,10.);mut,pvt,_=normal_components(model,mean[:8],torch.sqrt(var[:8]),query_u,zt);sct=score(mut,pvt,wt,base['actual_query'][:8])
                    results[name]=dict(**aggregate(sc),grid_refinement_nll_max_error=float((sc['nll'][:8]-scr['nll']).abs().max()),tail7_to10_nll_max_error=float((sct['nll']-scr['nll']).abs().max()))
                    saved[name+'_grid_mean']=mu.float();saved[name+'_variance']=pv
                    for key,value in sc.items():saved[name+'_'+key]=value
            path=OUT/c['raw_file'];np.savez_compressed(path,**{k:v.detach().cpu().numpy() for k,v in saved.items()});files.append(dict(path=path.name,sha256=sha(path)))
            cell=dict(kind=c['kind'],seed=c['checkpoint_seed'],n=c['context_size'],probe_u=probe_u,informative=c['physically_informative'],actual_query=results,
                ekf_update_seconds=ekf_seconds,map_update_seconds=map_seconds,map_grid_gap_max=float(gap.max()),map_grid_gap_mean=float(gap.mean()),map_normalized_gradient_max=float((gradient*torch.sqrt(postvar)).abs().max()),map_cost_increase_max=float((final_cost-initial_cost).max()),raw_file=path.name)
            cells.append(cell);print(json.dumps(dict(cells=len(cells),expected=48,elapsed=time.perf_counter()-start)),flush=True)
        assert sha(f)==record['checkpoint_sha256'];del model,encoder;torch.cuda.empty_cache()
    result=dict(posthoc=True,original_results_sha256=sha(original/'results.json'),plan_sha256=sha(ROOT/'POSTHOC_FILTER_CONTROL_PLAN.md'),script_sha256=sha(pathlib.Path(__file__)),core_sha256=sha(ROOT/'scripts/belief_core.py'),cells=cells,files=files,elapsed_seconds=time.perf_counter()-start,source_revision=verify_source(),new_training=False,accepted_novel_directions=0,
        inference_cost_note='Update times include likelihood/Jacobian/optimization; exclude original prior encoder and offline predictive density integration. No architectural compute superiority claim.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(cells=48,elapsed=result['elapsed_seconds'],max_map_grid_gap=max(x['map_grid_gap_max'] for x in cells))),flush=True)
if __name__=='__main__':main()
