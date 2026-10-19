import runtime_profile
import json,time,numpy as np,torch
from belief_core import *
OUT=ROOT/'outputs/ekf_density_repaired'
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);start=time.perf_counter()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=ROOT/'outputs/ordinary_filter_controls';r=json.loads((old/'results.json').read_text());base=ROOT/'outputs/native_belief';original=json.loads((base/'results.json').read_text())
    cells=[];files=[]
    for cr in original['checkpoints']:
        path=PREV/'outputs/native_context_weights'/cr['selected_path'];assert sha(path)==cr['checkpoint_sha256'];env,model,enc=load(cr['config'],path)
        for c in [c for c in r['cells'] if (c['kind'],c['seed'])==(cr['kind'],cr['seed'])]:
            with np.load(old/c['raw_file']) as f:loc=torch.tensor(f['ekf_loc'],device=DEVICE);scale=torch.tensor(f['ekf_scale'],device=DEVICE)
            oc=next(x for x in original['cells'] if (x['kind'],x['checkpoint_seed'],x['context_size'],x['probe_u'])==(c['kind'],c['seed'],c['n'],c['probe_u']))
            with np.load(base/oc['base_file']) as f:y=torch.tensor(f['actual_query'],device=DEVICE)
            u=0. if c['kind']==1 else 1.5;z,w=normal_grid(32769,37.);mu,var,_=normal_components(model,loc,scale,u,z);sc=score(mu,var,w,y)
            zr,wr=normal_grid(65537,37.);mur,vr,_=normal_components(model,loc[:8],scale[:8],u,zr);ref=score(mur,vr,wr,y[:8])
            zt,wt=normal_grid(65537,35.);mut,vt,_=normal_components(model,loc[:8],scale[:8],u,zt);tail=score(mut,vt,wt,y[:8])
            row=dict(kind=c['kind'],seed=c['seed'],n=c['n'],probe_u=c['probe_u'],informative=c['informative'],actual_query=summary_scores(sc),
                refined_nll_max_error=float((sc['nll'][:8]-ref['nll']).abs().max()),tail35_to37_nll_max_error=float((tail['nll']-ref['nll']).abs().max()),raw_file=c['raw_file'])
            f=OUT/c['raw_file'];np.savez_compressed(f,**{k:v.cpu().numpy() for k,v in sc.items()},ekf_loc=loc.cpu().numpy(),ekf_scale=scale.cpu().numpy(),refined_nll=ref['nll'].cpu().numpy(),tail35_nll=tail['nll'].cpu().numpy(),component_mean_first8=mu[:8].float().cpu().numpy(),component_variance=var.cpu().numpy(),grid_weights=w.cpu().numpy(),actual_query=y.cpu().numpy())
            files.append(dict(path=f.name,sha256=sha(f)));cells.append(row);print(json.dumps(dict(cells=len(cells),expected=48,elapsed=time.perf_counter()-start)),flush=True)
        del model,enc;torch.cuda.empty_cache()
    result=dict(original_results_sha256=sha(old/'results.json'),plan_sha256=sha(ROOT/'ADDITIONAL_ORDINARY_CONTROLS.md'),script_sha256=sha(pathlib.Path(__file__)),cells=cells,files=files,posterior_parameters_unchanged=True,elapsed_seconds=time.perf_counter()-start)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(cells=48,max_refinement=max(c['refined_nll_max_error'] for c in cells),max_tail=max(c['tail35_to37_nll_max_error'] for c in cells))),flush=True)
if __name__=='__main__':
    with torch.inference_mode():main()
