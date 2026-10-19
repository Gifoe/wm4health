import runtime_profile
import json,math,time,numpy as np,torch
from scipy.special import log_ndtr
from belief_core import *
OUT=ROOT/'outputs/full_tail_patches'
def arr(path,names):
    with np.load(path) as f:return {k:torch.as_tensor(f[k],device=DEVICE) for k in names}
def tail_bound(nll,var,limit,extra=None):
    logtau=math.log(2)+float(log_ndtr(-limit));logz=math.log1p(-math.exp(logtau));lpmax=float(-.5*torch.log(2*math.pi*var).sum())
    if extra is not None:lpmax+=float(-.5*torch.log(2*math.pi*extra).sum())
    return torch.maximum(torch.full_like(nll,-logz),torch.logaddexp(torch.full_like(nll,logz),nll+logtau+lpmax))
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);start=time.perf_counter();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bounds=json.loads((ROOT/'outputs/all_context_tail_bounds.json').read_text());todo=[x for x in bounds['rows'] if x['flagged_contexts']]
    original=ROOT/'outputs/native_belief';r=json.loads((original/'results.json').read_text());fr=json.loads((ROOT/'outputs/ordinary_filter_controls/results.json').read_text());pr=json.loads((ROOT/'outputs/particle_moment_control/results.json').read_text())
    cells=[];files=[]
    for cr in r['checkpoints']:
        f=PREV/'outputs/native_context_weights'/cr['selected_path'];assert sha(f)==cr['checkpoint_sha256'];env,model,encoder=load(cr['config'],f)
        for row in [x for x in todo if (x['kind'],x['seed'])==(cr['kind'],cr['seed'])]:
            oc=next(c for c in r['cells'] if (c['kind'],c['checkpoint_seed'],c['context_size'],c['probe_u'])==(row['kind'],row['seed'],row['n'],row['probe_u']))
            b=arr(original/oc['base_file'],['actual_query','loc','scale']);ids=torch.tensor(row['flagged_contexts'],device=DEVICE,dtype=torch.long);arm=row['arm']
            if arm=='no_update':loc=b['loc'];scale=b['scale'];source=original/oc['raw_file'];prefix='no_update_'
            elif arm=='native':
                source=original/oc['raw_file'];a=arr(source,['real_post_loc','real_post_scale']);loc=a['real_post_loc'];scale=a['real_post_scale'];prefix='native_'
            elif arm=='reweighted':loc=b['loc'];scale=b['scale'];source=original/oc['raw_file'];prefix='reweighted_'
            elif arm=='map_ggn_laplace':
                fc=next(c for c in fr['cells'] if (c['kind'],c['seed'],c['n'],c['probe_u'])==(row['kind'],row['seed'],row['n'],row['probe_u']))
                source=ROOT/'outputs/ordinary_filter_controls'/fc['raw_file'];a=arr(source,['map_loc','map_scale']);loc=a['map_loc'];scale=a['map_scale'];prefix='map_ggn_laplace_'
            else:
                count=int(arm.removeprefix('particle'));pc=next(c for c in pr['cells'] if (c['kind'],c['seed'],c['n'],c['probe_u'],c['importance_rep'])==(row['kind'],row['seed'],row['n'],row['probe_u'],row['rep']))
                source=ROOT/'outputs/particle_moment_control'/pc['raw_file'];a=arr(source,[f'c{count}_latent_loc',f'c{count}_latent_variance']);loc=a[f'c{count}_latent_loc'];scale=torch.sqrt(a[f'c{count}_latent_variance']);prefix=f'c{count}_'
            loc=loc[ids];scale=scale[ids];y=b['actual_query'][ids];u=0. if row['kind']==1 else 1.5
            def calculate(k):
                z,w=normal_grid(k,14.);mu,var,beta=normal_components(model,loc,scale,u,z);wp=w
                extras={}
                if arm=='reweighted':
                    pm,pv,_=normal_components(model,loc,scale,row['probe_u'],z);probe=arr(original/oc['raw_file'],['real_probe'])['real_probe'][ids]
                    wp=posterior_weights(torch.log(w)[None],pm,pv,probe)
                    pd=-torch.logsumexp(torch.log(w)[None]+likelihood(pm,pv,probe),-1)
                    extras=dict(probe_mean=pm,probe_variance=pv,probe=probe,prior_weights=w,probe_nll=pd)
                sc=score(mu,var,wp,y)
                if arm=='reweighted':bb=tail_bound(extras['probe_nll'],extras['probe_variance'],14.)+tail_bound(sc['nll']+extras['probe_nll'],var,14.,extra=extras['probe_variance'])
                else:bb=tail_bound(sc['nll'],var,14.)
                return sc,mu,var,wp,bb,extras
            sc,mu,var,wp,bb,extras=calculate(16385);sr,_,_,_,bbr,_=calculate(32769)
            ref=float((sc['nll']-sr['nll']).abs().max());assert ref<.01,ref;assert float(bb.max())<.01,float(bb.max())
            old=arr(source,[prefix+k for k in ['nll','mean','second','pit','error']]);full={}
            for key in sc:
                value=old[prefix+key].clone();value[ids]=sc[key];full[key]=value
            name=f"patch_{len(files):03d}.npz";file=OUT/name
            save=dict(**full,flagged_indices=ids,component_mean=mu.float(),component_variance=var,component_weights=wp,flagged_query=y,refined_nll=sr['nll'],conservative_tail_bound=bb,**extras)
            np.savez_compressed(file,**{k:v.cpu().numpy() for k,v in save.items()})
            files.append(dict(path=name,sha256=sha(file)))
            cells.append(dict(**row,patch_file=name,source_file=str(source.relative_to(ROOT)).replace('\\','/'),source_prefix=prefix,actual_query=summary_scores(full),refinement_nll_max_error=ref,new_max_tail_bound=float(bb.max()),original_source_sha256=sha(source)))
            print(json.dumps(dict(patched=len(files),expected=len(todo),elapsed=time.perf_counter()-start)),flush=True)
        assert sha(f)==cr['checkpoint_sha256'];del model,encoder;torch.cuda.empty_cache()
    result=dict(plan_sha256=sha(ROOT/'FULL_TAIL_REPAIR.md'),script_sha256=sha(pathlib.Path(__file__)),tail_audit_sha256=sha(ROOT/'outputs/all_context_tail_bounds.json'),patches=cells,files=files,elapsed_seconds=time.perf_counter()-start,
        semantics='Sparse numerical readout repair only. Original arrays/results retained. No changes to learned models, posterior Normal parameters, proposal samples, actual observations or scientific arms.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(patches=len(files),elapsed=result['elapsed_seconds'])),flush=True)
if __name__=='__main__':
    with torch.inference_mode():main()
