import runtime_profile
import json,math,time,numpy as np,torch
from belief_core import *
OUT=ROOT/'outputs/particle_moment_control'
def normal_logp(b,loc,scale):return -.5*((b-loc[:,None])/scale[:,None])**2-torch.log(scale[:,None])-.5*math.log(2*math.pi)
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);start=time.perf_counter()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/native_belief';r=json.loads((base/'results.json').read_text());z,w=normal_grid();cells=[];files=[]
    for cr in r['checkpoints']:
        checkpoint=PREV/'outputs/native_context_weights'/cr['selected_path'];assert sha(checkpoint)==cr['checkpoint_sha256'];env,model,encoder=load(cr['config'],checkpoint)
        with np.load(ROOT/f"outputs/data/action_squash_{cr['kind']}.npz") as f:states=tensor(f['states']);actions=tensor(f['actions'])
        for c in [c for c in r['cells'] if (c['kind'],c['checkpoint_seed'])==(cr['kind'],cr['seed'])]:
            with np.load(base/c['base_file']) as f:loc=torch.tensor(f['loc'],device=DEVICE).double();scale=torch.tensor(f['scale'],device=DEVICE).double();y=torch.tensor(f['actual_query'],device=DEVICE)
            with np.load(base/c['raw_file']) as f:probe=torch.tensor(f['real_probe'],device=DEVICE).double()
            query_u=0. if c['kind']==1 else 1.5
            for rep in range(3):
                torch.cuda.synchronize();t=time.perf_counter()
                pm,ps=append_evidence(encoder,states,actions,c['context_size'],probe.float(),c['probe_u']);pm=pm[:,0].double();ps=ps[:,0].double()
                generator=torch.Generator(device=DEVICE).manual_seed(202625400+c['kind']*10000+cr['seed']*1000+c['context_size']*10+int(c['probe_u']*2)+rep*100000)
                eps=torch.randn(64,4096,device=DEVICE,generator=generator).double()
                beta=torch.cat((loc[:,None]+scale[:,None]*eps[:,:2048],pm[:,None]+ps[:,None]*eps[:,2048:]),dim=1)
                pmean,pvar=decode(model,beta,c['probe_u']);lp=normal_logp(beta,loc,scale);lq=torch.logaddexp(lp,normal_logp(beta,pm,ps))-math.log(2)
                logweights=lp+likelihood(pmean.double(),pvar,probe)-lq
                torch.cuda.synchronize();max_update_seconds=time.perf_counter()-t
                saved=dict(beta=beta,probe_component_mean=pmean,probe_variance=pvar,log_prior=lp,log_mixture_proposal=lq,unnormalized_logweights=logweights,proposal_loc=pm,proposal_scale=ps)
                summaries={}
                for count in [256,1024,4096]:
                    ids=torch.cat((torch.arange(count//2,device=DEVICE),torch.arange(2048,2048+count//2,device=DEVICE)))
                    bb=beta[:,ids];ww=torch.softmax(logweights[:,ids],-1);m=(ww*bb).sum(-1);var=(ww*(bb-m[:,None])**2).sum(-1).clamp_min(1e-16)
                    mu,noise,_=normal_components(model,m,torch.sqrt(var),query_u,z);sc=score(mu,noise,w,y)
                    zr,wr=normal_grid(16385);mur,nr,_=normal_components(model,m[:8],torch.sqrt(var[:8]),query_u,zr);sr=score(mur,nr,wr,y[:8])
                    zt,wt=normal_grid(16385,10.);mut,nt,_=normal_components(model,m[:8],torch.sqrt(var[:8]),query_u,zt);st=score(mut,nt,wt,y[:8])
                    ess=1/(ww*ww).sum(-1)
                    summaries[str(count)]=dict(**summary_scores(sc),ess_mean=float(ess.mean()),ess_min=float(ess.min()),
                        grid_refinement_nll_max_error=float((sc['nll'][:8]-sr['nll']).abs().max()),tail7_to10_nll_max_error=float((sr['nll']-st['nll']).abs().max()))
                    saved[f'c{count}_weights']=ww;saved[f'c{count}_latent_loc']=m;saved[f'c{count}_latent_variance']=var;saved[f'c{count}_component_mean_first8']=mu[:8].float();saved[f'c{count}_component_noise']=noise
                    for key,value in sc.items():saved[f'c{count}_{key}']=value
                file=OUT/f"k{c['kind']}_s{cr['seed']}_n{c['context_size']}_p{int(c['probe_u']*2)}_r{rep}.npz"
                np.savez_compressed(file,**{k:v.cpu().numpy() for k,v in saved.items()});files.append(dict(path=file.name,sha256=sha(file)))
                cells.append(dict(kind=c['kind'],seed=cr['seed'],n=c['context_size'],probe_u=c['probe_u'],informative=c['physically_informative'],importance_rep=rep,actual_query=summaries,max4096_update_seconds=max_update_seconds,raw_file=file.name,base_file=c['base_file'],original_raw_file=c['raw_file']))
            print(json.dumps(dict(conditions=len(cells)//3,expected=48,elapsed=time.perf_counter()-start)),flush=True)
        assert sha(checkpoint)==cr['checkpoint_sha256'];del model,encoder;torch.cuda.empty_cache()
    result=dict(posthoc=True,plan_sha256=sha(ROOT/'ADDITIONAL_ORDINARY_CONTROLS.md'),script_sha256=sha(pathlib.Path(__file__)),original_results_sha256=sha(base/'results.json'),cells=cells,files=files,elapsed_seconds=time.perf_counter()-start,
        mechanism='Known defensive prior/native-posterior mixture importance correction, followed by Gaussian moment projection (assumed density filtering).',new_training=False,accepted_novel_directions=0,
        timing_note='4096-particle update cost includes proposal encoder, sampling and probe decoder/weights; excludes common prior encoder and offline query density integration. Smaller counts are not separately timed.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(cells=len(cells),elapsed=result['elapsed_seconds'])),flush=True)
if __name__=='__main__':
    with torch.inference_mode():main()
