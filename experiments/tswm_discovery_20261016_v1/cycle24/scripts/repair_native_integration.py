import runtime_profile
import copy,json,time,pathlib
import numpy as np
import torch
from belief_core import *
OUT=ROOT/'outputs/integration_repaired'
def moments_at(model,loc,scale,u,k):
    z,w=normal_grid(k);mu,var,_=normal_components(model,loc,scale,u,z);return moments(mu,var,w)
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);start=time.perf_counter()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    oldpath=ROOT/'outputs/native_belief/results.json';old=json.loads(oldpath.read_text());new=copy.deepcopy(old)
    wm=json.loads((PREV/'outputs/native_context_weights/manifest.json').read_text());files=[];diagnostics=[]
    for record in old['checkpoints']:
        original=next(r for r in wm['models'] if r['path']==record['selected_path'])
        checkpoint=PREV/'outputs/native_context_weights'/record['selected_path'];assert sha(checkpoint)==record['checkpoint_sha256']
        env,model,encoder=load(record['config'],checkpoint);kind=record['kind'];seed=record['seed'];u=0. if kind==1 else 1.5
        for cell in [c for c in new['cells'] if c['kind']==kind and c['checkpoint_seed']==seed]:
            with np.load(ROOT/'outputs/native_belief'/cell['base_file']) as f:b={k:torch.as_tensor(f[k],device=DEVICE) for k in f.files}
            with np.load(ROOT/'outputs/native_belief'/cell['raw_file']) as f:r={k:torch.as_tensor(f[k],device=DEVICE) for k in ['post_loc','post_scale','real_post_loc','real_post_scale','no_update_mean','no_update_second']}
            a,s=moments_at(model,r['post_loc'],r['post_scale'],u,513)
            ar,sr=moments_at(model,r['post_loc'][:8],r['post_scale'][:8],u,1025)
            prior_sd=torch.sqrt((b['prior_second']-b['prior_first']**2).clamp_min(1e-15))
            maxdiff=float(((a[:8]-ar).abs()/prior_sd[:8,None,:]).max())
            p1,p2=moments_at(model,b['loc'],b['scale'],u,513)
            prior_diff=float(((p1-b['prior_first']).abs()/prior_sd).max())
            real1,real2=moments_at(model,r['real_post_loc'],r['real_post_scale'],u,513)
            # Obtain exact original dense-grid real posterior moments directly.
            with np.load(ROOT/'outputs/native_belief'/cell['raw_file']) as f:
                dense_real1=torch.tensor(f['native_mean'],device=DEVICE);dense_real2=torch.tensor(f['native_second'],device=DEVICE)
            real_diff=float(((real1-dense_real1).abs()/prior_sd).max())
            check=dict(kind=kind,seed=seed,n=cell['context_size'],probe_u=cell['probe_u'],native_uniform513_to1025_max_first_in_prior_sd=maxdiff,
                native_uniform513_to1025_max_second=float((s[:8]-sr).abs().max()),native_uniform513_vs8193_prior_max_first_in_prior_sd=prior_diff,
                native_uniform513_vs8193_real_post_max_first_in_prior_sd=real_diff)
            file=OUT/cell['raw_file'];np.savez_compressed(file,native_post_first=a.cpu().numpy(),native_post_second=s.cpu().numpy(),
                refined_native_post_first=ar.cpu().numpy(),refined_native_post_second=sr.cpu().numpy(),prior_uniform_first=p1.cpu().numpy(),prior_uniform_second=p2.cpu().numpy(),
                real_uniform_first=real1.cpu().numpy(),real_uniform_second=real2.cpu().numpy())
            files.append(dict(path=file.name,sha256=sha(file)));diagnostics.append(check)
            cell['native_coherence']=summarize_drift(b['prior_first'],b['prior_second'],a,s)
            cell['numerics'].update({k:v for k,v in check.items() if k.startswith('native_uniform')})
            print(json.dumps(dict(repaired=len(files),expected=48,elapsed=time.perf_counter()-start)),flush=True)
        assert sha(checkpoint)==record['checkpoint_sha256'];del model,encoder;torch.cuda.empty_cache()
    new['original_results_sha256']=sha(oldpath);new['integration_repair_protocol_sha256']=sha(ROOT/'INTEGRATION_REPAIR.md')
    new['integration_repair_script_sha256']=sha(pathlib.Path(__file__));new['integration_repair_elapsed_seconds']=time.perf_counter()-start
    new['integration_repair_files']=files;new['integration_repair_diagnostics']=diagnostics
    new['new_integrator']='513-point normalized trapezoid Normal grid on[-7,7]; fixed first8 refinement1025; same frozen posterior Normal parameters.'
    new['real_scores_unchanged']=True;new['original_files_immutable']=True
    (OUT/'results.json').write_text(json.dumps(new,indent=2),encoding='utf-8')
    maximum=max(max(r[k] for k in ['native_uniform513_to1025_max_first_in_prior_sd','native_uniform513_vs8193_prior_max_first_in_prior_sd','native_uniform513_vs8193_real_post_max_first_in_prior_sd']) for r in diagnostics)
    print(json.dumps(dict(status='finished',max_normalized_first_error=maximum,predeclared_gate=.01,gate_pass=maximum<.01,elapsed=time.perf_counter()-start)),flush=True)
if __name__=='__main__':
    with torch.inference_mode():main()
