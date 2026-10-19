"""Continue both primary architectures to the same original update cap; validation only."""
import runtime_profile
import hashlib,json,pathlib,random,shutil,time
import numpy as np
import torch
from world_data import ROOT,OUT as DATA,sha,dump
from world_models_v2 import build
from train_world_comparison_v2 import setup,hashes,read_data,batch,validate,SEEDS,MAX_STEPS,BATCH,C,H
OUT=ROOT/'outputs/fixed_update_control_v1';KINDS=['compiled64','conditional128']

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def control_hashes():
    return dict(plan=sha(ROOT/'FIXED_UPDATE_CONTROL_PLAN.md'),source=sha(pathlib.Path(__file__)),original_sources=hashes(),data_manifest=sha(DATA/'manifest.json'))

def one(kind,seed,train,val):
    source=ROOT/'outputs/world_training_v2'/f'{kind}_{seed}';folder=OUT/'training'/f'{kind}_{seed}';folder.mkdir(parents=True,exist_ok=True);ch=control_hashes()
    original=read(source/'result.json');assert original['hashes']==hashes() and original['best_weight_sha256']==sha(source/'best.pt')
    if (folder/'result.json').exists():
        result=read(folder/'result.json');assert result['control_hashes']==ch;return result
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=build(kind).cuda();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    rng=np.random.default_rng(seed+1000);indices=rng.integers(0,len(train['y']),size=(MAX_STEPS,BATCH),dtype=np.int64);starts=rng.integers(0,512-C-H+1,size=(MAX_STEPS,BATCH),dtype=np.int64)
    schedule=hashlib.sha256(indices.tobytes()+starts.tobytes()).hexdigest();assert schedule==original['batch_schedule_sha256']
    indices=torch.tensor(indices,device='cuda');starts=torch.tensor(starts,device='cuda');extra_before=0.
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cuda',weights_only=True);assert state['control_hashes']==ch;extra_before=state['extra_elapsed_s']
    else:
        state=torch.load(source/'last.pt',map_location='cuda',weights_only=True);assert state['hashes']==hashes() and state['step']==original['steps']
        shutil.copy2(source/'best.pt',folder/'best.pt')
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    begin=int(state['step']);history=list(state['history']);best=float(state['best']);best_step=int(state['best_step']);assert 0<begin<=MAX_STEPS
    dump(folder/'origin.json',dict(kind=kind,seed=seed,source_result_sha256=sha(source/'result.json'),source_last_sha256=sha(source/'last.pt'),source_best_sha256=sha(source/'best.pt'),source_steps=original['steps'],batch_schedule_sha256=schedule,control_hashes=ch))
    start=time.monotonic();losses=[];model.train();torch.cuda.reset_peak_memory_stats()
    with (folder/'continuation.jsonl').open('a',encoding='utf-8') as stream:
        for k in range(begin,MAX_STEPS):
            cy,cu,fu,target=batch(train,indices[k],starts[k]);opt.zero_grad(set_to_none=True);pred=model(cy,cu,fu);loss=(pred-target).square().mean();assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
            if (k+1)%512==0:
                vl=validate(model,val);torch.cuda.synchronize();elapsed=time.monotonic()-start+extra_before
                row=dict(step=k+1,training_mse=float(np.mean(losses)),validation_mse=vl,elapsed_s=original['elapsed_s']+elapsed,grad_norm=float(norm));history.append(row);losses=[]
                if vl<best:
                    best=vl;best_step=k+1
                    torch.save(dict(model=model.state_dict(),kind=kind,seed=seed,step=k+1,validation_mse=vl,hashes=hashes(),control_hashes=ch),folder/'best.pt')
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=k+1,best=best,best_step=best_step,history=history,hashes=hashes(),control_hashes=ch,extra_elapsed_s=elapsed),folder/'last.pt')
                stream.write(json.dumps(row)+'\n');stream.flush();dump(folder/'status.json',dict(status='training',kind=kind,seed=seed,best_step=best_step,best_validation_mse=best,**row));print(json.dumps(dict(kind=kind,seed=seed,**row)),flush=True)
    torch.cuda.synchronize();elapsed=time.monotonic()-start+extra_before
    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);assert selected['validation_mse']==best==min(x['validation_mse'] for x in history)
    result=dict(status='completed',kind=kind,seed=seed,steps=MAX_STEPS,original_steps=original['steps'],extra_updates=MAX_STEPS-original['steps'],best_step=best_step,best_validation_mse=best,best_weight_sha256=sha(folder/'best.pt'),batch_schedule_sha256=schedule,original_elapsed_s=original['elapsed_s'],extra_elapsed_s=elapsed,history=history,control_hashes=ch,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),scope='fixed update cap;not matched training FLOPs or wall time')
    dump(folder/'result.json',result);dump(folder/'status.json',{k:v for k,v in result.items() if k!='history'});print(json.dumps({k:v for k,v in result.items() if k not in ['history','control_hashes']}),flush=True)
    del state,opt,model,selected;torch.cuda.empty_cache();return result

def main():
    setup();OUT.mkdir(parents=True,exist_ok=True);original=read(ROOT/'outputs/world_training_v2/summary.json');assert original['status']=='all_training_complete' and original['hashes']==hashes()
    proof=read(ROOT/'outputs/world_verification/model_comparison_v2.json');assert proof['status']=='passed' and proof['training_summary_sha256']==sha(ROOT/'outputs/world_training_v2/summary.json')
    for row in read(DATA/'manifest.json')['files']:assert sha(DATA/row['path'])==row['sha256']
    train=read_data('train');val=read_data('val');rows=[]
    for seed in SEEDS:
        for kind in KINDS:rows.append(one(kind,seed,train,val))
    summary=dict(status='all_training_complete',runs=6,rows=[{k:v for k,v in r.items() if k!='history'} for r in rows],control_hashes=control_hashes(),test_not_used_for_selection=True,accepted_final_directions=0)
    dump(OUT/'training_summary.json',summary);print(json.dumps(dict(status=summary['status'],runs=6)),flush=True)

if __name__=='__main__':main()
