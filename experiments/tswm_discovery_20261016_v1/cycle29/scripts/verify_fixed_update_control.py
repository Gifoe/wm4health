"""All control scores, validation-min checkpoints, shared schedules and neural replays."""
import runtime_profile
import csv,hashlib,json,pathlib,time
import numpy as np
import torch
from world_data import ROOT,OUT as DATA,C,sha,dump
from world_models_v2 import build
from train_world_comparison_v2 import setup,SEEDS,MAX_STEPS,BATCH,H as TRAIN_H,hashes
from train_fixed_update_control import OUT,KINDS,control_hashes
from verify_world_comparison_v2 import parse_name,rowkey,replay

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    setup();start=time.monotonic();directory=OUT/'evaluation';result=read(directory/'results.json');training=read(OUT/'training_summary.json')
    assert result['status']=='completed' and result['training_summary_sha256']==sha(OUT/'training_summary.json') and training['control_hashes']==control_hashes()
    schedules={};models=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed+1000);indices=rng.integers(0,1024,size=(MAX_STEPS,BATCH),dtype=np.int64);starts=rng.integers(0,512-C-TRAIN_H+1,size=(MAX_STEPS,BATCH),dtype=np.int64);digest=hashlib.sha256(indices.tobytes()+starts.tobytes()).hexdigest();schedules[seed]=digest
        for kind in KINDS:
            folder=OUT/'training'/f'{kind}_{seed}';r=read(folder/'result.json');origin=ROOT/'outputs/world_training_v2'/f'{kind}_{seed}';old=read(origin/'result.json');link=read(folder/'origin.json')
            assert r['steps']==MAX_STEPS==32768 and r['extra_updates']==MAX_STEPS-old['steps'] and r['batch_schedule_sha256']==old['batch_schedule_sha256']==digest and r['control_hashes']==control_hashes()
            assert link['source_result_sha256']==sha(origin/'result.json') and link['source_best_sha256']==sha(origin/'best.pt') and link['source_last_sha256']==sha(origin/'last.pt')
            assert r['history'][:len(old['history'])]==old['history'] and [x['step'] for x in r['history']]==list(range(512,MAX_STEPS+1,512))
            assert r['best_validation_mse']==min(x['validation_mse'] for x in r['history']) and r['best_weight_sha256']==sha(folder/'best.pt')
            state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);assert state['hashes']==hashes() and state['step']==r['best_step'] and state['validation_mse']==r['best_validation_mse']
            if 'control_hashes' in state:assert state['control_hashes']==control_hashes()
            else:assert sha(folder/'best.pt')==sha(origin/'best.pt')
            models.append(dict(kind=kind,seed=seed,steps=r['steps'],best_step=r['best_step'],best_validation_mse=r['best_validation_mse'],weight_sha256=r['best_weight_sha256'],extra_updates=r['extra_updates']))
    data={}
    manifest=read(DATA/'manifest.json')
    for row in manifest['files']:assert sha(DATA/row['path'])==row['sha256']
    for family in ['lti','wh']:
        with np.load(DATA/f'test_{family}.npz',allow_pickle=False) as f:data[family]={k:f[k] for k in ['y','u','clean']}
    with (directory/'per_system_metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    lookup={rowkey(r['family'],r['method'],int(r['seed']),int(r['system']),r['plan'],int(r['horizon'])):r for r in rows};assert len(lookup)==len(rows)==18432
    count=0;error=0.;saved={}
    for file in result['predictions']:
        path=directory/file['path'];assert sha(path)==file['sha256'];family,kind,seed=parse_name(path.name);assert kind in KINDS
        assert file['weight_sha256']==sha(OUT/'training'/f'{kind}_{seed}'/'best.pt')
        with np.load(path,allow_pickle=False) as f:pred=f['prediction'].astype('float64')
        assert pred.shape==(128,4,128) and np.isfinite(pred).all();saved[(family,kind,seed)]=pred
        clean=data[family]['clean'][:,:,C:].astype('float64');noisy=data[family]['y'][:,:,C:].astype('float64')
        for i in range(128):
            for j,plan in enumerate(['factual','zero','step','pulse']):
                for h in [32,64,128]:
                    r=lookup[rowkey(family,kind,seed,i,plan,h)];p=pred[i,j,:h];truth=clean[i,j,:h];pe=p-pred[i,0,:h];te=truth-clean[i,0,:h];den=np.dot(te,te)
                    values=dict(clean_rmse=np.linalg.norm(p-truth)/np.sqrt(h),noisy_rmse=np.linalg.norm(p-noisy[i,j,:h])/np.sqrt(h),effect_rmse=np.linalg.norm(pe-te)/np.sqrt(h))
                    if den>1e-8:values['effect_slope']=np.dot(pe,te)/den
                    else:assert r['effect_slope']==''
                    assert r['informative_effect']==str(bool(den>1e-8)) and r['finite']=='True'
                    for name,value in values.items():
                        difference=abs(float(r[name])-value)/(1+abs(value));assert difference<2e-5,(family,kind,seed,i,plan,h,name,difference);error=max(error,difference)
                    count+=1
    assert count==result['n_metric_rows']==18432
    replays=[]
    for modelrow in models:
        kind=modelrow['kind'];seed=modelrow['seed'];state=torch.load(OUT/'training'/f'{kind}_{seed}'/'best.pt',map_location='cpu',weights_only=True);model=build(kind).cuda().eval();model.load_state_dict(state['model'],strict=True)
        for family in ['lti','wh']:
            ids=[0,31,95];actual=replay(model,data[family],kind,ids);expected=saved[(family,kind,seed)][ids];err=float(np.max(np.abs(actual-expected)/(1+np.abs(expected))));assert err<2e-4,(family,kind,seed,err);replays.append(dict(family=family,kind=kind,seed=seed,systems=ids,relative_max_error=err))
        del model;torch.cuda.empty_cache()
    proof=dict(status='passed',results_sha256=sha(directory/'results.json'),training_summary_sha256=sha(OUT/'training_summary.json'),source_sha256=sha(pathlib.Path(__file__)),metric_rows_rescored=count,relative_metric_error_max=error,models=models,schedules=schedules,replays=replays,neural_relative_error_max=max(x['relative_max_error'] for x in replays),seconds=time.monotonic()-start,scope='fixed update control independent score/checkpoint/schedule/replay verification;not novelty proof')
    dump(OUT/'verification.json',proof);print(json.dumps({k:v for k,v in proof.items() if k not in ['models','replays','schedules']}),flush=True)

if __name__=='__main__':main()
