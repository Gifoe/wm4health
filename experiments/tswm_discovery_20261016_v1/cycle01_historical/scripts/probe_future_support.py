"""Structural reachability probes of existing, frozen, officially trained models.

These are training-context sensitivity diagnostics, not causal-effect estimates
on electricity data and not additional forecast-accuracy reproductions.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import hashlib,json,pathlib,sys,time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
(ROOT/'outputs').mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'upstream/WorldTS/DAG-bench'))
from ts_benchmark.baselines.worldts.models.worldts_model import WorldTSModel
torch.set_num_threads(4);torch.manual_seed(20261016)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def weights(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
results=[];provenance=[]
base=pathlib.Path('D:/baseline_reproduction_seed2021_v1')
for ds in ['NP','PJM']:
  for horizon in [24,360]:
    d=base/'results/worldts'/f'{ds}_H{horizon}_seed2021'
    cfg=json.loads((d/'resolved_config.json').read_text(encoding='utf-8'))
    ck=d/'stages/stage2_decoder_mix.pt'
    m=WorldTSModel(SimpleNamespace(**cfg)).cuda().eval()
    m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['Model'],strict=True)
    for p in m.parameters():p.requires_grad_(False)
    before=weights(m)
    frame=pd.read_csv(base/'data'/f'{ds}.csv')
    columns=frame['cols'].drop_duplicates().tolist();assert columns[-1]=='OT'
    frame['date']=pd.to_datetime(frame['date'])
    wide=frame.pivot(index='date',columns='cols',values='data')[columns]
    values=wide[['OT']+columns[:-1]].values
    assert values.shape[1]==cfg['input_dim']
    # All selected origins are in the first 2,000 training points. No test labels.
    train=values[:int(int(len(values)*.8)*.875)]
    mu=train.mean(0);sd=train.std(0).clip(1e-6)
    values=(values-mu)/sd
    L=cfg['seq_len'];origins=L+np.arange(8)*64
    hist=torch.tensor(np.stack([values[t-L:t] for t in origins]),dtype=torch.float32,device='cuda')
    future=torch.tensor(np.stack([values[t:t+horizon,1:] for t in origins]),dtype=torch.float32,device='cuda')
    with torch.inference_mode():basepred=m.forecast(hist,x_future=future).cpu().numpy()
    support=cfg['state_patch_len']+cfg['decoder_patch_len']-1
    arrays={}
    for channel in range(cfg['future_x_dim']):
      for pos in [0,horizon//2,horizon-1]:
        u=future.clone();u[:,pos,channel]+=.5
        with torch.inference_mode():delta=m.forecast(hist,x_future=u).cpu().numpy()-basepred
        allowed=np.zeros(horizon,bool);allowed[pos:min(horizon,pos+support)]=True
        rec={'dataset':ds,'horizon':horizon,'channel':channel,'source_position':pos,'theoretical_support_steps':support,
             'max_in_support_change':float(np.max(np.abs(delta[:,allowed]))),
             'max_outside_support_change':float(np.max(np.abs(delta[:,~allowed]))) if (~allowed).any() else 0,
             'observed_affected_positions':np.flatnonzero(np.abs(delta).max(axis=(0,2))>1e-7).tolist()}
        results.append(rec);arrays[f'channel{channel}_pos{pos}']=delta
    # Autodiff verifies exact structural zeros independently of finite differences.
    x=future[:1].clone().requires_grad_(True)
    far=support+2
    grad=torch.autograd.grad(m.forecast(hist[:1],x_future=x)[0,far,0],x)[0]
    assert float(grad[0,0].abs().max())==0
    assert weights(m)==before
    provenance.append({'dataset':ds,'horizon':horizon,'checkpoint_sha256':sha(ck),
        'resolved_config_sha256':sha(d/'resolved_config.json'),'resolved_config':cfg,
        'data_sha256':sha(base/'data'/f'{ds}.csv'),'model_state_sha256_before':before,
        'model_state_sha256_after':weights(m),'autodiff_distant_gradient_max':float(grad[0,0].abs().max()),
        'training_origins':origins.tolist(),'train_boundary':len(train),
        'channel_order':['OT']+columns[:-1],'data_layout':'long CSV pivot by named column',
        'normalization':'mean/std on exact 70% training partition'})
    np.savez_compressed(ROOT/'outputs'/f'existing_response_{ds}_{horizon}.npz',**arrays)
    print(ds,horizon,'support',support,'outside',max(r['max_outside_support_change'] for r in results if r['dataset']==ds and r['horizon']==horizon),flush=True)
    del m;torch.cuda.empty_cache()
(ROOT/'outputs/existing_future_support.json').write_text(json.dumps({'scope':'8 training contexts per frozen model; all covariates retain observational semantics','source_revision':'269426ea6217d7152a244da0e209d922acd46d4f','records':results,'provenance':provenance},indent=2),encoding='utf-8')
