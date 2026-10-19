"""Task-local source transfer for a TRAIN/VAL mechanism pilot, not paper replication."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]; BASE=ROOT.parent
sys.path.insert(0,str(BASE/'cycle29/scripts'))
import runtime_profile
import os, json, hashlib, types, importlib, copy, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from types import SimpleNamespace
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
T,H,D,B,SEED=168,96,32,32,2026351
ARMS=['N','JZ','JZ_N','PZ','PY','PZ_NC']
SRC=BASE/'upstream/WorldTS/DAG-bench'
CONFIG=dict(seq_len=T,pred_len=H,horizon=H,output_dim=1,latent_dim=D,
    state_patch_len=3,state_hidden_multiplier=2,decoder_type='causal_patch',
    decoder_patch_len=3,decoder_hidden_multiplier=2,future_x_dim=0,condition_dim=0,
    use_revin=True,revin_eps=1e-5,use_future_x_condition=False,use_time_condition=False,
    use_modal_condition=False,predictor_patch_len=48,predictor_patch_stride=24,
    predictor_d_model=64,predictor_n_heads=4,predictor_e_layers=1,predictor_d_ff=256,
    predictor_dropout=0.,latent_regularizer_type='legacy_vc')
DATA={
 'solar':('solar_with_weather/15T/train-00000-of-00001.parquet',
          ['target','global_horizontal_irradiance','temp','pressure','humidity','wind_speed','rain_1h','snow_1h','clouds_all','day_length'],
          'b44068eb97346134134015763d428ef6fe0b7ec91c38bdc49f987e15aa06ecbb'),
 'jena':('jena_weather/10T/train-00000-of-00001.parquet',['target_'+str(i) for i in range(21)],
         '22ce21756dfae541d740b0756330ab5181befcd797274aa5b27a0c3747cb5b0c')}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while a:=f.read(1048576):h.update(a)
    return h.hexdigest()

def state_sha(state):
    h=hashlib.sha256()
    for k,t in sorted(state.items()):
        h.update(k.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def dump(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def seed(value=SEED):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)

def native():
    for part in ['', '.baselines','.baselines.worldts','.baselines.worldts.models',
                 '.baselines.worldts.layers','.baselines.worldts.losses']:
        name='ts_benchmark'+part
        if name not in sys.modules:
            m=types.ModuleType(name);m.__path__=[str(SRC/name.replace('.','/'))];sys.modules[name]=m
    Model=importlib.import_module('ts_benchmark.baselines.worldts.models.worldts_model').WorldTSModel
    Loss=importlib.import_module('ts_benchmark.baselines.worldts.losses.latent_state_loss').LatentStateLoss
    return Model,Loss

def source_check():
    rows=json.loads((ROOT/'source_manifest.json').read_text())
    for r in rows:assert sha(BASE/r['path'])==r['sha256'],r['path']
    return {'files':len(rows),'manifest_sha256':sha(ROOT/'source_manifest.json')}

def load_data(name):
    rel,fields,expected=DATA[name];path=BASE/'cycle34/assets/fev'/rel
    assert sha(path)==expected
    raw=pd.read_parquet(path);assert len(raw)==1
    row=raw.iloc[0];a=np.stack([np.asarray(row[f],dtype=np.float64) for f in fields],1)
    ts=np.asarray(row['timestamp']);assert len(ts)==len(a)
    step=np.diff(ts).astype('timedelta64[s]').astype(np.int64)
    assert np.unique(step).tolist()==[900 if name=='solar' else 600]
    tr=int(.7*len(a));va=int(.8*len(a))
    mu=np.nanmean(a[:tr],axis=0);sd=np.nanstd(a[:tr],axis=0)
    assert np.isfinite(mu).all() and np.isfinite(sd).all() and (sd>1e-10).all()
    # Do not construct tensors or origins for TEST. No time-axis compression.
    allowed=a[:va].copy();finite=np.isfinite(allowed).all(1)
    bad=np.r_[0,np.cumsum(~finite)]
    origins={}
    for split,start,end,stride in [('train',T,tr,1),('val',tr,va,H)]:
        o=np.arange(start,end-H+1,stride,dtype=np.int64)
        o=o[(bad[o+H]-bad[o-T])==0];assert len(o)>20
        origins[split]=o
    scaled=((allowed-mu)/sd).astype(np.float32)
    metadata={'name':name,'data_sha256':expected,'fields':fields,'total_rows':len(a),
      'train_end':tr,'val_end':va,'test_scored':False,'mu':mu.tolist(),'sd':sd.tolist(),
      'valid_origin_counts':{k:len(v) for k,v in origins.items()},'id':str(row['id']),
      'first_timestamp':str(ts[0]),'val_last_timestamp':str(ts[va-1]),
      'nonfinite_before_val_end':int((~finite).sum()),'sampling_seconds':int(step[0])}
    return torch.tensor(scaled,device='cuda'),origins,metadata

def batch(values,origins):
    o=torch.as_tensor(origins,dtype=torch.long,device=values.device)
    assert int(o.min())>=T and int(o.max())+H<=len(values)
    hist=values[o[:,None]+torch.arange(-T,0,device=values.device)]
    future=values[o[:,None]+torch.arange(H,device=values.device),:1]
    assert bool(torch.isfinite(hist).all()) and bool(torch.isfinite(future).all())
    return hist[:,:,:1],hist[:,:,1:],future

class Arm(nn.Module):
    def __init__(self,teacher,arm,covdim):
        super().__init__();self.kind=arm;self.model=copy.deepcopy(teacher)
        for p in self.model.parameters():p.requires_grad_(False)
        for p in self.model.predictor.parameters():p.requires_grad_(True)
        self.g=None;self.project=None
        if arm!='N':
            self.g=nn.Sequential(nn.Linear(H*D+2*D+T*covdim,256),nn.GELU(),nn.Linear(256,H*D))
            nn.init.zeros_(self.g[-1].weight);nn.init.zeros_(self.g[-1].bias)
            if arm=='PY':self.project=nn.Linear(D,1,bias=False)
        self.cuda()

    def forward(self,yhist,covhist,future=None):
        out=self.model(yhist);n=out['h_pred'];hist=out['h_hist'];stats=out['target_revin_stats']
        numerical=self.model.decode(n,stats,hist);z=n;fused=numerical
        if self.g is not None:
            protected=self.kind in ['PZ','PY','PZ_NC'];anchor=n.detach() if protected else n
            c=torch.zeros_like(covhist) if self.kind=='PZ_NC' else covhist
            features=torch.cat([anchor.flatten(1),hist[:,-2:].flatten(1),c.flatten(1)],1)
            delta=self.g(features).reshape(-1,H,D);z=anchor+delta
            fused=(numerical.detach()+self.project(delta)*stats['stdev'] if self.kind=='PY'
                   else self.model.decode(z,stats,hist))
        target=None
        if future is not None:
            with torch.no_grad():
                target=self.model.encode_future_target((future-stats['mean'])/stats['stdev'],
                                                       (yhist-stats['mean'])/stats['stdev'])
        return {'fused':fused,'numerical':numerical,'z':z,'n':n,'target':target}

    def losses(self,out,future):
        fused=(out['fused']-future).abs().mean();num=(out['numerical']-future).abs().mean()
        latent=(out['z']-out['target']).abs().mean()
        if self.kind=='N':loss=num
        elif self.kind=='PY':loss=fused+num
        elif self.kind=='JZ':loss=fused+latent
        else:loss=fused+latent+num
        return loss,{'fused':fused,'num':num,'latent':latent}

    def groups(self):
        p=list(self.model.predictor.parameters())
        context=[] if self.g is None else list(self.g.parameters())
        if self.project is not None:context+=list(self.project.parameters())
        return p,context

@torch.no_grad()
def evaluate_arm(model,values,origins):
    model.eval();acc={k:[] for k in ['fused_mae','fused_mse','num_mae','num_mse','latent_mae','num_latent_mae']}
    preds=[];nums=[];targets=[]
    for i in range(0,len(origins),B):
        yh,ch,yf=batch(values,origins[i:i+B]);o=model(yh,ch,yf)
        for key,pred,func in [('fused_mae','fused','abs'),('fused_mse','fused','square'),
                              ('num_mae','numerical','abs'),('num_mse','numerical','square')]:
            err=o[pred]-yf;v=err.abs() if func=='abs' else err.square();acc[key].extend(v.mean((1,2)).cpu().tolist())
        acc['latent_mae'].extend((o['z']-o['target']).abs().mean((1,2)).cpu().tolist())
        acc['num_latent_mae'].extend((o['n']-o['target']).abs().mean((1,2)).cpu().tolist())
        preds.append(o['fused'].cpu().numpy());nums.append(o['numerical'].cpu().numpy());targets.append(yf.cpu().numpy())
    return {k:float(np.mean(v)) for k,v in acc.items()}, {
        'origins':np.asarray(origins),'pred':np.concatenate(preds),'num':np.concatenate(nums),'target':np.concatenate(targets)}
