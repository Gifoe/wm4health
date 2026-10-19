"""Frozen native core and ordinary readout definitions. No prior launcher import."""
import pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'cycle29/scripts'))
import runtime_profile
import os,json,hashlib,types,importlib,random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
OLD=pathlib.Path('D:/baseline_reproduction_seed2021_v1')
SRC=BASE/'upstream/WorldTS/DAG-bench'
CACHE=ROOT/'outputs/native_cache_v1'
FITS=ROOT/'outputs/readout_fits_v1'
SEEDS=[2026331,2026332,2026333]
ARMS=['native_mix','native_pred','wide_local_pred','global_pred']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()

def dump(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def state_sha(state):
    h=hashlib.sha256()
    for k,t in sorted(state.items()):h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)

def import_core():
    for name in ['ts_benchmark','ts_benchmark.baselines','ts_benchmark.baselines.worldts',
                 'ts_benchmark.baselines.worldts.models','ts_benchmark.baselines.worldts.layers']:
        if name not in sys.modules:
            m=types.ModuleType(name);m.__path__=[str(SRC/name.replace('.','/'))];sys.modules[name]=m
    model=importlib.import_module('ts_benchmark.baselines.worldts.models.worldts_model').WorldTSModel
    decoder=importlib.import_module('ts_benchmark.baselines.worldts.layers.state_to_observation_mlp_decoder').CausalPatchStateToObservationDecoder
    return model,decoder

def source_check():
    rows=json.loads((ROOT/'frozen_source_manifest.json').read_text())
    for row in rows:assert sha(BASE/row['path'])==row['sha256'],row['path']
    return {'checked_files':len(rows),'manifest_sha256':sha(ROOT/'frozen_source_manifest.json')}

def load_native(dataset):
    Model,_=import_core();folder=OLD/'results/worldts'/f'{dataset}_H24_seed2021'
    config=json.loads((folder/'resolved_config.json').read_text())
    assert config['seq_len']==168 and config['pred_len']==24 and config['output_dim']==1
    s1=torch.load(folder/'stages/stage1_latent.pt',map_location='cpu',weights_only=True)['Model']
    s2=torch.load(folder/'stages/stage2_decoder_mix.pt',map_location='cpu',weights_only=True)['Model']
    for k,t in s1.items():
        if not k.startswith('decoder.'):assert torch.equal(t,s2[k]),k
    model=Model(SimpleNamespace(**config));model.load_state_dict(s1,strict=True);model.eval().cuda()
    for p in model.parameters():p.requires_grad_(False)
    return model,config,s2,{'stage1_sha256':sha(folder/'stages/stage1_latent.pt'),
          'stage2_sha256':sha(folder/'stages/stage2_decoder_mix.pt'),'config_sha256':sha(folder/'resolved_config.json'),
          'frozen_nondecoder_equal':True,'stage1_state_sha256':state_sha(s1)}

def load_data(dataset,T=168,H=24):
    path=OLD/'data'/f'{dataset}.csv';raw=pd.read_csv(path);cols=raw['cols'].drop_duplicates().tolist()
    assert cols[-1]=='OT';raw['date']=pd.to_datetime(raw['date'])
    frame=raw.pivot(index='date',columns='cols',values='data')[cols]
    assert frame.shape==(52416,3) and frame.index.is_unique
    assert frame.index.is_monotonic_increasing and (frame.index[1:]-frame.index[:-1]==pd.Timedelta(hours=1)).all()
    a=frame.to_numpy(dtype=np.float64);assert np.isfinite(a).all()
    tv=int(len(a)*.8);tr=int(tv*.875)
    sy=StandardScaler().fit(a[:tr,-1:]);sc=StandardScaler().fit(a[:tr,:-1])
    values=np.c_[sy.transform(a[:,-1:]),sc.transform(a[:,:-1])].astype(np.float32)
    origins={k:np.arange(max(start,T),end-H+1,dtype=np.int64) for k,start,end in [('train',0,tr),('val',tr,tv),('test',tv,len(a))]}
    meta={'data_sha256':sha(path),'columns':['OT']+cols[:-1],'train_boundary':tr,'trainval_boundary':tv,'total_points':len(a),
          'y_mean':sy.mean_.tolist(),'y_scale':sy.scale_.tolist(),'cov_mean':sc.mean_.tolist(),'cov_scale':sc.scale_.tolist(),
          'counts':{k:len(o) for k,o in origins.items()}}
    saved=np.load(OLD/'results/worldts'/f'{dataset}_H24_seed2021/scaler.npz')
    for key,val in [('mean_y',sy.mean_),('scale_y',sy.scale_),('mean_c',sc.mean_),('scale_c',sc.scale_)]:assert np.array_equal(saved[key],val),key
    return values,origins,meta

def cache_arrays(dataset,split,device='cuda'):
    folder=CACHE/dataset/split
    out={}
    for name in ['pred','target','prefix','mean','scale','y']:
        arr=np.load(folder/(name+'.npy'),mmap_mode='r')
        out[name]=torch.tensor(np.asarray(arr),dtype=torch.float32,device=device)
    return out

class GlobalReadout(nn.Module):
    def __init__(self,input_dim,horizon,mean=None,scale=None):
        super().__init__();self.horizon=horizon
        self.register_buffer('feature_mean',torch.zeros(input_dim) if mean is None else mean.detach().clone())
        self.register_buffer('feature_scale',torch.ones(input_dim) if scale is None else scale.detach().clone())
        self.net=nn.Sequential(nn.Linear(input_dim,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,horizon))
    def forward(self,future,prefix,mean,scale):
        f=torch.cat([prefix.flatten(1),future.flatten(1),mean.flatten(1),torch.log(scale).flatten(1)],1)
        return self.net((f-self.feature_mean)/self.feature_scale).reshape(-1,self.horizon,1)

def global_features(pack):
    return torch.cat([pack['prefix'].flatten(1),pack['pred'].flatten(1),pack['mean'].flatten(1),torch.log(pack['scale']).flatten(1)],1)

def make_head(config,arm,train=None,global_stats=None):
    _,Decoder=import_core();d=config['latent_dim'];p=config['decoder_patch_len'];hidden=d*config['decoder_hidden_multiplier']
    if arm=='global_pred':
        n=(24+p-1)*d+2
        if global_stats is not None:mu,sd=global_stats
        elif train is not None:
            f=global_features(train);mu=f.mean(0);sd=f.std(0,unbiased=False).clamp_min(1e-4)
        else:mu=sd=None
        return GlobalReadout(n,24,mu,sd).cuda()
    return Decoder(d,1,p,hidden*(4 if arm=='wide_local_pred' else 1)).cuda()

def forward_head(head,pack,idx,kind='pred'):
    z=pack[kind][idx];prefix=pack['prefix'][idx];mean=pack['mean'][idx];scale=pack['scale'][idx]
    out=head(z,prefix,mean,scale) if isinstance(head,GlobalReadout) else head(z,prefix)
    return out*scale+mean

def lr_for_epoch(base,e):return base*(.9**max(0,e-3))
