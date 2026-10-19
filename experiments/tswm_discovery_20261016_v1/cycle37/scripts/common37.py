"""Fixed daily conditional-trajectory pressure audit, not a proposed method."""
import os,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'cycle29/scripts'))
import runtime_profile
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,hashlib,types,importlib,random
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.spatial.distance import pdist
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
Q=np.arange(1,20,dtype=np.float64)/20
STEPS=4000;BATCH=64;SEEDS=[2026371,2026372];H=24;L=168
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def seed(n):random.seed(n);np.random.seed(n%(2**32));torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def calendar(t):
    t=pd.DatetimeIndex(t);hour=t.hour.to_numpy()/24;dow=(t.dayofweek.to_numpy()+hour)/7;year=(t.dayofyear.to_numpy()-1+hour)/365.25
    return np.stack([f(2*np.pi*x) for x in [hour,dow,year] for f in [np.sin,np.cos]],axis=-1)
def dataset(name):
    path=ROOT/'data_v3'/f'{name}_zone1_canonical.csv'
    d=pd.read_csv(path,parse_dates=['valid_time','forecast_origin_day']);d=d.set_index('valid_time')
    meta=json.loads((ROOT/'data_v3/data_audit.json').read_text(encoding='utf-8'))['datasets'][name]
    weather=meta['weather_columns'];assert not any(x in weather for x in ['POWER','TARGETVAR','y'])
    origins=pd.DatetimeIndex(sorted(d.forecast_origin_day.unique()));n=len(origins)
    a,b,c=int(.56*n),int(.70*n),int(.85*n)
    cuts={'fit':(0,a-1),'cal':(a,b-1),'val':(b,c-1),'test':(c,n)}
    groups={};counts={}
    for part,(start,stop) in cuts.items():
        rows=[];dropped=[]
        for o in origins[start:stop]:
            ht=o-pd.to_timedelta(np.arange(L,0,-1),unit='h');ft=o+pd.to_timedelta(np.arange(1,H+1),unit='h')
            hist=d.reindex(ht);fut=d.reindex(ft)
            hy=hist.y.to_numpy(dtype=np.float64)[:,None];y=fut.y.to_numpy(dtype=np.float64)
            hx=np.concatenate([hist[weather].to_numpy(dtype=np.float64),calendar(ht)],axis=1)
            fx=np.concatenate([fut[weather].to_numpy(dtype=np.float64),calendar(ft)],axis=1)
            if not all(np.isfinite(x).all() for x in [hy,y,hx,fx]):dropped.append(str(o));continue
            assert ht.max()<o and ft.min()>o
            rows.append((str(o),hy,hx,fx,y))
        assert rows,part
        groups[part]={'origins':np.array([x[0] for x in rows]),'hy':np.stack([x[1] for x in rows]),
                      'hx':np.stack([x[2] for x in rows]),'fx':np.stack([x[3] for x in rows]),'y':np.stack([x[4] for x in rows])}
        counts[part]={'candidates':stop-start,'retained':len(rows),'excluded':len(dropped),'excluded_origins':dropped,
                      'first':rows[0][0],'last':rows[-1][0]}
    for before,after in [('fit','cal'),('cal','val'),('val','test')]:
        latest=pd.Timestamp(groups[before]['origins'][-1])+pd.Timedelta(hours=H+1)
        assert latest<=pd.Timestamp(groups[after]['origins'][0]),(before,after)
    tr=groups['fit'];xflat=np.concatenate([tr['hx'].reshape(-1,tr['hx'].shape[-1]),tr['fx'].reshape(-1,tr['fx'].shape[-1])])
    xm=xflat.mean(0);xs=np.maximum(xflat.std(0),1e-6);ym=float(tr['y'].mean());ys=max(float(tr['y'].std()),1e-6)
    for part,g in groups.items():
        g['hy']=((g['hy']-ym)/ys).astype('float32');g['hx']=((g['hx']-xm)/xs).astype('float32');g['fx']=((g['fx']-xm)/xs).astype('float32')
        g['yn']=((g['y']-ym)/ys).astype('float32')
    return groups,{'name':name,'canonical_sha256':sha(path),'counts':counts,'weather_columns':weather,
                   'calendar_columns':['hour_sin','hour_cos','dow_sin','dow_cos','year_sin','year_cos'],
                   'scaler':{'x_mean':xm.tolist(),'x_std':xs.tolist(),'y_mean':ym,'y_std':ys},
                   'boundaries':{'n_original_days':n,'indices':[a,b,c]},'no_test_scores':True}
def flat(g):return np.concatenate([g['hy'].reshape(len(g['hy']),-1),g['hx'].reshape(len(g['hy']),-1),g['fx'].reshape(len(g['hy']),-1)],axis=1).astype('float32')
def analogy(g):
    return np.concatenate([g['fx'].mean(1),g['fx'].std(1),g['hy'][:,-1,:],g['hy'].mean(1),g['hy'].std(1)],axis=1)
def neighbours(cal,val,k=32):
    c=analogy(cal);v=analogy(val);scale=np.maximum(c.std(0),1e-6)
    dist=np.mean(((v[:,None,:]-c[None,:,:])/scale)**2,axis=-1)
    return np.argsort(dist,axis=1,kind='stable')[:,:min(k,len(c))]
def kite_module(kind):
    tag='c37_'+kind
    if tag+'.models.KITEModel' in sys.modules:return sys.modules[tag+'.models.KITEModel']
    p=BASE/'cycle36/source_audit/KITE/ts_benchmark/baselines/kite'
    for sub,folder in [('',p),('.layers',p/'layers'),('.models',p/'models')]:
        m=types.ModuleType(tag+sub);m.__path__=[str(folder)];sys.modules[tag+sub]=m
    if kind=='kite_l2':
        orig=(p/'layers/flow_loss.py').read_text(encoding='utf-8')
        old='flow_loss = F.l1_loss(v_pred, v_target)';assert orig.count(old)==1
        changed=orig.replace(old,'flow_loss = F.mse_loss(v_pred, v_target)')
        folder=ROOT/'outputs/source_control';folder.mkdir(parents=True,exist_ok=True);dst=folder/'flow_loss_l2.py'
        if dst.exists():assert dst.read_text(encoding='utf-8')==changed
        else:dst.write_text(changed,encoding='utf-8')
        m=types.ModuleType(tag+'.layers.flow_loss');m.__package__=tag+'.layers';m.__file__=str(dst)
        sys.modules[m.__name__]=m;exec(compile(changed,str(dst),'exec'),m.__dict__)
    return importlib.import_module(tag+'.models.KITEModel')
class QuantileMLP(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,96),nn.GELU(),nn.Linear(96,96),nn.GELU(),nn.Linear(96,H*len(Q)))
    def forward(self,x):return self.net(x).reshape(-1,H,len(Q))
def make_model(kind,g):
    if kind=='quantile_mlp':return QuantileMLP(flat(g).shape[1]).cuda()
    config=types.SimpleNamespace(seq_len=L,horizon=H,flow_dim=64,num_sampling_steps=10,flow_depth=2,flow_head=4,
        omega=1.,noise_dropout=.1,input_dim=g['hx'].shape[-1]+1,output_dim=1,p_uncond=.1,structure_max=.9,rank=8,
        min_sigma=.15,fc_type='Linear',rate=2,use_future_exog=True,agg_method='mean',aux_loss_weight=.1,mlp_ratio=2.,prior_level='sample')
    return kite_module(kind).KITEModel(config).cuda()
def tensors(g):return {k:torch.as_tensor(g[k],device='cuda') for k in ['hy','hx','fx','yn']}
def loss(model,kind,g,index):
    if kind=='quantile_mlp':
        x=torch.cat([g['hy'][index].flatten(1),g['hx'][index].flatten(1),g['fx'][index].flatten(1)],dim=1)
        q=model(x);e=g['yn'][index,:,None]-q;levels=torch.as_tensor(Q,device='cuda',dtype=q.dtype)
        return torch.maximum(levels*e,(levels-1)*e).mean()
    return model.train_function(g['hy'][index],g['hx'][index],g['yn'][index,:,None],g['fx'][index])
def inverse_quantiles(q,n):
    q=np.sort(np.asarray(q,dtype=np.float64),axis=-1)
    ext=np.concatenate([(2*q[:,:,0]-q[:,:,1])[:,:,None],q,(2*q[:,:,-1]-q[:,:,-2])[:,:,None]],axis=-1)
    levels=np.r_[0,Q,1];u=(np.arange(n)+.5)/n;out=np.empty((len(q),n,H))
    for b in range(len(q)):
        for h in range(H):out[b,:,h]=np.interp(u,levels,ext[b,h])
    return out
@torch.no_grad()
def predict(model,kind,g,meta,n=128,rng_seed=7,return_quantiles=False):
    seed(rng_seed);model.eval();ys=meta['scaler']['y_std'];ym=meta['scaler']['y_mean'];parts=[];qs=[]
    for start in range(0,len(g['y']),16):
        sl=slice(start,start+16);t=tensors({k:v[sl] for k,v in g.items() if k in ['hy','hx','fx','yn']})
        if kind=='quantile_mlp':
            x=torch.cat([t['hy'].flatten(1),t['hx'].flatten(1),t['fx'].flatten(1)],dim=1)
            q=np.sort(model(x).cpu().numpy()*ys+ym,axis=-1);parts.append(inverse_quantiles(q,n));qs.append(q)
        else:
            b=len(t['hy'])
            y=model.inference(t['hy'].repeat_interleave(n,0),t['hx'].repeat_interleave(n,0),
                t['fx'].repeat_interleave(n,0),num_samples=1).reshape(b,n,H)
            parts.append(y.cpu().numpy().astype('float64')*ys+ym)
    return np.concatenate(qs if return_quantiles else parts,axis=0)
def fair_crps(s,y):
    s=np.asarray(s,dtype=np.float64);m=s.shape[1];z=np.sort(s,axis=1)
    coef=(2*np.arange(1,m+1)-m-1).reshape((1,m)+(1,)*(s.ndim-2))
    return np.mean(np.abs(s-np.expand_dims(y,1)),axis=1)-np.sum(coef*z,axis=1)/(m*(m-1))
def scores(samples,y):
    s=np.asarray(samples,dtype=np.float64);y=np.asarray(y,dtype=np.float64);m=s.shape[1];h=s.shape[2]
    crps=fair_crps(s,y).mean(-1)
    energy=np.array([(np.linalg.norm(a-b,axis=1).mean()-.5*pdist(a).mean())/np.sqrt(h) for a,b in zip(s,y)])
    i,j=np.triu_indices(h,1);w=1/(j-i);w=w/w.sum()
    vg=(((np.abs(y[:,i]-y[:,j])**.5)-np.mean(np.abs(s[:,:,i]-s[:,:,j])**.5,axis=1))**2*w).sum(-1)
    q=np.quantile(s,[.05,.95],axis=1);med=np.median(s,axis=1);mean=s.mean(1)
    return {'crps':crps,'daily_mean_crps':fair_crps(s.mean(-1),y.mean(-1)),
        'ramp_crps':fair_crps(np.abs(np.diff(s,axis=-1)).max(-1),np.abs(np.diff(y,axis=-1)).max(-1)),
        'energy':energy,'variogram':vg,'mae':np.abs(med-y).mean(-1),'mse':((mean-y)**2).mean(-1),
        'coverage90':((y>=q[0])&(y<=q[1])).mean(-1),'width90':(q[1]-q[0]).mean(-1)}
def summary(samples,y):return {k:float(v.mean()) for k,v in scores(samples,y).items()}
def reorder(grid,template,rng):
    # Random tie breaking; permutations preserve every marginal sample multiset.
    ranks=np.argsort(np.argsort(template+rng.uniform(-1e-11,1e-11,template.shape),axis=0,kind='stable'),axis=0,kind='stable')
    return np.take_along_axis(grid,ranks,axis=0)
def quantile_paths(quantiles,cal_quantiles,cal_y,nearest,n=128,rng_seed=8):
    grid=inverse_quantiles(quantiles,n);rng=np.random.default_rng(rng_seed);b=len(grid);pit=np.empty_like(cal_y)
    for i in range(len(cal_y)):
        for h in range(H):pit[i,h]=np.interp(cal_y[i,h],np.sort(cal_quantiles[i,h]),Q,left=0,right=1)
    out={k:np.empty_like(grid) for k in ['iid','shuffle','analogue']}
    for i in range(b):
        iid=np.stack([rng.permutation(n) for h in range(H)],axis=1);out['iid'][i]=np.take_along_axis(grid[i],iid,axis=0)
        rows=rng.integers(len(pit),size=n);out['shuffle'][i]=reorder(grid[i],pit[rows],rng)
        rows=rng.choice(nearest[i],size=n,replace=True);out['analogue'][i]=reorder(grid[i],pit[rows],rng)
    for v in out.values():assert np.array_equal(np.sort(v,axis=1),grid)
    return out
def sample_quantiles(s):return np.quantile(s,Q,axis=1).transpose(1,2,0)
def calibrate(s,cal_s,cal_y):
    qc=sample_quantiles(cal_s);qv=sample_quantiles(s)
    offset=np.stack([np.quantile(cal_y-qc[:,:,j],q,axis=0) for j,q in enumerate(Q)],axis=-1)
    grid=inverse_quantiles(qv+offset[None],s.shape[1]);out=np.empty_like(s)
    for i in range(len(s)):
        ranks=np.argsort(np.argsort(s[i],axis=0,kind='stable'),axis=0,kind='stable');out[i]=np.take_along_axis(grid[i],ranks,axis=0)
    return out,offset
