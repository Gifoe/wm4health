"""Frozen native meta-ID replication with calibration-only ordinary covariance controls."""
import runtime_profile
import argparse, hashlib, json, math, os, pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent / 'cycle16'
OUT = ROOT / 'outputs/native_gate_attempt2'
sys.path.insert(0, str(ROOT / 'runtime_dependencies'))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'outputs/runtime_numba_cache')
import numpy as np
import torch
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def predict(model, y, u, permutation=False):
    ctx, skip, nin = 400, 400, 30
    yc, uc = y[:, :ctx], u[:, :ctx]
    if permutation:
        yc, uc = yc.roll(1, 0), uc.roll(1, 0)
    q_u = u[:, ctx+skip:]
    q_y = y[:, ctx+skip:].clone(); q_y[:, nin:] = 0
    return model(yc, uc, q_u, q_y, nin)[:2]

def metrics(y, mean, std, covariance):
    # covariance is in standardized residual coordinates; exact Jacobian retained.
    residual = (y-mean) / std
    chol = np.linalg.cholesky(covariance)
    white = solve_triangular(chol, residual.T, lower=True).T
    logdet = 2*np.log(np.diag(chol)).sum()
    H = y.shape[1]
    joint = .5*(H*np.log(2*np.pi)+logdet+(white**2).sum(1))+np.log(std).sum(1)
    var = std**2 * np.diag(covariance)
    err = y-mean
    marginal = .5*(np.log(2*np.pi*var)+err**2/var)
    return {'marginal_nll': float(marginal.mean()), 'joint_nll_per_horizon':float(joint.mean()/H),
        'rmse':float(np.sqrt(np.mean(err**2))), 'mae':float(np.abs(err).mean()),
        'marginal_coverage95':float(np.mean(np.abs(err)<=1.959963984540054*np.sqrt(var))),
        'system_joint_nll_per_horizon':joint/H, 'marginal_variance_mean':float(var.mean())}

def functionals(y, mean, std, covariance, weights, names):
    error=(y-mean)@weights.T
    scaled=std[:,None,:]*weights[None,:,:]
    var=np.einsum('nkh,hj,nkj->nk', scaled, covariance, scaled)
    assert np.all(var>0)
    nll=.5*(np.log(2*np.pi*var)+error**2/var)
    return {name:{'coverage95':float(np.mean(np.abs(error[:,k])<=1.959963984540054*np.sqrt(var[:,k]))),
        'nll':float(nll[:,k].mean()), 'rmse':float(np.sqrt(np.mean(error[:,k]**2))),
        'variance_mean':float(var[:,k].mean())} for k,name in enumerate(names)}

def main():
    started=time.perf_counter();assert not OUT.exists();OUT.mkdir(parents=True)
    torch.set_num_threads(1);torch.manual_seed(2021)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=PREVIOUS/'upstream/sysid_benchmarks'
    manifest=json.loads((PREVIOUS/'outputs/source/metaid/manifest.json').read_text())
    for row in manifest['files']: assert sha(source/row['path'])==row['sha256']
    checkpoint=PREVIOUS/'outputs/external/metaid/ckpt_430_fine-tuned_3100000_lower_noise.pt'
    checkpoint_sha='e8324becad85ac610ec94f8ae09048938b343761a5bcd3967a86353a5bf28197'
    assert sha(checkpoint)==checkpoint_sha
    assert set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))<={'argparse.Namespace'}
    with torch.serialization.safe_globals([argparse.Namespace]):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    sys.path.insert(0,str(source))
    from dataset import WHDataset
    from transformer_sim import Config, TSTransformer
    device=torch.device('cuda:0');assert torch.cuda.is_available()
    model=TSTransformer(Config(**payload['model_args'])).to(device).eval()
    model.load_state_dict(payload['model'],strict=True)
    cfg=payload['cfg'];assert (cfg.seq_len_ctx,cfg.seq_len_skip,cfg.seq_len_n_in,cfg.seq_len_new)==(400,400,30,100)
    collections={}; invariance={}
    for label,seed in [('calibration',20211001),('probe',20212001)]:
        ds=WHDataset(nx=cfg.nx,nu=1,ny=1,seq_len=930,strictly_proper=True,
            system_seed=seed,input_seed=seed+1,noise_seed=seed+2,
            mag_range=cfg.mag_range,phase_range=cfg.phase_range)
        iterator=iter(ds);generated=[next(iterator) for _ in range(256)]
        y=torch.stack([p[0] for p in generated]);u=torch.stack([p[1] for p in generated])
        assert y.shape==u.shape==(256,930,1) and torch.isfinite(y).all() and torch.isfinite(u).all()
        means=[];stds=[];wrong_means=[];wrong_stds=[]
        for start in range(0,256,16):
            yy=y[start:start+16].to(device);uu=u[start:start+16].to(device)
            with torch.inference_mode():
                m,s=predict(model,yy,uu);wm,ws=predict(model,yy,uu,True)
                if start==0:
                    truth_query=yy[:,800:].clone()
                    a,b=model(yy[:,:400],uu[:,:400],uu[:,800:],truth_query,30)[:2]
                    ph=max(float((a-m).abs().max()),float((b-s).abs().max()))
                    changed=uu[:,800:].clone();changed[:,80:]+=5
                    a,b=model(yy[:,:400],uu[:,:400],changed,truth_query,30)[:2]
                    pf=max(float((a[:,:50]-m[:,:50]).abs().max()),float((b[:,:50]-s[:,:50]).abs().max()))
                    assert max(ph,pf)<2e-6,(label,ph,pf)
                    invariance[label]={'future_y_placeholder_error':ph,'future_u_prefix_error':pf}
            for tensor in [m,s,wm,ws]:assert torch.isfinite(tensor).all()
            assert (s>0).all() and (ws>0).all()
            means.append(m.cpu().numpy());stds.append(s.cpu().numpy())
            wrong_means.append(wm.cpu().numpy());wrong_stds.append(ws.cpu().numpy())
        collection={'generated_y':y.numpy(),'generated_u':u.numpy(),
            'mean':np.concatenate(means),'std':np.concatenate(stds),
            'wrong_context_mean':np.concatenate(wrong_means),'wrong_context_std':np.concatenate(wrong_stds),
            'system_seed':np.asarray(seed),'input_seed':np.asarray(seed+1),'noise_seed':np.asarray(seed+2)}
        for key in ['mean','std','wrong_context_mean','wrong_context_std']:
            assert collection[key].shape==(256,100,1)
        np.savez_compressed(OUT/(label+'.npz'),**collection);collections[label]=collection
        print(json.dumps({'phase':label,'systems':256,'elapsed':time.perf_counter()-started}),flush=True)
    cal=collections['calibration'];probe=collections['probe'];H=100
    cy=cal['generated_y'][:,830:,0].astype('float64');py=probe['generated_y'][:,830:,0].astype('float64')
    cm=cal['mean'][:,:,0].astype('float64');cs=cal['std'][:,:,0].astype('float64')
    pm=probe['mean'][:,:,0].astype('float64');ps=probe['std'][:,:,0].astype('float64')
    z=(cy-cm)/cs;temperature=float(np.mean(z*z));lw=LedoitWolf().fit(z)
    covariance=lw.covariance_;bias=lw.location_
    variants={'native':(pm,ps,np.eye(H)), 'temperature':(pm,ps,np.eye(H)*temperature),
        'bias_calibrated_diagonal':(pm+ps*bias,ps,np.diag(np.diag(covariance))),
        'bias_calibrated_full':(pm+ps*bias,ps,covariance)}
    names=['constant','alternating','half_contrast']+[f'dct{k}' for k in range(1,9)]
    weights=np.stack([np.ones(H),(-1.)**np.arange(H),np.r_[np.ones(H//2),-np.ones(H//2)],
        *[np.cos(np.pi*(np.arange(H)+.5)*k/H) for k in range(1,9)]])
    weights/=np.linalg.norm(weights,axis=1,keepdims=True)
    allmetrics={};system_nll={}
    for name,(m,s,cov) in variants.items():
        result=metrics(py,m,s,cov);system_nll[name]=result.pop('system_joint_nll_per_horizon')
        result['functionals']=functionals(py,m,s,cov,weights,names);allmetrics[name]=result
    wrong=metrics(py,probe['wrong_context_mean'][:,:,0].astype('float64'),
        probe['wrong_context_std'][:,:,0].astype('float64'),np.eye(H))
    wrong.pop('system_joint_nll_per_horizon')
    differences={};rng=np.random.default_rng(20213001);indices=rng.integers(0,256,size=(2000,256))
    for a,b in [('bias_calibrated_full','bias_calibrated_diagonal'),('temperature','native')]:
        delta=system_nll[a]-system_nll[b];boot=delta[indices].mean(1)
        differences[f'{a}_minus_{b}']={'mean':float(delta.mean()),
            'system_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'fraction_lower':float(np.mean(delta<0))}
    np.savez_compressed(OUT/'calibration_controls.npz',z=z,bias=bias,covariance=covariance,
        temperature=np.asarray(temperature),weights=weights,**{'nll_'+k:v for k,v in system_nll.items()})
    source_after={row['path']:sha(source/row['path']) for row in manifest['files']}
    assert all(source_after[row['path']]==row['sha256'] for row in manifest['files'])
    result={'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'script_sha256':sha(pathlib.Path(__file__)),
        'checkpoint_sha256':checkpoint_sha,'source_revision':manifest['revision'],'source_files_after':source_after,
        'systems':{'calibration':256,'probe':256},'invariance':invariance,
        'marginal_calibration_native':{k:v for k,v in metrics(cy,cm,cs,np.eye(H)).items() if k!='system_joint_nll_per_horizon'},
        'heldout_probe':allmetrics,'wrong_context_probe':wrong,
        'point_controls':{'zero_rmse':float(np.sqrt(np.mean(py**2))),
            'persistence_rmse':float(np.sqrt(np.mean((py-probe['generated_y'][:,829,0,None])**2)))},
        'temperature':temperature,'ledoit_wolf_shrinkage':float(lw.shrinkage_),
        'paired_system_nll_differences':differences,'elapsed_seconds':time.perf_counter()-started,
        'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
            'device':torch.cuda.get_device_name(0),'tf32':False},
        'limitations':['Native whole-sequence normalization includes future outputs; no raw-unit prospective claim.',
            'One released checkpoint, one system family; exploratory probe is not an untouched final evaluation.',
            'Ordinary Gaussian calibration and independent-vs-joint density are not new mechanisms.'],
        'new_training':False,'accepted_novel_directions':0,'objective_achieved':False}
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
