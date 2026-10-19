"""Independent SciPy density and native regeneration/replay verification."""
import runtime_profile
import argparse,hashlib,json,os,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];PREVIOUS=ROOT.parent/'cycle16'
sys.path.insert(0,str(ROOT/'runtime_dependencies'))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'outputs/runtime_numba_cache')
import numpy as np
import torch
from scipy.stats import multivariate_normal,norm
from sklearn.covariance import LedoitWolf

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.perf_counter();torch.set_num_threads(1)
    folder=ROOT/'outputs/native_gate_attempt2';receipt=folder/'independent_verification.json';assert not receipt.exists()
    result=json.loads((folder/'results.json').read_text());controls=np.load(folder/'calibration_controls.npz')
    cal=np.load(folder/'calibration.npz');probe=np.load(folder/'probe.npz')
    y=probe['generated_y'][:,830:,0].astype('float64');m=probe['mean'][:,:,0].astype('float64');s=probe['std'][:,:,0].astype('float64')
    z=(cal['generated_y'][:,830:,0].astype('float64')-cal['mean'][:,:,0])/cal['std'][:,:,0]
    fitted=LedoitWolf().fit(z);cov=fitted.covariance_;bias=fitted.location_
    assert np.max(np.abs(cov-controls['covariance']))<1e-12 and np.max(np.abs(bias-controls['bias']))<1e-12
    variants={'native':(m,np.eye(100)),'temperature':(m,np.eye(100)*np.mean(z*z)),
        'bias_calibrated_diagonal':(m+s*bias,np.diag(np.diag(cov))),'bias_calibrated_full':(m+s*bias,cov)}
    errors={};joint_vectors={};weights=controls['weights'];names=['constant','alternating','half_contrast']+[f'dct{k}' for k in range(1,9)]
    for label,(mu,R) in variants.items():
        residual=(y-mu)/s
        vector=(-multivariate_normal.logpdf(residual,mean=np.zeros(100),cov=R)+np.log(s).sum(1))/100
        joint_vectors[label]=vector
        expected=result['heldout_probe'][label]
        err=abs(vector.mean()-expected['joint_nll_per_horizon'])
        err=max(err,float(np.max(np.abs(vector-controls['nll_'+label]))))
        marginal_std=s*np.sqrt(np.diag(R))
        err=max(err,abs(float(-norm.logpdf(y,loc=mu,scale=marginal_std).mean())-expected['marginal_nll']))
        for k,name in enumerate(names):
            w=weights[k];v=[]
            for row in range(len(y)):
                a=s[row]*w;v.append(a@R@a)
            std=np.sqrt(v);deviation=(y-mu)@w
            covered=float(np.mean(np.abs(deviation)<=norm.ppf(.975)*std))
            nll=float(-norm.logpdf(deviation,scale=std).mean())
            err=max(err,abs(covered-expected['functionals'][name]['coverage95']),abs(nll-expected['functionals'][name]['nll']))
        assert err<1e-9,(label,err);errors[label]=err
    source=PREVIOUS/'upstream/sysid_benchmarks';sys.path.insert(0,str(source))
    from dataset import WHDataset
    from transformer_sim import Config,TSTransformer
    regenerated={}
    for label,saved,seed in [('calibration',cal,20211001),('probe',probe,20212001)]:
        ds=iter(WHDataset(nx=5,nu=1,ny=1,seq_len=930,strictly_proper=True,system_seed=seed,input_seed=seed+1,
            noise_seed=seed+2,mag_range=(.5,.97),phase_range=(0,np.pi/2)))
        pairs=[next(ds) for _ in range(8)]
        yy=torch.stack([p[0] for p in pairs]).numpy();uu=torch.stack([p[1] for p in pairs]).numpy()
        ey=float(np.max(np.abs(yy-saved['generated_y'][:8])));eu=float(np.max(np.abs(uu-saved['generated_u'][:8])))
        assert ey==eu==0;regenerated[label]={'samples':8,'y_error':ey,'u_error':eu}
    checkpoint=PREVIOUS/'outputs/external/metaid/ckpt_430_fine-tuned_3100000_lower_noise.pt'
    assert sha(checkpoint)==result['checkpoint_sha256']
    with torch.serialization.safe_globals([argparse.Namespace]):payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=TSTransformer(Config(**payload['model_args'])).cpu().eval();model.load_state_dict(payload['model'],strict=True)
    yy=torch.from_numpy(probe['generated_y'][:4]);uu=torch.from_numpy(probe['generated_u'][:4])
    with torch.inference_mode():pred,std,loss,rmse=model(yy[:,:400],uu[:,:400],uu[:,800:],yy[:,800:],30)
    em=float(np.max(np.abs(pred.numpy()-probe['mean'][:4])));es=float(np.max(np.abs(std.numpy()-probe['std'][:4])))
    assert max(em,es)<1e-4,(em,es)
    exact_loss=float(-norm.logpdf(yy.numpy()[:,830:],loc=pred.numpy(),scale=std.numpy()).mean())
    assert abs(exact_loss-float(loss))<1e-5
    manifest=json.loads((PREVIOUS/'outputs/source/metaid/manifest.json').read_text())
    for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
    receipt.write_text(json.dumps({'pass':True,'scipy_metric_errors':errors,'regeneration':regenerated,
        'CPU_vs_saved_GPU_mean_error':em,'CPU_vs_saved_GPU_std_error':es,'native_loss_vs_scipy_error':abs(exact_loss-float(loss)),
        'source_files_unchanged':len(manifest['files']),'covariance_min_eigenvalue':float(np.linalg.eigvalsh(cov).min()),
        'verified_array_sha256':{p.name:sha(p) for p in folder.glob('*.npz')},'script_sha256':sha(pathlib.Path(__file__)),
        'elapsed_seconds':time.perf_counter()-start,'objective_achieved':False},indent=2),encoding='utf-8')
    print(receipt.read_text())

if __name__=='__main__':main()
