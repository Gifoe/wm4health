"""Independent SciPy/NumPy score calculation, covariance and importance algebra."""
import runtime_profile
import hashlib,json,pathlib,math,time
import numpy as np
from scipy.special import ndtr,logsumexp
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/belief_audit';start=time.perf_counter()
r=json.loads((OUT/'results.json').read_text());assert len(r['rows'])==918
data=np.load(ROOT/'outputs/data/pendulum.npz');truth=data['query'][...,2].astype('float32').astype('float64');N=len(truth);M=r['mc_samples']
noise={}
for seed in r['mc_seeds']:
    gen=torch.Generator(device='cuda:0').manual_seed(seed)
    for _ in range(2):torch.randn(N,M,16,device='cuda:0',dtype=torch.float64,generator=gen)
    a=torch.randn(N,M,4,device='cuda:0',dtype=torch.float64,generator=gen).cpu().numpy();b=torch.randn(N,M,4,device='cuda:0',dtype=torch.float64,generator=gen).cpu().numpy();noise[seed]=(a,b)
def absolute(d,s):return s*np.sqrt(2/np.pi)*np.exp(-.5*(d/s)**2)+d*(2*ndtr(d/s)-1)
maxerr=0.;count=0;covmin=1.;verified=[]
for row in r['rows']:
    file=OUT/row['path'];assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256'];d=np.load(file);mu=d['mu'].astype('float64');v=d['variance_components'].astype('float64');mu2=d['mu2'].astype('float64');v2=d['variance_components2'].astype('float64')
    assert np.isfinite(mu).all() and np.isfinite(v).all() and v.min()>0
    mean=mu.mean(1);variance=(mu*mu+v).mean(1)-mean*mean;pit=ndtr((truth[:,None]-mu)/np.sqrt(v)).mean(1)
    crps=absolute(mu-truth[:,None],np.sqrt(v)).mean(1)-.5*absolute(mu-mu2,np.sqrt(v+v2)).mean(1)
    a,b=noise[row['mc_seed']];x=(mu+np.sqrt(v)*a)/8;x2=(mu2+np.sqrt(v2)*b)/8
    energy=np.linalg.norm(x-truth[:,None]/8,axis=-1).mean(1)-.5*np.linalg.norm(x-x2,axis=-1).mean(1)
    vg=np.zeros(N)
    for i in range(4):
        for j in range(i):vg+=(absolute((mu[...,i]-mu[...,j])/8,np.sqrt(v[...,i]+v[...,j])/8).mean(1)-abs(truth[:,i]-truth[:,j])/8)**2/6
    dm=mu-mean[:,None];cov=np.einsum('nmi,nmj->nij',dm,dm)/M
    for j in range(4):cov[:,j,j]+=v[...,j].mean(1)
    calc=dict(mean=mean,variance=variance,pit=pit,crps=crps,energy=energy,variogram=vg,covariance=cov,squared_error=(mean-truth)**2)
    for key,value in calc.items():
        e=float(np.max(abs(value-d[key])));maxerr=max(maxerr,e);assert e<2e-10,(row['path'],key,e);count+=1
    covmin=min(covmin,float(np.linalg.eigvalsh(cov).min()))
    aggregations=dict(crps=float(crps.mean()),query_crps=crps.mean(0),energy=float(energy.mean()),variogram=float(vg.mean()),rmse=float(np.sqrt(calc['squared_error'].mean())),query_rmse=np.sqrt(calc['squared_error'].mean(0)),coverage95=float(((pit>=.025)&(pit<=.975)).mean()),query_coverage95=((pit>=.025)&(pit<=.975)).mean(0),predictive_variance=float(variance.mean()))
    for key,value in aggregations.items():
        e=float(np.max(abs(np.asarray(value)-row[key])));assert e<2e-10,(row['path'],key,e);maxerr=max(maxerr,e);count+=1
    if 'ess' in d:assert np.min(d['ess'])>=1-1e-9
    verified.append(dict(path=row['path'],sha256=row['sha256']))
receipt=dict(status='pass',prediction_rows=len(r['rows']),scalar_or_array_checks=count,max_abs_error=maxerr,min_predictive_covariance_eigenvalue=covmin,raw_result_sha256=hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),data_sha256=hashlib.sha256((ROOT/'outputs/data/pendulum.npz').read_bytes()).hexdigest(),verifier_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.perf_counter()-start)
(OUT/'independent_score_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
