"""Exact latent-coordinate witness plus unmodified native simulator contracts."""
import runtime_profile
import contextlib,csv,datetime,hashlib,json,pathlib,sys,time,traceback
import numpy as np
import scipy.linalg as la
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/lag_mask';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT.parent/'upstream/dotime/src'))
class Tee:
 def __init__(self,*files):self.files=files
 def write(self,s):
  for f in self.files:f.write(s);f.flush()
 def flush(self):
  for f in self.files:f.flush()
def make_native(B,D):
 from dotime.tscm_sampler import TSCMSampler,TSCMStructure
 from dotime.temporal_graph import TemporalDAG
 from dotime.temporal_scm import TemporalSCM
 sampler=TSCMSampler(TSCMStructure.BOW_GRAPH,activations=[torch.nn.Identity()],noise_std=.3)
 native=sampler.sample(generator=torch.Generator().manual_seed(17))
 nodes=['U','A','Y'];assert native._topo==nodes
 dag=TemporalDAG(native.dag.G_0.copy(),[(np.abs(D)>1e-14).T.astype(np.float32)],1,nodes)
 with torch.no_grad():
  for i,name in enumerate(nodes):
   mechanism=native.mechanisms[name];mechanism.bias.zero_()
   for j,parent in enumerate(nodes):mechanism.weights_instant[parent].fill_(float(B[i,j]));mechanism.weights_lagged[0][parent].fill_(float(D[i,j]))
 return TemporalSCM(dag,native.mechanisms,native.noise,dtype=torch.float64)
def spectral(F,Q,w):
 C=np.eye(3)[1:];transfer=la.solve(np.eye(3)-F*np.exp(-1j*w),np.eye(3));return C@transfer@Q@transfer.conj().T@C.T
def sample(F,L,eps):
 values=[];state=np.zeros(3)
 for e in eps:state=F@state+L@e;values.append(state.copy())
 return np.asarray(values)
def main():
 torch.set_num_threads(1);started=time.time()
 cfg={'lag_scales':[0.,.001,.01,.1,1.],'seeds':[2021,2022,2023],'T':512,'burn_in':64,'onset':256,'a':1.,'b':.6,'c':-1.,'sigma':.3,'base_diagonal_lag':[.2,.5,.7],'source_revision':'8f227406a5b17c2040cfdcc680b493729ef47d00','scope':'Controlled linear source specialization; alternative permits off-diagonal lags, not a native released benchmark sample.'}
 (OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
 a,b,c=cfg['a'],cfg['b'],cfg['c'];B=np.array([[0,0,0],[a,0,0],[c,b,0.]])
 bp=b+2*a*c/(1+a*a);cp=-c;B2=np.array([[0,0,0],[a,0,0],[cp,bp,0.]])
 m=-2*c/(1+a*a+c*c);ell=-m*(b+a*c/(1+a*a));S=np.array([[1,ell,m],[0,1,0],[0,0,1.]])
 L=la.inv(np.eye(3)-B);L2=la.inv(np.eye(3)-B2);O=la.solve(L2,S@L);C=np.eye(3)[1:]
 assert np.max(np.abs(O@O.T-np.eye(3)))<1e-12
 assert np.max(np.abs(C@la.inv(S)-C))<1e-14
 invariants={'B':B.tolist(),'B_alternative':B2.tolist(),'hidden_state_transform_S':S.tolist(),'noise_rotation_O':O.tolist(),'noise_orthogonality_max_error':float(np.max(np.abs(O@O.T-np.eye(3)))),'observed_coordinates_preserved_max_error':float(np.max(np.abs(C@la.inv(S)-C))),'instant_slope_original':b,'instant_slope_alternative':bp,'counterfactual_signs_opposite':bool(b*bp<0)}
 (OUT/'exact_construction.json').write_text(json.dumps(invariants,indent=2),encoding='utf-8')
 rows=[];checks=[]
 for scale in cfg['lag_scales']:
  D=scale*np.diag(cfg['base_diagonal_lag']);F=L@D;F2=S@F@la.inv(S);D2=la.solve(L2,F2)
  Q=.09*L@L.T;Q2=.09*L2@L2.T;P=la.solve_discrete_lyapunov(F,Q);P2=la.solve_discrete_lyapunov(F2,Q2)
  spec=max(float(la.norm(spectral(F,Q,w)-spectral(F2,Q2,w))) for w in np.linspace(0,np.pi,257))
  check={'scale':scale,'rho_original':float(max(abs(la.eigvals(F)))),'rho_alternative':float(max(abs(la.eigvals(F2)))),'lag_original':D.tolist(),'lag_alternative':D2.tolist(),'offdiagonal_lag_frobenius':float(la.norm(D2-np.diag(np.diag(D2)))),'total_lag_difference_frobenius':float(la.norm(D2-D)),'stationary_observed_cov_max_error':float(np.max(np.abs(C@P@C.T-C@P2@C.T))),'heldout_spectral_frobenius_max_error':spec,'noise_cov_transform_max_error':float(np.max(np.abs(S@Q@S.T-Q2)))}
  assert check['rho_original']<1 and check['rho_alternative']<1
  assert spec<1e-11 and check['stationary_observed_cov_max_error']<1e-11
  checks.append(check)
  for seed in cfg['seeds']:
   from dotime.interventions import InterventionSpec,InterventionType
   rng=np.random.default_rng(seed);eps=rng.normal(size=(cfg['T']+cfg['burn_in'],3))*.3;eps2=eps@O.T
   first=make_native(B,D);second=make_native(B2,D2)
   first._frozen_noise={name:torch.tensor(eps[:,i],dtype=torch.float64) for i,name in enumerate(first._topo)}
   second._frozen_noise={name:torch.tensor(eps2[:,i],dtype=torch.float64) for i,name in enumerate(second._topo)}
   obs=first.sample_observational(cfg['T'],burn_in=cfg['burn_in']).numpy();obs2=second.sample_observational(cfg['T'],burn_in=cfg['burn_in']).numpy()
   direct=sample(F,L,eps)[cfg['burn_in']:];direct2=sample(F2,L2,eps2)[cfg['burn_in']:]
   directerr=float(np.max(np.abs(direct[:,1:]-direct2[:,1:])))
   contract=max(float(np.max(np.abs(obs-direct))),float(np.max(np.abs(obs2-direct2))))
   nativeerr=float(np.max(np.abs(obs[:,1:]-obs2[:,1:])))
   # Native mechanism parameters are float32 despite float64 buffers; tolerance accounts for that source contract.
   assert nativeerr<5e-6 and contract<5e-6 and directerr<1e-12
   value=float(obs[cfg['onset'],1]+1.)
   intervention=InterventionSpec(targets=[1],times=[cfg['onset']],values=value,intervention_type=InterventionType.HARD)
   cf=first.sample_interventional(cfg['T'],intervention,burn_in=cfg['burn_in']).numpy();cf2=second.sample_interventional(cfg['T'],intervention,burn_in=cfg['burn_in']).numpy()
   effect=float(cf[cfg['onset'],2]-obs[cfg['onset'],2]);effect2=float(cf2[cfg['onset'],2]-obs2[cfg['onset'],2])
   assert abs(effect-b)<5e-6 and abs(effect2-bp)<5e-6
   row={'scale':scale,'seed':seed,'native_observed_path_max_error':nativeerr,'native_vs_linear_contract_max_error':contract,'exact_linear_observed_path_max_error':directerr,'instant_effect_original':effect,'instant_effect_alternative':effect2,'offdiagonal_lag_frobenius':check['offdiagonal_lag_frobenius'],'native_max_amplitude':float(max(np.max(np.abs(obs)),np.max(np.abs(obs2))))}
   rows.append(row);print(json.dumps(row),flush=True)
   np.savez_compressed(OUT/f'paths_scale{scale:g}_seed{seed}.npz',observations=obs,observations_alternative=obs2,intervention=cf,intervention_alternative=cf2,noise=eps,noise_alternative=eps2)
 result={'construction':invariants,'matrix_checks':checks,'native_checks':rows,'elapsed_seconds':time.time()-started,'decision':'Exact observed-process equivalence can coexist with opposite intervention effects after relaxing native lag-mask restrictions. Generic causal nonidentification/sensitivity is established prior work; no new research direction or method accepted.'}
 (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'source':'../upstream/dotime','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2),encoding='utf-8')
 print('COMPLETE',result['elapsed_seconds'],flush=True)
if __name__=='__main__':
 with (OUT/'stdout.log').open('a',encoding='utf-8') as stdout,(OUT/'stderr.log').open('a',encoding='utf-8') as stderr,contextlib.redirect_stdout(Tee(sys.stdout,stdout)),contextlib.redirect_stderr(Tee(sys.stderr,stderr)):
  try:main()
  except BaseException:traceback.print_exc();raise
