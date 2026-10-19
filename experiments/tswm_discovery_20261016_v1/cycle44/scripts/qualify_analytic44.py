"""CPU-only algebra checks on an already selected ordinary model, VAL only."""
from analytic_query44 import *
import datetime
O=R/'outputs/analytic_qualification_v1';assert not O.exists();O.mkdir(parents=True)
folder=R/'outputs/fits_v1/gaussian_lgssm_2026441';summary=read(folder/'summary.json');assert sha(folder/'model.pt')==summary['checkpoint_sha256'];pca=dict(np.load(R/'data/gaussian/pca.npz'));model=LGSSM(pca,np.load(R/'data/gaussian/train.npz')['observations']).double();checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['selected_state']);model.eval();history=np.load(R/'data/gaussian/val.npz')['observations'][:1,:H];spec=read(R/'data/gaussian/functional_spec.json');moments=ssm_query_moments(model,history);got=ssm_analytic_readout(moments,spec)
# Independent full1024-dimensional Gaussian construction from fitted state law.
A=moments['A'].cpu().numpy();Q=moments['Q'].cpu().numpy();r=float(moments['R']);C=moments['loading'].cpu().numpy();m=moments['posterior_mean'].cpu().numpy();p=moments['posterior_covariance'].cpu().numpy();means=[];covs=[]
for _ in range(FUT):m=A@m;p=A@p@A.T+Q;means.append(m);covs.append(p.copy())
mu=np.stack(means)@C.T;covariance=np.empty((FUT*P,FUT*P))
for j in range(FUT):
 cross=covs[j].copy()
 for i in range(j,FUT):
  if i>j:cross=A@cross
  block=C@cross@C.T
  if i==j:block=block+r*np.eye(P)
  covariance[i*P:(i+1)*P,j*P:(j+1)*P]=block;covariance[j*P:(j+1)*P,i*P:(i+1)*P]=block.T
from scipy.special import ndtr
exact_mean=mu.mean();exact_occ=ndtr((mu.reshape(-1)-spec['point_threshold'])/np.sqrt(np.diag(covariance))).mean();rough=0.
for t in range(1,FUT):
 i=np.arange(t*P,(t+1)*P);j=i-P;rough+=np.mean((mu[t]-mu[t-1])**2+covariance[i,i]+covariance[j,j]-2*covariance[i,j])/(FUT-1)
truth={0:exact_mean/spec['scales'][0],2:rough/spec['scales'][2],3:exact_occ/spec['scales'][3]};errors={str(k):abs(float(got[k])-truth[k]) for k in truth};assert max(errors.values())<1e-9,errors
# Gaussian spatial-mean marginal equals independent dense projection.
project=np.zeros((FUT,FUT*P))
for t in range(FUT):project[t,t*P:(t+1)*P]=1/P
cov_error=float(np.max(np.abs(project@covariance@project.T-moments['spatial_covariance'].cpu().numpy())));assert cov_error<1e-9,cov_error
# Decoder conditional identities, independent ordinary Gaussian-noise MC.
ar=AR(pca).double();rng=np.random.default_rng(20264488);pc=torch.tensor(rng.normal(size=(2,FUT,ar.d)),dtype=torch.float64);rb=ar_conditional_readout(ar,pc,spec).cpu().numpy();U=ar.U.numpy();scale=ar.scale.numpy();noise=float(ar.tail_noise);mu=(pc.numpy()*scale)@U.T;records=[]
for context in range(2):
 draws=[]
 for block in range(32):
  eps=rng.normal(size=(2048,FUT,P));field=mu[context]+np.sqrt(noise)*(eps-(eps@U)@U.T);spatial=field.mean(-1);maximum=spatial.max(1);smooth=maximum+spec['smoothmax_tau']*np.log(np.exp((spatial-maximum[:,None])/spec['smoothmax_tau']).mean(1));functions=np.column_stack((spatial.mean(1),smooth,np.square(np.diff(field,axis=1)).mean((1,2)),(field>spec['point_threshold']).mean((1,2)),(maximum>spec['spatial_mean_threshold']).astype(float)));draws.append(functions/np.asarray(spec['scales']))
 draws=np.concatenate(draws);mean=draws.mean(0);se=draws.std(0,ddof=1)/np.sqrt(len(draws));indices=[0,2,3,4];delta=np.abs(mean-rb[context]);passed=[bool(delta[k]<=5*se[k]+1e-6) for k in indices];assert all(passed),(context,delta,se)
 records.append({'context':context,'draws':len(draws),'analytic_indices':indices,'conditional_mean':rb[context].tolist(),'independent_MC_mean':mean.tolist(),'independent_MC_SE':se.tolist(),'five_SE_checks':passed})
result={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dense_future_covariance_dimension':FUT*P,'exact_query_max_abs':max(errors.values()),'spatial_marginal_covariance_max_abs':cov_error,'conditional_decoder_checks':records,'selected_checkpoint_sha256':summary['checkpoint_sha256'],'eval_opened':False,'new_training_updates':0,'source_sha256':sha(pathlib.Path(__file__)),'analytic_module_sha256':sha(R/'scripts/analytic_query44.py'),'addendum_sha256':sha(R/'ANALYTIC_CONTROLS.md')};dump(O/'qualification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='conditional_decoder_checks'}),flush=True)
