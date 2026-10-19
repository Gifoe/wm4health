"""Independent observed trajectories; truth metadata excluded from learner files."""
from common44 import *
import datetime
O=R/'data';assert not O.exists();O.mkdir(parents=True);records=[]
rngB=np.random.default_rng(20264400);base=np.empty((P,4));base[:,0]=1/8;random_columns=rngB.normal(size=(P,3));random_columns-=base[:,[0]]@(base[:,[0]].T@random_columns);base[:,1:]=np.linalg.qr(random_columns)[0];C=2.5*base
A=np.array([[.90,.12,0,0],[-.12,.86,0,0],[0,0,.82,-.18],[0,0,.18,.82]],dtype=np.float64);q=np.array([.14,.12,.10,.10]);obs_std=.25
for case in ['gaussian','nonlinear']:
 folder=O/case;folder.mkdir();rng=np.random.default_rng(20264410+(case=='nonlinear'))
 def transition(z):
  if case=='gaussian':return z@A.T
  return np.stack([.82*z[:,0]+.25*np.tanh(2*z[:,0])+.05*z[:,1],.65*z[:,1]+.2*np.sin(z[:,0]),.85*z[:,2]-.18*z[:,3]+.10*np.tanh(z[:,0]*z[:,1]),.18*z[:,2]+.85*z[:,3]],axis=1)
 def generate(n):
  state=rng.normal(size=(n,4))
  for _ in range(100):state=transition(state)+rng.normal(size=(n,4))*q
  out=[]
  for t in range(LEN):
   out.append(state@C.T+rng.normal(0,obs_std,size=(n,P)));state=transition(state)+rng.normal(size=(n,4))*q
  return np.stack(out,1).astype('float32')
 train=generate(2048);mean=train.astype('float64').mean((0,1));scale=float(np.std(train.astype('float64')-mean));normalized=(train-mean)/scale;pca=pca_fit(normalized);np.savez_compressed(folder/'pca.npz',**pca)
 normalization={'mean':mean.tolist(),'scale':scale,'fit_split':'TRAIN only','history':H,'future':FUT,'channels':P};dump(folder/'normalization.json',normalization)
 for phase,x in [('train',train),('val',generate(128)),('eval',generate(64))]:
  data=((x-mean)/scale).astype('float32');np.savez_compressed(folder/(phase+'.npz'),observations=data);records.append({'case':case,'phase':phase,'shape':list(data.shape),'sha256':sha(folder/(phase+'.npz'))})
 future=normalized[:,H:];threshold=float(np.quantile(future,.9));cross=float(np.quantile(future.mean(-1),.9));m=future.mean(-1);tau=.25;smooth=tau*(np.log(np.mean(np.exp((m-m.max(-1,keepdims=True))/tau),axis=-1))+m.max(-1)/tau)
 functions=np.stack([future.mean((1,2)),smooth,np.square(np.diff(future,axis=1)).mean((1,2)),(future>threshold).mean((1,2)),(m.max(1)>cross).astype(float)],axis=1)
 dump(folder/'functional_spec.json',{'names':['mean','smoothmax_spatial_mean','temporal_roughness','occupancy','crossing'],'point_threshold':threshold,'spatial_mean_threshold':cross,'smoothmax_tau':tau,'scales':np.maximum(functions.std(0),1e-3).tolist(),'fit_split':'TRAIN only'})
 dump(folder/'truth_model.json',{'case':case,'A':A.tolist(),'C':C.tolist(),'basis':base.tolist(),'process_std':q.tolist(),'observation_std':obs_std,'initial_covariance':'I4','burn_transitions':100,'nonlinear_transition':'See fixed data44.py; no latent states stored in learner arrays','for_evaluation_only':True})
dump(O/'DATA_MANIFEST.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records,'pca_rank':{case:int(np.load(O/case/'pca.npz')['rank']) for case in ['gaussian','nonlinear']},'protocol_sha256':sha(R/'PROTOCOL.md'),'generator_sha256':sha(pathlib.Path(__file__)),'old_real_test_scored':False,'eval_not_scored':True});print(json.dumps({'data_splits':len(records),'rank':{case:int(np.load(O/case/'pca.npz')['rank']) for case in ['gaussian','nonlinear']}}))
