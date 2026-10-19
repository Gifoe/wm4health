"""Privileged TRAIN/VAL-only state probes; never model selection or EVAL input."""
from common46 import *
from control46 import AlignedSSM,regular
from scipy.linalg import expm,solve_continuous_lyapunov
import datetime
torch.set_num_threads(1)

def main():
 assert read(R/'outputs/supervisor_v1.json')['status']=='FITS_COMPLETE_EVAL_UNOPENED';out=R/'outputs/train_val_diagnostics_v1';assert not out.exists();out.mkdir(parents=True);train=dict(np.load(R/'data/train.npz'));val=dict(np.load(R/'data/val.npz'));truth=read(R/'data/GENERATOR_TRUTH_NOT_LEARNER_INPUT.json');norms=read(R/'data/normalizers.json');latent={};max_replay=0.
 for split,collection,base_seed in [('train',train,20264610),('val',val,20264630)]:
  for key,y in collection.items():
   v=int(key[1]);s=int(key[-1]);rng=np.random.default_rng(base_seed+v*4+s);Fs=[np.asarray(f) for f in truth['F_primitives']];F=Fs[0]+(s&1)*Fs[1]+((s>>1)&1)*Fs[2];A=expm(F*.25);P=solve_continuous_lyapunov(F,-.25*np.eye(D));Q=P-A@P@A.T;L=np.linalg.cholesky((Q+Q.T)/2);z=rng.normal(size=(len(y),D));zs=[z]
   for t in range(1,T):z=z@A.T+rng.normal(size=z.shape)@L.T;zs.append(z)
   zs=np.stack(zs,1);info=truth['views'][v];u=zs@np.asarray(info['rotation']).T;raw=((u+.3*u**3)@np.asarray(info['projection']).T+rng.normal(size=y.shape)*info['noise_std']).astype('float32');replay=(raw-np.asarray(norms[str(v)]['mean'],dtype='float32'))/norms[str(v)]['scalar_std'];err=float(np.max(np.abs(replay-y)));max_replay=max(max_replay,err);assert err<5e-6,(key,split,err);latent[(split,key)]=zs
 records=[]
 def retained_probe(tag,states,futures,reconstruction):
  X=np.concatenate([states[('train',k)] for k in sorted(train)]);Z=np.concatenate([latent[('train',k)][:,HIST-1] for k in sorted(train)]);features=np.column_stack([X,np.ones(len(X))]);coef=np.linalg.solve(features.T@features+1e-6*np.eye(D+1),features.T@Z);XV=np.concatenate([states[('val',k)] for k in sorted(val)]);ZV=np.concatenate([latent[('val',k)][:,HIST-1] for k in sorted(val)]);pred=np.column_stack([XV,np.ones(len(XV))])@coef;future=np.concatenate([futures[k] for k in sorted(val)]);Zfuture=np.concatenate([latent[('val',k)][:,-1] for k in sorted(val)]);fpred=np.column_stack([future,np.ones(len(future))])@coef;state_mse=float(np.mean((pred-ZV)**2));r2=float(1-np.sum((pred-ZV)**2)/np.sum((ZV-ZV.mean(0))**2));fmse=float(np.mean((fpred-Zfuture)**2));path=out/(tag+'.npz');np.savez_compressed(path,train_state=X,train_true_z=Z,val_state=XV,val_true_z=ZV,probe_coefficients=coef,val_predicted_z=pred,val_true_future_z=Zfuture,val_predicted_future_z=fpred);records.append({'model':tag,'val_state_linear_probe_mse':state_mse,'val_state_linear_probe_r2':r2,'val_future_state_probe_mse':fmse,'val_instantaneous_qmean_decoder_reconstruction_mse':reconstruction,'raw_sha256':sha(path)})
 for folder in sorted((R/'outputs/fits').iterdir()):
  cfg=read(folder/'config.json');summary=read(folder/'fit_summary.json');assert sha(folder/'selected.pt')==summary['selected_sha256'];seed(cfg['seed']);model=World(cfg['obs'],cfg['dyn']);model.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)['model']);model.eval();before=checksum(model);states={};futures={};rec=[]
  with torch.no_grad():
   for split,collection in [('train',train),('val',val)]:
    for k,y in collection.items():
     model.set_context(int(k[1]),int(k[-1]));values=torch.from_numpy(y[:,:HIST]);mask=torch.ones(len(values),HIST);times=torch.arange(HIST)*.25;aux,_,_=model.native.aux_inference_net(values,mask,num_samples=1,deterministic=True);m=model.native.base_ssm.filter(aux[0],mask,times)['last_filtered_dist'][0];states[(split,k)]=m.numpy()
     if split=='val':futures[k]=(m@torch.matrix_exp(model.native.base_ssm.dynamics.F*(FUTURE*.25)).T).numpy();rec.append(float((model.native.y_emission_net(aux[0]).mean-values).square().mean()))
  retained_probe(folder.name,states,futures,float(np.mean(rec)));assert before==checksum(model) and sha(folder/'selected.pt')==summary['selected_sha256']
 selection=read(R/'outputs/ordinary_control_v1/selection.json');control=AlignedSSM(train,np.load(R/'data/anchors.npz')['X'],selection['selected_ridge']);states={};futures={}
 for split,collection in [('train',train),('val',val)]:
  for k,y in collection.items():
   view=control.views[int(k[1])];system=control.systems[int(k[-1])];A=system['A'];H=view['H'];Q=system['Q'];obs_R=view['R'];ms=[]
   for history in y[:,:HIST]:
    m=control.m0.copy();P=control.P0.copy()
    for t,value in enumerate(history):
     if t:m=A@m;P=A@P@A.T+Q
     K=np.linalg.solve(H@P@H.T+obs_R,H@P).T;m=m+K@(value-view['offset']-H@m);left=np.eye(D)-K@H;P=regular(left@P@left.T+K@obs_R@K.T)
    ms.append(m)
   states[(split,k)]=np.asarray(ms)
   if split=='val':futures[k]=np.asarray(ms)@np.linalg.matrix_power(A,FUTURE).T
 retained_probe('ordinary_aligned_PCA_LGSSM',states,futures,None)
 result={'status':'TRAIN_VAL_PRIVILEGED_DIAGNOSTICS_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'generator_replay_max_abs':max_replay,'records':records,'notes':['True latent states are auditor-only, not model inputs/selection labels. Affine probe fits TRAIN states and scores independent seen-cell VAL states.','Linear-probe performance is not a universal information/identifiability test and nonlinear gauge changes can affect it.','Native LTI posterior mean is linear in auxiliary history, so deterministic auxiliary means give the mean of the fitted variational mixture state. No oracle latent is fed to forecasting.','No sealed EVAL file opened here; no weights, stopping rules or hyperparameters changed. These are failure-attribution diagnostics, not new-method evidence.'],'source_sha256':sha(__file__)};dump(out/'diagnostics.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
