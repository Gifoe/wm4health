from gaussian64 import *
assert read(OUT/'QUALIFICATION64_V1.json')['status']=='PASS'
pp=paths();corpora=[];fits=[];evaluation=[];t0=__import__('time').perf_counter()
def generate(N,k,s,idx,split):
 rng=np.random.default_rng(np.random.SeedSequence([64000,s,idx,split]));u=np.empty((N,64));y=np.empty_like(u);last=np.full(N,Y0)
 for t in range(64):u[:,t]=-k*last+TAU*rng.normal(size=N);y[:,t]=RHO*last+BETA*u[:,t]+SIGMA*rng.normal(size=N);last=y[:,t]
 return u,y
def regress(X,y):
 coef=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@coef;return coef,float(np.mean(res*res))
def obs_nll(U,Y,jm,jc):
 h=Y.shape[1];_,C,B=conditional(jm,jc,np.zeros(h));pred=jm[:h]+(U-jm[h:])@B.T;return float(-logpdf(Y-pred,np.zeros(h),C).mean()/h)
for i,k in enumerate(GAINS):
 for s in SEEDS:
  U,Y=generate(16384,k,s,i,0);Uv,Yv=generate(4096,k,s,i,1);key='k'+str(k)+'_s'+str(s);datafile=pp['data_root']/(key+'.npz');np.savez_compressed(datafile,U=U,Y=Y,U_validation=Uv,Y_validation=Yv);corpora.append({'key':key,'gain':k,'data_seed':s,'path':str(datafile),'sha256':sha(datafile),'training_records':len(U),'validation_records':len(Uv),'max_horizon':64})
  for h in HORIZONS:
   state('fit_state64_v1.json',{'status':'FITTING','gain':k,'seed':s,'horizon':h,'completed_fits':len(fits)})
   u,y=U[:,:h],Y[:,:h];joint=np.concatenate([y,u],axis=1);jm=joint.mean(0);center=joint-jm;jc=sym(center.T@center/len(joint))+1e-10*np.eye(2*h)
   prev=np.concatenate([np.full((len(u),1),Y0),y[:,:-1]],1).reshape(-1);flat_u=u.reshape(-1);flat_y=y.reshape(-1)
   ar,noise_var=regress(np.stack([prev,flat_u,np.ones_like(prev)],1),flat_y);pol,policy_var=regress(np.stack([prev,np.ones_like(prev)],1),flat_u);rho,beta,b=ar;c,d=pol
   modelpath=pp['checkpoint_root']/(key+'_H'+str(h)+'.npz');np.savez_compressed(modelpath,joint_mean=jm,joint_covariance=jc,ar=ar,noise_variance=noise_var,policy=pol,policy_variance=policy_var)
   truejm,truejc=structural_joint(h,-k);arjm,arjc=structural_joint(h,c,rho,beta,np.sqrt(noise_var),np.sqrt(policy_var),b,d)
   fit={'key':key,'gain':k,'data_seed':s,'horizon':h,'path':str(modelpath),'sha256':sha(modelpath),'data_sha256':corpora[-1]['sha256'],'ar':ar.tolist(),'noise_variance':noise_var,'policy':pol.tolist(),'policy_variance':policy_var,'validation_observed_conditional_NLL_per_step':{'oracle':obs_nll(Uv[:,:h],Yv[:,:h],truejm,truejc),'block_Gaussian':obs_nll(Uv[:,:h],Yv[:,:h],jm,jc),'causal_AR_reconditioned':obs_nll(Uv[:,:h],Yv[:,:h],arjm,arjc)}};fits.append(fit)
   for pk in ['zero','sine']:
    query=plan(h,pk);truth=canonical(query,-k);mo,Co,_=conditional(jm,jc,query);Qo=inverse(Co);ho=Qo@mo
    arlaw=canonical(query,c,rho,beta,np.sqrt(noise_var),np.sqrt(policy_var),b,d)
    alternatives=[('observational_block',{'valid':True,'mean':mo,'cov':Co}),('causal_AR',{'valid':True,'mean':arlaw['do_mean'],'cov':arlaw['do_cov']})]
    for label,pc,pv,pd in [('known_policy',-k,TAU**2,0.),('estimated_policy',c,policy_var,d)]:
     pdiag=np.r_[np.repeat(pc**2/pv,h-1),0.];ell=np.r_[pc*(query[1:]-pd)/pv,0.];Q=Qo-np.diag(pdiag);hh=ho-ell;law=moments_from_precision(Q,hh);alternatives.append(('ratio_'+label,law))
     if label=='estimated_policy':
      ev,V=la.eigh(sym(Q));floorQ=(V*np.maximum(ev,1e-3))@V.T;alternatives.append(('fixed_eigenvalue_floor',moments_from_precision(floorQ,hh)))
    for method,law in alternatives:
     row={'gain':k,'data_seed':s,'horizon':h,'query':pk,'method':method,'valid':law['valid'],'minimum_precision_eigenvalue':law.get('minimum_precision_eigenvalue')}
     if law['valid']:row.update(metrics(law['mean'],law['cov'],truth['do_mean'],truth['do_cov']))
     evaluation.append(row)
dump(OUT/'FIT_AND_GAUSSIAN_EVALUATION64_V1.json',{'utc':utc(),'status':'ALL_GAUSSIAN_MLE_FITS_AND_FIXED_QUERIES_SCORED','corpora':corpora,'fits':fits,'evaluation':evaluation,'unique_training_corpora':12,'related_horizon_fit_cells':48,'fitted_model_classes':['full_joint_Gaussian','causal_linear_Gaussian'],'fitted_behavior_policy_regressions':48,'solver':'Closed-form train-only MLE; no iterative optimizer or model selection','protocol_sha256':sha(ROOT/'PROTOCOL64_V1.md'),'elapsed_seconds':__import__('time').perf_counter()-t0,'accepted_final_directions':0})
state('fit_state64_v1.json',{'status':'ALL_48_RELATED_MLE_CELLS_CLOSED','evaluation_rows':len(evaluation),'unique_corpora':12});print(json.dumps({'status':'ALL_48_RELATED_MLE_CELLS_CLOSED','rows':len(evaluation),'invalid_corrected_rows':sum(not r['valid'] for r in evaluation)}))
