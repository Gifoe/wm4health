from gaussian64 import *
f=read(OUT/'FIT_AND_GAUSSIAN_EVALUATION64_V1.json');rows=[];art={};pp=paths()
for fit in f['fits']:
 assert sha(fit['path'])==fit['sha256'];z=np.load(fit['path']);h=fit['horizon'];order=np.array([q for t in range(h) for q in [h+t,t]]);mu=z['joint_mean'][order];C=z['joint_covariance'][np.ix_(order,order)];chol=la.cholesky(C,lower=True);L=la.solve_triangular(chol,np.eye(2*h),lower=True)
 A=np.eye(h);B=np.zeros((h,h));b=np.zeros(h);v=np.zeros(h)
 for t in range(h):
  j=2*t+1;coef=-L[j,:j]/L[j,j];b[t]=mu[j]-coef@mu[:j];B[t,:t+1]=coef[0::2];A[t,:t]=-coef[1::2];v[t]=1/L[j,j]**2
 Ai=la.solve_triangular(A,np.eye(h),lower=True);Cd=sym((Ai*v)@Ai.T);key=fit['key']+'_H'+str(h);art[key+'_A']=A;art[key+'_B']=B;art[key+'_b']=b;art[key+'_variance']=v
 for pk in ['zero','sine']:
  u=plan(h,pk);md=la.solve_triangular(A,b+B@u,lower=True);truth=canonical(u,-fit['gain']);rows.append({'method':'same_frozen_joint_temporal_causal_factorization','gain':fit['gain'],'data_seed':fit['data_seed'],'horizon':h,'query':pk,'valid':True,'new_fit_parameters':0,'new_training_data':0,**metrics(md,Cd,truth['do_mean'],truth['do_cov'])})
p=pp['artifact_root']/'CAUSAL_FACTORIZATION64_V1.npz';np.savez_compressed(p,**art)
dump(OUT/'CAUSAL_FACTORIZATION64_V1.json',{'utc':utc(),'status':'ALL_SAME_FROZEN_JOINT_CONTROLS_SCORED','rows':rows,'new_model_fits':0,'new_data':0,'artifact':str(p),'artifact_sha256':sha(p),'original_evaluation_sha256':sha(OUT/'FIT_AND_GAUSSIAN_EVALUATION64_V1.json'),'scope':'Classical temporal Gaussian factorization of retained full joint; not a proposed new mechanism or a conditional-only generator guarantee.'});print(json.dumps({'status':'ALL_SAME_FROZEN_JOINT_CONTROLS_SCORED','rows':len(rows),'new_fits':0}))
