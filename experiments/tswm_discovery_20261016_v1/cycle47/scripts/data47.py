"""Fresh latent Markov diffusion/marked Cox data. Truth is auditor-only."""
from common47 import *
from scipy.special import ndtr
def probabilities(interior,mean,scale):
 cdf=ndtr((np.asarray(interior)-np.asarray(mean)[...,None])/scale);p=np.diff(np.concatenate([np.zeros_like(cdf[...,:1]),cdf,np.ones_like(cdf[...,:1])],-1),axis=-1);p=np.maximum(p,1e-12);return p/p.sum(-1,keepdims=True)
def true_components(case):
 grid=np.linspace(-3.5,3.5,K);mid=(grid[:-1]+grid[1:])/2;drift=-.6*grid if case=='ou' else 1.2*np.tanh(1.7*grid)-.75*grid;sig=.7 if case=='ou' else .55;P=probabilities(mid,grid+DT*drift,sig*math.sqrt(DT));rate=.3+3.7/(1+np.exp(-2*grid));mark=probabilities(np.array([-1.5,-1.,-.5,0.,.5,1.,1.5]),grid,OBS);prior=probabilities(mid,0.,.8);return grid,P,rate,mark,prior
def generate(case,n,draw_seed,rhos):
 grid,P,rate,mark,prior=true_components(case);gen=np.random.default_rng(draw_seed);states=np.empty((n,T+1),dtype='int16');states[:,0]=gen.choice(K,size=n,p=prior)
 for t in range(T):states[:,t+1]=(gen.random(n)[:,None]>np.cumsum(P[states[:,t]],-1)).sum(-1)
 initial=(gen.random(n)[:,None]>np.cumsum(mark[states[:,0]],-1)).sum(-1).astype('int64');dense=(grid[states[:,HIST:T]]+OBS*gen.normal(size=(n,FUTURE))).astype('float32');results=[]
 for rho in rhos:
  rho=np.broadcast_to(np.asarray(rho),(n,));mu=DT*rho[:,None,None]*rate[states[:,:T],None]*mark[states[:,:T]];counts=gen.poisson(mu).astype('int16');results.append({'initial':initial,'counts':counts,'rho':rho.astype('float32')})
 return results,{'state_index':states,'scheduled_future':dense}
def main():
 out=R/'data';assert not out.exists();out.mkdir(parents=True);weights=feature_bank(512,20264700);np.savez_compressed(out/'feature_bank.npz',weights=weights);records={}
 for j,case in enumerate(CASES):
  caseout=out/case;caseout.mkdir()
  for i,(split,n) in enumerate([('train',512),('val',256)]):
   rho=np.repeat(np.array(RHOS),n//2);results,audit=generate(case,n,20264710+j*100+i,[rho]);values=results[0];np.savez_compressed(caseout/(split+'.npz'),**values)
   if split=='train':
    phis=empirical_features(values['initial'],values['counts'],weights);mu=np.stack([phis[rho==r].mean(0) for r in RHOS]);var=np.stack([phis[rho==r].var(0,ddof=1) for r in RHOS]);np.savez_compressed(caseout/'train_features.npz',mean=mu,variance=var)
   # TRAIN/VAL true paths are deliberately not saved or used for selection.
  results,audit=generate(case,256,20264780+j*100,[.25,1.,2.]);save={}
  for q,values in enumerate(results):save.update({f'policy{q}_{key}':v for key,v in values.items()})
  np.savez_compressed(caseout/'sealed_eval_observations.npz',**save);np.savez_compressed(caseout/'sealed_eval_auditor_truth.npz',**audit);grid,P,rate,mark,prior=true_components(case);np.savez_compressed(caseout/'auditor_generating_parameters.npz',grid=grid,P=P,rate=rate,mark=mark,prior=prior)
 for p in out.rglob('*'):
  if p.is_file():records[p.relative_to(out).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
 dump(out/'MANIFEST.json',{'status':'FROZEN_NEW_DATA','utc':now(),'files':records,'protocol_sha256':sha(R/'PROTOCOL.md'),'source_sha256':sha(__file__),'unique_EVAL_physical_trajectories_per_case':256,'paired_EVAL_clock_conditions':[.25,1.,2.],'TRAIN_VAL_latent_labels_saved':False,'no_C46_reuse':True});print(json.dumps({'status':'FROZEN_NEW_DATA','files':len(records)}),flush=True)
if __name__=='__main__':main()
