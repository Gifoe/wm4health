from common89_v1 import *
import itertools,time,math,torch,numpy as np
from torch.nn import functional as F
from scipy.special import logsumexp

torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
PI=np.array([.4,.2,.1,.08,.07,.05,.04,.035,.025],np.float64)
ARMS=['gaussian_distinct','gaussian_independent','count_independent']

def population(K,policy):
    ids=np.array(list(itertools.product(range(9),repeat=K)),np.int64);same=np.all(ids==ids[:,:1],axis=1);distinct=np.array([len(set(r))==K for r in ids]);proposal=PI[ids].prod(1)
    positive=np.where(same,PI[ids[:,0]],0.)
    negative=proposal.copy()
    if policy=='distinct':negative*=distinct;negative/=negative.sum()
    assert abs(positive.sum()-1)<1e-12 and abs(negative.sum()-1)<1e-12
    mixture=.5*(positive+negative);post=np.divide(positive,positive+negative,out=np.zeros_like(positive),where=(positive+negative)>0)
    lower=-.5*np.sum(positive*np.log(np.maximum(post,1e-300))+negative*np.log(np.maximum(1-post,1e-300)))
    return dict(K=K,ids=ids,same=same,proposal=proposal,positive=positive,negative=negative,mixture=mixture,post=post,bayes_bce=float(lower))

POPS={(K,policy):population(K,policy) for K in [2,3,4] for policy in ['distinct','independent']}

class GaussianProduct(torch.nn.Module):
    def __init__(self):
        super().__init__();self.prior_mean=torch.nn.Parameter(torch.zeros(32));self.prior_logstd=torch.nn.Parameter(torch.zeros(32));self.precision_raw=torch.nn.Parameter(torch.full((9,32),-1.));self.natural_delta=torch.nn.Parameter(torch.randn(9,32)*.2)
    def forward(self,ids):
        K=ids.shape[1];prior_prec=torch.exp(-2*self.prior_logstd.clamp(-4,2));prior_nat=self.prior_mean*prior_prec
        prec=prior_prec+F.softplus(self.precision_raw[ids]);nat=prior_nat+self.natural_delta[ids]
        combined_prec=prec.sum(1)-(K-1)*prior_prec;combined_nat=nat.sum(1)-(K-1)*prior_nat
        return .5*(prec.log().sum(1)-(K-1)*prior_prec.log()-combined_prec.log()+combined_nat.square()/combined_prec-(nat.square()/prec).sum(1)+(K-1)*prior_nat.square()/prior_prec).sum(-1)

class CountMLP(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(10,128),torch.nn.SiLU(),torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,1))
    def forward(self,ids):
        counts=F.one_hot(ids,9).float().sum(1);k=torch.full((len(ids),1),ids.shape[1],dtype=counts.dtype)
        return self.net(torch.cat([counts,k],-1)).squeeze(-1)

def draw(K,labels,policy,rpos,rneg):
    n=len(labels);ids=np.empty((n,K),np.int64);ids[:]=rpos.choice(9,size=(n,1),p=PI)
    ix=np.flatnonzero(~labels)
    if policy=='independent':ids[ix]=rneg.choice(9,size=(len(ix),K),p=PI)
    else:
        todo=ix
        while len(todo):
            candidates=rneg.choice(9,size=(max(64,2*len(todo)),K),p=PI);good=candidates[np.array([len(set(r))==K for r in candidates])];take=min(len(todo),len(good));ids[todo[:take]]=good[:take];todo=todo[take:]
    return torch.as_tensor(ids),torch.as_tensor(labels.astype(np.float32))

@torch.no_grad()
def val(model,policy):
    model.eval();rows=[]
    for K in [2,3,4]:
        p=POPS[K,policy];ids=torch.as_tensor(p['ids']);score=torch.cat([model(ids[b:b+1024]) for b in range(0,len(ids),1024)]).numpy().astype(np.float64)
        bce=float(.5*np.sum(p['positive']*np.logaddexp(0,-score)+p['negative']*np.logaddexp(0,score)))
        rows.append(dict(K=K,bce=bce,bayes_bce=p['bayes_bce'],excess_bce=bce-p['bayes_bce']))
    return dict(bce=float(np.mean([r['bce'] for r in rows])),excess_bce=float(np.mean([r['excess_bce'] for r in rows])),by_K=rows)

@torch.no_grad()
def coupling(model):
    model.eval();rows=[];marginals={}
    for K in [2,3,4]:
        p=POPS[K,'independent'];ids=torch.as_tensor(p['ids']);score=torch.cat([model(ids[b:b+1024]) for b in range(0,len(ids),1024)]).numpy().astype(np.float64)
        for gamma in [0.,.01,.1,1.]:
            logw=np.log(p['proposal'])+gamma*score;weights=np.exp(logw-logsumexp(logw));marginal=np.stack([np.bincount(p['ids'][:,k],weights=weights,minlength=9) for k in range(K)])
            marginal_tv=.5*abs(marginal-PI).sum(1)
            rows.append(dict(K=K,gamma=gamma,coherence=float(weights[p['same']].sum()),anchor_marginal=marginal[0].tolist(),anchor_tv=float(marginal_tv[0]),max_branch_tv=float(marginal_tv.max())))
            marginals[K,gamma]=marginal[0]
    drift=[dict(gamma=g,anchor_tv_K2_to_K4=float(.5*abs(marginals[2,g]-marginals[4,g]).sum())) for g in [0.,.01,.1,1.]]
    return rows,drift

def fit(arm,seed):
    policy='distinct' if arm=='gaussian_distinct' else 'independent';torch.manual_seed(seed);model=CountMLP() if arm.startswith('count') else GaussianProduct();opt=torch.optim.Adam(model.parameters(),lr=.01)
    rbase=np.random.default_rng(seed);rpos=np.random.default_rng(seed+100);rneg=np.random.default_rng(seed+200);trace=[];best=float('inf');started=time.perf_counter();dest=Path(STORAGE['checkpoint_root'])/'cycle89/critic_stress_v1';dest.mkdir(parents=True,exist_ok=True);selected=dest/f'{arm}_{seed}_selected.pt';assert not selected.exists()
    for step in range(10001):
        if step in [0,1000,3000,6000,10000]:
            value=val(model,policy);trace.append(dict(step=step,**value))
            if value['bce']<best:
                best=value['bce'];torch.save(dict(model=model.state_dict(),step=step,arm=arm,seed=seed),selected)
            log(dict(stage='critic_population_validation',arm=arm,seed=seed,step=step,**value,elapsed_seconds=time.perf_counter()-started))
        if step==10000:break
        K=int(rbase.choice([2,3,4]));labels=rbase.random(256)<.5;ids,y=draw(K,labels,policy,rpos,rneg);model.train();opt.zero_grad(set_to_none=True);s=model(ids);loss=F.binary_cross_entropy_with_logits(s,y);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(norm);opt.step()
    saved=torch.load(selected,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);rows,drift=coupling(model)
    return dict(arm=arm,seed=seed,policy=policy,parameters=sum(p.numel() for p in model.parameters()),updates=10000,selected_step=saved['step'],selected_population=val(model,policy),trace=trace,coupling=rows,query_drift=drift,checkpoint_path=str(selected),checkpoint_sha256=sha(selected),elapsed_seconds=time.perf_counter()-started)

def exact_controls():
    rows=[]
    for K in [2,3,4]:
        p=POPS[K,'independent'];same=p['same'];hard=p['proposal']*same;hard/=hard.sum();correct=p['positive'];
        for name,w in [('independent',p['proposal']),('hard_agreement',hard),('exact_bayes_ratio',correct)]:
            marginal=np.bincount(p['ids'][:,0],weights=w,minlength=9);tv=float(.5*abs(marginal-PI).sum())
            if name=='exact_bayes_ratio':assert tv<1e-12
            rows.append(dict(control=name,K=K,coherence=float(w[same].sum()),anchor_marginal=marginal.tolist(),anchor_tv=tv))
    return rows

def main():
    assert not (O/'CRITIC_STRESS89_V1.json').exists();models=[]
    for seed in [2026896,2026897,2026898]:
        for arm in ARMS:models.append(fit(arm,seed))
    result=dict(utc=utc(),status='FINITE_ORACLE_CRITIC_STRESS_COMPLETE_NOT_NATIVE_WORLD_MODEL_REPRODUCTION',prior=PI.tolist(),worlds=9,models=models,exact_controls=exact_controls(),privileged_oracle_mechanism_ID_inputs=True,learned_stochastic_world_models_executed=0,posterior_calibration_metric_used_in_selection=False,terminal_Gibbs_not_continuous_flow_guidance=True,protocol_sha256=sha(C/'QUERY_SET_STRESS_PROTOCOL89_V1.md'),accepted_research_directions=0,overall_goal='ACTIVE_UNACHIEVED')
    write(O/'CRITIC_STRESS89_V1.json',result);log(dict(stage='critic_stress_closed',fits=len(models),goal='ACTIVE_UNACHIEVED'))

if __name__=='__main__':main()
