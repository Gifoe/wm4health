from csdi_core89_v1 import *
import itertools,time,warnings
from scipy.special import logsumexp
from scipy.cluster.vq import kmeans2

EXPONENTS=np.array([e for degree in range(6) for e in itertools.product(range(6),repeat=3) if sum(e)==degree])
assert len(EXPONENTS)==56

def polynomial(x):
    with np.errstate(over='raise',invalid='raise'):
        return np.prod(np.power(np.asarray(x,np.float64)[...,None,:],EXPONENTS),axis=-1)

def statistics(states,actions,state_scale,feature_scale=None):
    future=states[:,CONTEXT:];previous=np.concatenate([states[:,CONTEXT-1:CONTEXT],future[:,:-1]],axis=1)
    inputs=np.concatenate([previous,actions[:,CONTEXT:,None]],axis=-1)
    phi=polynomial(inputs)
    if feature_scale is None:feature_scale=np.maximum(np.sqrt(np.mean(phi**2,axis=(0,1))),1e-3)
    phi/=feature_scale
    y=(future-previous)/state_scale
    xx=np.einsum('ntf,ntg->nfg',phi,phi,optimize=True);xy=np.einsum('ntf,ntd->nfd',phi,y,optimize=True);yy=np.sum(y*y,axis=(1,2))
    lin=np.concatenate([np.ones((*inputs.shape[:2],1)),inputs],axis=-1)
    lxx=np.einsum('ntf,ntg->nfg',lin,lin,optimize=True)+np.eye(4)[None]*1e-6
    lxy=np.einsum('ntf,ntd->nfd',lin,y,optimize=True)
    descriptors=np.linalg.solve(lxx,lxy).reshape(len(states),-1)
    return dict(xx=xx,xy=xy,yy=yy,descriptors=descriptors,feature_scale=feature_scale)

def residual(stats,beta):
    out=stats['yy'][:,None]-2*np.einsum('nfd,kfd->nk',stats['xy'],beta,optimize=True)+np.einsum('kfd,nfg,kgd->nk',beta,stats['xx'],beta,optimize=True)
    # Cancellation can produce a tiny negative SSE near a deterministic fit.
    assert out.min()>-1e-5,float(out.min())
    return np.maximum(out,0.)

def density(stats,beta,pi,var):
    sse=residual(stats,beta)
    lp=np.log(pi)[None]-64*np.log(2*np.pi*var)[None]-sse/(2*var[None])
    normalizer=logsumexp(lp,axis=1)
    return np.exp(lp-normalizer[:,None]),float(-normalizer.mean()),sse

def initialize(descriptors):
    x=(descriptors-descriptors.mean(0))/np.maximum(descriptors.std(0),1e-6)
    best=None
    for trial in range(20):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            centers,labels=kmeans2(x,9,iter=30,minit='++',missing='warn',seed=20268924+trial)
        if len(np.unique(labels))!=9:continue
        value=float(np.sum((x-centers[labels])**2))
        if best is None or value<best[0]:best=(value,labels)
    assert best is not None
    return np.eye(9)[best[1]],best[0]

def fit_family(train,cal,scale,family):
    a=statistics(train['states'],train['actions'],scale);b=statistics(cal['states'],cal['actions'],scale,a['feature_scale'])
    responsibility,inertia=initialize(a['descriptors']);trace=[];best=None;started=time.perf_counter()
    for iteration in range(1,41):
        mass=responsibility.sum(0);assert (mass>1e-8).all()
        xx=np.einsum('nk,nfg->kfg',responsibility,a['xx'],optimize=True)
        xy=np.einsum('nk,nfd->kfd',responsibility,a['xy'],optimize=True)
        xx+=np.eye(56)[None]*(1e-6*mass*64)[:,None,None]
        beta=np.linalg.solve(xx,xy)
        sse=residual(a,beta);var=np.maximum(np.sum(responsibility*sse,axis=0)/(mass*64*2),1e-7)
        pi=np.maximum(mass/len(responsibility),1e-6);pi/=pi.sum()
        responsibility,nll,_=density(a,beta,pi,var)
        if iteration in [1,5,10,20,40]:
            _,cnll,_=density(b,beta,pi,var)
            row=dict(family=family,iteration=iteration,train_trajectory_nll=nll,cal_trajectory_nll=cnll,mass=mass.tolist(),elapsed_seconds=time.perf_counter()-started)
            trace.append(row);log(dict(stage='persistent_mixture_fit',**row))
            if best is None or cnll<best['cal_trajectory_nll']:best=dict(beta=beta.copy(),pi=pi.copy(),var=var.copy(),feature_scale=a['feature_scale'].copy(),iteration=iteration,cal_trajectory_nll=cnll)
    return best,trace,inertia

def rollout(model,actions,state_scale):
    n=len(actions);state=np.zeros((n,9,2),np.float64);out=np.zeros((n,9,LENGTH,2))
    for t in range(CONTEXT,LENGTH):
        inputs=np.concatenate([state,np.broadcast_to(actions[:,None,t,None],(n,9,1))],axis=-1)
        phi=polynomial(inputs)/model['feature_scale']
        increment=np.einsum('nkf,kfd->nkd',phi,model['beta'],optimize=True)*state_scale
        state=state+increment
        if not np.isfinite(state).all():raise FloatingPointError('All rollout failures must be retained, not filtered')
        out[:,:,t]=state
    return out

def main():
    started=time.perf_counter();assert not (O/'PERSISTENT_MIXTURE_FIT89_V1.json').exists()
    m=read(O/'CSDI_DATA89_V1.json');paths={r['split']:r for r in m['files']};splits={}
    for split in ['train','cal']:
        assert sha(paths[split]['path'])==paths[split]['sha256']
        with np.load(paths[split]['path']) as f:
            # Ground-truth world labels and parameters deliberately not accessed.
            splits[split]={k:f[k].copy() for k in ['states','actions','family','scale','center']}
    scale=splits['train']['scale'];models=[];traces=[];rows=[]
    for family in [0,1]:
        subsets={sp:{k:splits[sp][k][splits[sp]['family']==family] for k in ['states','actions']} for sp in ['train','cal']}
        model,trace,inertia=fit_family(subsets['train'],subsets['cal'],scale,family);models.append(model);traces.extend(trace)
        rows.append(dict(family=family,selected_iteration=model['iteration'],cal_trajectory_nll=model['cal_trajectory_nll'],mixture_probabilities=model['pi'].tolist(),kmeans_train_inertia=inertia))
    dest=Path(STORAGE['checkpoint_root'])/'cycle89/persistent_mixture_v1';dest.mkdir(parents=True,exist_ok=True);p=dest/'selected.npz'
    arrays={f'f{i}_{k}':model[k] for i,model in enumerate(models) for k in ['beta','pi','var','feature_scale']}
    with p.open('xb') as f:np.savez_compressed(f,**arrays,state_scale=scale,state_center=splits['train']['center'],exponents=EXPONENTS)
    receipt=dict(utc=utc(),status='ORDINARY_PERSISTENT_MIXTURE_FIT_COMPLETE_EVAL_PENDING',families=rows,trace=traces,checkpoint_path=str(p),checkpoint_sha256=sha(p),protocol_sha256=sha(C/'PERSISTENT_MIXTURE_CONTROL89_V1.md'),source_data_manifest_sha256=sha(O/'CSDI_DATA89_V1.json'),hidden_mechanism_labels_read_by_fit=False,evaluation_queries_read=False,known_number_of_components=9,degree=5,parameters=2*(9*56*2+9+9),new_method_claim=False,elapsed_seconds=time.perf_counter()-started,accepted_research_directions=0)
    write(O/'PERSISTENT_MIXTURE_FIT89_V1.json',receipt);log(dict(stage='persistent_mixture_fit_closed',families=rows,elapsed_seconds=receipt['elapsed_seconds']))

if __name__=='__main__':main()
