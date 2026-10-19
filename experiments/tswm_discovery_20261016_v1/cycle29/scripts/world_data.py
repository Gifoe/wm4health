"""Fresh multi-system data. Fixed-reference units; never query-dependent normalization."""
import runtime_profile
import hashlib,json,os,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'outputs/runtime_numba_cache')
sys.path.insert(0,str(BASE/'cycle26/vendor/native_control'))
sys.path.insert(0,str(BASE/'cycle16/upstream/sysid_benchmarks'))
from lti import drss_matrices,dlsim,nn_fun
BURN=200;C=128;H=128
OUT=ROOT/'outputs/world_data'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def draw(rng,family):
    args=dict(inputs=1,outputs=1,rng=rng,mag_range=(.5,.97),phase_range=(0,np.pi/2))
    if family=='lti':
        return {k:v for k,v in zip('ABCD',drss_matrices(states=int(rng.integers(1,6)),strictly_proper=False,**args))}
    w1=rng.normal(size=(32,1))*5/3;b1=rng.normal(size=(1,32))
    w2=rng.normal(size=(1,32))/np.sqrt(32);b2=rng.normal(size=(1,1))
    g1=drss_matrices(states=int(rng.integers(1,6)),strictly_proper=True,**args)
    g2=drss_matrices(states=int(rng.integers(1,6)),strictly_proper=False,**args)
    return dict(w1=w1,b1=b1,w2=w2,b2=b2,**{'g1_'+k:v for k,v in zip('ABCD',g1)},**{'g2_'+k:v for k,v in zip('ABCD',g2)})
def lin(p,prefix,u):return dlsim(*(p[prefix+k] for k in 'ABCD'),np.ascontiguousarray(u,dtype='float64'))
def moments(x):return x.mean(0),x.std(0)+1e-6
def reference(p,family,u):
    if family=='lti':y=lin(p,'',u);c={}
    else:
        v=lin(p,'g1_',u);m,s=moments(v[BURN:]);y=lin(p,'g2_',nn_fun((v-m)/s,p['w1'],p['b1'],p['w2'],p['b2']));c=dict(internal_mean=m,internal_scale=s)
    um,us=moments(u[BURN:]);ym,ys=moments(y[BURN:]);return dict(c,u_mean=um,u_scale=us,y_mean=ym,y_scale=ys)
def simulate(p,family,u,c):
    if family=='lti':y=lin(p,'',u)
    else:
        v=(lin(p,'g1_',u)-c['internal_mean'])/c['internal_scale']
        y=lin(p,'g2_',nn_fun(v,p['w1'],p['b1'],p['w2'],p['b2']))
    return ((y[BURN:]-c['y_mean'])/c['y_scale'])[:,0],((u[BURN:]-c['u_mean'])/c['u_scale'])[:,0]
def signal(rng,n,kind):
    amp=rng.uniform(.5,2.)
    if kind==0:x=rng.normal(size=n)
    elif kind==1:
        z=rng.normal(size=n);x=np.zeros(n)
        for t in range(1,n):x[t]=.8*x[t-1]+.6*z[t]
    elif kind==2:
        dwell=int(rng.integers(3,25));x=np.repeat(rng.choice([-1.,1.],size=(n+dwell-1)//dwell),dwell)[:n]
    else:
        t=np.arange(n);x=np.sin(t*rng.uniform(.02,.6)+rng.uniform(0,2*np.pi))
        x+=np.repeat(rng.uniform(-1,1,size=(n+31)//32),32)[:n]*.5
    return (amp*x)[:,None]
def aslists(mapping):return {k:v.tolist() for k,v in mapping.items()}
def make_one(split,family,index):
    splitseed={'train':202629101,'val':202629102,'test':202629103}[split]
    seed=np.random.SeedSequence([splitseed,int(family=='wh'),index]);rng=np.random.default_rng(seed)
    p=draw(rng,family);ref=rng.normal(size=(4096+BURN,1));c=reference(p,family,ref)
    if split!='test':
        kind=int(rng.integers(4));u=signal(rng,512+BURN,kind);y,u=simulate(p,family,u,c);noise=rng.normal(size=512)*.01
        arrays=dict(u=u.astype('float32'),y=(y+noise).astype('float32'),clean=y.astype('float32'))
    else:
        raw=rng.normal(size=(BURN+C+H,1));noise=rng.normal(size=C+H)*.01;arrays={};yy=[];uu=[];clean=[]
        for plan in range(4):
            b=raw.copy()
            if plan==1:b[BURN+C:]=0
            elif plan==2:b[BURN+C:]=1.5
            elif plan==3:b[BURN+C:,0]=1.5*np.where((np.arange(H)//16)%2==0,1.,-1.)
            yc,uc=simulate(p,family,b,c);yy.append(yc+noise);uu.append(uc);clean.append(yc)
        arrays=dict(u=np.array(uu,dtype='float32'),y=np.array(yy,dtype='float32'),clean=np.array(clean,dtype='float32'),raw_u=raw[:,0],reference_u=ref[:,0],noise=noise)
        assert np.max(np.abs(arrays['y'][:,:C]-arrays['y'][0,:C]))==0
        assert np.max(np.abs(arrays['u'][:,:C]-arrays['u'][0,:C]))==0
    assert all(np.isfinite(v).all() for v in arrays.values())
    metadata=dict(split=split,family=family,index=index,seed_components=[splitseed,int(family=='wh'),index],parameters=aslists(p),reference=aslists(c))
    return arrays,metadata

def main():
    import time
    start=time.monotonic();assert not OUT.exists();OUT.mkdir(parents=True)
    source=json.loads((BASE/'cycle16/outputs/source/metaid/manifest.json').read_text())
    for row in source['files']:assert sha(BASE/'cycle16/upstream/sysid_benchmarks'/row['path'])==row['sha256']
    metadata=[];files=[]
    for split,n in [('train',512),('val',64),('test',128)]:
        for family in ['lti','wh']:
            rows=[]
            for i in range(n):
                arrays,meta=make_one(split,family,i);rows.append(arrays);metadata.append(meta)
            keys=list(rows[0]);path=OUT/f'{split}_{family}.npz';np.savez_compressed(path,**{k:np.stack([r[k] for r in rows]) for k in keys})
            files.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size,n_worlds=n,shapes={k:list(rows[0][k].shape) for k in keys}))
            print(json.dumps(dict(phase='data',split=split,family=family,n=n,elapsed=time.monotonic()-start)),flush=True)
    dump(OUT/'systems.json',metadata);files.append(dict(path='systems.json',sha256=sha(OUT/'systems.json'),bytes=(OUT/'systems.json').stat().st_size))
    result=dict(source_revision=source['revision'],source_manifest_sha256=sha(BASE/'cycle16/outputs/source/metaid/manifest.json'),script_sha256=sha(pathlib.Path(__file__)),plan_sha256=sha(ROOT/'MODEL_COMPARISON_PLAN.md'),files=files,elapsed_s=time.monotonic()-start,prefix_max_abs=0,units='independent fixed-reference; per-context model scaling happens later',test_unseen_worlds=True)
    dump(OUT/'manifest.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
