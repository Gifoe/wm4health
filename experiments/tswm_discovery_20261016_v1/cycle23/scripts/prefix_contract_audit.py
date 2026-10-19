"""Native generator decomposition; prospective branch-prefix audit, no training."""
import runtime_profile
import hashlib,json,os,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'cycle17/runtime_dependencies'))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'outputs/runtime_numba_cache')
import numpy as np
import torch
SOURCE=BASE/'cycle16/upstream/sysid_benchmarks'
sys.path.insert(0,str(SOURCE))
from lti import drss_matrices,dlsim,nn_fun
from dataset import WHDataset
OUT=ROOT/'outputs/prefix_contract'
T=930;BURN=200;CUTS=[400,744]
ARMS=['raw_causal','native_internal_only','native_complete','fixed_reference']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def draw(rng):
    w1=rng.normal(size=(32,1))*5/3;b1=rng.normal(size=(1,32))
    w2=rng.normal(size=(1,32))/np.sqrt(32);b2=rng.normal(size=(1,1))
    args=dict(inputs=1,outputs=1,rng=rng,mag_range=(.5,.97),phase_range=(0,np.pi/2))
    g1=drss_matrices(states=rng.integers(1,6),strictly_proper=True,**args)
    g2=drss_matrices(states=rng.integers(1,6),strictly_proper=False,**args)
    return dict(w1=w1,b1=b1,w2=w2,b2=b2,**{'g1_'+k:v for k,v in zip('ABCD',g1)},**{'g2_'+k:v for k,v in zip('ABCD',g2)})
def linear(p,which,u):return dlsim(*(p[which+'_'+k] for k in 'ABCD'),u)
def nonlinear(p,y):return nn_fun(y,p['w1'],p['b1'],p['w2'],p['b2'])
def moments(x):return x.mean(axis=0),x.std(axis=0)+1e-6
def calibration(p,u):
    v=linear(p,'g1',u);m,s=moments(v[BURN:]);z=linear(p,'g2',nonlinear(p,(v-m)/s))
    um,us=moments(u[BURN:]);ym,ys=moments(z[BURN:])
    return dict(internal_mean=m,internal_scale=s,u_mean=um,u_scale=us,y_mean=ym,y_scale=ys)
def simulate(p,u,arm,c=None):
    v=linear(p,'g1',u)
    if arm in ('native_internal_only','native_complete'):
        m,s=moments(v[BURN:]);v=(v-m)/s
    elif arm=='fixed_reference':v=(v-c['internal_mean'])/c['internal_scale']
    y=linear(p,'g2',nonlinear(p,v))[BURN:];u=u[BURN:].copy()
    if arm=='native_complete':
        m,s=moments(u);u=(u-m)/s;m,s=moments(y);y=(y-m)/s
    elif arm=='fixed_reference':
        u=(u-c['u_mean'])/c['u_scale'];y=(y-c['y_mean'])/c['y_scale']
    return y,u
def main():
    start=time.perf_counter();assert not OUT.exists();OUT.mkdir(parents=True)
    manifest=json.loads((BASE/'cycle16/outputs/source/metaid/manifest.json').read_text())
    for row in manifest['files']:assert sha(SOURCE/row['path'])==row['sha256'],row['path']
    replay=[]
    for k in range(8):
        ss=202623100+k*10;ii=ss+1;nn=ss+2
        p=draw(np.random.default_rng(ss));u=np.random.default_rng(ii).normal(size=(T+BURN,1))
        e=np.random.default_rng(nn).normal(size=(T,1))*.01
        for normalize,arm in [(False,'native_internal_only'),(True,'native_complete')]:
            ds=WHDataset(nx=5,seq_len=T,normalize=normalize,system_seed=ss,input_seed=ii,noise_seed=nn,mag_range=(.5,.97),phase_range=(0,np.pi/2))
            yy,uu=next(iter(ds));a,b=simulate(p,u,arm);a=(a+e).astype('float32');b=b.astype('float32')
            assert np.array_equal(yy.numpy(),a) and np.array_equal(uu.numpy(),b),(k,normalize)
            replay.append(dict(system_seed=ss,normalize=normalize,y_max_error=float(np.max(np.abs(yy.numpy()-a))),u_max_error=float(np.max(np.abs(uu.numpy()-b)))))
    sr=np.random.default_rng(20262301);ur=np.random.default_rng(20262302)
    vr=np.random.default_rng(20262303);cr=np.random.default_rng(20262304);nr=np.random.default_rng(20262305)
    rows=[];files=[]
    for system in range(128):
        p=draw(sr);u=ur.normal(size=(T+BURN,1));alt=vr.normal(size=(T+BURN,1));ref=cr.normal(size=(T+BURN,1));noise=nr.normal(size=(T,1))*.01
        c=calibration(p,ref)
        arrays={**p,**{'cal_'+k:v for k,v in c.items()},'original_input':u,'alternative_input':alt,'reference_input':ref,'noise':noise}
        for arm in ARMS:
            y0,u0=simulate(p,u,arm,c);y0+=noise
            arrays[arm+'_original_y']=y0;arrays[arm+'_original_u']=u0
            for cut in CUTS:
                branch=u.copy();branch[BURN+cut:]=alt[BURN+cut:]
                y1,u1=simulate(p,branch,arm,c);y1+=noise
                dy=y1[:cut]-y0[:cut];du=u1[:cut]-u0[:cut]
                scale=float(y0.std());rel=float(np.sqrt(np.mean(dy**2))/(scale+1e-12))
                row=dict(system=system,arm=arm,cut=cut,relative_prefix_y_rms=rel,
                    prefix_y_max=float(np.max(np.abs(dy))),prefix_u_max=float(np.max(np.abs(du))),
                    prefix_u_rms=float(np.sqrt(np.mean(du**2))),original_y_std=scale,
                    first_changed_input_output_difference=float(abs(y1[cut,0]-y0[cut,0])))
                rows.append(row);arrays[f'{arm}_{cut}_y']=y1;arrays[f'{arm}_{cut}_u']=u1
        f=OUT/f'system_{system:03d}.npz';np.savez_compressed(f,**arrays);files.append(dict(path=f.name,sha256=sha(f),bytes=f.stat().st_size))
        if (system+1)%32==0:print(json.dumps(dict(systems=system+1,elapsed=time.perf_counter()-start)),flush=True)
    summary=[]
    for arm in ARMS:
        for cut in CUTS:
            rr=[r for r in rows if r['arm']==arm and r['cut']==cut];v=np.array([r['relative_prefix_y_rms'] for r in rr])
            summary.append(dict(arm=arm,cut=cut,systems=len(rr),relative_y_rms_median=float(np.median(v)),
                relative_y_rms_mean=float(v.mean()),relative_y_rms_p95=float(np.quantile(v,.95)),
                fraction_relative_above_1e8=float(np.mean(v>1e-8)),prefix_y_max=max(r['prefix_y_max'] for r in rr),
                prefix_u_rms_median=float(np.median([r['prefix_u_rms'] for r in rr]))))
    for row in manifest['files']:assert sha(SOURCE/row['path'])==row['sha256']
    result=dict(protocol_sha256=sha(ROOT/'PROTOCOL.md'),script_sha256=sha(pathlib.Path(__file__)),source_revision=manifest['revision'],source_manifest_sha256=sha(BASE/'cycle16/outputs/source/metaid/manifest.json'),
        native_replay=replay,summary=summary,system_rows=rows,files=files,elapsed_seconds=time.perf_counter()-start,
        seeds=dict(systems=20262301,input=20262302,alternative=20262303,reference=20262304,noise=20262305),
        semantics='Fresh prospective audit; normalization repair is an ordinary control, not a proposed new method.',
        no_training=True,accepted_novel_directions=0,environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__))
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['system_rows','files']}),flush=True)
if __name__=='__main__':main()
