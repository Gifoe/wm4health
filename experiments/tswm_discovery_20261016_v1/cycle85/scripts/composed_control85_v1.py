from common85_v1 import *
import common85_v1 as common
import run85_v1 as old
import weak_control85_v1 as weak
import attribution85_v1 as attribution
from scipy.optimize import least_squares

STAGE=C/'outputs'/'composed_control_v1';STAGE.mkdir(parents=True,exist_ok=True);common.O=STAGE
DEST=Path(CONFIG['artifact_root'])/'cycle85'/'composed_control_v1';DEST.mkdir(parents=True,exist_ok=True)

def affine_fit(x,y):return np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
def affine(x,w):return np.column_stack([np.ones(len(x)),x])@w
def kernel(x,centers,scale,bw):return np.exp(-np.sum(((x[:,None]-centers[None])/scale)**2,axis=-1)/(2*bw*bw))
def fit_sensor(s,y,kind,bw=None):
    if kind=='oracle_family':
        w=affine_fit(s,y);p0=np.r_[w[1:].T.ravel(),w[0],0.]
        def residual(p):return ((s+p[6]*np.tanh(s))@p[:4].reshape(2,2).T+p[4:6]-y).ravel()
        r=least_squares(residual,p0,bounds=(np.r_[np.full(6,-np.inf),-.5],np.r_[np.full(6,np.inf),.5]),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=2000)
        assert r.success and abs(np.linalg.det(r.x[:4].reshape(2,2)))>.01
        return dict(kind=kind,p=r.x.tolist(),anchor_residual_rms=float(np.sqrt(np.mean(residual(r.x)**2))))
    model=dict(kind=kind,bw=bw)
    for name,x,z in [('forward',s,y),('inverse',y,s)]:
        w=affine_fit(x,z);part=dict(w=w.tolist())
        if kind=='rbf_residual':
            scale=np.maximum(x.std(axis=0),.05);k=kernel(x,x,scale,bw);coef=np.linalg.solve(k+.01*np.eye(len(x)),z-affine(x,w));part.update(centers=x.tolist(),scale=scale.tolist(),coef=coef.tolist())
        model[name]=part
    return model
def mapping(x,model,inverse=False):
    shape=x.shape;xx=np.asarray(x).reshape(-1,2)
    if model['kind']=='oracle_family':
        p=np.array(model['p']);m=p[:4].reshape(2,2)
        if inverse:
            z=(xx-p[4:6])@np.linalg.inv(m).T;v=z.copy()
            for _ in range(16):v-=(v+p[6]*np.tanh(v)-z)/(1+p[6]*(1-np.tanh(v)**2))
            out=v
        else:out=(xx+p[6]*np.tanh(xx))@m.T+p[4:6]
    else:
        part=model['inverse' if inverse else 'forward'];out=affine(xx,np.array(part['w']))
        if model['kind']=='rbf_residual':out+=kernel(xx,np.array(part['centers']),np.array(part['scale']),model['bw'])@np.array(part['coef'])
    return out.reshape(shape)
def simulate(y0,u,coef,spec,sensor):return mapping(weak.simulate(mapping(y0,sensor,True),u,coef,'polynomial3',spec),sensor)
def source(seed):
    if seed!=2026854:return attribution.source_data(seed)
    x,u=generate('brusselator',seed,1000,128,64);xv,uv=generate('brusselator',seed,2000,16,128);mu=x[:,:-1].mean(axis=(0,1));scale=x[:,:-1].std(axis=(0,1));sd=dict(y=(x-mu)/scale,u=u,val_y=(xv-mu)/scale,val_u=uv,mu=mu,scale=scale)
    p=DEST/f'brusselator_{seed}_source_v1.npz';old.save_npz(p,**sd);return sd,p
def main():
    log(dict(kind='composed_start',pid=os.getpid(),protocol_sha256=sha(C/'COMPOSED_CONTROL85_V1.md')));results=[];source_checks=[]
    for seed in SEEDS+[2026853,2026854]:
        sd,source_path=source(seed);spec=old.dictionary_specs(4);options=[]
        for length in [8,16]:
            a,b=weak.weak_rows(sd['y'],sd['u'],'polynomial3',spec,length);n=a.shape[1];coef=np.linalg.lstsq(np.vstack([a,np.sqrt(1e-6)*np.eye(n)]),np.vstack([b,np.zeros((n,2))]),rcond=None)[0]
            pred=weak.simulate(sd['val_y'][:,0],sd['val_u'],coef,'polynomial3',spec);s32=finite_score(float(np.median(old.path_scores(pred,sd['val_y'],32))));s128=finite_score(float(np.median(old.path_scores(pred,sd['val_y'],128))));qualified=s32<=.15 and s128<=.35
            source_checks.append(dict(seed=seed,window=length,median32=s32,median128=s128,qualified=qualified))
            if qualified:options.append((length,coef))
        assert options
        for change in SHIFT_CASES:
            base=attribution.allocations(seed,sd,change)['long'];sensors=[fit_sensor(base['anchor_source'],base['anchor_target'],'affine')]+[fit_sensor(base['anchor_source'],base['anchor_target'],'rbf_residual',bw) for bw in [.5,1.,2.,4.]]+[fit_sensor(base['anchor_source'],base['anchor_target'],'oracle_family')]
            records=[];coefs={}
            for si,sensor in enumerate(sensors):
                transformed=mapping(base['y'],sensor,True)
                for length,cs in options:
                    a,b=weak.weak_rows(transformed,base['u'],'polynomial3',spec,length);mask=np.sum(spec['exponents'],axis=1)<=1
                    for lam in [None,1e-6,.001,.1,10.]:
                        coef=cs.copy()
                        if lam is not None:
                            design=a[:,mask];coef[mask]+=np.linalg.solve(design.T@design+lam*np.eye(int(mask.sum())),design.T@(b-a@cs))
                        pred=simulate(base['val_y'][:,0],base['val_u'],coef,spec,sensor);key=f'{si}_{length}_{lam}';coefs[key]=coef
                        records.append(dict(key=key,sensor_index=si,sensor_kind=sensor['kind'],bandwidth=sensor.get('bw'),source_window=length,ridge=lam,val32=finite_score(nrmse(pred,base['val_y'],32)),val128=finite_score(nrmse(pred,base['val_y'],128))))
            selected={}
            for group in ['ordinary','oracle']:
                subset=[r for r in records if (r['sensor_kind']=='oracle_family')==(group=='oracle')]
                for sel in ['val32','val128']:selected[group+'_'+sel]=min(subset,key=lambda r:r[sel])
            tag=f'brusselator_{seed}_{change}';calibration=DEST/f'{tag}_calibration_v1.npz';old.save_npz(calibration,**base)
            coefpath=DEST/f'{tag}_coefficients_v1.npz';old.save_npz(coefpath,**{label+'_coef':coefs[r['key']] for label,r in selected.items()},**spec)
            seal=STAGE/f'SELECTION_SEALED85_COMPOSED_{tag}_V1.json';write(seal,dict(utc=utc(),seed=seed,case=change,selected=selected,candidates=records,sensors=sensors,source_path=str(source_path),source_sha256=sha(source_path),calibration_path=str(calibration),calibration_sha256=sha(calibration),coefficient_path=str(coefpath),coefficient_sha256=sha(coefpath),eval_generated=False,unopened_seed=seed==2026854,oracle_privileged_function_family=True))
            x,u=generate('brusselator',seed,6000,8,128,change in ['dynamics','joint'],paired=True);truth=observe((x-sd['mu'])/sd['scale'],change in ['observation','joint']);rng=np.random.default_rng(np.random.SeedSequence([seed,10000]));initial=truth[:,0]+np.repeat(rng.normal(0,.01,(8,2)),2,axis=0)
            previous=C/'outputs'/'attribution_v1'/f'CASE85_ATTRIBUTION_{tag}_V1.json'
            if previous.exists():
                d=np.load(json.loads(previous.read_text())['array_path'],allow_pickle=False);assert np.array_equal(d['reference'],truth) and np.array_equal(d['actions'],u) and np.array_equal(d['initial_observation'],initial)
            else:assert seed==2026854
            saved=dict(reference=truth,actions=u,initial_observation=initial,physical_reference=x,mu=sd['mu'],scale=sd['scale']);rows=[];summaries={}
            for label,r in selected.items():
                pred=simulate(initial,u,coefs[r['key']],spec,sensors[r['sensor_index']]);saved[label+'_prediction']=pred;rr,ss=weak.score(pred,truth);rows.extend(dict(method=label,**z) for z in rr);summaries[label]=ss
            path=DEST/f'{tag}_evaluation_v1.npz';old.save_npz(path,**saved);record=dict(utc=utc(),seed=seed,case=change,summary=summaries,rows=rows,array_path=str(path),array_sha256=sha(path),selection_seal_sha256=sha(seal),unopened_seed=seed==2026854,new_method=False,oracle_privileged_function_family=True)
            write(STAGE/f'CASE85_COMPOSED_{tag}_V1.json',record);results.append({k:v for k,v in record.items() if k!='rows'});log(dict(kind='composed_case_closed',seed=seed,case=change,summary=summaries))
    write(STAGE/'RESULT85_COMPOSED_V1.json',dict(status='EXISTING_COMPOSITION_COMPLETE',utc=utc(),cases=results,source_checks=source_checks,protocol_sha256=sha(C/'COMPOSED_CONTROL85_V1.md'),new_method=False,qualified_final_directions=0))
if __name__=='__main__':main()
