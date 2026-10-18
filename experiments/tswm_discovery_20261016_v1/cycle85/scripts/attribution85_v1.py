from common85_v1 import *
import common85_v1 as common
import run85_v1 as old
import weak_control85_v1 as weak

STAGE=C/'outputs'/'attribution_v1';STAGE.mkdir(parents=True,exist_ok=True);common.O=STAGE
DEST=Path(CONFIG['artifact_root'])/'cycle85'/'attribution_v1';DEST.mkdir(parents=True,exist_ok=True)
ALLSEEDS=SEEDS+[2026853]

def source_data(seed):
    if seed in SEEDS:
        path=Path(CONFIG['data_root'])/'cycle85'/'anchored_screen_v1'/f'brusselator_{seed}'/'source_trainval_v1.npz'
        d=np.load(path,allow_pickle=False);return {k:d[k] for k in d.files},path
    x,u=generate('brusselator',seed,1000,128,64);xv,uv=generate('brusselator',seed,2000,16,128);mu=x[:,:-1].mean(axis=(0,1));scale=x[:,:-1].std(axis=(0,1))
    d=dict(y=(x-mu)/scale,u=u,val_y=(xv-mu)/scale,val_u=uv,mu=mu,scale=scale);p=DEST/f'brusselator_{seed}_source_v1.npz';old.save_npz(p,**d);return d,p

def extend(seed,namespace,paths,u,changed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,namespace]));extra=np.repeat(rng.uniform(-1,1,(len(paths),8,2)),8,axis=1)
    x=[paths[:,-1]]
    for t in range(64):x.append(step(x[-1],extra[:,t],'brusselator',changed))
    return np.concatenate([paths,np.stack(x[1:],axis=1)],axis=1),np.concatenate([u,extra],axis=1)

def allocations(seed,sd,change):
    short=target_data('brusselator',seed,sd['mu'],sd['scale'],change);dyn=change in ['dynamics','joint'];obs=change in ['observation','joint']
    xt,ut=generate('brusselator',seed,3000,4,64,dyn);xv,uv=generate('brusselator',seed,5000,2,64,dyn)
    xl,ul=extend(seed,3001,xt[:2],ut[:2],dyn);xvl,uvl=extend(seed,5001,xv[:1],uv[:1],dyn)
    r=np.random.default_rng(np.random.SeedSequence([seed,7001]));yextra=observe((xl[:,65:]-sd['mu'])/sd['scale'],obs)+r.normal(0,.01,xl[:,65:].shape)
    r=np.random.default_rng(np.random.SeedSequence([seed,7002]));vextra=observe((xvl[:,65:]-sd['mu'])/sd['scale'],obs)+r.normal(0,.01,xvl[:,65:].shape)
    long=dict(y=np.concatenate([short['y'][:2],yextra],axis=1),u=ul,val_y=np.concatenate([short['val_y'][:1],vextra],axis=1),val_u=uvl,anchor_source=short['anchor_source'],anchor_target=short['anchor_target'])
    assert long['y'].shape==(2,129,2) and long['val_y'].shape==(1,129,2)
    assert np.array_equal(long['y'][:,:65],short['y'][:2]) and np.array_equal(long['u'][:,:64],short['u'][:2])
    assert np.array_equal(long['val_y'][:,:65],short['val_y'][:1]) and np.array_equal(long['val_u'][:,:64],short['val_u'][:1])
    return {'short':short,'long':long}

def main():
    log(dict(kind='attribution_start',pid=os.getpid(),protocol_sha256=sha(C/'ATTRIBUTION_PROTOCOL85_V1.md')));results=[];source_checks=[]
    for seed in ALLSEEDS:
        sd,source_path=source_data(seed);spec=old.dictionary_specs(4);options=[]
        for kind in ['polynomial3','rff128']:
            for length in [8,16]:
                a,b=weak.weak_rows(sd['y'],sd['u'],kind,spec,length);n=a.shape[1];cs=np.linalg.lstsq(np.vstack([a,np.sqrt(1e-6)*np.eye(n)]),np.vstack([b,np.zeros((n,2))]),rcond=None)[0]
                pred=weak.simulate(sd['val_y'][:,0],sd['val_u'],cs,kind,spec);s32=finite_score(float(np.median(old.path_scores(pred,sd['val_y'],32))));s128=finite_score(float(np.median(old.path_scores(pred,sd['val_y'],128))))
                record=dict(seed=seed,kind=kind,window=length,median32=s32,median128=s128,qualified=s32<=.15 and s128<=.35);source_checks.append(record)
                if record['qualified']:options.append(dict(kind=kind,window=length,source_coef=cs))
        assert options
        for change in SHIFT_CASES:
            allocation_data=allocations(seed,sd,change);chosen={};all_records=[];coefficients={}
            for allocation,data in allocation_data.items():
                for family in ['full_field','affine_residual']:
                    records=[]
                    for option in options:
                        kind,L,cs=option['kind'],option['window'],option['source_coef'];a,b=weak.weak_rows(data['y'],data['u'],kind,spec,L);n=a.shape[1]
                        configs=[(lam,tau) for lam in [1e-6,1e-3,.1] for tau in [0.,1e-3,.1,10.]] if family=='full_field' else [(lam,0.) for lam in [1e-6,1e-3,.1,10.]]
                        for lam,tau in configs:
                            if family=='full_field':coef=np.linalg.solve(a.T@a+(lam+tau)*np.eye(n),a.T@b+tau*cs)
                            else:
                                mask=np.sum(spec['exponents'],axis=1)<=1 if kind=='polynomial3' else np.arange(n)<5
                                design=a[:,mask];delta=np.linalg.solve(design.T@design+lam*np.eye(int(mask.sum())),design.T@(b-a@cs));coef=cs.copy();coef[mask]+=delta
                            pred=weak.simulate(data['val_y'][:,0],data['val_u'],coef,kind,spec);h=data['val_u'].shape[1]
                            key=f'{allocation}_{family}_{kind}_{L}_{lam}_{tau}';r=dict(key=key,allocation=allocation,model_family=family,kind=kind,window=L,ridge=lam,source_prior=tau,val32=finite_score(nrmse(pred,data['val_y'],32)),val_long=finite_score(nrmse(pred,data['val_y'],h)),val_long_horizon=h);records.append(r);coefficients[key]=coef
                    all_records.extend(records)
                    for selector,score_key in [('val32','val32'),('val_long','val_long')]:chosen[f'{allocation}_{family}_{selector}']=min(records,key=lambda r:r[score_key])
            tag=f'brusselator_{seed}_{change}';calibration=DEST/f'{tag}_calibration_v1.npz';old.save_npz(calibration,**{allocation+'_'+k:v for allocation,d in allocation_data.items() for k,v in d.items()})
            coefficients_path=DEST/f'{tag}_coefficients_v1.npz';old.save_npz(coefficients_path,**{label+'_coef':coefficients[r['key']] for label,r in chosen.items()},**spec)
            seal=STAGE/f'SELECTION_SEALED85_ATTRIBUTION_{tag}_V1.json';write(seal,dict(utc=utc(),seed=seed,case=change,selected=chosen,candidates=all_records,calibration_path=str(calibration),calibration_sha256=sha(calibration),coefficient_path=str(coefficients_path),coefficient_sha256=sha(coefficients_path),source_path=str(source_path),source_sha256=sha(source_path),eval_generated=False,unopened_seed=seed==2026853))
            x,u=generate('brusselator',seed,6000,8,128,change in ['dynamics','joint'],paired=True);truth=observe((x-sd['mu'])/sd['scale'],change in ['observation','joint']);r=np.random.default_rng(np.random.SeedSequence([seed,10000]));initial=truth[:,0]+np.repeat(r.normal(0,.01,(8,2)),2,axis=0)
            previous=C/'outputs'/'weak_controls_v1'/f'CASE85_WEAK_{tag}_V1.json'
            if previous.exists():
                p=json.loads(previous.read_text());d=np.load(p['array_path'],allow_pickle=False);assert np.array_equal(truth,d['reference']) and np.array_equal(u,d['actions']) and np.array_equal(initial,d['initial_observation'])
            else:assert seed==2026853
            saved=dict(reference=truth,actions=u,initial_observation=initial,physical_reference=x,mu=sd['mu'],scale=sd['scale']);rows=[];summaries={}
            for label,r in chosen.items():
                prediction=weak.simulate(initial,u,coefficients[r['key']],r['kind'],spec);saved[label+'_prediction']=prediction;rr,ss=weak.score(prediction,truth);rows.extend(dict(method=label,**z) for z in rr);summaries[label]=ss
            path=DEST/f'{tag}_evaluation_v1.npz';old.save_npz(path,**saved);record=dict(utc=utc(),seed=seed,case=change,summary=summaries,rows=rows,array_path=str(path),array_sha256=sha(path),selection_seal_sha256=sha(seal),unopened_seed=seed==2026853,new_method=False,post_screen_attribution=seed!=2026853)
            write(STAGE/f'CASE85_ATTRIBUTION_{tag}_V1.json',record);results.append({k:v for k,v in record.items() if k!='rows'});log(dict(kind='attribution_case_closed',seed=seed,case=change,summary=summaries))
    write(STAGE/'RESULT85_ATTRIBUTION_V1.json',dict(status='ORDINARY_ATTRIBUTION_COMPLETE',utc=utc(),source_checks=source_checks,cases=results,protocol_sha256=sha(C/'ATTRIBUTION_PROTOCOL85_V1.md'),qualified_final_directions=0,new_method=False))

if __name__=='__main__':main()
