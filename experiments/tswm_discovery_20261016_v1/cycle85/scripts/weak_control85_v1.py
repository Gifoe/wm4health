from common85_v1 import *
import common85_v1 as common
import run85_v1 as old
import time

STAGE=C/'outputs'/'weak_controls_v1';STAGE.mkdir(parents=True,exist_ok=True);common.O=STAGE
DEST=Path(CONFIG['artifact_root'])/'cycle85'/'weak_controls_v1';DEST.mkdir(parents=True,exist_ok=True)

def weak_rows(y,u,kind,spec,length):
    rows=[];targets=[];h=np.arange(length+1)/length;m=(np.arange(length)+.5)/length;duration=length*DT
    psi=lambda s:16*s*s*(1-s)*(1-s)
    derivative=lambda s:32*s*(1-s)*(1-2*s)/duration
    p=psi(h);pm=psi(m);dp=derivative(h);dpm=derivative(m)
    norm=float(np.sum(DT/6*(p[:-1]+4*pm+p[1:])))
    for path in range(len(y)):
        for start in range(0,u.shape[1]-length+1,4):
            left=y[path,start:start+length];right=y[path,start+1:start+length+1];middle=(left+right)/2;commands=u[path,start:start+length]
            fl=old.features(np.column_stack([left,commands]),kind,spec);fr=old.features(np.column_stack([right,commands]),kind,spec);fm=old.features(np.column_stack([middle,commands]),kind,spec)
            rows.append(np.sum(DT/6*(p[:-1,None]*fl+4*pm[:,None]*fm+p[1:,None]*fr),axis=0)/norm)
            targets.append(-np.sum(DT/6*(dp[:-1,None]*left+4*dpm[:,None]*middle+dp[1:,None]*right),axis=0)/norm)
    return np.asarray(rows),np.asarray(targets)

def simulate(y0,u,coef,kind,spec):
    y=y0.copy();out=[]
    def vf(a,command):return old.features(np.column_stack([a,command]),kind,spec)@coef
    with np.errstate(over='ignore',invalid='ignore'):
        for t in range(u.shape[1]):
            bad=(~np.isfinite(y).all(axis=1))|(np.max(np.abs(y),axis=1)>1e4);a=y.copy();a[bad]=0;command=u[:,t]
            k1=vf(a,command);k2=vf(a+.5*DT*k1,command);k3=vf(a+.5*DT*k2,command);k4=vf(a+DT*k3,command)
            y=y+DT*(k1+2*k2+2*k3+k4)/6;y[bad]=np.nan;out.append(y.copy())
    return np.stack(out,axis=1)

def score(pred,truth):
    rows=[]
    for h in [32,128]:
        vals=old.path_scores(pred,truth,h)
        for i,val in enumerate(vals):rows.append(dict(kind='trajectory',horizon=h,path=i,score=finite_score(float(val)),failed=not np.isfinite(val)))
    response=[]
    for j in range(8):
        delta=pred[2*j+1]-pred[2*j];true=truth[2*j+1,1:]-truth[2*j,1:];bad=(not np.isfinite(delta).all()) or np.max(np.abs(pred[2*j:2*j+2]))>1e4
        val=float('inf') if bad else float(np.sqrt(np.mean((delta-true)**2))/max(float(np.sqrt(np.mean(true**2))),.05))
        response.append(val);rows.append(dict(kind='action_response',horizon=128,path=j,score=finite_score(val),failed=not np.isfinite(val)))
    h128=[z for z in rows if z['kind']=='trajectory' and z['horizon']==128]
    summary=dict(median128=float(np.median([z['score'] for z in h128])),median_response=finite_score(float(np.median(response))),failed128=sum(z['failed'] for z in h128),failed_response=sum(not np.isfinite(z) for z in response))
    summary['suffices']=summary['failed128']==0 and summary['failed_response']==0 and summary['median128']<=.2 and summary['median_response']<=.3
    return rows,summary

def main():
    log(dict(kind='weak_control_start',pid=os.getpid(),protocol_sha256=sha(C/'WEAK_CONTROL_PROTOCOL85_V1.md')));cases=[];source_records=[]
    for seed in SEEDS:
        srcpath=Path(CONFIG['data_root'])/'cycle85'/'anchored_screen_v1'/f'brusselator_{seed}'/'source_trainval_v1.npz';sd=np.load(srcpath,allow_pickle=False);spec=old.dictionary_specs(4);options=[]
        for kind in ['polynomial3','rff128']:
            for length in [8,16]:
                a,b=weak_rows(sd['y'],sd['u'],kind,spec,length);width=a.shape[1];cs=np.linalg.lstsq(np.vstack([a,np.sqrt(1e-6)*np.eye(width)]),np.vstack([b,np.zeros((width,2))]),rcond=None)[0]
                pred=simulate(sd['val_y'][:,0],sd['val_u'],cs,kind,spec);p32=finite_score(float(np.median(old.path_scores(pred,sd['val_y'],32))));p128=finite_score(float(np.median(old.path_scores(pred,sd['val_y'],128))))
                qualified=p32<=.15 and p128<=.35;record=dict(seed=seed,kind=kind,window=length,median32=p32,median128=p128,qualified=qualified)
                source_records.append(record)
                if qualified:options.append(dict(kind=kind,window=length,source_coef=cs))
        assert options,'No competent ordinary source field; do not score target'
        for change in SHIFT_CASES:
            base=target_data('brusselator',seed,sd['mu'],sd['scale'],change);records=[];coefficients={}
            for option in options:
                kind,length,cs=option['kind'],option['window'],option['source_coef'];a,b=weak_rows(base['y'],base['u'],kind,spec,length);width=a.shape[1]
                for lam in [1e-6,1e-3,.1]:
                    for prior in [0.,1e-3,.1,10.]:
                        w=np.linalg.solve(a.T@a+(lam+prior)*np.eye(width),a.T@b+prior*cs);pred=simulate(base['val_y'][:,0],base['val_u'],w,kind,spec)
                        key=f'{kind}_{length}_{lam}_{prior}';records.append(dict(key=key,kind=kind,window=length,ridge=lam,source_prior=prior,val32=finite_score(nrmse(pred,base['val_y'],32)),val64=finite_score(nrmse(pred,base['val_y'],64))))
                        coefficients[key]=w
            selected={key:min(records,key=lambda r:r[criterion]) for key,criterion in [('weak_val32','val32'),('weak_val64','val64')]}
            tag=f'brusselator_{seed}_{change}';arrays={};meta={}
            for label,r in selected.items():arrays[label+'_coef']=coefficients[r['key']];meta[label]=r
            training_path=DEST/f'{tag}_calibration_v1.npz';old.save_npz(training_path,**base)
            coefficient_path=DEST/f'{tag}_coefficients_v1.npz';old.save_npz(coefficient_path,**arrays,**spec)
            seal=STAGE/f'SELECTION_SEALED85_WEAK_{tag}_V1.json';write(seal,dict(utc=utc(),seed=seed,case=change,selected=meta,candidates=records,source_options=[{k:v for k,v in z.items() if k!='source_coef'} for z in options],source_data_sha256=sha(srcpath),calibration_path=str(training_path),calibration_sha256=sha(training_path),coefficient_path=str(coefficient_path),coefficient_sha256=sha(coefficient_path),eval_generated=False,first_unopened_eval_seed=seed==2026852))
            x,u=generate('brusselator',seed,6000,8,128,change in ['dynamics','joint'],paired=True);truth=observe((x-sd['mu'])/sd['scale'],change in ['observation','joint']);rng=np.random.default_rng(np.random.SeedSequence([seed,10000]));initial=truth[:,0]+np.repeat(rng.normal(0,.01,(8,2)),2,axis=0)
            previous=C/'outputs'/'source_polish_v3'/f'CASE85_{tag}_V1.json'
            if previous.exists():
                oldcase=json.loads(previous.read_text());oldarray=np.load(oldcase['array_path'],allow_pickle=False)
                assert np.array_equal(truth,oldarray['reference']) and np.array_equal(u,oldarray['actions']) and np.array_equal(initial,oldarray['initial_observation'])
            else:assert seed==2026852
            output=dict(reference=truth,actions=u,initial_observation=initial,physical_reference=x,mu=sd['mu'],scale=sd['scale']);metrics=[];summaries={}
            for label,r in selected.items():
                prediction=simulate(initial,u,coefficients[r['key']],r['kind'],spec);output[label+'_prediction']=prediction;rows,summary=score(prediction,truth);metrics.extend(dict(method=label,**z) for z in rows);summaries[label]=summary
            arraypath=DEST/f'{tag}_evaluation_v1.npz';old.save_npz(arraypath,**output)
            record=dict(utc=utc(),seed=seed,case=change,summary=summaries,rows=metrics,selection_seal_sha256=sha(seal),array_path=str(arraypath),array_sha256=sha(arraypath),first_unopened_eval_seed=seed==2026852,post_screen_control=True,new_method=False)
            write(STAGE/f'CASE85_WEAK_{tag}_V1.json',record);cases.append({k:v for k,v in record.items() if k!='rows'});log(dict(kind='weak_case_closed',seed=seed,case=change,summary=summaries))
    write(STAGE/'RESULT85_WEAK_V1.json',dict(status='ORDINARY_WEAK_CONTROL_COMPLETE',utc=utc(),source_checks=source_records,cases=cases,protocol_sha256=sha(C/'WEAK_CONTROL_PROTOCOL85_V1.md'),new_method_tested=False,qualified_final_directions=0))

if __name__=='__main__':main()
