"""Frozen native-law diagnostic; no training, no target truth scoring."""
import pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'cycle34/scripts'))
from sundial_common34 import *
import pandas as pd,time,datetime,traceback
O=R/'outputs/native_queries_v2';assert not O.exists();O.mkdir(parents=True)
N=256;M=8;SPLITS=[96,352,360];REPS=[2026391,2026392]
def stats(prefix,suffix,mu,scale):
    # suffix [outer, inner, time]; computes E[g], not g(E[path]).
    u=(np.asarray(prefix,dtype='float64')-mu)/scale;v=(np.asarray(suffix,dtype='float64')-mu)/scale
    w=np.stack([np.ones(len(u)),np.tanh(u.mean(-1)),np.tanh(u[:,-1])],axis=-1)
    g=np.stack([v.mean(-1),v[:,:,-1],np.mean(v*v,axis=-1)],axis=-1).mean(axis=1)
    return (w[:,:,None]*g[:,None,:]).reshape(len(u),9)
def summarize(direct,composed,meta):
    delta=direct-composed;avg=delta.mean(0);se=delta.std(0,ddof=1)/np.sqrt(len(delta));rows=[]
    for j in range(9):rows.append({**meta,'probe':j,'w':j//3,'g':j%3,'direct':float(direct[:,j].mean()),
       'composed':float(composed[:,j].mean()),'difference':float(avg[j]),'mc_se_outer':float(se[j]),
       'mc95_low':float(avg[j]-1.96*se[j]),'mc95_high':float(avg[j]+1.96*se[j]),
       'mc_z':float(avg[j]/se[j]) if se[j]>0 else None})
    return rows
def fixtures():
    rng=np.random.default_rng(2026390);k=32768;rho=.8;u=rng.normal(size=(k,1));d=(rho*u+np.sqrt(1-rho*rho)*rng.normal(size=(k,1)))[:,None,:]
    coherent=rho*u[:,None,:]+np.sqrt(1-rho*rho)*rng.normal(size=(k,M,1));independent=rng.normal(size=(k,M,1))
    a=stats(u,d,0.,1.);good=stats(u,coherent,0.,1.);bad=stats(u,independent,0.,1.)
    rows=summarize(a,good,{'control':'AR1_correct'})+summarize(a,bad,{'control':'AR1_independent_refresh'})
    assert all(abs(x['mc_z'])<5 for x in rows if x['control']=='AR1_correct')
    assert abs(rows[9+3]['mc_z'])>10
    nodes,weights=np.polynomial.hermite.hermgauss(64);expect=float(np.sum(weights*(np.sqrt(2)*nodes)*np.tanh(np.sqrt(2)*nodes))/np.sqrt(np.pi))
    empirical=float(a[:,3].mean());assert abs(empirical-rho*expect)<.03
    np.savez_compressed(O/'statistic_fixtures.npz',prefix=u,direct=d,coherent=coherent,independent=independent)
    return {'passed':True,'rows':rows,'rho':rho,'quadrature_E_U_tanhU':expect,'expected_weighted_cross':rho*expect}
@torch.inference_mode()
def cache(model,hist,reference=None):
    if reference is None:mu=hist.mean(-1,keepdim=True);scale=hist.std(-1,keepdim=True,unbiased=False)+1e-5
    else:mu=torch.full((len(hist),1),reference[0],device='cuda');scale=torch.full((len(hist),1),reference[1],device='cuda')
    z=model.model(input_ids=(hist-mu)/scale,return_dict=True,use_cache=False).last_hidden_state[:,-1]
    return {'z':z.detach(),'mean':mu,'scale':scale}
@torch.inference_mode()
def native_noise(model,hist,noise,horizon,reference=None):
    original=torch.randn;calls=[]
    def controlled(*shape,**kwargs):
        dims=tuple(shape[0]) if len(shape)==1 and isinstance(shape[0],(tuple,list)) else tuple(shape)
        assert dims==tuple(noise.shape) and not kwargs;calls.append(dims);return noise.clone()
    torch.randn=controlled
    try:
        if reference is None:out=model.generate(hist,max_new_tokens=horizon,num_samples=len(noise)//len(hist))
        else:
            out=model.generate((hist-reference[0])/reference[1],max_new_tokens=horizon,num_samples=len(noise)//len(hist),revin=False)
            out=out*reference[1]+reference[0]
    finally:torch.randn=original
    assert len(calls)==1;return out
@torch.inference_mode()
def qualify(model):
    t=torch.arange(1032,device='cuda',dtype=torch.float32);allhist=torch.stack([torch.sin(t/13)+t/500,2*torch.cos(t/31)-t/1000+5])
    errors=[]
    for length in [672,768,1032]:
        h=allhist[:,:length];seed(20263900+length);e=torch.randn(6,720)
        for label,ref in [('native',None),('fixed_reference',(1.5,2.))]:
            a=native_noise(model,h,e,192,ref);b=from_noise(model,cache(model,h,ref),e.cuda().reshape(3,2,720),192)
            err=float((a-b).abs().max())/(1+float(a.abs().max()));assert err<2e-5
            errors.append({'history_length':length,'mode':label,'relative_max':err})
    return errors
def inputs():
    records=[]
    qualification_path=B/'cycle34/outputs/qualification_v1/result.json'
    assert sha(qualification_path)=='f804c7fe754bd0e8a8d33b37f333e1a57e32e57efad2ddaf1c0b066797ee0c35'
    prior_qualification=json.loads(qualification_path.read_text(encoding='utf-8'))
    origin_start={x['name']:x['history_start_index'] for x in prior_qualification['records'] if 'history_start_index' in x}
    for name,rel,field in [('solar','solar_with_weather/15T/train-00000-of-00001.parquet','target'),
                           ('jena','jena_weather/10T/train-00000-of-00001.parquet','target_0')]:
        p=B/'cycle34/assets/fev'/rel;df=pd.read_parquet(p);assert len(df)==1
        row=df.iloc[0];y=np.asarray(row[field],dtype='float32');timestamps=np.asarray(row['timestamp']);kept=0
        for end in [origin_start[name]+672+offset for offset in [0,336,672,1008]]:
            meta={'dataset':name,'end_index_exclusive':end,'raw_sha256':sha(p),'field':field,'series_id':str(row['id']),'qualification_anchor_start':origin_start[name],'qualification_anchor_sha256':sha(qualification_path)}
            if end>len(y) or not np.isfinite(y[end-672:end]).all():
                records.append((None,{**meta,'excluded':'insufficient or nonfinite history'}));continue
            times=timestamps[end-672:end];gaps=np.diff(times);assert np.all(gaps==gaps[0]) and gaps[0]>np.timedelta64(0,'ns')
            records.append((y[end-672:end].copy(),{**meta,'first_timestamp':str(times[0]),'last_timestamp':str(times[-1]),'excluded':None}));kept+=1
        assert kept>=2
    return records
def main():
    start=time.monotonic();locks=validate_sources();fixture=fixtures();records=inputs();model=load_model();before=state_sha(model)
    torch.cuda.reset_peak_memory_stats();qualification=qualify(model);rows=[];files=[]
    for index,(hist,meta) in enumerate(records):
        if hist is None:continue
        folder=O/f"{meta['dataset']}_{meta['end_index_exclusive']}";folder.mkdir()
        h=torch.tensor(hist[None],device='cuda');base=cache(model,h);mu=float(base['mean'].item());scale=float(base['scale'].item())
        np.save(folder/'history.npy',hist)
        for rep in REPS:
            seed(rep+index*100);noise=torch.randn(N,1,720,device='cuda')
            with torch.inference_mode():direct=from_noise(model,base,noise,horizon=720).squeeze(0).cpu().numpy()
            np.save(folder/f'direct_{rep}.npy',direct)
            for split in SPLITS:
                prefix=direct[:,:split];target=direct[:,split:2*split];d=stats(prefix,target[:,None,:],mu,scale)
                composed={mode:np.empty((N,M,split),dtype='float32') for mode in ['native','fixed_reference']}
                # Separate independent seed stream; no reuse of base flow noise.
                seed(rep+index*100+split*10000+7000000)
                inner_noise=torch.randn(M,N,720,device='cuda')
                with torch.inference_mode():
                    for begin in range(0,N,16):
                        stop=min(begin+16,N);extended=np.concatenate([np.repeat(hist[None],stop-begin,axis=0),prefix[begin:stop]],axis=1)
                        ext=torch.tensor(extended,device='cuda');e=inner_noise[:,begin:stop]
                        for mode in composed:
                            ref=None if mode=='native' else (mu,scale)
                            val=from_noise(model,cache(model,ext,ref),e,horizon=split).cpu().numpy();composed[mode][begin:stop]=val
                arrays={'prefix':prefix,'direct_suffix':target,'direct_probes':d}
                for mode,pred in composed.items():
                    c=stats(prefix,pred,mu,scale);arrays[mode+'_suffix']=pred;arrays[mode+'_probes']=c
                    rows.extend(summarize(d,c,{**meta,'rep':rep,'split':split,'mode':mode,'outer':N,'inner':M,'history_mean':mu,'history_scale':scale}))
                p=folder/f'query_{rep}_{split}.npz';np.savez_compressed(p,**arrays)
                files.append({'path':p.relative_to(O).as_posix(),'sha256':sha(p)})
            dump(O/'status.json',{'status':'running','last_context':meta,'rep':rep,'rows_complete':len(rows),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            print(json.dumps({'context':folder.name,'rep':rep,'rows':len(rows)}),flush=True)
        dump(folder/'input.json',{**meta,'history_mean':mu,'history_scale':scale,'history_sha256':sha(folder/'history.npy')})
    after=state_sha(model);assert after==before
    out={'status':'completed_pending_independent_verification','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'seconds':time.monotonic()-start,'source_locks':locks,'runtime':runtime(),'qualification':qualification,'statistic_fixture':fixture,
       'model_state_before':before,'model_state_after':after,'outer':N,'inner':M,'splits':SPLITS,'rep_seeds':REPS,
       'inputs':[x[1] for x in records],'rows':rows,'prediction_files':files,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
       'test_scored':False,'future_truth_scored':False,'new_training_updates':0,'accepted_directions':0,
       'plan_sha256':sha(R/'NATIVE_QUERY_PLAN_V2.md'),'script_sha256':sha(pathlib.Path(__file__))}
    dump(O/'result.json',out);print(json.dumps({'status':out['status'],'rows':len(rows),'seconds':out['seconds']}),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:dump(O/'failure.json',{'error':traceback.format_exc()});raise
