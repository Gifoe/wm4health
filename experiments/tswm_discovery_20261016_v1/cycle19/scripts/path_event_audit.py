"""Prespecified event-law audit of retained native simulations and an old analogue control."""
import runtime_profile
import hashlib,json,pathlib,time
import numpy as np
from scipy.spatial import cKDTree
ROOT=pathlib.Path(__file__).resolve().parents[1];PREVIOUS=ROOT.parent/'cycle18'
OUT=ROOT/'outputs/event_audit_attempt1';HORIZONS=np.asarray([10,50,100]);LAGS=np.asarray([1,5,10])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fractions(numerator,denominator):
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan,dtype='float64'),where=denominator>0)
def summarize(x,context):
    # Primary events use only context thresholds. Rank diagnostic deliberately uses path-only evaluation quantiles.
    x=np.asarray(x,dtype='float64');blocks=x.reshape(20,500)
    results={};counts={}
    for label,threshold_data in [('context_thresholds',context),('own_rank_diagnostic',x)]:
        lower,upper=np.quantile(threshold_data,[.5,.9]);windows=blocks.reshape(20,5,100)
        eligible=windows[:,:,0]<=lower
        hits=np.stack([np.any(windows[:,:,:h]>=upper,axis=2)&eligible for h in HORIZONS],axis=2)
        denominator=eligible.sum(1);numerator=hits.sum(1)
        counts[label]=(numerator,denominator)
        crossings=np.flatnonzero((x[:-1]<=lower)&(x[1:]>lower))+1;intervals=np.diff(crossings)
        results[label]={'thresholds':[float(lower),float(upper)],'eligible_landmarks':int(denominator.sum()),
            'hit_counts':numerator.sum(0).tolist(),'first_passage_CDF':fractions(numerator.sum(0),denominator.sum()).tolist(),
            'median_upcrossings':int(len(crossings)),'completed_return_intervals':int(len(intervals)),
            'return_CDF_completed_only':[float(np.mean(intervals<=h)) if len(intervals) else None for h in HORIZONS],
            'final_return_right_censored':bool(len(crossings)>0),
            'last_censored_return_lower_bound':int(len(x)-1-crossings[-1]) if len(crossings) else None,
            'occupancy_above_upper':float(np.mean(x>=upper)),
            'first_half_CDF':fractions(numerator[:10].sum(0),denominator[:10].sum()).tolist(),
            'second_half_CDF':fractions(numerator[10:].sum(0),denominator[10:].sum()).tolist(),
            'first_half_eligible':int(denominator[:10].sum()),'second_half_eligible':int(denominator[10:].sum())}
        # Post-first-attempt diagnostic: coarse stride100 can resonate with periodic paths.
        # Keep the frozen primary statistic; show every full100-sample window separately.
        dense_windows=np.lib.stride_tricks.sliding_window_view(blocks,100,axis=1)
        dense_eligible=dense_windows[:,:,0]<=lower
        dense_hits=np.stack([np.any(dense_windows[:,:,:h]>=upper,axis=2)&dense_eligible for h in HORIZONS],axis=2)
        dense_n=dense_hits.sum(1);dense_d=dense_eligible.sum(1)
        counts[label+'_all_offsets']=(dense_n,dense_d)
        results[label+'_all_offsets']={'eligible_windows':int(dense_d.sum()),
            'first_passage_CDF':fractions(dense_n.sum(0),dense_d.sum()).tolist(),
            'role':'Exploratory all-offset diagnostic added after coarse-grid missing-denominator failure; not preregistered confirmation.'}
    centered=(blocks-np.mean(context))/np.std(context,ddof=1)
    moments=np.stack([np.mean(centered[:,:-lag]**2*centered[:,lag:]-centered[:,:-lag]*centered[:,lag:]**2,axis=1) for lag in LAGS],axis=1)
    results['reversal_odd_moment']=moments.mean(0).tolist();return results,counts,moments
def compare(candidate,reference,rng):
    result={};cmet,ccount,cmom=candidate;rmet,rcount,rmom=reference
    ci=rng.integers(0,20,size=(2000,20));ri=rng.integers(0,20,size=(2000,20))
    for name in ['context_thresholds','own_rank_diagnostic','context_thresholds_all_offsets','own_rank_diagnostic_all_offsets']:
        cn,cd=ccount[name];rn,rd=rcount[name]
        delta=fractions(cn[ci].sum(1),cd[ci].sum(1)[:,None])-fractions(rn[ri].sum(1),rd[ri].sum(1)[:,None])
        point=np.asarray(cmet[name]['first_passage_CDF'])-np.asarray(rmet[name]['first_passage_CDF'])
        result[name]={'CDF_difference':point.tolist(),'descriptive_bootstrap95':np.nanquantile(delta,[.025,.975],axis=0).tolist(),
            'max_absolute_CDF_difference':float(np.nanmax(np.abs(point))),
            'bootstrap_replicates_with_zero_eligible_count':int(np.sum(~np.isfinite(delta).all(1)))}
    delta=cmom[ci].mean(1)-rmom[ri].mean(1)
    result['reversal_odd_moment_difference']=(cmom.mean(0)-rmom.mean(0)).tolist()
    result['reversal_odd_descriptive_bootstrap95']=np.quantile(delta,[.025,.975],axis=0).tolist()
    return result
def markov_witness():
    P=np.asarray([[.1,.8,.1],[.1,.1,.8],[.8,.1,.1]]);R=P.T;pi=np.ones(3)/3
    observables=np.vstack([np.asarray([0.,1.,2.]),np.eye(3)])
    errors=[]
    for x in observables:
        centered=x-pi@x
        for k in range(33):errors.append(abs((pi*centered)@np.linalg.matrix_power(P,k)@centered-(pi*centered)@np.linalg.matrix_power(R,k)@centered))
    fpt=[];means=[]
    for transition in [P,R]:
        Q=transition[np.ix_([0,2],[0,2])]
        fpt.append([float(1-(np.linalg.matrix_power(Q,int(h))@np.ones(2))[0]) for h in HORIZONS])
        means.append(float(np.linalg.solve(np.eye(2)-Q,np.ones(2))[0]))
    assert max(errors)<1e-14 and abs(means[0]-1/.73)<1e-14 and abs(means[1]-1.7/.73)<1e-14
    return {'P':P.tolist(),'time_reversal':R.tolist(),'stationary_distribution':pi.tolist(),
        'max_all_scalar_autocovariance_difference_lags0to32':float(max(errors)),
        'directed_0_to_1_first_step_probability':[float(P[0,1]),float(R[0,1])],
        'directed_0_to_1_FPT_CDF':fpt,'directed_0_to_1_MFPT':means,
        'ordinary_transition_likelihood_KL':float(np.sum(pi[:,None]*P*np.log(P/R))),
        'interpretation':'Classical time reversal witness; ordinary transition likelihood already distinguishes it, not a new method gap.'}
def main():
    started=time.perf_counter();assert not OUT.exists();OUT.mkdir(parents=True)
    receipt=json.loads((PREVIOUS/'outputs/native_demo/independent_verification.json').read_text());assert receipt['pass']
    assert sha(PREVIOUS/'ARTIFACT_MANIFEST.json')=='1820dda9443e66613563a81df7bf118feaca9c9c9e6a0ef31baa48b7abf224a6'
    rng=np.random.default_rng(20261901);systems={}
    for name in ['lorenz63','chua','sprottM']:
        source=PREVIOUS/'outputs/native_demo'/(name+'.npz');assert sha(source)==receipt['verified_arrays_sha256'][name+'.npz']
        saved=np.load(source,allow_pickle=False);cl=int(saved['context_length'])
        context=saved['data'][:cl].astype('float32').astype('float64')
        truth=saved['data'][cl:cl+10000].astype('float32').astype('float64')
        reference=summarize(truth[:,0],context[:,0]);paths={'context_replay':saved['context_replay']}
        for i,seed in enumerate(saved['seeds']):paths['native_'+str(seed)]=saved['predictions'][i]
        scale=context.std(0,ddof=1);assert np.all(scale>0)
        tree=cKDTree(context[:-1]/scale);_,neighbors=tree.query(context/scale,k=8)
        controls=[];states=[]
        for seed in [2021,2022,2023]:
            generator=np.random.default_rng(seed);indices=np.empty(10000,dtype='int64');current=cl-1
            for t in range(10000):
                origin=neighbors[current,generator.integers(0,8)];assert origin<cl-1
                current=int(origin)+1;indices[t]=current
            trajectory=context[indices].astype('float32');paths['analogue_'+str(seed)]=trajectory;controls.append(trajectory);states.append(indices)
        np.savez_compressed(OUT/(name+'_analogue.npz'),trajectories=np.stack(controls),context_indices=np.stack(states),neighbors=neighbors,
            context=context,seeds=np.asarray([2021,2022,2023]))
        evaluated={}
        for label,path in paths.items():
            assert path.shape==(10000,3) and np.isfinite(path).all()
            summary=summarize(path[:,0],context[:,0]);evaluated[label]={'summaries':summary[0],'versus_reference':compare(summary,reference,rng)}
        systems[name]={'context_length':cl,'reference':reference[0],'paths':evaluated}
        print(json.dumps({'completed':name,'native_context_threshold_max_CDF_difference':{
            key:value['versus_reference']['context_thresholds']['max_absolute_CDF_difference'] for key,value in evaluated.items()}}),flush=True)
    result={'systems':systems,'markov_witness':markov_witness(),'horizons_samples':HORIZONS.tolist(),'moment_lags':LAGS.tolist(),
        'reference_source':'cycle18 exact released native demonstration arrays, not newly sampled physical systems',
        'source_array_sha256':receipt['verified_arrays_sha256'],'script_sha256':sha(pathlib.Path(__file__)),
        'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'elapsed_seconds':time.perf_counter()-started,
        'environment':{'numpy':np.__version__},'new_training':False,'accepted_novel_directions':0,'objective_achieved':False,
        'limitations':['One reference per system;3stochastic draws are not training/system replications.',
            'Bootstrap is descriptive; mixing and coverage are not proved.',
            'Completed-return CDF is censor-biased and secondary.',
            'Event law/kinetic fidelity and the Markov witness are not new concepts.']}
    missing=[]
    def clean(value,path=''):
        if isinstance(value,float) and not np.isfinite(value):missing.append(path);return None
        if isinstance(value,dict):return {k:clean(v,path+'/'+str(k)) for k,v in value.items()}
        if isinstance(value,list):return [clean(v,path+'/'+str(i)) for i,v in enumerate(value)]
        return value
    result=clean(result);result['unavailable_nonfinite_stat_paths']=missing
    result['unavailable_stat_reason']='A conditional denominator/eligible block cohort is zero; null is not a zero probability.'
    result['analogue_npz_sha256']={p.name:sha(p) for p in OUT.glob('*.npz')}
    result['attempt0_analogue_npz_identical']={p.name:sha(p)==sha(ROOT/'outputs/event_audit'/p.name) for p in OUT.glob('*.npz')}
    assert all(result['attempt0_analogue_npz_identical'].values())
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'finished':True,'seconds':result['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
