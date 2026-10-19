"""Independent next-hit-index arithmetic, past-only analogue checks and Markov formulas."""
import runtime_profile
import hashlib,json,pathlib,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent/'cycle18';FOLDER=ROOT/'outputs/event_audit_attempt1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def counts(path,lo,hi,dense):
    starts=np.arange(401) if dense else np.arange(0,500,100)
    numerators=[];denominators=[]
    for block in path.reshape(20,500):
        target_indices=np.r_[np.flatnonzero(block>=hi),1000000]
        next_hit=target_indices[np.searchsorted(target_indices,starts)]
        eligible=block[starts]<=lo
        denominators.append(int(eligible.sum()))
        numerators.append([int(np.sum(eligible&(next_hit-starts<h))) for h in [10,50,100]])
    return np.asarray(numerators),np.asarray(denominators)
def main():
    start=time.perf_counter();receipt=FOLDER/'independent_verification.json';assert not receipt.exists()
    result=json.loads((FOLDER/'results.json').read_text());max_error=0.;cohorts=0;analogue_points=0
    for name in ['lorenz63','chua','sprottM']:
        original=BASE/'outputs/native_demo'/(name+'.npz');assert sha(original)==result['source_array_sha256'][name+'.npz']
        saved=np.load(original,allow_pickle=False);cl=int(saved['context_length']);context=saved['data'][:cl].astype('float32').astype('float64')
        truth=saved['data'][cl:cl+10000].astype('float32').astype('float64')
        control=np.load(FOLDER/(name+'_analogue.npz'),allow_pickle=False)
        assert sha(FOLDER/(name+'_analogue.npz'))==result['analogue_npz_sha256'][name+'_analogue.npz']
        assert np.array_equal(control['context'],context)
        for j in range(3):
            indices=control['context_indices'][j]
            assert np.array_equal(context[indices].astype('float32'),control['trajectories'][j])
            previous=np.r_[cl-1,indices[:-1]];origin=indices-1
            assert np.all((origin>=0)&(origin<cl-1))
            assert np.all(np.any(control['neighbors'][previous]==origin[:,None],axis=1))
            analogue_points+=len(indices)
        scaled=context/context.std(0,ddof=1)
        for i in np.linspace(0,cl-1,17).astype(int):
            distances=np.sum((scaled[:-1]-scaled[i])**2,axis=1)
            assert np.max(distances[control['neighbors'][i]])<=np.partition(distances,7)[7]+1e-10
        paths={'reference':truth,'context_replay':saved['context_replay']}
        paths.update({'native_'+str(seed):saved['predictions'][j] for j,seed in enumerate(saved['seeds'])})
        paths.update({'analogue_'+str(seed):control['trajectories'][j] for j,seed in enumerate(control['seeds'])})
        for label,path in paths.items():
            x=path[:,0].astype('float64')
            expected=result['systems'][name]['reference'] if label=='reference' else result['systems'][name]['paths'][label]['summaries']
            for mode,threshold_data in [('context_thresholds',context[:,0]),('own_rank_diagnostic',x)]:
                lo,hi=np.quantile(threshold_data,[.5,.9])
                for dense in [False,True]:
                    key=mode+('_all_offsets' if dense else '');num,den=counts(x,lo,hi,dense)
                    actual=num.sum(0)/den.sum();target=np.asarray(expected[key]['first_passage_CDF'])
                    error=float(np.max(np.abs(actual-target)));max_error=max(max_error,error);assert error==0
                    assert int(den.sum())==expected[key]['eligible_windows' if dense else 'eligible_landmarks'];cohorts+=1
                primary_num,primary_den=counts(x,lo,hi,False)
                for half,slice_ in [('first',slice(0,10)),('second',slice(10,20))]:
                    n=int(primary_den[slice_].sum());assert n==expected[mode][half+'_half_eligible']
                    values=expected[mode][half+'_half_CDF']
                    if n==0:assert values==[None,None,None]
            z=(x.reshape(20,500)-np.mean(context[:,0]))/np.std(context[:,0],ddof=1)
            moments=[]
            for lag in [1,5,10]:
                a=z[:,:-lag];b=z[:,lag:]
                moments.append(float(np.sum(a*b*(a-b))/a.size))
            e=float(np.max(np.abs(np.asarray(moments)-expected['reversal_odd_moment'])));max_error=max(max_error,e);assert e<1e-12
    witness=result['markov_witness'];P=np.asarray(witness['P']);R=P.T;pi=np.ones(3)/3;mfpt=[]
    for matrix in [P,R]:
        assert np.max(np.abs(pi@matrix-pi))<1e-15
        Z=np.linalg.inv(np.eye(3)-matrix+np.ones((3,1))*pi)
        mfpt.append(float((Z[1,1]-Z[0,1])/pi[1]))
    assert np.max(np.abs(np.asarray(mfpt)-witness['directed_0_to_1_MFPT']))<1e-14
    info={'pass':True,'event_cohorts_exactly_verified':cohorts,'all_event_CDF_count_errors':0,
        'maximum_moment_arithmetic_error':max_error,'past_only_analogue_transitions_verified':analogue_points,
        'nearest_neighbor_distance_checks_per_case':17,'independent_stationary_fundamental_matrix_MFPT':mfpt,
        'empty_conditional_halves_verified_as_null':True,'script_sha256':sha(pathlib.Path(__file__)),
        'elapsed_seconds':time.perf_counter()-start,'limits':'Arithmetic verification does not prove bootstrap coverage, physical stationarity, novelty, or generality.',
        'objective_achieved':False}
    receipt.write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
