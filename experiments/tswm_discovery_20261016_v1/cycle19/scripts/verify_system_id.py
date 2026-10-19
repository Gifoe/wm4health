"""Independent event indexing, validation data-boundary and integration checks."""
import runtime_profile
import hashlib,json,pathlib,time
import numpy as np
from verify_event_audit import counts
from system_id_control import rollout,select,numerical_checks,features
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/system_id_control'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    tick=time.perf_counter();path=OUT/'independent_verification.json';assert not path.exists()
    result=json.loads((OUT/'results.json').read_text());checked=0;selection_replays=0;replay_errors=[];max_mase_error=0.;equation_residuals={}
    for row in result['rows']:
        file=OUT/row['array_file'];assert sha(file)==row['array_sha256'];data=np.load(file,allow_pickle=False)
        prediction=data['prediction'];context=data['context'];truth=data['truth'];reference=data['reference_context']
        # Refit from context alone and reproduce all12 candidate choices without reading future.
        model,selection=select(context.copy());assert selection==row['selection'];selection_replays+=1
        for name in model:assert np.array_equal(model[name],data[name])
        pred,failed=rollout(context[-1],64,model);error=float(np.max(np.abs(pred[:,0,:]-prediction[:64])));assert error==0;replay_errors.append(error)
        chosen=selection['selected'];assert chosen['validation_MSE']==min(r['validation_MSE'] for r in selection['candidates'] if r['validation_MSE'] is not None)
        assert selection['train_prefix']<=min(selection['validation_initial_state_indices'])
        assert max(selection['validation_initial_state_indices'])+64<len(context)
        # Recompute normalized ridge normal equations directly from retained coefficients.
        z=(context-model['mean'])/model['scale'];phi=features(z[1:-1],model['powers'],model['knots']);target=(z[2:]-z[:-2])/2
        scales=np.maximum(np.sqrt(np.mean(phi**2,axis=0)),1e-12);design=phi/scales
        penalty=np.eye(design.shape[1])*chosen['ridge'];constant=np.flatnonzero(model['powers'].sum(1)==0)[0];penalty[constant,constant]=0
        standardized_coeff=model['coeff']*scales[:,None]
        residual=float(np.max(np.abs((design.T@design/len(design)+penalty)@standardized_coeff-design.T@target/len(design))))
        assert residual<1e-10;equation_residuals[row['task']]=residual
        if row['diverged_at_future_index'] is None:
            assert prediction.shape==truth.shape==(10000,3) and np.isfinite(prediction).all()
            x=prediction[:,0]
            for label,values in [('context_thresholds',reference[:,0]),('own_rank_diagnostic',x)]:
                lower,upper=np.quantile(values,[.5,.9])
                for dense in [False,True]:
                    num,den=counts(x,lower,upper,dense);got=num.sum(0)/den.sum()
                    expected=np.asarray(row['event_summaries'][label+('_all_offsets' if dense else '')]['first_passage_CDF'])
                    assert np.array_equal(got,expected);checked+=1
            # Metric worker casts to float32; match those inputs before independent float64 arithmetic.
            p=prediction.astype('float32').astype('float64');t=truth.astype('float32').astype('float64')
            mase=float(np.mean(np.mean(np.abs(p[:10]-t[:10]),axis=0)/(np.mean(np.abs(np.diff(t,axis=0)),axis=0)+1e-8)))
            error=abs(mase-row['global_metrics']['MASE_true_denominator']);assert error<2e-6;max_mase_error=max(max_mase_error,error)
    info=dict(pass_=True,arrays_hash_verified=len(result['rows']),past_only_selection_replays_exact=selection_replays,
        event_cohorts_exact=checked,prefix_replay_max_error=max(replay_errors),normal_equation_residuals=equation_residuals,
        max_MASE_arithmetic_error=max_mase_error,numerical_checks=numerical_checks(),script_sha256=sha(pathlib.Path(__file__)),
        seconds=time.perf_counter()-tick,limits='Selection/refit replay shares model code; event counts and MASE use independent arithmetic. No independent physical reference, bootstrap coverage or novelty is proved.')
    info['pass']=info.pop('pass_');path.write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
