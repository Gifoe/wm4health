import runtime_profile
import json,pathlib,time,sys,math
import numpy as np
from scipy.special import logsumexp,log_ndtr
from verify_belief_results import arrays,mix,aggregate,sha
ROOT=pathlib.Path(__file__).resolve().parents[1];PREV=ROOT.parent/'cycle23'
def main():
    start=time.perf_counter();OUT=ROOT/'outputs/full_tail_patches';r=json.loads((OUT/'results.json').read_text());ba=json.loads((ROOT/'outputs/all_context_tail_bounds.json').read_text())
    assert sum(bool(x['flagged_contexts']) for x in ba['rows'])==len(r['patches'])==58
    assert sha(ROOT/'FULL_TAIL_REPAIR.md')==r['plan_sha256'];errors=[];scerrors=[];weight_errors=[];tailerrors=[];scalars=0
    for row,file in zip(r['patches'],r['files']):
        path=OUT/file['path'];assert sha(path)==file['sha256'];a=arrays(path);oldpath=ROOT/row['source_file'];assert sha(oldpath)==row['original_source_sha256']
        old=arrays(oldpath);ids=a['flagged_indices'];sc=mix(a['component_mean'],a['component_variance'],a['component_weights'],a['flagged_query'])
        for k,v in sc.items():scerrors.append(float(abs(v-a[k][ids]).max()));assert np.array_equal(a[k][np.setdiff1d(np.arange(64),ids)],old[row['source_prefix']+k][np.setdiff1d(np.arange(64),ids)])
        for k,v in aggregate({k:a[k] for k in ['nll','mean','second','pit','error']}).items():errors.append(float(abs(np.asarray(v)-row['actual_query'][k]).max()));scalars+=np.asarray(v).size
        z=np.linspace(-14,14,16385);w=np.exp(-.5*z*z);w[[0,-1]]*=.5;w/=w.sum()
        if row['arm']=='reweighted':
            lp=-.5*(np.log(2*np.pi*a['probe_variance'])+(a['probe'][:,None,:]-a['probe_mean'])**2/a['probe_variance']).sum(-1);l=np.log(w)[None]+lp;ww=np.exp(l-logsumexp(l,axis=-1,keepdims=True))
            weight_errors.append(float(abs(ww-a['component_weights']).max()));pn=-logsumexp(l,axis=-1)
        else:weight_errors.append(float(abs(w-a['component_weights']).max()))
        logtau=np.log(2)+log_ndtr(-14);logz=np.log1p(-np.exp(logtau))
        def bnd(nll,var,extra=None):
            lmax=-.5*np.log(2*np.pi*var).sum()
            if extra is not None:lmax-=.5*np.log(2*np.pi*extra).sum()
            return np.maximum(-logz,np.logaddexp(logz,logtau+lmax+nll))
        if row['arm']=='reweighted':bb=bnd(pn,a['probe_variance'])+bnd(sc['nll']+pn,a['component_variance'],a['probe_variance'])
        else:bb=bnd(sc['nll'],a['component_variance'])
        tailerrors.append(float(abs(bb-a['conservative_tail_bound']).max()));assert bb.max()<.01
        assert abs(sc['nll']-a['refined_nll']).max()<.01
    assert max(errors)<2e-8 and max(scerrors)<2e-8 and max(weight_errors)<2e-8 and max(tailerrors)<2e-8,(max(errors),max(scerrors),max(weight_errors),max(tailerrors))
    receipt=dict(status='pass',result_sha256=sha(OUT/'results.json'),verifier_sha256=sha(pathlib.Path(__file__)),patches=len(r['patches']),all_flagged_rows_covered=True,
        aggregate_scalars_replayed=scalars,aggregate_max_error=max(errors),score_max_error=max(scerrors),weight_max_error=max(weight_errors),tail_bound_max_error=max(tailerrors),
        max_refinement_nll_error=max(x['refinement_nll_max_error'] for x in r['patches']),max_new_tail_bound=max(x['new_max_tail_bound'] for x in r['patches']),unflagged_raw_values_unchanged=True,elapsed_seconds=time.perf_counter()-start,
        scope='All sparse patches and source hashes; independent SciPy component score/weight algebra for every replaced context, full aggregates and conservative tail bounds. Unchanged checkpoints were verified in native/ordinary replay receipts.')
    (OUT/'independent_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
