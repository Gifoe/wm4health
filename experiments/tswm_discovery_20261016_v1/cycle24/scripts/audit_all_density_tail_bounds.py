"""All-context conservative Normal-tail bounds for the frozen mixture scores."""
import json,math,pathlib,time
import numpy as np
from scipy.special import log_ndtr,logsumexp
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(path,names):
    with np.load(path) as f:return {k:f[k] for k in names}
def bound(nll,var,limit=7.,extra_var=None):
    logtau=math.log(2)+log_ndtr(-limit);logz=np.log1p(-np.exp(logtau))
    lpmax=-.5*np.log(2*np.pi*np.asarray(var,dtype='float64')).sum()
    if extra_var is not None:lpmax-=.5*np.log(2*np.pi*np.asarray(extra_var,dtype='float64')).sum()
    return np.maximum(-logz,np.logaddexp(logz,logtau+lpmax+np.asarray(nll)))
rows=[];start=time.perf_counter();base=ROOT/'outputs/native_belief';r=json.loads((base/'results.json').read_text())
for c in r['cells']:
    b=read(base/c['base_file'],['grid_weight','query_variance']);a=read(base/c['raw_file'],['no_update_nll','native_nll','reweighted_nll','real_post_query_variance','probe_grid_mean','probe_variance','real_probe'])
    common=dict(kind=c['kind'],seed=c['checkpoint_seed'],n=c['context_size'],probe_u=c['probe_u'])
    for arm,key,var in [('no_update','no_update_nll',b['query_variance']),('native','native_nll',a['real_post_query_variance'])]:
        bb=bound(a[key],var);rows.append(dict(**common,arm=arm,max_nll=float(a[key].max()),max_tail_nll_bound=float(bb.max()),flagged_contexts=np.where(bb>.01)[0].tolist()))
    ll=-.5*(np.log(2*np.pi*a['probe_variance'])+(a['real_probe'][:,None,:]-a['probe_grid_mean'])**2/a['probe_variance']).sum(-1)
    probe_nll=-logsumexp(np.log(b['grid_weight'])[None]+ll,axis=-1)
    bb=bound(probe_nll,b['query_variance']*0+a['probe_variance'])+bound(a['reweighted_nll']+probe_nll,b['query_variance'],extra_var=a['probe_variance'])
    rows.append(dict(**common,arm='reweighted',max_nll=float(a['reweighted_nll'].max()),max_tail_nll_bound=float(bb.max()),flagged_contexts=np.where(bb>.01)[0].tolist()))
filt=ROOT/'outputs/ordinary_filter_controls';fr=json.loads((filt/'results.json').read_text())
for c in fr['cells']:
    a=read(filt/c['raw_file'],['map_ggn_laplace_nll','map_ggn_laplace_variance']);bb=bound(a['map_ggn_laplace_nll'],a['map_ggn_laplace_variance'])
    rows.append(dict(kind=c['kind'],seed=c['seed'],n=c['n'],probe_u=c['probe_u'],arm='map_ggn_laplace',max_nll=float(a['map_ggn_laplace_nll'].max()),max_tail_nll_bound=float(bb.max()),flagged_contexts=np.where(bb>.01)[0].tolist()))
ekf=ROOT/'outputs/ekf_density_repaired';er=json.loads((ekf/'results.json').read_text())
for c in er['cells']:
    a=read(ekf/c['raw_file'],['nll','component_variance']);bb=bound(a['nll'],a['component_variance'],37.)
    rows.append(dict(kind=c['kind'],seed=c['seed'],n=c['n'],probe_u=c['probe_u'],arm='ekf_repaired',max_nll=float(a['nll'].max()),max_tail_nll_bound=float(bb.max()),flagged_contexts=np.where(bb>.01)[0].tolist()))
particle=ROOT/'outputs/particle_moment_control';pr=json.loads((particle/'results.json').read_text())
for c in pr['cells']:
    with np.load(particle/c['raw_file']) as f:
        for count in [256,1024,4096]:
            nll=f[f'c{count}_nll'];bb=bound(nll,f[f'c{count}_component_noise'])
            rows.append(dict(kind=c['kind'],seed=c['seed'],n=c['n'],probe_u=c['probe_u'],rep=c['importance_rep'],arm=f'particle{count}',max_nll=float(nll.max()),max_tail_nll_bound=float(bb.max()),flagged_contexts=np.where(bb>.01)[0].tolist()))
result=dict(rows=rows,threshold_nats=.01,flagged_rows=sum(bool(r['flagged_contexts']) for r in rows),elapsed_seconds=time.perf_counter()-start,
    interpretation='Conservative absolute log-density truncation error bound using omitted Normal tail mass times maximum Gaussian likelihood; reweighted conditional score bounds numerator+denominator. Does not replace body quadrature refinement checks. A loose bound is not proof of an actual error.')
(ROOT/'outputs/all_context_tail_bounds.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(rows=len(rows),flagged_rows=result['flagged_rows'],max_bound=max(x['max_tail_nll_bound'] for x in rows),max_nll=max(x['max_nll'] for x in rows),elapsed=result['elapsed_seconds'])),flush=True)
