"""Standard Bayes squared-error arithmetic and sequence bootstrap, no model access."""
import numpy as np

def decompose(pred,oracle,truth):
    assert pred.shape==oracle.shape==truth.shape and pred.ndim==3
    delta=pred-oracle;noise=oracle-truth
    result={'factual':((pred-truth)**2).mean(-1),'oracle_noise':(noise**2).mean(-1),'excess_conditional':(delta**2).mean(-1),'finite_sample_cross':2*(delta*noise).mean(-1)}
    assert all(np.isfinite(v).all() for v in result.values())
    reconstructed=result['oracle_noise']+result['excess_conditional']+result['finite_sample_cross']
    error=float(np.max(abs(result['factual']-reconstructed)))
    assert error<1e-10*max(1.,float(np.max(abs(result['factual']))))
    return result,error

def ratio(numerator,denominator):
    if denominator<=0:return None
    value=float(numerator/denominator)
    return value if np.isfinite(value) else None

def percentile(values,q):
    a=np.sort(np.asarray(values));assert a.ndim==1 and len(a)>0 and np.isfinite(a).all()
    position=(len(a)-1)*q;i=int(np.floor(position));j=int(np.ceil(position))
    return float(a[i]+(a[j]-a[i])*(position-i))

def paired_interval(delta,energy,indices):
    assert delta.shape==energy.shape and delta.ndim==2 and indices.ndim==2
    points=delta.mean(0);denominator=energy.mean(0)
    boots=delta[indices].mean(1);energy_boots=energy[indices].mean(1)
    rows=[]
    for h in range(delta.shape[1]):
        normalized=boots[:,h]/energy_boots[:,h] if np.all(energy_boots[:,h]>0) else None
        if normalized is not None and not np.isfinite(normalized).all():normalized=None
        rows.append({'delta_mean':float(points[h]),'pointwise95_interval':[percentile(boots[:,h],.025),percentile(boots[:,h],.975)],'predictable_energy':float(denominator[h]),'normalized_delta':ratio(points[h],denominator[h]),'normalized_pointwise95_interval':[percentile(normalized,.025),percentile(normalized,.975)] if normalized is not None else None})
    return rows
