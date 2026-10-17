"""Gaussian fixed-horizon scores; no fitting, covariance repair or model access."""
import numpy as np

def scores(mean, covariance, oracle_mean, oracle_covariance, truth, cutoff):
    assert mean.shape == oracle_mean.shape == truth.shape
    assert covariance.shape == oracle_covariance.shape == mean.shape + (mean.shape[-1],)
    K = mean.shape[-1]
    arrays = [mean, covariance, oracle_mean, oracle_covariance, truth]
    assert all(np.isfinite(a).all() for a in arrays)
    assert max(np.max(abs(covariance-covariance.swapaxes(-1,-2))), np.max(abs(oracle_covariance-oracle_covariance.swapaxes(-1,-2)))) < 1e-8
    ev = np.linalg.eigvalsh(covariance); eo = np.linalg.eigvalsh(oracle_covariance)
    valid = ev.min(-1) > 0
    assert np.all(eo > 0), 'The true conditional oracle covariance must be positive definite'
    shape = mean.shape[:-1]
    result = {k: np.full(shape, np.nan) for k in ['gaussian_kl', 'mean_kl', 'covariance_kl', 'learned_nll', 'oracle_nll', 'realized_nll_excess', 'ellipsoid_covered', 'mahalanobis_squared']}
    result['valid_covariance'] = valid
    result['minimum_covariance_eigenvalue'] = ev.min(-1)
    if not np.any(valid):return result
    cm = covariance[valid]; co = oracle_covariance[valid]
    mm = mean[valid]; mo = oracle_mean[valid]; z = truth[valid]
    _, lm = np.linalg.slogdet(cm); _, lo = np.linalg.slogdet(co)
    delta = mm-mo
    mean_term = .5*np.einsum('ni,ni->n', delta, np.linalg.solve(cm, delta[...,None])[...,0])
    cov_term = .5*(np.trace(np.linalg.solve(cm, co), axis1=-2, axis2=-1)-K+lm-lo)
    assert np.min(cov_term) > -1e-8, 'Negative covariance KL beyond arithmetic roundoff'
    dm = z-mm; do = z-mo
    md = np.einsum('ni,ni->n', dm, np.linalg.solve(cm, dm[...,None])[...,0])
    od = np.einsum('ni,ni->n', do, np.linalg.solve(co, do[...,None])[...,0])
    learned_nll = .5*(K*np.log(2*np.pi)+lm+md)
    oracle_nll = .5*(K*np.log(2*np.pi)+lo+od)
    values = dict(gaussian_kl=mean_term+cov_term, mean_kl=mean_term, covariance_kl=cov_term, learned_nll=learned_nll, oracle_nll=oracle_nll, realized_nll_excess=learned_nll-oracle_nll, ellipsoid_covered=(md <= cutoff).astype(float), mahalanobis_squared=md)
    for key, value in values.items():result[key][valid] = value
    return result
