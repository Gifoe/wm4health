"""Attribute frozen reference sensitivity to native parameter blocks, train only."""
from common54 import *
import ctypes, pickle, time
from dataclasses import replace
from jax.flatten_util import ravel_pytree
from rp_ssm.datasets import TrainData
from per_target_auxiliary54_v3 import per_target_loss

fn = ctypes.windll.kernel32.SetProcessAffinityMask
fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert fn(ctypes.c_void_p(-1), ctypes.c_size_t(0xff))

def initialized(seed, B, em):
    energy, cfg = rp_model('linear_small', seed)
    cfg = replace(cfg, batch_size=B, em=em)
    energy.init(jr.PRNGKey(54), TrainData(obs=(jnp.zeros((B, 100, 5)),)), cfg)
    fun = jax.jit(jax.value_and_grad(lambda p, x: energy.loss(p, TrainData(obs=(x,)), 1., em, jr.PRNGKey(54))[0]))
    return energy, fun

def metrics(g, b):
    a = np.asarray(ravel_pytree(g)[0]); c = np.asarray(ravel_pytree(b)[0])
    na = float(np.linalg.norm(a)); nc = float(np.linalg.norm(c))
    return {'dimension': int(a.size), 'full_norm': na, 'balanced_mean_norm': nc,
            'difference_norm': float(np.linalg.norm(c-a)),
            'relative_difference_norm': float(np.linalg.norm(c-a)/max(na, 1e-300)),
            'cosine': float(a@c/(na*nc)) if na*nc > 0 else None,
            'max_abs_difference': float(np.max(np.abs(c-a)))}

def maxdiff(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    return max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))

def main():
    source_check(); out = OUT/'encoder_reference_audit_v1'; assert not out.exists(); out.mkdir()
    with np.load(data_path('linear_small')) as ar: x = ar['train_x']
    start = time.perf_counter(); records = []
    for seed in [0, 1]:
        run = f'rp_linear_small_seed{seed}'
        receipt = json.loads((OUT/'runs'/run/'result.json').read_text())
        checkpoint = pathlib.Path(receipt['checkpoint']); before = sha(checkpoint)
        assert before == receipt['checkpoint_sha256']
        params = pickle.loads(checkpoint.read_bytes()); xx = jnp.asarray(x)
        with np.load(ART/'reference_control_audit_v2'/f'{run}.npz') as ar: ids = ar['balanced_indices']
        assert ids.shape == (25, 32) and np.array_equal(np.bincount(ids.ravel(), minlength=200), np.full(200, 4))
        modes = []; controls = {}
        for em in [False, True]:
            energy, full = initialized(seed, 200, em)
            value, grad = full(params, xx)
            _, batch = initialized(seed, 32, em)
            gs = []; vals = []
            for chosen in ids:
                v, g = batch(params, jnp.asarray(x[chosen])); gs.append(g); vals.append(float(v))
            bg = jax.tree.map(lambda *v: jnp.stack(v).mean(0), *gs)
            leafs = []
            for (key, a), (_, b) in zip(jax.tree_util.tree_flatten_with_path(grad)[0], jax.tree_util.tree_flatten_with_path(bg)[0]):
                leafs.append({'path': jax.tree_util.keystr(key), 'shape': list(a.shape), **metrics(a, b)})
            if not em:
                fv, fg = jax.jit(jax.value_and_grad(lambda p: per_target_loss(energy, p, xx).mean()))(params)
                controls['False_native_vs_per_target_forward_error'] = abs(float(value)-float(fv))
                controls['False_native_vs_per_target_all_gradient_max_error'] = maxdiff(grad, fg)
                assert max(controls.values()) < 1e-7, controls
            else:
                prior = energy.model.prior.update(params[0])
                chain, sites, posterior = energy.get_posterior(None, prior, params[1:], TrainData(obs=(xx,)))
                def frozen_rec_objective(rec):
                    new_sites = energy.model.get_factors((rec,), (xx,))
                    kf, lg, kp = energy.get_loss_terms(chain, new_sites, posterior)
                    return -(lg-kf-kp)/(200*100)
                fv, fg = jax.jit(jax.value_and_grad(frozen_rec_objective))(params[1])
                controls['True_native_vs_fixed_posterior_forward_error'] = abs(float(value)-float(fv))
                controls['True_native_vs_fixed_posterior_rec_gradient_max_error'] = maxdiff(grad[1], fg)
                assert max(controls.values()) < 1e-7, controls
            modes.append({'em': em, 'full_value': float(value), 'balanced_mean_value': float(np.mean(vals)),
                          'A': metrics(grad[0], bg[0]), 'recognition': metrics(grad[1:], bg[1:]),
                          'per_leaf': leafs})
        record = {'run': run, 'checkpoint_sha256': before, 'modes': modes, 'controls': controls,
                  'checkpoint_unchanged': sha(checkpoint) == before,
                  'latent_truth_read': False, 'heldout_read': False, 'optimizer_updates': 0,
                  'protocol_sha256': sha(ROOT/'TRAIN_ONLY_ENCODER_REFERENCE54_V1.md'),
                  'script_sha256': sha(__file__)}
        assert record['checkpoint_unchanged']; records.append(record); dump(out/(run+'.json'), record)
        print(json.dumps(record), flush=True)
    dump(out/'summary.json', {'utc': utc(), 'status': 'COMPLETE_FROZEN_ENCODER_REFERENCE_ATTRIBUTION',
                             'records': records, 'runtime_seconds': time.perf_counter()-start,
                             'native_source_files_unchanged': source_check(),
                             'scope': 'Training-only gradient attribution; no trained quality or novelty claim.'})
    print('ENCODER_REFERENCE_ATTRIBUTION_COMPLETE', flush=True)

if __name__ == '__main__': main()
