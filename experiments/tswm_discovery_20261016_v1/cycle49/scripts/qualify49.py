"""No-training numerical qualification; not evidence of predictive advantage."""
import datetime
import hashlib
import json
import math
import pathlib
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R.parent / 'cycle29' / 'scripts'))
import runtime_profile
import numpy as np
import torch
from measurement49 import conditional_features, observed_features, unbiased_score


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = R / 'outputs' / 'qualification_v1'
    assert not out.exists()
    out.mkdir(parents=True)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2026491)
    start = time.perf_counter()
    means = torch.tensor([[[0.35, -0.8], [0.2, 0.6]]])
    rates = torch.tensor([[0.7, 1.4]])
    alpha = torch.tensor([[0.4, -0.2], [-0.1, 0.6], [0.0, 0.0]])
    omega = torch.tensor([[[0.3, -0.7], [0.2, 0.1]], [[-0.4, 0.5], [0.9, 0.2]], [[0.0, 0.0], [0.0, 0.0]]])
    sigma = torch.tensor([0.3, 0.5])
    native = conditional_features(means, rates, alpha, omega, sigma).numpy()[0]
    # Independent scalar Poisson summation and Gauss-Hermite integration.
    nodes, weights = np.polynomial.hermite.hermgauss(32)
    weights = weights / math.sqrt(math.pi)
    enumeration = []
    for f in range(len(alpha)):
        product = 1 + 0j
        for t in range(2):
            value = 0j
            rate = float(rates[0, t])
            probability = math.exp(-rate)
            for n in range(25):
                noise = 1 + 0j
                for d in range(2):
                    if n:
                        points = n * float(means[0, t, d]) + math.sqrt(2 * n) * float(sigma[d]) * nodes
                        noise *= np.dot(weights, np.exp(1j * float(omega[f, t, d]) * points))
                value += probability * np.exp(1j * n * float(alpha[f, t])) * noise
                probability *= rate / (n + 1)
            product *= value
        enumeration.append(product)
    enumeration_error = float(np.max(np.abs(native - enumeration)))
    assert enumeration_error < 2e-12
    assert abs(native[-1] - 1) < 1e-15
    # Independently check the raw count/sum feature packing.
    counts = torch.tensor([[2., 0.]])
    sums = torch.tensor([[[0.9, -1.1], [0., 0.]]])
    observed = observed_features(counts, sums, alpha, omega)
    packed = np.array([complex(np.exp(1j * sum(float(alpha[f, t]) * float(counts[0, t])
                        + sum(float(omega[f, t, d]) * float(sums[0, t, d]) for d in range(2)) for t in range(2)))) for f in range(3)])
    assert np.max(abs(observed.numpy()[0] - packed)) < 1e-14
    # The scalar perturbation changes BOTH the observed mark and its acquisition rate.
    paths = torch.tensor([[[0.35, -0.8], [0.2, 0.6]], [[-0.4, 0.1], [0.8, -0.2]], [[0.6, 0.3], [-0.2, 0.4]]])
    direction = torch.tensor([[[0.2, -0.3], [0.7, 0.2]]])

    def loss(theta):
        signal = paths + theta * direction
        count_rate = 0.2 + torch.nn.functional.softplus(0.6 * signal[..., 0])
        feature = conditional_features(signal, count_rate, alpha, omega, sigma)
        return unbiased_score(feature, observed[0]).mean()

    theta = torch.tensor(0.15, requires_grad=True)
    value = loss(theta)
    grad = float(torch.autograd.grad(value, theta)[0])
    eps = 1e-5
    finite = float((loss(theta.detach() + eps) - loss(theta.detach() - eps)) / (2 * eps))
    gradient_error = abs(grad - finite)
    assert gradient_error < 2e-9
    # Exhaustive all ensembles for a discrete latent law proves expectation and
    # exposes the finite-M diagonal bias without relying on Monte Carlo tolerances.
    phi = conditional_features(paths, 0.2 + torch.nn.functional.softplus(0.6 * paths[..., 0]), alpha, omega, sigma)
    population = phi.mean(0)
    expected = population.abs().square() - 2 * (population * observed[0].conj()).real
    unbiased = []
    biased = []
    for a in range(3):
        for b in range(3):
            ensemble = phi[[a, b]]
            unbiased.append(unbiased_score(ensemble, observed[0]))
            empirical = ensemble.mean(0)
            biased.append(empirical.abs().square() - 2 * (empirical * observed[0].conj()).real)
    expectation_error = float((torch.stack(unbiased).mean(0) - expected).abs().max())
    bias = torch.stack(biased).mean(0) - expected
    predicted_bias = (phi.abs().square().mean(0) - population.abs().square()) / 2
    assert expectation_error < 1e-14 and float((bias - predicted_bias).abs().max()) < 1e-14
    assert float(bias.max()) > 1e-5
    # Record actual end-to-end forward/backward throughput on both devices.
    # No optimizer update, no trained weights and no evaluation set are created.
    timings = []
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        dt = torch.float32
        x = torch.randn(16, 8, 24, 2, device=device, dtype=dt, requires_grad=True)
        aa = torch.randn(64, 24, device=device, dtype=dt) / math.sqrt(24)
        ww = torch.randn(64, 24, 2, device=device, dtype=dt) / math.sqrt(48)
        ss = sigma.to(device=device, dtype=dt)
        target = torch.ones(16, 64, device=device, dtype=torch.complex64)
        times = []
        for i in range(8):
            if device == 'cuda':
                torch.cuda.synchronize()
            begin = time.perf_counter()
            p = conditional_features(x, 0.2 + torch.nn.functional.softplus(x[..., 0]), aa, ww, ss)
            objective = unbiased_score(p, target).mean()
            g = torch.autograd.grad(objective, x)[0]
            assert torch.isfinite(g).all()
            if device == 'cuda':
                torch.cuda.synchronize()
            if i >= 3:
                times.append(time.perf_counter() - begin)
        timings.append({'device': device, 'median_forward_backward_seconds': float(np.median(times)),
                        'scope': 'observation transform/score only; no encoder or SDE, not a training speedup'})
    result = {'status': 'PASS_MARKED_OBSERVATION_TRANSFORM_QUALIFICATION_ONLY',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'formal_training_updates': 0, 'eval_opened': False,
              'max_independent_poisson_gaussian_quadrature_error': enumeration_error,
              'coupled_signal_clock_gradient': grad, 'finite_difference_gradient': finite,
              'gradient_absolute_error': gradient_error,
              'finite_ensemble_unbiased_expectation_error': expectation_error,
              'finite_ensemble_naive_square_positive_bias': bias.tolist(),
              'timings': timings, 'elapsed_seconds': time.perf_counter() - start,
              'classical_components_not_novel': ['marked Poisson characteristic functional', 'Rao-Blackwellization', 'kernel score', 'U-statistic'],
              'does_not_prove': ['new research gap', 'trained future-law improvement', 'physical latent identification', 'complete-clock likelihood weakness'],
              'measurement_source_sha256': sha(R / 'scripts' / 'measurement49.py'),
              'source_sha256': sha(pathlib.Path(__file__))}
    (out / 'qualification.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
