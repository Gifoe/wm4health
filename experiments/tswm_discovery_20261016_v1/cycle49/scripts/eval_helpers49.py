"""Instrument the unchanged predictive recursion, retaining actual innovations."""
from world49_v2 import *


def prediction_with_noise(model, count, sums, density, samples, horizon):
    mu, logsd, _ = model.encode(count, sums, density)
    first = torch.randn(len(mu), samples, 8, device=mu.device)
    z = mu[:, None] + logsd.exp()[:, None] * first
    noises, paths = [first], []
    for t in range(horizon):
        innovation = torch.randn_like(z)
        z = z + DT * model.evolve(z) + math.sqrt(DT) * model.innovation() * innovation
        paths.append(model.decoder(z))
        noises.append(innovation)
    return torch.stack(paths, -2), torch.stack(noises, -2)


def energy(paths, truth):
    x = paths.double().reshape(len(paths), paths.shape[1], -1)
    y = truth.double().reshape(len(paths), 1, -1)
    first = torch.linalg.vector_norm(x - y, dim=-1).mean(-1)
    pair = torch.cdist(x, x)
    pair.diagonal(dim1=-2, dim2=-1).zero_()
    return (first - .5 * pair.sum((1, 2)) / (x.shape[1] * (x.shape[1] - 1))) / math.sqrt(x.shape[-1])


def observed_score(paths, counts, sums, density, alpha, omega):
    score = torch.zeros(len(paths), device=paths.device, dtype=torch.float64)
    for start in range(0, len(alpha), 32):
        aa, ww = alpha[start:start + 32], omega[start:start + 32]
        phi = conditional_features(paths, rates(paths, density[:, None, None]), aa, ww, SIGMA)
        target = observed_features(counts, sums, aa, ww)
        score += unbiased_score(phi, target).double().sum(-1)
    return score / len(alpha)
