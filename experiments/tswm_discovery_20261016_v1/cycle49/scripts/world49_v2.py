"""Original shared-core conditional latent Markov model for a controlled screen."""
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
from torch import nn
from measurement49 import conditional_features, observed_features, unbiased_score

CASES = ['rotating', 'bistable']
ARMS = ['rb_score', 'full_clock_elbo']
SEEDS = [2026491, 2026492]
T = 24
DT = 0.15
SIGMA = 0.35


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def net(din, dout, hidden=64):
    return nn.Sequential(nn.Linear(din, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, dout))


def rates(signal, density):
    return density * (0.3 + 1.2 * torch.sigmoid(0.9 * signal[..., 0] + 0.4 * signal[..., 1]))


def probes(n, scale, device, generator=None):
    scale = torch.as_tensor(scale, device=device, dtype=torch.float32)
    count = torch.zeros(n, T, device=device)
    mark = torch.zeros(n, T, 2, device=device)
    spans = [1, 4, 8, 24]
    factors = [0.3, 1., 3.]
    for j in range(n):
        span = spans[j % 4]
        offset = int(torch.randint(T - span + 1, (1,), device=device, generator=generator))
        factor = factors[(j // 4) % 3] / math.sqrt(span)
        count[j, offset:offset + span] = torch.randn(span, device=device, generator=generator) * factor / scale[0]
        mark[j, offset:offset + span] = torch.randn(span, 2, device=device, generator=generator) * factor / scale[1:]
    return count, mark


def load(case, split, device):
    # Observation files deliberately do not contain clean physical truth.
    z = np.load(R / 'data' / case / (split + '_observed.npz'))
    assert set(z.files) == {'counts', 'sums', 'density', 'physical_id'}
    return {k: torch.tensor(z[k], device=device, dtype=torch.float32) for k in ['counts', 'sums', 'density']}


class World(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        self.encoder = nn.GRU(4, 32, batch_first=True)
        self.initial = nn.Linear(32, 16)
        self.drift = net(8, 8)
        self.raw_innovation = nn.Parameter(torch.full((8,), -1.8))
        self.decoder = nn.Linear(8, 2)

    def encode(self, counts, sums, density):
        a = density.reshape(-1, 1, 1).expand(-1, counts.shape[-1], 1)
        inputs = torch.cat((counts[..., None] / self.scale[0], sums / self.scale[1:], a), -1)
        _, hidden = self.encoder(inputs)
        context = hidden[-1]
        mu, logsd = self.initial(context).chunk(2, -1)
        return mu, logsd.clamp(-3, 1), context

    def innovation(self):
        return 0.02 + torch.nn.functional.softplus(self.raw_innovation)

    def evolve(self, z):
        return 2 * torch.tanh(self.drift(z))

    def predict(self, counts, sums, density, samples, horizon=T):
        mu, logsd, _ = self.encode(counts, sums, density)
        z = mu[:, None] + logsd.exp()[:, None] * torch.randn(len(mu), samples, 8, device=mu.device)
        paths = []
        for t in range(horizon):
            z = z + DT * self.evolve(z) + math.sqrt(DT) * self.innovation() * torch.randn_like(z)
            paths.append(self.decoder(z))
        return torch.stack(paths, -2)


class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.future = nn.GRU(3, 32, batch_first=True)
        self.initial = nn.Linear(64, 16)
        self.control = net(72, 8)
        nn.init.zeros_(self.initial.weight)
        nn.init.zeros_(self.initial.bias)
        nn.init.zeros_(self.control[-1].weight)
        nn.init.zeros_(self.control[-1].bias)


def observation_loglik(signal, counts, sums, density):
    expected = rates(signal, density[:, None, None])
    n = counts[:, None]
    likelihood = n * expected.log() - expected - torch.lgamma(n + 1)
    variance = n.clamp_min(1)[..., None] * SIGMA**2
    error = sums[:, None] - n[..., None] * signal
    gaussian = -0.5 * (error.square() / variance + torch.log(2 * math.pi * variance))
    likelihood = likelihood + torch.where(n[..., None] > 0, gaussian, 0).sum(-1)
    return likelihood.sum(-1)


def elbo_loss(model, recognition, batch):
    n, s, a = batch['counts'], batch['sums'], batch['density']
    pm, pl, history = model.encode(n[:, :T], s[:, :T], a)
    future_inputs = torch.cat((n[:, T:, None] / model.scale[0], s[:, T:] / model.scale[1:]), -1)
    reverse, _ = recognition.future(future_inputs.flip(1))
    ctx = reverse.flip(1)
    dm, dl = recognition.initial(torch.cat((history, ctx[:, 0]), -1)).chunk(2, -1)
    qm, ql = pm + dm, (pl + dl).clamp(-3, 1)
    initial_kl = (pl - ql + (torch.exp(2 * ql) + (qm - pm).square()) / (2 * torch.exp(2 * pl)) - .5).sum(-1)
    z = qm[:, None] + ql.exp()[:, None] * torch.randn(len(qm), 4, 8, device=qm.device)
    transition_kl = torch.zeros(len(qm), 4, device=qm.device)
    paths = []
    for t in range(T):
        info = torch.cat((z, history[:, None].expand(-1, 4, -1), ctx[:, t, None].expand(-1, 4, -1)), -1)
        correction = 2 * torch.tanh(recognition.control(info))
        transition_kl = transition_kl + .5 * DT * (correction / model.innovation()).square().sum(-1)
        z = z + DT * (model.evolve(z) + correction) + math.sqrt(DT) * model.innovation() * torch.randn_like(z)
        paths.append(model.decoder(z))
    loglik = observation_loglik(torch.stack(paths, -2), n[:, T:], s[:, T:], a)
    return (initial_kl[:, None] + transition_kl - loglik).mean() / T


def rb_loss(model, batch, alpha, omega, samples=8):
    n, s, a = batch['counts'], batch['sums'], batch['density']
    paths = model.predict(n[:, :T], s[:, :T], a, samples)
    phi = conditional_features(paths, rates(paths, a[:, None, None]), alpha, omega, SIGMA)
    truth = observed_features(n[:, T:], s[:, T:], alpha, omega)
    return unbiased_score(phi, truth).mean()


def validate(model, data, alpha, omega):
    values = []
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if alpha.device.type == 'cuda' else []):
        seed(902649)
        for i in range(0, len(data['counts']), 16):
            batch = {k: v[i:i + 16] for k, v in data.items()}
            n, s, a = batch['counts'], batch['sums'], batch['density']
            y = model.predict(n[:, :T], s[:, :T], a, 32)
            phi = conditional_features(y, rates(y, a[:, None, None]), alpha, omega, SIGMA)
            target = observed_features(n[:, T:], s[:, T:], alpha, omega)
            values.extend(unbiased_score(phi, target).mean(-1).cpu().tolist())
    return float(np.mean(values))
