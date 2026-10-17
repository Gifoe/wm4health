"""Independent NumPy replay and observation scoring; no Torch model imports."""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def replay(weights, counts, sums, density, noise):
    w = {k: np.asarray(v, dtype='float64') for k, v in weights.items()}
    inputs = np.concatenate((counts[..., None] / w['scale'][0], sums / w['scale'][1:],
                             np.broadcast_to(density[:, None, None], (*counts.shape, 1))), -1)
    hidden = np.zeros((len(counts), 32))
    for t in range(counts.shape[1]):
        left = inputs[:, t] @ w['encoder.weight_ih_l0'].T + w['encoder.bias_ih_l0']
        right = hidden @ w['encoder.weight_hh_l0'].T + w['encoder.bias_hh_l0']
        ir, iz, inn = np.split(left, 3, -1)
        hr, hz, hn = np.split(right, 3, -1)
        reset, update = sigmoid(ir + hr), sigmoid(iz + hz)
        candidate = np.tanh(inn + reset * hn)
        hidden = (1 - update) * candidate + update * hidden
    mean, logsd = np.split(hidden @ w['initial.weight'].T + w['initial.bias'], 2, -1)
    z = mean[:, None] + np.exp(np.clip(logsd, -3, 1))[:, None] * noise[:, :, 0]
    innovation = .02 + np.logaddexp(0, w['raw_innovation'])
    paths = []
    for t in range(noise.shape[-2] - 1):
        field = z
        for j in [0, 2, 4]:
            field = field @ w[f'drift.{j}.weight'].T + w[f'drift.{j}.bias']
            if j != 4:
                field = np.tanh(field)
        z = z + .15 * 2 * np.tanh(field) + np.sqrt(.15) * innovation * noise[:, :, t + 1]
        paths.append(z @ w['decoder.weight'].T + w['decoder.bias'])
    return np.stack(paths, -2)


def observed_score(paths, counts, sums, density, alpha, omega):
    n = paths.shape[1]
    answer = np.zeros(len(paths), dtype='float64')
    rate = density[:, None, None] * (.3 + 1.2 * sigmoid(.9 * paths[..., 0] + .4 * paths[..., 1]))
    total_rate = np.sum(rate, -1)
    for start in range(0, len(alpha), 32):
        aa, ww = alpha[start:start + 32], omega[start:start + 32]
        phase = np.einsum('bmtd,ftd->bmft', paths, ww) + aa
        attenuation = np.exp(-.5 * .35**2 * np.sum(ww**2, -1))
        weight = rate[:, :, None] * attenuation
        real = np.sum(weight * np.cos(phase), -1) - total_rate[:, :, None]
        imaginary = np.sum(weight * np.sin(phase), -1)
        phi = np.exp(real) * (np.cos(imaginary) + 1j * np.sin(imaginary))
        total = np.sum(phi, 1)
        pair = (np.abs(total)**2 - np.sum(np.abs(phi)**2, 1)) / (n * (n - 1))
        target = np.exp(1j * (np.einsum('bt,ft->bf', counts, aa) + np.einsum('btd,ftd->bf', sums, ww)))
        answer += np.sum(pair - 2 * np.real(np.mean(phi, 1) * np.conjugate(target)), -1)
    return answer / len(alpha)


def energy(paths, truth):
    x = paths.reshape(len(paths), paths.shape[1], -1).astype('float64')
    y = truth.reshape(len(paths), 1, -1).astype('float64')
    distance = np.sqrt(np.sum((x - y)**2, -1)).mean(-1)
    square = np.sum(x**2, -1)
    pairs = np.maximum(square[:, :, None] + square[:, None, :] - 2 * x @ x.transpose(0, 2, 1), 0)
    pairs = np.sqrt(pairs)
    idx = np.arange(x.shape[1])
    pairs[:, idx, idx] = 0
    return (distance - .5 * pairs.sum((1, 2)) / (x.shape[1] * (x.shape[1] - 1))) / np.sqrt(x.shape[-1])
