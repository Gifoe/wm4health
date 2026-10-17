"""Fresh reproducible controlled episodes; physical future stored separately."""
from world49_v2 import *


def source(case, count, rng):
    x = rng.normal(size=(count, 2))
    path = []
    for k in range(64 + 48 * 3):
        if case == 'rotating':
            drift = np.stack((-.4 * x[:, 0] - 1.8 * x[:, 1], 1.8 * x[:, 0] - .4 * x[:, 1]), -1)
            sd = np.array([.4, .4])
        else:
            drift = np.stack((.8 * x[:, 0] - .5 * x[:, 0]**3 + .4 * x[:, 1], -.7 * x[:, 1] - .4 * x[:, 0]), -1)
            sd = np.array([.45, .3])
        x = x + .05 * drift + math.sqrt(.05) * sd * rng.normal(size=x.shape)
        assert np.isfinite(x).all()
        if k >= 64 and (k - 64) % 3 == 2:
            path.append(x.copy())
    return np.stack(path, 1).astype('float32')


def save_unchanged(path, **arrays):
    if path.exists():
        previous = np.load(path)
        assert set(previous.files) == set(arrays)
        assert all(np.array_equal(previous[k], v) for k, v in arrays.items()), str(path)
    else:
        np.savez_compressed(path, **arrays)


def main():
    assert not (R / 'data/MANIFEST.json').exists()
    records, description = [], {}
    for ci, case in enumerate(CASES):
        folder = R / 'data' / case
        folder.mkdir(parents=True, exist_ok=True)
        for si, (split, count) in enumerate([('train', 1024), ('val', 128), ('eval', 256)]):
            rng = np.random.default_rng(20264900 + 100 * ci + si)
            physical = source(case, count, rng)
            if split == 'eval':
                density = np.repeat(np.array([.25, 1., 4.]), count).astype('float32')
                ids = np.tile(np.arange(count), 3)
                signal = physical[ids]
            else:
                density = (rng.choice([.5, 1., 2.], count) if split == 'train' else np.ones(count)).astype('float32')
                ids = np.arange(count)
                signal = physical
            rate = density[:, None] * (.3 + 1.2 / (1 + np.exp(-(.9 * signal[..., 0] + .4 * signal[..., 1]))))
            counts = rng.poisson(rate).astype('float32')
            sums = counts[..., None] * signal + np.sqrt(counts[..., None]) * SIGMA * rng.normal(size=signal.shape)
            path = folder / (split + '_observed.npz')
            save_unchanged(path, counts=counts, sums=sums.astype('float32'), density=density, physical_id=ids)
            truth_path = folder / (split + '_physical_truth.npz')
            save_unchanged(truth_path, signal=physical)
            for item in [path, truth_path]:
                records.append({'path': item.relative_to(R / 'data').as_posix(), 'bytes': item.stat().st_size, 'sha256': sha(item)})
            if split == 'train':
                scales = [max(float(np.sqrt(np.mean(counts**2))), .1)] + np.maximum(np.sqrt(np.mean(sums**2, axis=(0, 1))), .1).tolist()
                description[case] = {'encoder_probe_scale_train_only': scales, 'train_physical_paths': 1024, 'val_physical_paths': 128, 'eval_physical_paths': 256, 'eval_paired_density_contexts': 768}
                gen = torch.Generator().manual_seed(20264901 + ci)
                alpha, omega = probes(256, scales, 'cpu', gen)
                pp = folder / 'validation_probes.npz'
                save_unchanged(pp, count=alpha.numpy(), mark=omega.numpy())
                records.append({'path': pp.relative_to(R / 'data').as_posix(), 'bytes': pp.stat().st_size, 'sha256': sha(pp)})
    dump(R / 'data' / 'MANIFEST.json', {'status': 'FROZEN_FRESH_CONTROLLED_DATA', 'utc': now(), 'protocol_sha256': sha(R / 'PROTOCOL.md'), 'source_sha256': sha(pathlib.Path(__file__)), 'files': records, 'description': description})
    print(json.dumps({'status': 'FROZEN_FRESH_CONTROLLED_DATA', 'files': len(records), 'bytes': sum(r['bytes'] for r in records)}), flush=True)


if __name__ == '__main__':
    main()
