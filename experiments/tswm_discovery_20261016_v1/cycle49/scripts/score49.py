"""Locked fresh physical-future and observed-future evaluation after all selections."""
from eval_helpers49 import *


def main():
    assert read(R / 'outputs/supervisor_v1.json')['status'] == 'ALL_EIGHT_FITS_COMPLETE_EVAL_UNOPENED'
    qualification = read(R / 'outputs/replay_qualification_v1/qualification.json')
    assert qualification['status'] == 'PASS_TRAIN_ONLY_INDEPENDENT_REPLAY'
    assert qualification['helper_sha256'] == sha(R / 'scripts/eval_helpers49.py')
    assert qualification['independent_source_sha256'] == sha(R / 'scripts/independent49.py')
    fit_paths = sorted((R / 'outputs/fits').glob('*/fit_summary.json'))
    assert len(fit_paths) == 8
    fits = [read(p) for p in fit_paths]
    for fit, fp in zip(fits, fit_paths):
        assert sha(fp.parent / 'selected.pt') == fit['selected_sha256']
    for case in CASES:
        for se in SEEDS:
            pair = [f for f in fits if f['case'] == case and f['seed'] == se]
            assert len(pair) == 2 and len(set(f['initial_predictive_checksum'] for f in pair)) == 1
            assert len(set(f['schedule_sha256'] for f in pair)) == 1
    out = R / 'outputs/eval_v1'
    assert not out.exists()
    out.mkdir()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        # Process-local precision setting; trained CPU weights are unchanged.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dump(out / 'OPENING_RECEIPT.json', {'utc': now(), 'selected': {f['tag']: f['selected_sha256'] for f in fits},
                                       'protocol_sha256': sha(R / 'PROTOCOL.md'), 'EVAL_used_for_selection': False,
                                       'evaluation_device': device, 'source_sha256': sha(pathlib.Path(__file__))})
    manifest = read(R / 'data/MANIFEST.json')
    summaries, rows, raw, bundles = [], [], [], []
    for ci, case in enumerate(CASES):
        data = load(case, 'eval', device)
        # This is the first access to clean EVAL future truth in the pipeline.
        clean = np.load(R / 'data' / case / 'eval_physical_truth.npz')['signal']
        identity = np.load(R / 'data' / case / 'eval_observed.npz')['physical_id']
        truth_all = torch.tensor(clean[identity], device=device)
        scale = manifest['description'][case]['encoder_probe_scale_train_only']
        gen = torch.Generator(device=device).manual_seed(20264990 + ci)
        aa, ww = probes(512, scale, device, gen)
        probe_path = out / (case + '__eval_probes.npz')
        np.savez_compressed(probe_path, count=aa.cpu().numpy(), mark=ww.cpu().numpy())
        for fit in [f for f in fits if f['case'] == case]:
            model = World(scale).to(device).eval()
            checkpoint_path = R / 'outputs/fits' / fit['tag'] / 'selected.pt'
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)['world'])
            weight_path = out / (fit['tag'] + '__weights.npz')
            np.savez_compressed(weight_path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
            bundles.append({'model': fit['tag'], 'path': weight_path.name, 'sha256': sha(weight_path), 'selected_sha256': fit['selected_sha256']})
            for reveal in [0, 12]:
                horizon = T - reveal
                count = data['counts'][:, reveal:T + reveal]
                sums = data['sums'][:, reveal:T + reveal]
                density = data['density']
                target_n = data['counts'][:, T + reveal:]
                target_s = data['sums'][:, T + reveal:]
                truth = truth_all[:, T + reveal:]
                seed(fit['seed'] + ci * 100 + reveal + 701)
                parts = {'prediction': [], 'noise': [], 'energy': [], 'tail12_energy': [], 'mse': [],
                         'tail12_mse': [], 'observation_score': [], 'tail12_observation_score': []}
                begin = time.perf_counter()
                with torch.no_grad():
                    for start in range(0, len(count), 8):
                        sl = slice(start, start + 8)
                        prediction, noise = prediction_with_noise(model, count[sl], sums[sl], density[sl], 128, horizon)
                        values = {'prediction': prediction, 'noise': noise,
                                  'energy': energy(prediction, truth[sl]),
                                  'tail12_energy': energy(prediction[:, :, -12:], truth[sl, -12:]),
                                  'mse': (prediction.mean(1) - truth[sl]).double().square().mean((1, 2)),
                                  'tail12_mse': (prediction[:, :, -12:].mean(1) - truth[sl, -12:]).double().square().mean((1, 2)),
                                  'observation_score': observed_score(prediction, target_n[sl], target_s[sl], density[sl], aa[:, -horizon:], ww[:, -horizon:]),
                                  'tail12_observation_score': observed_score(prediction[:, :, -12:], target_n[sl, -12:], target_s[sl, -12:], density[sl], aa[:, -12:], ww[:, -12:])}
                        for key, value in values.items():
                            assert torch.isfinite(value).all(), (fit['tag'], reveal, key)
                            parts[key].append(value.cpu().numpy())
                arrays = {k: np.concatenate(v, 0) for k, v in parts.items()}
                path = out / (fit['tag'] + f'__reveal{reveal}.npz')
                np.savez_compressed(path, **arrays, truth=truth.cpu().numpy(), counts=count.cpu().numpy(),
                                    sums=sums.cpu().numpy(), density=density.cpu().numpy(), physical_id=identity,
                                    target_counts=target_n.cpu().numpy(), target_sums=target_s.cpu().numpy())
                raw.append({'model': fit['tag'], 'case': case, 'arm': fit['arm'], 'seed': fit['seed'], 'reveal': reveal,
                            'horizon': horizon, 'path': path.name, 'sha256': sha(path), 'contexts': len(count),
                            'probe_path': probe_path.name, 'probe_sha256': sha(probe_path), 'elapsed_seconds': time.perf_counter() - begin})
                metric_keys = ['energy', 'tail12_energy', 'mse', 'tail12_mse', 'observation_score', 'tail12_observation_score']
                for d in [.25, 1., 4.]:
                    mask = arrays['energy'] == arrays['energy']
                    mask &= density.cpu().numpy() == d
                    group = {'model': fit['tag'], 'case': case, 'arm': fit['arm'], 'seed': fit['seed'], 'reveal': reveal,
                             'horizon': horizon, 'density': d, 'physical_paths': int(mask.sum()),
                             **{k: float(arrays[k][mask].mean()) for k in metric_keys}}
                    summaries.append(group)
                    print(json.dumps(group), flush=True)
                for i in range(len(count)):
                    rows.append({'model': fit['tag'], 'case': case, 'arm': fit['arm'], 'seed': fit['seed'], 'reveal': reveal,
                                 'density': float(density[i].cpu()), 'physical_id': int(identity[i]), **{k: float(arrays[k][i]) for k in metric_keys}})
                assert sha(checkpoint_path) == fit['selected_sha256']
    dump(out / 'model_bundles.json', bundles)
    dump(out / 'raw_manifest.json', raw)
    dump(out / 'all_rows.json', rows)
    dump(out / 'summary.json', {'status': 'SCORED_PENDING_INDEPENDENT_VERIFICATION', 'utc': now(), 'groups': summaries,
                               'raw_groups': len(raw), 'model_context_reveal_entries': len(rows), 'unique_physical_paths': 512,
                               'paired_density_history_contexts': 1536, 'independent_real_domains': 0,
                               'all_rows_sha256': sha(out / 'all_rows.json'), 'raw_manifest_sha256': sha(out / 'raw_manifest.json'),
                               'source_sha256': sha(pathlib.Path(__file__))})


if __name__ == '__main__':
    main()
