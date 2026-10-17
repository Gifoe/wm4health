"""TRAIN-only qualification of evaluation instrumentation and NumPy replay."""
from eval_helpers49 import *
from independent49 import replay, observed_score as numpy_score, energy as numpy_energy


def main():
    out = R / 'outputs/replay_qualification_v1'
    assert not out.exists()
    out.mkdir(parents=True)
    manifest = read(R / 'data/MANIFEST.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = {k: v[:2] for k, v in load('rotating', 'train', device).items()}
    model = World(manifest['description']['rotating']['encoder_probe_scale_train_only']).to(device).eval()
    n, s, a = data['counts'], data['sums'], data['density']
    records = []
    for horizon in [12, 24]:
        seed(20264903 + horizon)
        with torch.no_grad():
            native = model.predict(n[:, :T], s[:, :T], a, 8, horizon)
            seed(20264903 + horizon)
            primary, noise = prediction_with_noise(model, n[:, :T], s[:, :T], a, 8, horizon)
            assert torch.equal(primary, native)
            weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            independent = replay(weights, n[:, :T].cpu().numpy(), s[:, :T].cpu().numpy(), a.cpu().numpy(), noise.cpu().numpy())
            model_error = float(np.max(abs(primary.cpu().numpy() - independent)))
            assert np.allclose(primary.cpu().numpy(), independent, atol=3e-4, rtol=3e-4)
            aa, ww = probes(64, model.scale, device)
            aa, ww = aa[:, -horizon:], ww[:, -horizon:]
            nc, sc = n[:, -horizon:], s[:, -horizon:]
            raw = primary.double().cpu().numpy()
            ps = observed_score(primary, nc, sc, a, aa, ww).cpu().numpy()
            ns = numpy_score(raw, nc.cpu().numpy(), sc.cpu().numpy(), a.cpu().numpy(), aa.cpu().numpy(), ww.cpu().numpy())
            score_error = float(np.max(abs(ps - ns)))
            assert score_error < 2e-5
            # An observed mark sum is only a numeric target for this arithmetic check.
            truth = sc
            pe = energy(primary, truth).cpu().numpy()
            ne = numpy_energy(raw, truth.cpu().numpy())
            energy_error = float(np.max(abs(pe - ne)))
            assert energy_error < 2e-8
            records.append({'horizon': horizon, 'native_instrumented_difference': 0., 'numpy_model_max_error': model_error,
                            'numpy_observation_score_error': score_error, 'numpy_energy_error': energy_error})
    dump(out / 'qualification.json', {'status': 'PASS_TRAIN_ONLY_INDEPENDENT_REPLAY', 'utc': now(), 'device': device,
                                     'records': records, 'formal_weights_used': False, 'eval_used': False,
                                     'numpy_model_tolerances': {'atol': 3e-4, 'rtol': 3e-4},
                                     'observation_score_absolute_tolerance': 2e-5, 'energy_absolute_tolerance': 2e-8,
                                     'source_sha256': sha(pathlib.Path(__file__)),
                                     'independent_source_sha256': sha(R / 'scripts/independent49.py'),
                                     'helper_sha256': sha(R / 'scripts/eval_helpers49.py')})
    print(json.dumps({'status': 'PASS_TRAIN_ONLY_INDEPENDENT_REPLAY', 'records': records}), flush=True)


if __name__ == '__main__':
    main()
