"""Discarded end-to-end model/likelihood checks before fresh formal fitting."""
from world49 import *


def main():
    assert read(R / 'outputs/qualification_v1/qualification.json')['status'] == 'PASS_MARKED_OBSERVATION_TRANSFORM_QUALIFICATION_ONLY'
    out = R / 'outputs/model_qualification_v1'
    assert not out.exists()
    out.mkdir(parents=True)
    manifest = read(R / 'data/MANIFEST.json')
    scale = manifest['description']['rotating']['encoder_probe_scale_train_only']
    # Independent scalar likelihood arithmetic, including empty observation bins.
    signal = torch.tensor([[[[.2, -.3], [.5, .7]]]], dtype=torch.float64)
    count = torch.tensor([[0., 2.]], dtype=torch.float64)
    sums = torch.tensor([[[0., 0.], [.6, 1.1]]], dtype=torch.float64)
    density = torch.tensor([.7], dtype=torch.float64)
    native = float(observation_loglik(signal, count, sums, density)[0, 0])
    reference = 0.
    for t in range(2):
        y = signal[0, 0, t].numpy()
        rate = .7 * (.3 + 1.2 / (1 + math.exp(-(.9 * y[0] + .4 * y[1]))))
        n = int(count[0, t])
        reference += n * math.log(rate) - rate - math.lgamma(n + 1)
        if n:
            for d in range(2):
                error = float(sums[0, t, d]) - n * y[d]
                reference += -.5 * (error**2 / (n * SIGMA**2) + math.log(2 * math.pi * n * SIGMA**2))
    assert abs(native - reference) < 1e-12
    # Gaussian-step KL: compare the exact formula to independent log-density ratios.
    gen = np.random.default_rng(2026492)
    pmean = np.array([.1, -.2]); delta = np.array([.25, -.1]); sd = np.array([.3, .4])
    qmean = pmean + DT * delta
    qdraw = qmean + math.sqrt(DT) * sd * gen.normal(size=(100000, 2))
    logratio = (((qdraw - pmean) / sd)**2 - ((qdraw - qmean) / sd)**2).sum(-1) / (2 * DT)
    exact_kl = float(.5 * DT * ((delta / sd)**2).sum())
    se = float(logratio.std(ddof=1) / math.sqrt(len(logratio)))
    assert abs(float(logratio.mean()) - exact_kl) < 5 * se
    records = []
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        data = load('rotating', 'train', device)
        batch = {k: v[:32] for k, v in data.items()}
        for arm in ARMS:
            seed(2026491)
            model = World(scale).to(device)
            recognition = Recognizer().to(device) if arm == 'full_clock_elbo' else None
            params = list(model.parameters()) + (list(recognition.parameters()) if recognition is not None else [])
            opt = torch.optim.Adam(params, lr=1e-3)
            times = []
            for i in range(3):
                if device == 'cuda':
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                if arm == 'rb_score':
                    aa, ww = probes(128, scale, device)
                    loss = rb_loss(model, batch, aa, ww)
                else:
                    loss = elbo_loss(model, recognition, batch)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
                assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()) for module in [model.encoder, model.initial, model.drift, model.decoder])
                assert model.raw_innovation.grad is not None and model.raw_innovation.grad.abs().sum() > 0
                torch.nn.utils.clip_grad_norm_(params, 1.)
                opt.step()
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - begin)
            records.append({'device': device, 'arm': arm, 'discarded_updates': 3, 'last_loss': float(loss.detach().cpu()),
                            'predictive_parameters': sum(p.numel() for p in model.parameters()),
                            'extra_recognition_parameters': sum(p.numel() for p in recognition.parameters()) if recognition is not None else 0,
                            'median_step_seconds': float(np.median(times[1:]))})
    devices = sorted(set(r['device'] for r in records))
    costs = {d: sum(r['median_step_seconds'] for r in records if r['device'] == d) for d in devices}
    selected = min(costs, key=costs.get)
    dump(out / 'qualification.json', {'status': 'QUALIFIED_FRESH_CONDITIONAL_WORLD_SCREEN', 'utc': now(),
                                      'likelihood_scalar_error': abs(native - reference),
                                      'exact_discrete_step_kl': exact_kl, 'independent_logratio_mean': float(logratio.mean()), 'independent_logratio_standard_error': se,
                                      'records': records, 'selected_device': selected, 'discarded_updates': sum(r['discarded_updates'] for r in records),
                                      'formal_warm_start': False, 'data_manifest_sha256': sha(R / 'data/MANIFEST.json'),
                                      'world_source_sha256': sha(R / 'scripts/world49.py'), 'source_sha256': sha(pathlib.Path(__file__))})
    print(json.dumps({'status': 'QUALIFIED_FRESH_CONDITIONAL_WORLD_SCREEN', 'selected_device': selected, 'records': records}), flush=True)


if __name__ == '__main__':
    main()
