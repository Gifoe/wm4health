"""Fresh fits of identical predictive cores, selection on observed VAL only."""
from world49 import *
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = p.parse_args()
    q = read(R / 'outputs/model_qualification_v1/qualification.json')
    assert q['status'] == 'QUALIFIED_FRESH_CONDITIONAL_WORLD_SCREEN'
    assert q['world_source_sha256'] == sha(R / 'scripts/world49.py')
    assert q['data_manifest_sha256'] == sha(R / 'data/MANIFEST.json')
    device = q['selected_device']
    folder = R / f'outputs/fits/{args.case}__{args.arm}__seed{args.seed}'
    assert not folder.exists()
    folder.mkdir(parents=True)
    manifest = read(R / 'data/MANIFEST.json')
    scale = manifest['description'][args.case]['encoder_probe_scale_train_only']
    train, val = load(args.case, 'train', device), load(args.case, 'val', device)
    zz = np.load(R / 'data' / args.case / 'validation_probes.npz')
    aa = torch.tensor(zz['count'], device=device); ww = torch.tensor(zz['mark'], device=device)
    indices = np.random.default_rng(args.seed + 17).integers(1024, size=(3000, 32))
    np.savez_compressed(folder / 'schedule.npz', indices=indices)
    seed(args.seed)
    model = World(scale).to(device)
    initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    recognition = Recognizer().to(device) if args.arm == 'full_clock_elbo' else None
    parameters = list(model.parameters()) + (list(recognition.parameters()) if recognition is not None else [])
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    config = {'case': args.case, 'arm': args.arm, 'seed': args.seed, 'device': device, 'utc': now(),
              'predictive_parameters': sum(p.numel() for p in model.parameters()),
              'recognition_parameters': sum(p.numel() for p in recognition.parameters()) if recognition is not None else 0,
              'initial_predictive_checksum': initial, 'schedule_sha256': sha(folder / 'schedule.npz'),
              'world_sha256': sha(R / 'scripts/world49.py'), 'measurement_sha256': sha(R / 'scripts/measurement49.py'),
              'protocol_sha256': sha(R / 'PROTOCOL.md'), 'data_manifest_sha256': sha(R / 'data/MANIFEST.json'),
              'source_sha256': sha(pathlib.Path(__file__)), 'clean_physical_truth_used': False, 'eval_used_for_selection': False}
    dump(folder / 'config.json', config)
    best, best_step, stale = float('inf'), 0, 0
    records, losses = [], []
    begin = time.perf_counter()
    for i in range(3000):
        step = i + 1
        model.train()
        batch = {k: v[indices[i]] for k, v in train.items()}
        optimizer.zero_grad(set_to_none=True)
        if args.arm == 'rb_score':
            a, w = probes(128, scale, device)
            objective = rb_loss(model, batch, a, w)
        else:
            objective = elbo_loss(model, recognition, batch)
        assert torch.isfinite(objective), (step, 'nonfinite loss')
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert torch.isfinite(norm), (step, 'nonfinite gradient')
        optimizer.step()
        losses.append(float(objective.detach().cpu()))
        if step % 50 == 0:
            dump(folder / 'status.json', {'status': 'training', 'pid': __import__('os').getpid(), 'step': step, 'best_step': best_step, 'best_observed_val_score': best if math.isfinite(best) else None, 'utc': now()})
        if step % 200:
            continue
        value = validate(model, val, aa, ww)
        assert math.isfinite(value)
        relative = (best - value) / max(abs(best), 1e-9) if math.isfinite(best) else None
        checkpoint = {'world': model.state_dict(), 'step': step, 'observed_val_score': value, 'config': config}
        torch.save(checkpoint, folder / f'validation_step{step}.pt')
        if value < best:
            best, best_step = value, step
            torch.save(checkpoint, folder / 'selected.pt')
        stale = 0 if relative is None or relative >= .002 else stale + 1
        row = {'step': step, 'observed_val_score': value, 'best_step': best_step, 'best_score': best,
               'relative_improvement': relative, 'stale_checks': stale, 'mean_training_loss_last200': float(np.mean(losses)),
               'elapsed_seconds': time.perf_counter() - begin}
        records.append(row)
        dump(folder / 'validation.json', records)
        print(json.dumps({'tag': folder.name, **row}), flush=True)
        losses = []
        if step >= 1500 and stale >= 6:
            break
    torch.save({'world': model.state_dict(), 'recognition': recognition.state_dict() if recognition is not None else None,
                'optimizer': optimizer.state_dict(), 'step': step, 'config': config,
                'torch_rng': torch.random.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else None}, folder / 'last.pt')
    dump(folder / 'fit_summary.json', {'status': 'FIT_COMPLETE_EVAL_UNOPENED', 'tag': folder.name, 'case': args.case,
                                      'arm': args.arm, 'seed': args.seed, 'updates': step, 'selected_step': best_step,
                                      'selected_observed_val_score': best, 'hard_cap_hit': step == 3000,
                                      'selected_at_final_update': best_step == step, 'elapsed_seconds': time.perf_counter() - begin,
                                      'selected_sha256': sha(folder / 'selected.pt'), 'last_sha256': sha(folder / 'last.pt'),
                                      'initial_predictive_checksum': initial, 'schedule_sha256': sha(folder / 'schedule.npz'), 'utc': now()})
    dump(folder / 'status.json', {'status': 'completed', 'step': step, 'utc': now()})


if __name__ == '__main__':
    main()
