"""Pre-registered time-budget frontier; primary fits/data/selection stay unchanged."""
from eval_helpers49 import *
from independent49 import replay, energy as numpy_energy
import csv


def main():
    seal = json.loads((R / 'outputs/COST_RULE_SEAL.json').read_text(encoding='utf-8-sig'))
    assert seal['primary_eval_opened'] is False
    assert seal['budgets_seconds'] == [60, 120, 180]
    assert seal['supplement_sha256'] == sha(R / 'MATCHED_COST_SUPPLEMENT.md')
    assert seal['training_protocol_sha256'] == sha(R / 'PROTOCOL.md')
    primary = R / 'outputs/eval_v1'
    assert read(primary / 'verification.json')['status'] == 'PASS_FRESH_CONDITIONAL_WORLD_SCREEN'
    opening = read(primary / 'OPENING_RECEIPT.json')
    assert seal['utc'][:19] < opening['utc'][:19], 'Cost rule was not sealed before primary EVAL'
    out = R / 'outputs/cost_frontier_v1'
    assert not out.exists()
    out.mkdir()
    fits = [read(p) for p in sorted((R / 'outputs/fits').glob('*/fit_summary.json'))]
    primary_raw = {(e['model'], e['reveal']): e for e in read(primary / 'raw_manifest.json')}
    selections = []
    for fit in fits:
        folder = R / 'outputs/fits' / fit['tag']
        curve = read(folder / 'validation.json')
        for budget in seal['budgets_seconds']:
            eligible = [c for c in curve if c['elapsed_seconds'] <= budget]
            if not eligible:
                selections.append({'model': fit['tag'], 'budget_seconds': budget, 'available': False})
                continue
            selected = min(eligible, key=lambda c: (c['observed_val_score'], c['step']))
            path = folder / ('selected.pt' if selected['step'] == fit['selected_step'] else f"validation_step{selected['step']}.pt")
            selections.append({'model': fit['tag'], 'case': fit['case'], 'seed': fit['seed'], 'arm': fit['arm'],
                               'budget_seconds': budget, 'available': True, 'step': selected['step'],
                               'selected_val_score': selected['observed_val_score'], 'measured_selection_seconds': selected['elapsed_seconds'],
                               'checkpoint': str(path.relative_to(R)).replace('\\', '/'), 'checkpoint_sha256': sha(path),
                               'matches_primary_selected': sha(path) == fit['selected_sha256']})
    dump(out / 'PLAN.json', {'utc': now(), 'rule_seal_sha256': sha(R / 'outputs/COST_RULE_SEAL.json'),
                             'selections': selections, 'selection_inputs': 'Only frozen VAL scores and measured fit time; no EVAL selection',
                             'source_sha256': sha(pathlib.Path(__file__))})
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    manifest = read(R / 'data/MANIFEST.json')
    cache, raw_entries, groups, all_rows = {}, [], [], []
    max_model, max_energy, max_mse = 0., 0., 0.
    for selected in selections:
        if not selected['available']:
            continue
        key = selected['model'], selected['checkpoint_sha256']
        if key not in cache:
            if selected['matches_primary_selected']:
                ent = primary_raw[selected['model'], 0]
                path = primary / ent['path']
                assert sha(path) == ent['sha256']
                with np.load(path) as z:
                    values = {k: z[k].copy() for k in ['energy', 'mse', 'physical_id', 'density']}
                raw_entries.append({'model': selected['model'], 'step': selected['step'], 'checkpoint_sha256': selected['checkpoint_sha256'],
                                    'path': '../eval_v1/' + path.name, 'sha256': ent['sha256'], 'reused_primary_verified': True})
            else:
                case = selected['case']
                scale = manifest['description'][case]['encoder_probe_scale_train_only']
                data = load(case, 'eval', device)
                identity = np.load(R / 'data' / case / 'eval_observed.npz')['physical_id']
                clean = np.load(R / 'data' / case / 'eval_physical_truth.npz')['signal'][identity, T:]
                truth = torch.tensor(clean, device=device)
                model = World(scale).to(device).eval()
                checkpoint_path = R / selected['checkpoint']
                assert sha(checkpoint_path) == selected['checkpoint_sha256']
                model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)['world'])
                weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
                weight_path = out / (selected['model'] + f"__step{selected['step']}__weights.npz")
                np.savez_compressed(weight_path, **weights)
                seed(selected['seed'] + ['rotating', 'bistable'].index(case) * 100 + 701)
                collected = {k: [] for k in ['prediction', 'noise', 'energy', 'mse']}
                with torch.no_grad():
                    for start in range(0, len(identity), 8):
                        sl = slice(start, start + 8)
                        n, s, a = data['counts'][sl, :T], data['sums'][sl, :T], data['density'][sl]
                        prediction, noise = prediction_with_noise(model, n, s, a, 128, T)
                        pe = energy(prediction, truth[sl]).cpu().numpy()
                        pm = (prediction.mean(1) - truth[sl]).double().square().mean((1, 2)).cpu().numpy()
                        pp, nn = prediction.cpu().numpy(), noise.cpu().numpy()
                        independent = replay(weights, n.cpu().numpy(), s.cpu().numpy(), a.cpu().numpy(), nn)
                        difference = float(np.max(abs(pp - independent)))
                        max_model = max(max_model, difference)
                        assert np.allclose(pp, independent, atol=3e-4, rtol=3e-4), (selected['model'], selected['step'], start, difference)
                        ne = numpy_energy(pp, clean[sl])
                        nm = np.mean((pp.astype('float64').mean(1) - clean[sl])**2, (1, 2))
                        eerror, merror = float(np.max(abs(pe - ne))), float(np.max(abs(pm - nm)))
                        max_energy, max_mse = max(max_energy, eerror), max(max_mse, merror)
                        assert eerror < 2e-8 and merror < 2e-5
                        for k, value in [('prediction', pp), ('noise', nn), ('energy', pe), ('mse', pm)]:
                            collected[k].append(value)
                arrays = {k: np.concatenate(v) for k, v in collected.items()}
                arrays.update({'truth': clean, 'physical_id': identity, 'density': data['density'].cpu().numpy(),
                               'counts': data['counts'][:, :T].cpu().numpy(), 'sums': data['sums'][:, :T].cpu().numpy()})
                path = out / (selected['model'] + f"__step{selected['step']}.npz")
                np.savez_compressed(path, **arrays)
                raw_entries.append({'model': selected['model'], 'step': selected['step'], 'checkpoint_sha256': selected['checkpoint_sha256'],
                                    'path': path.name, 'sha256': sha(path), 'weight_path': weight_path.name,
                                    'weights_sha256': sha(weight_path), 'reused_primary_verified': False,
                                    'independent_all_path_and_score_replay': True})
                values = {k: arrays[k] for k in ['energy', 'mse', 'physical_id', 'density']}
            cache[key] = values
        values = cache[key]
        for density in [.25, 1., 4.]:
            mask = values['density'] == density
            group = {k: selected[k] for k in ['model', 'case', 'seed', 'arm', 'budget_seconds', 'step', 'measured_selection_seconds']}
            group.update({'density': density, 'physical_paths': int(mask.sum()), 'energy': float(values['energy'][mask].mean()), 'mse': float(values['mse'][mask].mean())})
            groups.append(group)
            print(json.dumps(group), flush=True)
        for i, pid in enumerate(values['physical_id']):
            all_rows.append({**{k: selected[k] for k in ['model', 'case', 'seed', 'arm', 'budget_seconds', 'step']},
                             'density': float(values['density'][i]), 'physical_id': int(pid),
                             'energy': float(values['energy'][i]), 'mse': float(values['mse'][i])})
    lookup = {(g['case'], g['seed'], g['arm'], g['budget_seconds'], g['density']): g for g in groups}
    pairs = []
    for case in ['rotating', 'bistable']:
        for seed_value in [2026491, 2026492]:
            for budget in [60, 120, 180]:
                for density in [.25, 1., 4.]:
                    left = lookup.get((case, seed_value, 'rb_score', budget, density))
                    right = lookup.get((case, seed_value, 'full_clock_elbo', budget, density))
                    if left is None or right is None:
                        continue
                    pairs.append({'case': case, 'seed': seed_value, 'budget_seconds': budget, 'density': density,
                                  'rb_energy': left['energy'], 'elbo_energy': right['energy'],
                                  'energy_rb_minus_elbo': left['energy'] - right['energy'],
                                  'rb_mse': left['mse'], 'elbo_mse': right['mse'], 'mse_rb_minus_elbo': left['mse'] - right['mse']})
    dump(out / 'raw_manifest.json', raw_entries)
    dump(out / 'all_rows.json', all_rows)
    dump(out / 'summary.json', {'status': 'PASS_PREREGISTERED_TIME_FRONTIER_WITH_INDEPENDENT_REPLAY', 'utc': now(),
                               'groups': groups, 'paired_differences': pairs, 'unique_weight_evaluations': len(cache),
                               'summary_groups': len(groups), 'model_context_budget_entries': len(all_rows),
                               'unavailable_selections': [s for s in selections if not s['available']],
                               'max_independent_model_error': max_model, 'max_independent_energy_error': max_energy,
                               'max_independent_mse_error': max_mse, 'rule_seal_sha256': sha(R / 'outputs/COST_RULE_SEAL.json'),
                               'raw_manifest_sha256': sha(out / 'raw_manifest.json'), 'all_rows_sha256': sha(out / 'all_rows.json'),
                               'source_sha256': sha(pathlib.Path(__file__)), 'goal_complete': False})
    with (out / 'paired_differences.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)


if __name__ == '__main__':
    main()
