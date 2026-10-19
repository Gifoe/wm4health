"""Close the completed evidence screen, retaining prior progress reports."""
import datetime
import hashlib
import json
import pathlib

R = pathlib.Path(__file__).resolve().parents[1]
ROOT = R.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    assert not (R / 'FINAL_REPORT_MANIFEST.json').exists()
    analysis = read(R / 'outputs/analysis_v1/analysis.json')
    verification = read(R / 'outputs/eval_v1/verification.json')
    assert verification['status'] == 'PASS_REAL_PROCESS_MEMORY_SCREEN'
    assert analysis['verification_sha256'] == sha(R / 'outputs/eval_v1/verification.json')
    assert analysis['source_summary_sha256'] == sha(R / 'outputs/eval_v1/summary.json')
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    names = ['literature_review.md', 'baseline_registry.md', 'experimental_results.md', 'negative_evidence.md',
             'hypothesis_space.md', 'research_ideas.md', 'final_research_direction.md', 'CURRENT_STATE.md']
    archive = R / 'outputs/progress_history_v1'
    archive.mkdir()
    old = {}
    for name in names:
        data = (R / name).read_bytes()
        (archive / name).write_bytes(data)
        old[name] = hashlib.sha256(data).hexdigest()
    evidence = (f"UTC {utc}.16 fresh fits,40750 formal updates;44 raw groups /1694 model-origin-reveal scores from77 chronological origins across2 real processes. "
                f"Independent CSV/ZIP, windows, model and metric reconstruction PASS. Max model difference {verification['max_independent_model_forecast_error']:.3g}; "
                'window and score arithmetic differences0. No hard-cap or final-step selections,0 numerical-refinement flags. Origins are temporally dependent.')
    aggregates = {(a['case'], a['method'], a['reveal']): a for a in analysis['aggregate']}
    lines = ['| Process | Existing arm | Open-loop48h MSE | After true24h reveal: final24h MSE |', '|---|---|---:|---:|']
    for case in ['ett', 'jena']:
        for method in ['node', 'fixed_dde', 'learned_dde', 'gru', 'ridge', 'persistence', 'seasonal24']:
            lines.append(f"| {case} | {method} | {aggregates[case, method, 0]['mse_mean']:.7f} | {aggregates[case, method, 24]['mse_mean']:.7f} |")
    table = '\n'.join(lines)
    decision = ('Deprioritize the current explicit-delay route as a proposed new framework. It provides no stable advantage over the competent history-aware controls. '
                'On Jena, GRU is better than learned delay in both seeds for open-loop and updated forecasts. On ETT, learned versus GRU depends on seed; '
                'the weak scalar NODE has the lowest mean open-loop MSE among these arms. This does not rule out useful memory generally or establish NODE as a universal world model.')
    updated = ('Real new observations improve the same final24-hour target with ordinary controls: GRU tail MSE ETT0.0832203 ->0.0431851 and '
               'Jena0.1992946 ->0.1104120; ridge ETT0.0946744 ->0.0452926 and Jena0.2392916 ->0.1188755. '
               'Means are over two GRU seeds and one deterministic ridge fit. These are point-forecast updates, not Bayesian-law consistency proofs.')
    literature = (R / 'RESEARCH_INTAKE.md').read_text(encoding='utf-8')
    (R / 'literature_review.md').write_text('# C48 source and innovation boundaries\n\n'+literature+'\n\nThe frozen source comparison is now complete. '+decision+'\n', encoding='utf-8')
    fits = [read(p) for p in sorted((R / 'outputs/fits').glob('*/fit_summary.json'))]
    registry = ['# C48 completed baseline registry', '', evidence, '',
                'Article-linked source model9b573c4bed5efee9fe9c6b765044dbd506b5bf07 / solver4888c63dc84e966f6c6553277adc3851a2cca4fc.98 Git blobs verified. Source MLP/GRU forward bodies and native solver unchanged.',
                'New calendar packing, scalar residual GRU readout, H96/F48 task and optimizer protocol are adaptations. This is cross-benchmark transfer, not headline reproduction. NODE3329/fixedDDE3521/learnedDDE3523/GRU3585 trainable parameters; ridge4752 coefficients.',
                'Two seeds; same allowed history/known clock; shared minibatch schedules per case/seed; fixed/learned DDE identical initial weights. Parameter counts, objectives and effective costs are not exactly matched across all families.', '',
                '| Fit | Updates | Selected step | VAL MSE |', '|---|---:|---:|---:|']
    for fit in fits:
        registry.append(f"| {fit['tag']} | {fit['updates']} | {fit['selected_step']} | {fit['selected_val_mse']:.7f} |")
    registry += ['', f"Sum per-fit wall time {analysis['sum_fit_wall_seconds_not_elapsed_or_gpu_hours']:.3f}s, not elapsed time or GPU hours. Actual discarded CPU/GPU throughput selected2 CPU workers;24 qualification updates excluded.",
                 'Full checkpoints at every250-step validation, selected/last weights, optimizer/RNG, source, data and logs retained on server.']
    (R / 'baseline_registry.md').write_text('\n'.join(registry)+'\n', encoding='utf-8')
    results = '# C48 completed real-process memory comparison\n\n'+evidence+'\n\n'+table+'\n\n'
    results += 'MSE is in TRAIN-standardized units. Neural rows average two optimization seeds; deterministic controls have one fit. The two columns score different horizons and must not be compared as reveal gains. See same_target_reveal.csv for identical-target comparisons.\n\n'+updated+'\n\n'+decision+'\n\n'
    results += ('Full per-origin/seed values, paired differences and seed ranges: outputs/analysis_v1/RESULTS.md, aggregate.csv, paired_model_differences.csv and same_target_reveal.csv. '
                'Qualification/verification are computation evidence, not novelty evidence. These are two univariate conditional-mean tasks; probability calibration and reusable stochastic-world learning were not tested.\n')
    (R / 'experimental_results.md').write_text(results, encoding='utf-8')
    (R / 'negative_evidence.md').write_text('# C48 negative evidence and attribution limits\n\n'+decision+'\n\n'+updated+'\n\n'
        'The stronger claim that existing models cannot incorporate fresh evidence is contradicted in these tasks. Learned delays are already an existing mechanism; their addition alone cannot supply novelty. '
        'No hard caps/final-step selections or numerical flags were observed, but optimization, capacity and two-process coverage still limit any universal conclusion. A scalar NODE doing well does not prove history is unnecessary in other systems.\n\n'
        'Prior C47 fixed-feature Laplace/hybrid and C46 simple neural factorization remain deprioritized. New C49 is a different conditional learning-objective feasibility screen with fresh trajectories; it is not a retune of these EVALs.\n', encoding='utf-8')
    (R / 'hypothesis_space.md').write_text('# C48 adjudicated hypotheses\n\n'
        'H0 ordinary history-aware controls already cover useful forecast/update behavior: supported within this two-process protocol.\n\n'
        'H1 current explicit delays reliably beat recurrent state: unsupported; Jena favors GRU, ETT varies by seed.\n\n'
        'H2 learned delay locations improve beyond fixed locations: context-dependent; Jena update mean improves but open-loop mean worsens. This is existing-method utility, not a new learning interaction.\n\n'
        'H3 the scored differences are explained by a detected solver/window/replay failure: not supported by qualified numerical checks; no numerical flags. This is not a claim that all training confounds are excluded.\n\n'
        'H4 static learned associative slots substitute for dynamic episode memory: not tested here. The inspected gradient functional cannot identify missing information or causality.\n\n'
        'H5 a new framework-level memory interaction has independent-domain utility: untested and not accepted.\n', encoding='utf-8')
    (R / 'research_ideas.md').write_text('# C48 development decision\n\n'+decision+'\n\n'
        'Adding explicit delays is not retained as the main new-method proposal. The broader state/evolution/update question remains open, but this screen supplies no necessary new mechanism.\n\n'
        'The active separate C49 feasibility screen investigates observation-marginalized conditional stochastic-world learning versus complete-clock variational learning with a shared predictive core. '
        'Its measurement identity, scoring rule and conditional path learning are existing ingredients; a distinct useful learning interaction, competent comparisons and independent real-domain evidence remain to be established.\n', encoding='utf-8')
    (R / 'final_research_direction.md').write_text('# C48 final decision: no new direction accepted\n\n'
        '**Research Question:** Does explicit learned history delay give a reusable world-model memory mechanism useful forecast and observation-update capabilities beyond ordinary full-history/recurrent controls?\n\n'
        '**Evidence:** '+evidence+'\n\n'+decision+'\n\n'
        '**Existing Methods:** GRU/ridge already use fresh histories effectively; learned and state-dependent delays are prior art.\n\n'
        '**Proposed Idea:** No new memory method accepted. Current explicit-delay recipe is deprioritized.\n\n'
        '**Falsification:** The expected stable advantage over strong ordinary controls did not occur. This falsifies that restricted improvement claim, not the utility of memory generally.\n\n'
        '**Feasibility:** Server execution/source/data/checkpoints/raw forecasts/logs retained; independent replay PASS. Two real processes and conditional-mean metrics do not certify a broad stochastic-world framework.\n\n'
        '**Next Experiment:** Finish the separate frozen C49 conditional neural stochastic-learning screen. A positive synthetic result would remain a candidate mechanism requiring current prior-art differentiation and fresh real-domain validation.\n\n'
        'Accepted final directions0; full discovery goal ACTIVE_UNACHIEVED.\n', encoding='utf-8')
    (R / 'CURRENT_STATE.md').write_text('# C48 screen complete; full research goal remains active\n\n'+evidence+'\n\n'+decision+'\n\n'
        'All C48 fit/evaluation supervisors are terminal, independent verification passed and descriptive analysis is complete. Do not restart them. C49 is the separate active experiment. Progress snapshots were archived unchanged under outputs/progress_history_v1.\n', encoding='utf-8')
    (R / 'SCIENTIFIC_DECISION.md').write_text('# C48 scientific decision\n\n'+decision+'\n\n'+updated+'\n\n'
        'No final research direction is accepted; no new model was tested here. The complete results and independent computation support deprioritizing this particular delay route, while preserving the broader discovery goal.\n', encoding='utf-8')
    dump(R / 'FINAL_REPORT_MANIFEST.json', {'utc': utc, 'status': 'C48_COMPLETE_SCREEN_GOAL_UNACHIEVED', 'old_progress_sha256': old,
                                          'new_reports': {name: sha(R / name) for name in names + ['SCIENTIFIC_DECISION.md']},
                                          'analysis_sha256': sha(R / 'outputs/analysis_v1/analysis.json'),
                                          'verification_sha256': sha(R / 'outputs/eval_v1/verification.json'), 'source_sha256': sha(pathlib.Path(__file__)), 'goal_complete': False})
    print(json.dumps({'status': 'C48_COMPLETE_SCREEN_GOAL_UNACHIEVED', 'formal_updates': 40750, 'accepted_final_directions': 0}), flush=True)


if __name__ == '__main__':
    main()
