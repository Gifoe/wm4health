"""Close verified evidence with an explicit human-agent scientific decision file."""
import datetime
import hashlib
import json
import pathlib

R = pathlib.Path(__file__).resolve().parents[1]
ROOT = R.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    assert not (R / 'FINAL_REPORT_MANIFEST.json').exists()
    verification = read(R / 'outputs/eval_v1/verification.json')
    analysis = read(R / 'outputs/analysis_v1/analysis.json')
    frontier = read(R / 'outputs/cost_frontier_v1/summary.json')
    costs = read(R / 'outputs/cost_frontier_v1/paired_analysis.json')
    supervisor = read(R / 'outputs/evaluation_supervisor_v3.json')
    assert verification['status'] == 'PASS_FRESH_CONDITIONAL_WORLD_SCREEN'
    assert supervisor['status'] == 'VERIFICATION_TIME_FRONTIER_ANALYSIS_AND_PLOTS_COMPLETE'
    assert analysis['verification_sha256'] == sha(R / 'outputs/eval_v1/verification.json')
    assert frontier['status'] == 'PASS_PREREGISTERED_TIME_FRONTIER_WITH_INDEPENDENT_REPLAY'
    assert costs['source_summary_sha256'] == sha(R / 'outputs/cost_frontier_v1/summary.json')
    assert costs['status'] == 'ALL_FIXED_TIME_POINTS_REPORTED_NO_EVAL_SELECTION'
    assert (R / 'outputs/analysis_v1/conditional_world_comparison.png').is_file()
    decision = (R / 'SCIENTIFIC_DECISION.md').read_text(encoding='utf-8')
    assert any(tag in decision for tag in ['Decision: DEPRIORITIZE_CURRENT_RECIPE', 'Decision: RETAIN_CANDIDATE_NEEDS_CONFIRMATION'])
    assert 'Accepted final directions: 0' in decision
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    names = ['literature_review.md', 'baseline_registry.md', 'experimental_results.md', 'negative_evidence.md',
             'hypothesis_space.md', 'research_ideas.md', 'final_research_direction.md', 'CURRENT_STATE.md']
    archive = R / 'outputs/progress_history_final_v1'
    archive.mkdir()
    previous = {}
    for name in names:
        data = (R / name).read_bytes()
        (archive / name).write_bytes(data)
        previous[name] = hashlib.sha256(data).hexdigest()
    evidence = (f"UTC {utc}.8 fresh fits /{analysis['formal_updates']} formal updates. Primary:16 raw groups /48 summary groups /12288 model-context-reveal scores, "
                '512 independent synthetic physical paths and1536 paired density history contexts,128 iid future draws per history. '
                f"Independent all-path/all-score replay PASS: max model difference{verification['max_independent_model_error']:.3g}, "
                f"energy arithmetic{verification['max_energy_error']:.3g}, observation score{verification['max_observed_score_error']:.3g}.0 real domains.")
    lookup = {(g['case'], g['arm'], g['density'], g['reveal']): g for g in analysis['aggregate']}
    main_table = ['| Process | Density | Arm | Full24-bin energy | Full24-bin MSE |', '|---|---:|---|---:|---:|']
    reveal_table = ['| Process | Density | Arm | Original final12 energy | Updated final12 energy |', '|---|---:|---|---:|---:|']
    for case in ['rotating', 'bistable']:
        for density in [.25, 1., 4.]:
            for arm in ['rb_score', 'full_clock_elbo']:
                a, b = lookup[case, arm, density, 0], lookup[case, arm, density, 12]
                main_table.append(f"| {case} | {density} | {arm} | {a['energy_mean']:.7f} | {a['mse_mean']:.7f} |")
                reveal_table.append(f"| {case} | {density} | {arm} | {a['tail12_energy_mean']:.7f} | {b['tail12_energy_mean']:.7f} |")
    literature = '\n\n'.join((R / name).read_text(encoding='utf-8') for name in ['RESEARCH_INTAKE.md', 'LITERATURE_SUPPLEMENT_20261017.md', 'NOVELTY_CHALLENGE_20261017.md'])
    (R / 'literature_review.md').write_text('# C49 completed literature boundary\n\n'+literature+'\n\n'+decision+'\n', encoding='utf-8')
    registry = ['# C49 completed baseline registry', '', evidence, '',
                'This is an original controlled shared-core experiment, not an original-paper headline reproduction. Kernel/variational/marked-observation methods are prior mechanisms.',
                'Both predictive cores9458 parameters; complete-clock ELBO additionally uses13944 training-only future-recognition parameters. Forecasts use only observed history. Same predictive initialization/data schedules per case/seed were independently checked.',
                'Source physics: rotating OU and coupled bistability,2D signals. Learned latent8, neural autonomous discrete transitions, learned signal decoder; known calibrated sensor and density. TRAIN1024/VAL128/EVAL256 independent paths per process. Clean physical truth separated from training observation files; no EVAL model selection.',
                'Formal device CPU selected by close discarded whole-step timings; CPU/CUDA ranking differed by less than1% and is not an efficiency conclusion.12 discarded qualification updates excluded.', '',
                '| Fit | Updates | Selected step | Selected-step seconds | Whole-fit seconds |', '|---|---:|---:|---:|---:|']
    for fit in analysis['fit_costs']:
        registry.append(f"| {fit['model']} | {fit['updates']} | {fit['selected_step']} | {fit['selected_training_and_selection_seconds']:.3f} | {fit['terminal_fit_seconds']:.3f} |")
    registry += ['', f"Sum whole-fit time{analysis['sum_terminal_fit_seconds_not_gpu_hours']:.3f}s, not GPU hours. Hard caps:{analysis['hard_cap_fits']}; final-step selections:{analysis['selected_at_final_update']}.",
                 'All200-step validation checkpoints, selected/last states, terminal optimizer/recognizer/RNG, schedules and raw forecasts retained. Selected validation checkpoints store predictive core weights; selected-time recognition/optimizer snapshots are not separately stored. A full matched-time replay of training would be needed to reconstruct them if required for posterior diagnostics.']
    (R / 'baseline_registry.md').write_text('\n'.join(registry)+'\n', encoding='utf-8')
    results = '# C49 completed conditional stochastic-world learning screen\n\n'+evidence+'\n\n'+'\n'.join(main_table)+'\n\n'+'\n'.join(reveal_table)+'\n\n'
    results += ('Rows average two optimization seeds. Energy uses an independent-pair ensemble score and Euclidean distance divided by sqrt(target scalar coordinates). '
                'Across reveals, only identical final12 targets are compared; full24 versus remaining12 is a different task. Density changes also alter available history information.\n\n')
    results += (f"The pre-EVAL-sealed60/120/180-second frontier contains{frontier['summary_groups']} summary groups /{frontier['model_context_budget_entries']} scored model-context-budget entries. "
                f"Unique weight evaluations{frontier['unique_weight_evaluations']}; identical primary checkpoints reuse their verified raw bytes. All novel frontier predictions were independently replayed. "
                'See outputs/cost_frontier_v1/RESULTS.md, aggregate.csv and paired_analysis.json; all points are reported.\n\n')
    results += decision+'\n\n'
    results += ('Complete per-episode/seed values and episode-bootstrap intervals are in outputs/analysis_v1. Density views stay together when resampling physical episodes; intervals condition on two fitted seeds, not on training-seed/real-domain variability. '
                'Observed kernel scores omit a constant and may be negative; only differences within matched contexts are interpreted.\n')
    (R / 'experimental_results.md').write_text(results, encoding='utf-8')
    (R / 'negative_evidence.md').write_text('# C49 negative and unresolved evidence\n\n'+decision+'\n\n'
        'The tested objective is an observation-kernel integration of a proper conditional score. Existing convMMD/generative deconvolution, Ambient methods and conditional Neural SDE scoring occupy its major ingredients. '
        'Any observed two-arm difference also changes the objective and use of a future recognizer, so it does not isolate analytic averaging as the cause.\n\n'
        'C48 explicit-delay route and C47 fixed-feature Laplace recipe remain deprioritized; no old EVAL was retuned. '
        'A constructed native TSWM Greenhouse preprocessing fixture exposed possible cross-boundary interpolation leakage. Real-data prevalence and ranking effects are unmeasured; it is an implementation contract issue, not a new world-model mechanism.\n', encoding='utf-8')
    (R / 'hypothesis_space.md').write_text('# C49 adjudication and competing explanations\n\n'+decision+'\n\n'
        'H0 ordinary complete-clock variational learning covers the useful conditional-future behavior: assess under the full reported quality/cost points, not a weak sampled-event baseline.\n\n'
        'H1 analytic observation averaging supplies an independent practical benefit: requires an effect beyond a generic objective/recognizer change. The current two-arm design cannot isolate it.\n\n'
        'H2 finite-ensemble bias, probe selection, optimization or baseline capacity explains a difference: independent-pair scoring, fixed choices, shared predictive capacity and time points address parts; caps/final-step selections and approximation family still limit attribution.\n\n'
        'H3 observed-law fit identifies a useful physical simulation: only the specified calibrated sensor and simulated signal queries were tested; no physical latent coordinate identification.\n\n'
        'H4 reusable framework utility on independent real domains: not tested.0 real domains remain a material missing requirement.\n', encoding='utf-8')
    (R / 'research_ideas.md').write_text('# C49 development outcome\n\n'+decision+'\n\n'
        'A method built from known components can still contribute through a distinct useful learning interaction. An unprecedented operator is not required, but ordinary observation-kernel integration alone cannot establish one. '
        'Future direction choice must start from a reproducible failure of competent models and a discriminating explanation, rather than from another arbitrarily added loss.\n\n'
        'No new direction is accepted by this completed screen. Before a real-data baseline comparison, quantify the native TSWM interpolation-boundary issue and preserve original versus corrected information protocols separately. '
        'Do not infer causality from observational control trajectories or treat synthetic density contexts as independent real-world domains.\n', encoding='utf-8')
    (R / 'final_research_direction.md').write_text('# C49 final scientific disposition; overall discovery incomplete\n\n'
        '**Research Question:** Can observation-marginalized conditional training improve a stochastic time-series world model under informative noisy sensing beyond complete-clock variational inference?\n\n'
        '**Evidence:** '+evidence+' The fixed time-budget frontier and independent replay are complete.\n\n'
        '**Existing Methods:** Conditional kernel Neural SDE learning, observation-aware latent models and convMMD/deconvolution cover major ingredients.\n\n'
        '**Proposed Idea:** The implemented conditional observation-averaged objective; disposition below.\n\n'+decision+'\n\n'
        '**Falsification:** Lack of useful quality/cost advantage over competent complete-clock inference, or attribution solely to generic scoring/approximate-inference differences, removes the intended mechanism claim.\n\n'
        '**Feasibility:** Executed8 fits on the server with the single5090 available; fixed fresh data, source/config/checkpoints/raw futures/innovations and receipts retained. This is a two-process controlled screen.\n\n'
        '**Next Experiment:** Return to source-grounded real-world failure discovery or require a genuinely distinct mechanism with fresh confirmatory data. A real baseline must first satisfy its information contract. Further EVAL tuning of this recipe is not authorized by the scientific protocol.\n\n'
        'Accepted final directions: 0. Overall goal ACTIVE_UNACHIEVED.\n', encoding='utf-8')
    (R / 'CURRENT_STATE.md').write_text('# C49 complete; full discovery goal remains active\n\n'+evidence+'\n\n'+decision+'\n\n'
        'The authoritative completed chain is outputs/evaluation_supervisor_v3.json. Earlier verifier supervisors retain intentional runtime-replacement failures. '
        'All primary fitting/scoring bytes were retained; source versions, equivalence qualification and targeted termination receipts explain the I/O/kernel/thread changes. '
        'Do not rerun completed fits/evaluators or dated progress writers.\n', encoding='utf-8')
    dump(R / 'FINAL_REPORT_MANIFEST.json', {'utc': utc, 'status': 'C49_COMPLETE_SCREEN_GOAL_UNACHIEVED',
                                          'old_progress_sha256': previous, 'new_reports': {n: sha(R / n) for n in names + ['SCIENTIFIC_DECISION.md']},
                                          'primary_verification_sha256': sha(R / 'outputs/eval_v1/verification.json'),
                                          'cost_frontier_sha256': sha(R / 'outputs/cost_frontier_v1/summary.json'),
                                          'cost_analysis_sha256': sha(R / 'outputs/cost_frontier_v1/paired_analysis.json'),
                                          'source_sha256': sha(pathlib.Path(__file__)), 'goal_complete': False})
    archive = R / 'outputs/root_history_final_v1'
    archive.mkdir()
    header = f'''<!-- cycle49-final-index-start -->
# C49已完成：条件随机世界模型评分与固定时间预算比较

UTC {utc}。8组训练/18800正式updates；主评分16组原始轨迹/12288模型—历史—揭示分数，512条独立合成物理轨迹。独立NumPy重放全部预测及指标通过，同预算60/120/180秒规则在EVAL前固定并全部报告。模型、优化器、原始轨迹、日志和版本修复凭据保留。见[完整结果](cycle49/experimental_results.md)、[科学裁决](cycle49/SCIENTIFIC_DECISION.md)、[同预算结果](cycle49/outputs/cost_frontier_v1/RESULTS.md)。

本轮是原创受控实验，非原论文headline复现；0真实域。已有convMMD、条件路径评分和观测感知模型约束其创新边界。当前接受最终方向仍为0，整体ACTIVE_UNACHIEVED。TSWM原生缺失插值的构造边界测试发现输入契约风险，发生频率和实际效果未测，不能作为新模型贡献。以下为历史。
<!-- cycle49-final-index-end -->

'''
    root_records = []
    for name in ['CURRENT_RESEARCH_STATUS.md', 'literature_review.md', 'baseline_registry.md', 'experimental_results.md',
                 'negative_evidence.md', 'hypothesis_space.md', 'research_ideas.md', 'final_research_direction.md']:
        path = ROOT / name
        old = path.read_bytes()
        (archive / name).write_bytes(old)
        assert path.read_bytes() == old
        path.write_bytes(header.encode('utf-8') + old)
        root_records.append({'path': name, 'before': hashlib.sha256(old).hexdigest(), 'after': sha(path)})
    dump(R / 'ROOT_FINAL_INDEX_RECEIPTS.json', {'utc': utc, 'files': root_records, 'goal_complete': False})
    print(json.dumps({'status': 'C49_COMPLETE_SCREEN_GOAL_UNACHIEVED', 'formal_updates': analysis['formal_updates'], 'accepted_final_directions': 0}), flush=True)


if __name__ == '__main__':
    main()
