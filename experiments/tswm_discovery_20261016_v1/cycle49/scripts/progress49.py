"""Dated evidence-backed progress reports; no EVAL performance is invented."""
from world49_v2 import *


def main():
    assert not (R / 'PROGRESS_REPORT_MANIFEST.json').exists()
    q = read(R / 'outputs/model_qualification_v1/qualification.json')
    transform = read(R / 'outputs/qualification_v1/qualification.json')
    replay = read(R / 'outputs/replay_qualification_v1/qualification.json')
    state = read(R / 'outputs/supervisor_v1.json')
    fits = [read(p) for p in sorted((R / 'outputs/fits').glob('*/fit_summary.json'))]
    utc = now()
    evidence = (f"UTC {utc}.8 formal fresh fits queued, {len(fits)} terminal at this snapshot; supervisor PID{state['pid']}, state{state['status']}. "
                'No EVAL performance is available. Actual process handles were independently inspected; do not restart from a stale report. '
                'Current counts must be read from supervisor/status files and actual processes.')
    boundary = ('The experiment tests a conditional state-inference/dynamics/observation-objective interaction. '
                'Conditional kernel world learning, variational inference, observation-operator learning, Rao-Blackwellization and marked-Poisson identities already exist. '
                'A useful new framework contribution has not yet been demonstrated.')
    (R / 'literature_review.md').write_text('# C49 literature and novelty boundaries\n\n'+(R / 'RESEARCH_INTAKE.md').read_text(encoding='utf-8')+'\n', encoding='utf-8')
    registry = '# C49 baseline registry\n\n'+evidence+'\n\n'
    registry += ('Both arms use the same9458-parameter predictive core. The complete-clock ELBO control additionally has13944 future-recognition parameters. '
                 'Recognition sees future observations during training only; prediction uses only observed history. Same initial predictive bytes and minibatch schedules are retained for each process/seed. '
                 'No original-paper headline reproduction is claimed. This shared-core implementation is an original controlled experiment, with established variational and proper-scoring mechanisms.\n\n'
                 'Two independent2D process families, fresh1024/128/256 TRAIN/VAL/EVAL physical paths each,24 observed bins ->24 future bins. Known acquisition density and Gaussian noise calibration. '
                 'Training observation files contain no physical truth. Three EVAL density contexts are paired views of each physical path, not independent episodes.0 real domains validated.\n\n')
    registry += f"Formal training device{q['selected_device']};12 discarded end-to-end qualification updates excluded. CPU/CUDA aggregate step timings differ by less than1%, so this is not a robust device-efficiency finding.\n"
    (R / 'baseline_registry.md').write_text(registry, encoding='utf-8')
    result = '# C49 experiment progress\n\n'+evidence+'\n\n'
    result += (f"Analytic marked-observation feature versus independent Poisson summation/Gaussian quadrature: max error{transform['max_independent_poisson_gaussian_quadrature_error']:.3g}. "
               f"Coupled signal/acquisition gradient versus finite differences: absolute difference{transform['gradient_absolute_error']:.3g}. "
               f"Exhaustive finite-ensemble unbiased-score expectation error{transform['finite_ensemble_unbiased_expectation_error']:.3g}. "
               'The corresponding naive squared feature average has positive finite-ensemble bias. These are implementation facts, not new mathematical results or trained predictive improvements.\n\n'
               f"Complete likelihood scalar arithmetic error{q['likelihood_scalar_error']:.3g}. Exact discrete Gaussian-step KL was checked against independent log-density-ratio sampling. "
               'Both objective paths have finite nonzero encoder/initial-state/dynamics/decoder/innovation gradients in discarded updates.\n\n'
               'TRAIN-only independent replay matches native and instrumented predictions exactly, then NumPy GRU/Markov evolution within fixed tolerance. '
               f"Max replay qualification discrepancy{max(r['numpy_model_max_error'] for r in replay['records']):.3g}; no formal selected weights or EVAL inputs were used.\n\n")
    for f in fits:
        result += f"Completed {f['tag']}: updates{f['updates']}, selected{f['selected_step']}, observed VAL score{f['selected_observed_val_score']:.7f}; this is not an EVAL future-quality result.\n\n"
    result += ('All8 checkpoint selections must seal before score49.py opens EVAL. Independent verification will replay every saved future innovation/model forecast and every energy/observation-score entry. '
               'Final12 targets are matched across real-prefix reveal conditions.\n')
    (R / 'experimental_results.md').write_text(result, encoding='utf-8')
    (R / 'negative_evidence.md').write_text('# C49 inherited negatives and unsupported claims\n\n'
        'C48 explicit delays lacked stable advantage over GRU/full-history controls; that route is deprioritized. C47 fixed-feature Laplace/hybrid lost to complete likelihood on its controlled bistable screen. '
        'This experiment changes the conditional learned-state formulation and uses fresh trajectories; it does not retune old EVALs.\n\n'+boundary+'\n\n'
        'Estimator qualification alone does not demonstrate a useful research gap, stochastic future improvement, physical-state identification or failure of complete likelihood. '
        'Sparse observations may be nonidentifying; a known calibrated sensor is an explicit assumption.2 synthetic processes and no real domain cannot establish a reusable practical framework.\n', encoding='utf-8')
    (R / 'hypothesis_space.md').write_text('# C49 competing hypotheses\n\n'
        'H0 competent complete-clock variational learning already matches or beats the alternative per total cost. Primary shared-core comparison tests this.\n\n'
        'H1 removing discrete observation randomness improves useful conditional state/dynamics learning; needs trained held-out future-law evidence, not only lower estimator variance.\n\n'
        'H2 apparent gains arise from ensemble bias, selected probes, optimizer/capacity differences or a weak likelihood control. Use the unbiased score, fixed TRAIN/VAL contract, complete count likelihood and disclose extra recognition parameters/actual cost.\n\n'
        'H3 observation-law fit leaves physical future queries unidentified. Keep clean physical scoring independent and avoid latent-state identification claims.\n\n'
        'H4 a framework-level benefit transfers to independent real domains. Not tested in this controlled screen and remains required for an accepted direction.\n', encoding='utf-8')
    (R / 'research_ideas.md').write_text('# C49 concrete candidate, not accepted direction\n\n'
        'Research question: can a history-conditioned stochastic world model learn reusable evolution from informative noisy observations more effectively by analytically averaging over its observation process during training?\n\n'
        'Implemented mechanism: history -> Gaussian latent state -> shared autonomous neural stochastic transitions -> signal; the calibrated marked-count observation law is integrated inside an unbiased conditional kernel score. '
        'Complete-clock variational learning uses the same predictive core, with a stronger training-only future recognizer.\n\n'+boundary+'\n\n'
        'Continue only if actual independent future-law and cost evidence supports the interaction. A positive screen still needs current prior-art differentiation, external strong baselines and fresh real-domain confirmation.\n', encoding='utf-8')
    (R / 'final_research_direction.md').write_text('# C49 final direction remains unaccepted\n\n'
        '**Research Question:** Can observation-marginalized conditional learning improve a stochastic time-series world model under informative sparse sensing?\n\n'
        '**Evidence:** '+evidence+' Analytic/gradient/likelihood/model replay qualifications passed; formal future-quality comparison pending.\n\n'
        '**Existing Methods:** Conditional signature-kernel Neural SDEs, AmbientGAN/AmbientFlow, latent variational dynamics and marked-Cox inference cover the ingredients.\n\n'
        '**Proposed Idea:** The implemented conditional observation-marginalized learning interaction; novelty and utility remain unestablished.\n\n'
        '**Falsification:** Comparable complete-clock variational prediction at equal/lower cost, probe-only gains, nonidentification or failure to transfer removes the intended claim. No EVAL retuning.\n\n'
        '**Feasibility:**8 server fits, shared9458-parameter predictive core, fixed fresh data and two seeds. Single5090 available; tiny whole-step timings selected CPU.\n\n'
        '**Next Experiment:** Complete frozen fits, locked physical/observation future scoring and independent all-path replay. If promising, define a fresh real-domain/strong-baseline protocol before accepting the direction.\n\n'
        'Accepted final directions0; overall goal ACTIVE_UNACHIEVED.\n', encoding='utf-8')
    (R / 'CURRENT_STATE.md').write_text('# C49 training snapshot\n\n'+evidence+'\n\n'
        'Active versions world49_v2/setup49_v2/qualify_model49_v2/train49_v2/capture49_v2. Pre-training scale-list repair is recorded in RUNTIME_REPAIR.md; original code and partial arrays retained. '
        'PROTOCOL.md remained unchanged. score49/verify49/capture_eval49 implement the sealed evaluation queue. C48 is complete and must not be restarted.\n', encoding='utf-8')
    dump(R / 'PROGRESS_REPORT_MANIFEST.json', {'utc': utc, 'status': 'C49_TRAINING_PROGRESS_GOAL_UNACHIEVED', 'completed_fits_at_snapshot': len(fits),
                                             'formal_updates_completed_fits': sum(f['updates'] for f in fits),
                                             'reports': {p.name: sha(p) for p in R.glob('*.md')}, 'source_sha256': sha(pathlib.Path(__file__)), 'goal_complete': False})
    # One dated root index; preserve exact preceding bytes and check no concurrent edit.
    receipt = R / 'ROOT_INDEX_RECEIPTS.json'
    assert not receipt.exists()
    archive = R / 'outputs/root_history_v1'
    archive.mkdir()
    records = []
    header = f'''<!-- cycle49-progress-index-start -->
# C49：新的条件随机世界模型学习比较正在执行

UTC {utc}。整体ACTIVE_UNACHIEVED，接受最终方向0。研究目标仍是WorldTS贡献范围的可复用时间序列world-model框架。

C48已经完成16组训练/40750正式updates、44组原始预测/1694条模型—起点—揭示评分，独立CSV/模型/指标重算PASS。显式延迟没有稳定超越GRU；普通GRU/ridge也能利用新观测。该具体路线降优先级，结果见[完整结果](cycle48/experimental_results.md)和[裁决](cycle48/SCIENTIFIC_DECISION.md)。勿重启已完成队列。

C49实现“在条件随机世界模型训练中解析积分观测过程”与完整计数似然变分学习的shared-core比较。两组预测核心同为9458参数，变分组另有13944训练期识别参数。独立数学/梯度/似然/NumPy模型核查通过，12discarded updates不计正式训练。冻结8组新训练，当前完成{len(fits)}组；无EVAL收益主张。详见[协议](cycle49/PROTOCOL.md)、[当前状态](cycle49/CURRENT_STATE.md)、[文献边界](cycle49/RESEARCH_INTAKE.md)。

条件路径评分、观测模型积分等组件已有先例，具体框架收益与新颖性仍待证明。当前为两类独立合成过程的可证伪筛查，尚无真实域验证。历史索引如下。
<!-- cycle49-progress-index-end -->

'''
    for name in ['CURRENT_RESEARCH_STATUS.md', 'literature_review.md', 'baseline_registry.md', 'experimental_results.md',
                 'negative_evidence.md', 'hypothesis_space.md', 'research_ideas.md', 'final_research_direction.md']:
        path = R.parent / name
        previous = path.read_bytes()
        (archive / name).write_bytes(previous)
        assert path.read_bytes() == previous
        path.write_bytes(header.encode('utf-8') + previous)
        records.append({'path': name, 'before': hashlib.sha256(previous).hexdigest(), 'after': sha(path)})
    dump(receipt, {'utc': utc, 'files': records, 'goal_complete': False})
    print(json.dumps({'status': 'C49_PROGRESS_REPORTS_SAVED', 'completed_fits_at_snapshot': len(fits), 'accepted_final_directions': 0}), flush=True)


if __name__ == '__main__':
    main()
