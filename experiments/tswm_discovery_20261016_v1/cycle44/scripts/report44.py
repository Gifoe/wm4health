"""Evidence-bound progress reports. Preserve each earlier snapshot verbatim."""
import pathlib,json,hashlib,datetime,time
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs'
def load(p):
 if not p.exists():return None
 for attempt in range(4):
  try:return json.loads(p.read_text(encoding='utf-8'))
  except json.JSONDecodeError:
   if attempt==3:raise
   time.sleep(.05)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat();q=load(O/'qualification_v1/qualification.json');assert q['status']=='PASS';status=load(O/'fits_v1/status.json');training=load(O/'fits_v1/training_summary.json');summaries=training['summaries'] if training else [load(p) for p in sorted((O/'fits_v1').glob('*/summary.json'))];pilot=load(O/'pilot_v1/status.json');quality=load(O/'quality_v1/quality_summary.json');quality_verify=load(O/'quality_verification_v1.json');truth=load(O/'reference_eval_v1/reference_eval.json');reference=load(O/'reference_validation_v1/reference_validation.json');analytic=load(O/'analytic_qualification_v2/qualification.json');cache=load(O/'query_qualification_v1/qualification.json')
actual=f"UTC {now}。C44本轮是机制筛查PROGRESS；整体ACTIVE_UNACHIEVED，最终接受方向0。正式训练{len(summaries)}/16完成，训练状态{status['status']}；已完成模型的正式updates共{sum(s['stopped_step'] for s in summaries)}。VAL分配阶段{pilot['status'] if pilot else 'pending'}。独立EVAL质量阶段{'scored, independently verified' if quality and quality_verify else 'scores produced, verification pending' if quality else 'pending/running'}；EVAL truth reference {truth['status'] if truth else 'pending'}。同拟合law high-sample reference、16-replicate estimator比较与实际allocated-call timing尚未完成，不能宣称新方法收益或完成研究目标。"
scope="用户明确目标为可复用的时间序列world-model框架，WorldTS作为贡献范围参照。单数据集提分、ordinary采样器包装或implementation repair不足作为最终贡献。具体idea由机制证据决定，未要求新数学算子、Loop或固定某一数据集。"
registry='# C44 Baselines and reproducibility\n\n'+actual+'\n\n'+scope+'\n\n两种独立生成的noisy known-process carriers，各两初始化，四个完整模型：原生SPF、仅条件packing启用梯度的ordinary控制、TRAIN-PCA上的两层GRU256/4-component full-covariance MDN、联合估计的LGSSM。所有模型只有8步历史观测；真值状态不提供给learner。来源commit、数据、信息、优化和selection规则见PROTOCOL.md。不同family目标/预算/容量不同，不宣称隔离架构优劣。\n\n|case|family|seed|selected update|stopped update|nominal params|last-gradient params|train+VAL seconds|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
for s in summaries:registry+=f"|{s['case']}|{s['family']}|{s['seed']}|{s['selected_step']}|{s['stopped_step']}|{s['nominal_parameters']}|{s['parameters_with_last_gradient']}|{s['training_and_validation_seconds']:.2f}|\n"
registry+='\nSelected checkpoints与last states、固定minibatch indices、曲线与SHA都在outputs/fits_v1。原生SPF继承的unused parent参数保留并计入nominal；不从inner width推断参数数。原论文气候headline未复现。\n'
results='# C44 Actual evidence and unfinished comparisons\n\n'+actual+f"\n\n资格检查：energy独立误差{q['independent_energy_max_abs']:.9g}；条件梯度普通修复前向差{q['native_autograd_only_forward_delta']}，启用后梯度max {q['ordinary_condition_gradient_max']:.9g}；mixture logprob独立差{q['independent_mixture_logprob_max_abs']:.9g}；Kalman likelihood独立差{q['independent_Kalman_nll_max_abs']:.9g}。30步资格训练全部丢弃，不用于效果。\n\nVAL truth reference状态{reference['status'] if reference else 'pending'}：Gaussian exact Kalman与两particle counts/两rep核查；nonlinear count convergence核查只说明这些VAL histories数值可用，不能保证EVAL。强普通analytic/conditional-noise readout的代数资格状态{analytic['status'] if analytic else 'pending'}。普通prefix/coarse路径资格{cache['status'] if cache else 'pending'}。这些不是新方法效果。\n"
if quality:
 results+='\n独立held-out full-law scores（不是查询收益或框架创新证明）：\n\n|model|joint energy ↓|mean MSE ↓|marginal90% coverage|\n|---|---:|---:|---:|\n'
 for s in quality['rows']:results+=f"|{s['model']}|{s['joint_energy_mean']:.7f}|{s['mean_prediction_MSE']:.7f}|{s['marginal_90pct_coverage']:.5f}|\n"
 results+='\n每行64独立histories/64 iid futures，原始paths/targets保留。所有family在各自预算下训练，不能单凭排名归因架构。\n'
results+='\n本轮人工known-law EVAL与历史真实数据TEST不同；不要把其执行状态混同。详见各outputs/status.json原始时间戳。\n'
lit='''# C44 Literature boundary

This round executes a falsification screen, without claiming a new literature discovery. The pinned [SPF author implementation](https://github.com/stanfordmlgroup/spf/tree/a7624ae25b47b474d5a31ca22b3ac6e6907795dd) and [arXiv Methods](https://arxiv.org/html/2512.02268v1) already provide multiscale conditional generation. C43 qualified the actual numerical core, scheduler, collator and author sampler bodies in a disclosed context adapter. It did not reproduce the full original climate pipeline.

Ordinary common-random-number MLMC requires matching coarse marginals in its independent and correction terms; it does not require coarse outputs to equal aggregated fine outputs. Antithetic sampling, exact Gaussian expectations and integrating independent decoder noise are established ordinary controls. Their success would not establish a new world model. Their omission would create an artificially weak comparator.

WorldTS remains a contribution-scope reference. Observation/dynamics modular reuse remains a distinct candidate, constrained by NCDSSM, modular dynamics, LIFTS and ordinary structured SSMs documented in C41–C43. A generic model's expressivity does not automatically invalidate a useful new learning interaction; actual equivalence or a strong complete implementation must be checked.
'''
negative='''# C44 Negative evidence and limits

'''+actual+'''

- No new framework has been accepted. Training completion and numerical qualification are not innovation evidence.
- The native conditioning projection's detached packing has a simple gradient-enabled control. Forward equivalence before training is verified; this is not a proposed method or proof that the original ignores conditioning.
- Learned Gaussian models allow exact expectations of several diagnostic functions. Zero estimator variance still leaves learned-law bias; a new hierarchy must compete with this control.
- Conditional decoder-noise integration is ordinary Rao–Blackwellization. A gain from it does not establish an architecture gap.
- The original analytic qualifier omitted detach in a diagnostic CPU conversion and failed before numerical comparisons. Original source/attempt retained; V2 fixes diagnostic conversion only. See FAILURE_LEDGER.md.
- Controlled synthetic carriers, four query histories/carrier and two initialization seeds cannot establish real-domain generality or publication readiness. Keep model bias, estimator error and reference uncertainty separate.
'''
hyp='''# C44 Competing mechanisms

H1: SPF plus common-RNG/antithetic/ordinary MLMC already provides useful queries. Same selected weights, independent coarse marginals, fitted-law references,16 replicates and actual total cost distinguish this from an architecture need.

H2: a cheap fitted state-space/AR world law and analytic/conditional readouts absorb the utility. Independent full-law quality and known-process conditional expectations distinguish low estimator variance from a wrong model. If H2 covers the target, end the corresponding new-hierarchy claim.

H3: only trivial linear queries benefit, model bias dominates, or allocation overhead erases the gain. All five TRAIN-defined functions, including roughness and threshold functions, remain mandatory. No EVAL-driven query/weight selection.

H4: a useful new learning/inference interaction remains after these controls. No current effect establishes H4. It would require a concrete mechanism, matched information/compute controls and independent cross-domain/task validation before becoming a framework contribution.

The observation-evidence/dynamics-interface route is separate; C42 ordinary HMM's anchored held-out signal is evidence of a structural inductive bias, not an accepted new model. Do not expand the same synthetic setting simply to accumulate results.
'''
ideas='''# C44 Candidate research scope

'''+scope+'''

1. Observation-evidence × reusable dynamics: a framework candidate about learning transferable probabilistic state/dynamics interfaces across observed sensor–system compositions. Existing structured SSMs and modular inference are strong complete contenders. A necessary new learning interaction and neural/real-domain evidence remain missing. Merely replacing HMM tables with MLPs is insufficient.
2. Query-adaptive stochastic world laws: a reserve route about answering nontrivial future path queries at useful cost. C44 tests whether strong existing generators and ordinary estimation already cover it. It is not yet a method; no special consistency loss or new architecture is authorized by the evidence alone.

These are development priorities, not final accepted recommendations. Maximum final recommendations remains three; accepted count is zero. A known-component recipe can qualify if its necessary interaction and useful effect are established, without requiring invention of a mathematical operator.
'''
final='''# C44 Current research judgment

'''+actual+'\n\n'+scope+'''

**Research Question**：如何学习可复用的概率状态与演化规律，使时间序列world model支持独立任务上的未来推演，而不局限于固定数据集/固定horizon误差？当前筛查的问题是按查询需求生成未来是否还需要新机制。

**Evidence**：真实服务器运行、独立数值核查和训练日志见experimental_results.md；尚无被验证的新框架收益。当前已知ordinary解析/条件积分能力必须纳入强对照。

**Existing Methods**：SPF已有多尺度采样；AR/SSM已有廉价随机演化与强读出；观测/动态分离近邻另有NCDSSM/LIFTS/modular模型。不能因遗漏组合对照声称研究缺口。

**Proposed Idea**：具体新机制待筛查决定，不把经典MLMC、梯度修复或普通structured SSM重新命名。保留观测证据接口×动态复用作为不同候选。

**Falsification**：ordinary完整方法覆盖目标效用；只在平凡均值函数/同一数据集获益；bias或额外预算解释提升；没有独立任务泛化。发生时淘汰相应创新主张。

**Feasibility**：单RTX5090实际训练；数据、split、seeds、source commit、权重与原始日志保留。人工carrier只决定资格，跨真实域还需后续证据。

**Next Experiment**：完成已冻结的VAL分配、held-out full-law与truth核查，再做同拟合law高样本reference、16独立估计replicates及真实总成本。若普通方法已充分，结束query-hierarchy包装，转向具体有缺失证据的框架学习交互。
'''
decision='# C44 Scientific decision — progress, not a framework acceptance\n\n'+actual+'\n\n'+scope+'\n\nDo not interpret any pending stage as completed. No final scientific direction or submission readiness is asserted.\n'
reports={'literature_review.md':lit,'baseline_registry.md':registry,'experimental_results.md':results,'negative_evidence.md':negative,'hypothesis_space.md':hyp,'research_ideas.md':ideas,'final_research_direction.md':final,'CURRENT_STATE.md':'# C44 live evidence snapshot\n\n'+actual+'\n\n'+scope+'\n','SCIENTIFIC_DECISION.md':decision}
old=R/'REPORT_MANIFEST.json'
if old.exists():
 archive=O/'report_snapshots'/now.replace(':','-');archive.mkdir(parents=True)
 for name in list(reports)+['REPORT_MANIFEST.json']:
  p=R/name
  if p.exists():(archive/name).write_bytes(p.read_bytes())
for name,body in reports.items():(R/name).write_text(body,encoding='utf-8')
manifest={'utc':now,'goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0,'fits_completed_at_snapshot':len(summaries),'formal_updates_completed_fits':sum(s['stopped_step'] for s in summaries),'reports':{n:sha(R/n) for n in reports},'protocol_sha256':sha(R/'PROTOCOL.md'),'source_sha256':sha(pathlib.Path(__file__)),'status_snapshot':{'training':status,'pilot':pilot,'quality_complete':quality is not None,'quality_independently_verified':quality_verify is not None,'truth_status':truth['status'] if truth else None}};(R/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'reports':len(reports),'fits_completed_at_snapshot':len(summaries),'goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}))
