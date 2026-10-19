"""Report verified native contracts and the actual remaining framework question."""
import pathlib,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();q=read(O/'native_qualification_v1/qualification.json');v=read(O/'verification_v1.json');assert q['status']=='PASS_WITH_EXPLICIT_SAMPLING_CONTRACTS' and v['status']=='PASS' and v['qualification_sha256']==sha(O/'native_qualification_v1/qualification.json');assert not (R/'REPORT_MANIFEST.json').exists();now=datetime.datetime.now(datetime.timezone.utc).isoformat()
summary=f"UTC {now}。本轮PROGRESS；整体ACTIVE_UNACHIEVED，最终接受方向0。固定官方NCDSSM commit {q['source_revision']}，48个选定source/config/license文件/Gitblob+SHA核查。原生线性推断与独立NumPy/dense Gaussian一致；非线性原生core完成10 discarded updates/862参数与预测回放，0正式比较fits。普通联合Markov路径控制核查了线性采样定义区别，不是新模型收益。"
lit='''# C45 Literature and innovation boundary

The [NCDSSM ICML2023 paper](https://proceedings.mlr.press/v202/ansari23a.html) already separates auxiliary recognition from continuous-discrete dynamic inference. Its [pinned author source](https://github.com/clear-nus/NCDSSM/tree/7212bba9616d81c948f507eef9eb1f7d5cf1954e) is now actually executed in the present environment, rather than cited as a schematic comparator. Its recognition/dynamics separation is not a new primitive.

The [C44 targeted primary-source screen](../cycle44/literature-search-20261017-observation-interface/papers.md) adds implemented exposure-aware inference and incremental observation-scale alignment as close boundaries. WorldTS remains the user's contribution-scope example; multimodality, Loop, physical-state identifiability and a new mathematical operator are not mandatory. Related model-class expressivity does not alone invalidate a new useful training interaction.

Modular dynamics and ordinary structured SSMs remain strong complete contenders. Reusing observation evidence together with dynamics under held-out combinations remains a development question, not an established global gap. A new framework would need a concrete distinct learning recipe, fair complete controls and independent tasks/domains. No sampling-interface correction alone earns that claim.
'''
registry=f'''# C45 Native baseline registry

{summary}

|Control|Actual execution|What it establishes|What it does not establish|
|---|---|---|---|
|Author BaseLTI|fixed known2D parameters; native matrix-exponential prediction/filter|numerical likelihood/mean/covariance/gradient contract|training quality or a new model|
|Native Base.forecast|16384 default state draws, same filtered history|timewise-marginal sampling behavior|a joint Markov trajectory sample contract|
|Ordinary joint Markov sampler|same linear process parameters/posterior,16384 draws|standard complete joint-law control|a new architecture or nonlinear repair|
|Author NCDSSMNL|z6/aux2/y2/MLP64/RK4.05;10 discarded updates; CPU|real core forward/backward/forecast and checkpoint replay|convergence, paper-headline reproduction or a trained framework benefit|

Source subset excludes data/datasets directories, media and checkpoints; core numerical files are unmodified. All loaded native module origins were checked inside this pinned cache in a fresh process. ServerPython3.10/torch2.8cu128/torchdiffeq0.2.3 differs from declaredtorch~=1.11; no global downgrade, dependency installation or numerical source patch was required. Core execution did not require the optional TensorboardX training shell. CUDA was never initialized by these tests.

Control inputs are explicitly unimplemented in this source constructor (`u_dim==0`). No action-conditioned capability is inferred. Nonlinear joint SDE trajectory sampling with independent outer auxiliary-history draws still needs an ordinary disclosed adapter and numerical convergence checks before any full-law method comparison.
'''
results=f'''# C45 Actual execution evidence

{summary}

Native linear max absolute differences: mean{q['linear_inference_max_abs']['mean']:.3g}, covariance{q['linear_inference_max_abs']['covariance']:.3g}, log likelihood{q['linear_inference_max_abs']['log_likelihood']:.3g}; independent finite-difference gradient difference{q['independent_gradient_abs']:.3g}. A separate dense-block Gaussian conditioning implementation yields mean difference{v['independent_dense_conditioning_mean_max_abs']:.3g} and covariance difference{v['independent_dense_conditioning_covariance_max_abs']:.3g}. Retained sample statistics reconstruct exactly.

Both samplers match the known linear future marginal moments within predeclared MC tolerances. Their path dependence differs:

|Sampler|Cov(state1 at first,last future)|Exact joint covariance|max discrepancy in joint MC-SE units|Var(last−first), state1/state2|
|---|---:|---:|---:|---|
'''
for row in q['sampling']:results+=f"|{row['kind']}|{row['cross_time_covariance_0_2'][0][0]:.7f}|{row['exact_joint_cross_time_covariance_0_2'][0][0]:.7f}|{row['max_cross_discrepancy_in_exact_joint_MC_SE']:.3f}|{row['temporal_difference_variance'][0]:.6f} / {row['temporal_difference_variance'][1]:.6f}|\n"
results+=f'''
Exact joint temporal-difference variances are{q['sampling'][0]['exact_joint_temporal_difference_variance'][0]:.6f}/{q['sampling'][0]['exact_joint_temporal_difference_variance'][1]:.6f}. This is a fixed-parameter numerical fixture, not a quality comparison between trained methods. The ordinary control resolves the discrepancy without changing learned dynamics.

The outer native forecast traced one auxiliary-history sample for B1 before generating16 futures; repeated-history B16/one-future calls traced16 independent auxiliary histories. This addresses the outer ensemble contract only; it does not itself turn the native marginal-time forecast into a nonlinear joint simulator.

Nonlinear qualification ran10 updates,862 parameters, all finite gradients/forecasts, checkpoint replay max{q['checkpoint_replay_max_abs']}. Loss values on stochastic preflight draws are not convergence evidence. Original source and logs retained. Diagnostic scalar conversion emitted a nonfatal requires-grad warning; no numerical repair was made.

See [sampling-contract diagnostic](outputs/figures_v1/sampling_contract.png) and [C44 controlled quality/cost](outputs/figures_v1/controlled_quality_cost.png). The figures describe numerical/utility controls, not a new-method result.
'''
negative='''# C45 Negative evidence and ordinary fixes

'''+summary+'''

- Recognition–dynamics separation is already an operational complete baseline. It cannot be introduced as an absent primitive.
- Correct per-time means/variances do not imply joint path dependence. The native linear forecast's cross-time covariance differs from the generative joint law; a standard Markov sampler with identical information/parameters resolves it. No new model is justified by this interface issue.
- A repeated-history outer draw control supplies independent auxiliary histories; it does not solve the separate within-path temporal law. Keep these contracts distinct.
- No formal neural candidate or cross-domain effect was trained. Ten preflight steps cannot support a framework claim.
- C44's full query protocol was deliberately stopped for priority reasons and remains incomplete. Five completion-sealed groups are partial evidence, not a replacement for16 groups or positive-result selection.
- Startup/guard traces in EXECUTION_LEDGER are operational events, not model failures or scientific falsifications.
'''
hyp='''# C45 Hypotheses to distinguish before a new framework claim

H0: a strong direct structured NCDSSM/modular learner already provides the desired observation/dynamics reuse. It must receive the same histories, synchronization anchors, noise information and optimization opportunities. If it covers the utility, do not rename it.

H1: an apparent probability/path failure comes only from marginal-time or shared-outer sampling semantics. The linear control already demonstrates an ordinary resolution. A full nonlinear comparison needs the corresponding standard generative-path adapter before attributing failure to learned states.

H2: a distinct learning interaction helps observation-evidence and dynamic-module composition under finite data. The four-state HMM signal from C42 is not proof of a neural framework. A full neural test must measure held-out composition effects and seen-task tradeoffs, rather than only showing that the most structured model wins.

H3: any improvement is explained by extra anchors, capacity, gradient steps, inference accuracy or a mismatched evaluation law. Match information and parameter/compute opportunities, and keep model approximation, inference and query error separate.

No current result establishes H2 or a conference-ready research gap. Do not expand dataset count before an executable mechanism and discriminating test exist.
'''
ideas='''# C45 Research development priority

The user's target remains one reusable time-series world-model framework. The most worthwhile remaining development question is **learning observation-to-state evidence and dynamic mechanisms that can be recomposed for held-out sensor–system combinations**. Same framework/rules must transfer across tasks; a per-dataset improvement is insufficient.

The current ingredients overlap NCDSSM, modular dynamic inference, LIFTS, observation-scale alignment and ordinary hierarchical SSMs. A drawing of their composition is not a new contribution. A useful known-component recipe remains eligible if its necessary interaction, complete training/inference protocol and independent generalization value are demonstrated. Generic SSM expressivity alone is not a rejection criterion.

The next missing element is a concrete learning recipe that differs from the strongest complete direct composition. Possible training interactions still need prior-art checks and falsification; none is named a new method here. Keep the observation/dynamics interaction as the primary development question, not a validated recommendation. C44 query-hierarchy packaging is deprioritized, with full query protocol explicitly incomplete. No second or third final direction is recommended; accepted count remains0.
'''
final='''# C45 Current framework research direction — not yet accepted

'''+summary+'''

**Research Question**：一个时间序列world model能否用同一训练配方，把观测证据接口与动力学机制学成可复用组件，并在未见sensor–system组合保持可靠的概率未来推演？

**Evidence**：C42只有普通因子化HMM的人工held-out结构信号及seen代价。C44完成16模型的分布评分和强普通控制筛查，未支持新增query层级机制；余下查询提前停止。C45已实际运行强完整SSM core并明确其采样合同，但没有新方法效果。

**Existing Methods**：NCDSSM已有auxiliary recognition/dynamics分离；modular模型已有动态模块复用；LIFTS已有joint inference/generation；积分观测和尺度对齐另有完整近邻。它们的直接强组合必须保留。采样接口差异有普通处理，不能制造新框架优势。

**Proposed Idea**：开发重点是观测证据×动态组合之间的必要学习交互。具体新配方仍缺证据，不把ordinary structured SSM或Markov采样改名。已知组件可以构成新配方，但必须有明确区别和真实独立效用。

**Falsification**：强直接组合覆盖未见组合收益；交互不成立；只改善seen cells；收益由额外信息/容量/预算解释；或需要每个数据集换机制。发生时放弃相应创新主张。

**Feasibility**：单RTX5090，原生NCDSSM数值core已在现有环境执行。下一步先完善普通非线性generative joint-path对照，再冻结完整神经任务、容量/预算和信息合同。资格权重全部排除，不在旧EVAL上调方法。

**Next Experiment**：先使非线性强基线具有明确的联合轨迹与独立outer-history抽样定义；随后检验一个具体新学习交互对比完整direct composition。需要未见组合、seen权衡和独立任务证据；不能再以“11最好”代替机制证明。最终idea和顶会潜力尚未验证，整体目标继续ACTIVE_UNACHIEVED。
'''
reports={'literature_review.md':lit,'baseline_registry.md':registry,'experimental_results.md':results,'negative_evidence.md':negative,'hypothesis_space.md':hyp,'research_ideas.md':ideas,'final_research_direction.md':final,'CURRENT_STATE.md':'# C45 native qualification and independent verification complete\n\n'+summary+'\n\nFull nonlinear joint-path adaptation and new framework evidence remain outstanding. See final_research_direction.md.\n','SCIENTIFIC_DECISION.md':'# C45 scientific decision\n\n'+summary+'\n\nThe ordinary joint-path control resolves the verified linear sampling-contract difference. This is a baseline correction, not a new world model. Prioritize an actual observation/dynamics learning interaction; no final research direction is accepted.\n'}
for name,body in reports.items():(R/name).write_text(body,encoding='utf-8')
manifest={'utc':now,'goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0,'comparative_fits':0,'discarded_updates':10,'reports':{n:sha(R/n) for n in reports},'qualification_sha256':sha(O/'native_qualification_v1/qualification.json'),'verification_sha256':sha(O/'verification_v1.json'),'source_tree_sha256':sha(R/'source_archive/ncdssm/SOURCE_TREE.json'),'source_sha256':sha(pathlib.Path(__file__))};(R/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'reports':len(reports),'native_discarded_updates':10,'comparative_fits':0,'accepted_final_directions':0,'goal':'ACTIVE_UNACHIEVED'}))
