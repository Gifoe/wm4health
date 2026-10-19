import pathlib,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent
O=ROOT/'outputs/compatibility_witness_v1'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
r=json.loads((O/'result.json').read_text());v=json.loads((O/'verification.json').read_text())
assert v['status']=='passed' and v['result_sha256']==sha(O/'result.json')
assert r['arrays_sha256']==sha(O/'arrays.npz')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
head=f'UTC {now}。通用时间序列 world-model 框架发现任务 ACTIVE_UNACHIEVED；接受最终方向0。\n\n'
reports={}
reports['literature_review.md']=head+'''# 目标协议与创新边界

本轮在 ccf-idea-optimizer exploratory 与 deep-research 三视角下，按冻结 RQ 核查未来目标协议、既有修复和可执行区别。完整来源/比较在 [主流](target_models32.md)、[方法学](target_critic32.md)、[相邻理论](target_adjacent32.md)。没有用三人结论投票代替证据。

**RQ1。** [WorldTS Eq18–21](https://arxiv.org/html/2609.31162v1#S4.SS5) 实际使用 MAE+cosine+VICReg，target 之前有历史 prefix，future normalization 用历史统计，decoder 在预测与编码潜态上训练；不能以 future-only、future-stat leakage 或 decoder 只见真编码批评它。[LatentTSF](https://arxiv.org/html/2602.00297v2) 的冻结 AE 与平方 latent loss 是另一协议。两者均无需先证明 physical state identification 才有 forecasting utility。

**RQ2。** PSR/PSIM 的未来特征期望、[calibrated surrogate/link 理论](https://arxiv.org/abs/1902.01958)、近期 [CF-JEPA](https://arxiv.org/abs/2606.07031)、[Orthogonal JEPA](https://arxiv.org/abs/2608.20065)、[Latent-Foresight](https://arxiv.org/abs/2610.01942)、[VJEPA](https://proceedings.mlr.press/v306/huang26ba.html) 等已覆盖 broad predictive-state/history/multi-horizon/stochastic/grounding 动机。不能把 squared-loss 条件均值解释套给原生 MAE+cosine。通用 surrogate 兼容性文献是先例，不声称其 smooth-convex theorem 直接覆盖本轮非凸 cosine 命题。

**RQ3。** 固定编码时，样本可逆与回归摘要的 task compatibility 是两件事。下述有限 witness 可执行地分开它们；但更宽的普通 decoder、native prediction-only decoder、raw observation supervision 和 protected anchor 都是现有强修复。下一个有意义判断必须是这些控制后仍存在的 actual learned-model residual，而不是更多固定 toy。

四个已核对本地 WorldTS 核心文件与官方 pinned revision 的规范化文本相等；VISReg/v46B 字样本身就在公开代码中。不能把无 .git 或这些字符串当作污染证据。本轮三个 loss 组件又按 public Git blob 下载、hash 核验并在服务器实际调用。
'''
reports['baseline_registry.md']=head+f'''# 模型、来源和执行状态

| 对象 | 身份与状态 |
|---|---|
| WorldTS revision {r['source_revision']} | 三个 loss 原组件，public blob/SHA 核对，128 次 native 对独立 NumPy 公式比较；本轮没有整模型训练 |
| Native recipe | 10×MAE +15×cosine，固定 target；reg0 与 .1 两个精确诊断；H1/40行，不冒充 H24/H360 官方任务 |
| LatentTSF、PSR/PSIM、2026 JEPA 等 | 本轮一手协议/创新边界核查；没有伪报新执行 |
| C11/C21 | 已有 learned-target/future-driver 负证据继续保留；本轮不重复其训练 |
| C31 NormWear-2 | 上轮已执行并修复接口；本轮未重复下载/推理 |

源码见 upstream/worldts_loss/manifest.json。科学脚本 sha `{r['script_sha256']}`；计划在执行前固定。raw GitHub 网络连接失败后，公开 Git blob API 正常 TLS 获取相同源码，整文件 SHA 上传服务器；没有关闭 TLS 或改变系统网络。原失败 acquisition 脚本与操作记录保留。

服务器使用既有 Python/torch2.8+cu128 环境与 process-local 线程/affinity；诊断实际 CPU运行，GPU没有新训练进程。40行是精确有限分布，不是40个独立世界或实际训练样本效率结果。
'''
reports['experimental_results.md']=head+f'''# 实际执行：固定编码的任务兼容性

Y∈{{−1,0,1}}；等概率 contexts A/B，条件概率分别(.45,.40,.15)/(.60,.10,.30)。每context20行精确计数，合40行。固定编码 phi(−1)=(100,100)、phi(0)=(103,100)、phi(1)=(100,103)。所有 future target 的逐样本 Y 可用 `(phi1+2phi2−300)/3−1` 精确恢复。

| 项 | 结果 |
|---|---|
| 样本线性重构误差 | {r['exact_linear_reconstruction_max_abs']:.1g} |
| Target 两坐标 std | {r['target_std']}，均>1 |
| Target covariance eigenvalues | {r['target_covariance_eigenvalues']}，均>0 |
| 原生 loss 固定编码全局最优摘要 | contexts A/B 都是(100,100)，reg0与.1均有严格全局margin证书 |
| 原观测 MAE 最优预测 | A=0、B=−1 |
| 共用摘要最佳 decoder MAE | {r['best_shared_decoder_MAE']:.2f} |
| 区分contexts的 Bayes MAE | {r['context_Bayes_MAE']:.2f} |
| 不可避免 excess | {r['unavoidable_excess_MAE']:.2f}=1/20 |
| 普通 protected raw anchor | 读取0/−1，MAE恢复.65；没有声称未保护cos/reg耦合anchor的保证 |
| 普通平方 target +重拟合 MAE读出 | Ephi区分两概率分布，正确读出0/−1，MAE恢复.65 |

作用域是固定编码、H1、给predictor的两个context、decoder额外side information相同。WorldTS实际decoder还接收历史状态与RevIN统计；这些若包含context，可以修复collision。因此本例不是全模型失败。更不说明自由联合训练会选择此编码。

源组件与独立NumPy公式128次比较，最大relative error {max(x['native_formula_max_relative'] for x in r['perturbation_checks']):.3g}；reg0/.1各2304个预定随机扰动均满足正risk gap。大扰动quartic covariance造成最大absolute差约7.63e−6，实际relative差约1.67e−15，使用执行前的abs+relative容差。全局最优由解析margin证明，扰动/数值优化不是证明替代。

独立 verifier 用 Fraction 有理数重算1/20、独立解线性重构系数、枚举4001个raw决策、检查2个全局证书、128个原组件比较以及普通读出；全部通过。科学执行 {r['seconds']:.3f}s，verifier {v['seconds']:.3f}s。见 [结果](outputs/compatibility_witness_v1/result.json)、[验证](outputs/compatibility_witness_v1/verification.json)。

这一步证明的是一个范围受限的数学现象，没有真实benchmark分数或新方法优势。
'''
reports['negative_evidence.md']=head+'''# 淘汰的宽泛解释与保留边界

- Future target 局部就必然丢失预测状态：不成立。平方loss时包含原观测的target特征已可保留MSE Bayes act；原生WorldTS还含history/covariates及预测态decoder。
- 普通 Jensen 不等式就证明WorldTS坏：不成立。它的decoder在预测态上适配，且原生不是纯平方loss。
- 重构好就保证下游好：本轮固定编码witness否定；可逆、满秩、目标std>1仍能在surrogate Bayes summary处合并。缺口在summary，不在sample encoding。
- 这已证明真实WorldTS失败：不成立。编码由人为固定，H1、decoder side information相同；自由joint training、实际history prefix与finite budget均未测。
- 这已需要新架构：不成立。两项普通控制精确修复，calibrated surrogate/PSR理论已提供基础。未证明新框架的必要性或收益。
- latent/readout复杂loss和epoch1状态选择再启动：已有NP/PJM历史NO-GO/INCONCLUSIVE，不由这个数学例子推翻。

原先对本地VISReg/v46B疑似修改的推测已由public pinned文件逐字核查纠正，不能传播为源码污染事实。网络错误没有产生科学失败或放宽检查。
'''
reports['hypothesis_space.md']=head+'''# 竞争解释

| 假设 | 本轮证据 | 下一项能区分的证据 |
|---|---|---|
| 样本编码丢失原始值 | 本例排除，精确线性重构0误差 | 真实learned target的独立readout及actual side info |
| surrogate Bayes摘要丢失任务答案 | 固定编码例得到严格正reg/无reg证书 | 实际trained target上出现forecast-relevant residual，而非只抽象反例 |
| shared decoder mix负迁移 | 本轮未测；native已有prediction-only route | 同固定stage1/decoderinit/order预算，mix .3 vs pred0 |
| 目标权重、容量或优化不足 | 本轮未排除真实任务上的这些解释 | ordinary supervision/weight/capacity envelope，匹配总预算 |
| 需要新compatible目标机制 | 暂无支持，ordinary控制修复witness | 相同输入/预算下超过这些修复且跨任务/骨干可重复 |

若actual decoder side info已区分两个context，摘要collision不能证明forecast不足。若普通pred-only/observation loss修好，停止新target架构叙事。若容量/收敛修好，可研究明确的有限预算效率，但不能继续声称原则信息不足。
'''
reports['research_ideas.md']=head+'''# 探索分支与救援路线

| 分支/谱系 | 可发展的具体问题 | 现状和最强对照 |
|---|---|---|
| A，refine：future target→task-compatible report | 可否在保持latent预测收益时，使学习到的统计摘要稳定支持native MAE/MSE读出？ | 本轮witness确认定义有意义；protected raw anchor、pureMSE+refit、普通obs监督已修复。尚无新机制，needs-native-evidence |
| B，instrument：同一stage1的decoder监督来源 | mix true/pred codes何时帮助、何时伤害实际预测？ | native stage2_decoder_pred已实现；可用作A的归因控制，不能只改alpha成为框架贡献 |
| C，combine：多尺度/随机目标与预算 | 哪种目标与predictor耦合在固定总预算下比简单监督更有效？ | CF-JEPA、OrthogonalJEPA、Latent-Foresight等直接先例；未指定新的兼容操作，needs-mechanism |

当前优先级只是A的**原生残差资格判断**，不是推荐论文方向。结构不同的fallback B用于排查decoder，而非承诺另写一篇paper。没有将“组件已知”作为无限预算否决；有限预算方法仍允许，但必须给出超过普通组合的具体交互与实测。

不继续扩大同一40行toy或反复调其编码。新问题应在真实允许输入、完整decoder side information、匹配loss的已知方法下成立；单数据集涨点不足以完成用户目标。
'''
reports['final_research_direction.md']=head+'''# 当前推荐：尚无已验证最终方向

**Research Question：** 可复用时间序列world model的未来编码目标，经过实际训练surrogate后，是否仍保留最终预测任务所需的统计答案？这是本轮精确化问题，不是已确认的未解决gap。

**Evidence：** 服务器执行一个固定、可逆且精确线性重构的编码反例。两个不同raw-MAE Bayes acts在原生MAE+cos loss下合为同一最优摘要；.1原生VICReg亦纳入严格证书。128个原组件比较、4608个扰动和有理数独立验证通过。没有实际learned模型失败。

**Existing Methods：** surrogate/link compatibility、PSR未来特征、raw observation supervision、protected anchor、平方latent+重新适配读出均是强先例/普通控制；后两项精确修复本例。WorldTS本身已含预测潜态解码与prediction-only route，不能忽视。

**Proposed Idea：** 目前只保留task-compatible报告的可证伪救援路线；尚未形成区别于既有anchor/auxiliary loss的新机制。不会给普通修复取新架构名称。

**Falsification：** 若原生实际side info已恢复信息，或decoder refit、普通监督/权重/容量对照消除差距，该target机制动机撤销。固定人为反例成立不能越过此门槛。

**Feasibility：** 既有单5090，WorldTS/LatentTSF代码和NP/PJM旧checkpoints保留；本轮只做CPU精确诊断。源native执行版本、完整输入、输出、证书、SHA链和独立验证均保存。

**Next Experiment：** 若继续这一精确问题，先做bounded native残差归因，而非长训练新架构。优先核实一个此前未用于选方法的数据协议及其split；matched decoder refit与ordinary supervision envelope先于protected-anchor候选。服务器当前旧baseline数据只有NP/PJM、四个原checkpoints，Energy/ETTm1尚未在本轮核实为可直接用；不可写成已备齐或已运行。

方向A尚属needs-evidence，最终接受数0，整体研究目标仍未完成。此轮缩小了问题并排除错误动机，没有达到顶会研究方向的最终交付标准。
'''
for n,text in reports.items():(ROOT/n).write_text(text,encoding='utf-8')
state={'updated_utc':now,'goal_status':'ACTIVE_UNACHIEVED','accepted_directions':0,
       'previous_goal_turn':'PROGRESS','current_turn':'PROGRESS',
       'evidence':'fixed injective target plus native MAE/cos/VICReg report-compatibility witness; no learned-model failure',
       'scientific_result_sha256':sha(O/'result.json'),'verification_sha256':sha(O/'verification.json'),
       'new_training':False,'new_method':False,'live_jobs':0,'builder_sha256':sha(__file__)}
(ROOT/'CURRENT_STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
snapshot=ROOT/'outputs/previous_root_reports';snapshot.mkdir(exist_ok=True)
for name in ['CURRENT_RESEARCH_STATUS.md']+list(reports):
    path=BASE/name;old=path.read_text(encoding='utf-8') if path.exists() else ''
    backup=snapshot/name
    if not backup.exists():backup.write_text(old,encoding='utf-8')
    begin='<!-- cycle32-latest-start -->';end='<!-- cycle32-latest-end -->'
    if begin in old:
        a=old.index(begin);b=old.index(end,a)+len(end);old=(old[:a]+old[b:]).lstrip()
    prefix=f'''{begin}
# 当前 cycle32：潜态目标与任务答案兼容性

UTC {now}。已执行固定可逆target的原生loss组件数学诊断与独立复核；不等于实际WorldTS训练失败。普通修复已解本例，仍需原生残差证据。最终方向0，目标ACTIVE_UNACHIEVED。

见[实测](cycle32/experimental_results.md)、[方向判断](cycle32/final_research_direction.md)及[范围约束](cycle31/SCOPE_CORRECTION_20261017.md)。以下历史中的“当前”仅指当时，各轮原产物保留。
{end}

'''
    path.write_text(prefix+old,encoding='utf-8')
print(json.dumps(state,ensure_ascii=False),flush=True)
