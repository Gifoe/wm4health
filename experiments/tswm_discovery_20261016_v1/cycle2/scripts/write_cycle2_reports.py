"""Build the seven live research ledgers from completed, saved evidence only."""
import datetime,hashlib,json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def evaluations(folder):return [load(p) for p in sorted((OUT/folder).glob('*_evaluation.json'))]
def training(folder):return [load(p) for p in sorted((OUT/folder).glob('*_seed*.json')) if not p.name.endswith('_evaluation.json')]
def table(results,methods,points):
 lines=['|条件|方法|已完成初始化种子|MSE|Gaussian NLL|95%覆盖率|预测尾部概率|真实/噪声条件期望尾部概率|','|---|---|---:|---:|---:|---:|---:|---:|']
 for kind,L,H,subset in points:
  for method in methods:
   rows=[r for result in results if result['dataset']==kind for r in result['metrics'] if r['method']==method and r['prefix']==L and r['horizon']==H and r['subset']==subset]
   if not rows:continue
   fmt=lambda key:f"{statistics.mean(r[key] for r in rows):.5f}"
   lines.append('|'+f'{kind}; L={L}, H={H}, {subset}|{method}|{len(rows)}|'+ '|'.join(fmt(k) for k in ['mse','moment_gaussian_nll','moment_gaussian_95_coverage','mean_predicted_abs_state_gt2','mean_realized_or_noise_expected_abs_state_gt2'])+'|')
 return '\n'.join(lines)
def main():
 initial=evaluations('native_alarm');extended=evaluations('native_alarm_extended');generative=evaluations('generative_alarm');t0=training('native_alarm');t1=training('native_alarm_extended');tg=training('generative_alarm');tr=training('rpm_exact');er=evaluations('rpm_exact')
 tail=load(OUT/'rpm_tail_probe.json');compat=load(OUT/'native_compatible_rpm_probe.json');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 common=f'更新时间：{now}。这是第二轮工作记录；上一轮目录及结论未改写。研究总目标仍未完成。所有数值只从已保存结果读取。\n\n'
 points=[('state_dependent_alarm',1,0,'alarm_no_anchor'),('independent_alarm_control',1,0,'alarm_no_anchor'),('state_dependent_alarm',16,1,'alarm_no_anchor'),('state_dependent_alarm',16,8,'alarm_no_anchor')]
 (ROOT/'experimental_results.md').write_text('# 第二轮实验事实\n\n'+common+f'''已保存：1000步原生训练 {len(t0)}/6，评测 {len(initial)}/6；10000步训练 {len(t1)}/6，评测 {len(extended)}/6；普通生成式SSM训练 {len(tg)}/6，评测 {len(generative)}/6。

## 任务与比较边界

这些是新的受控机制实验，不是原论文排行榜复现。状态为已知正确的 AR(1)，rho=.95，平稳方差1；训练4096条、验证512条、测试4096条，各16步。20%的时刻有带噪线性状态测量，其余测量缺失；另有依赖状态平方的稀有二元告警。独立告警对照与状态无关。三颗种子仅为训练初始化重复，使用相同的数据集，不能当作三套独立数据。

RP-GSSM使用原生MLP、目标、训练器和后验组件；正确prior以零学习率固定。潜状态真值未进入训练目标或checkpoint选择。训练集前1024条只在模型冻结后拟合线性物理状态读出。所有模型同一划分。

## 初始1000步结果（均值）

'''+table(initial,['native','exact_same_readout','known_observer_oracle'],points)+'''

## 统一10000步预算对照（完成多少写多少）

'''+table(extended,['native','exact_same_readout','known_observer_oracle'],points)+'''

## 普通非线性观测SSM（已有方法强基线）

'''+table(generative,['generative_ssm'],points)+f'''

生成式基线学习线性Gaussian传感器、全局观测噪声和普通MLP Bernoulli告警似然，使用常规数值边际似然训练。没有提供真实告警函数或潜状态标签。它有结构化传感器似然假设，不能宣称与原生RP具有完全相同的归纳偏置；匹配的是数据、更新数、batch大小与大致网络规模，未匹配CPU/GPU壁钟时间。

## 固定概率模型审计

通用Gaussian辅助界：{len(tail['records'])}个固定参数cell，精确积分与160/320阶Gauss-Hermite Gaussian VI。原生正精度site兼容反例：{len(compat['records'])}个cell。原始JSON包含全部正/负结果。V=1.5的低于容量上界对照接近零gap；当common和rare recognition完全相同时，真实posterior等于prior，而固定的收缩site仍会给出过窄posterior。这说明固定反例不能替代学习后的证据。

## 指标和数值限制

NLL与95%区间对数值后验和oracle使用其矩匹配Gaussian；不是精确分位数区间。尾部概率对数值后验保留完整grid分布。未来指标在给定真实当前状态后，对新的转移噪声解析平均，没有把未知未来噪声交给模型。

原生数值smoother会在方差上额外加1e-5；组件容量检查扣除了它。401点[-8,8]对照801点[-10,10]，原生冻结模型的固定64个样本均检查均值/方差误差小于1e-4。生成式基线另按5e-4检查。通用概率模型/辅助界5项独立检查与生成式似然/梯度2项检查在服务器通过。

## 经验reference的限定

原生冻结模型的F只由训练集组成。对未见连续观测，称为“训练混合reference下的plug-in ratio posterior”；原子经验p0对未见观测本身赋零质量，不能未经说明就称为一个连续观测生成模型的精确posterior。给定ratio时数值归一化是精确的，reference估计与模型错设仍可能有误差。较大batch/reference、容量、其他数据种子尚待控制。
''',encoding='utf-8')
 (ROOT/'literature_review.md').write_text('# 第二轮文献与创新边界\n\n'+common+'''重点核查了与当前机制直接相邻的原始论文，而非用“没有搜索到”证明首创。

|原始来源|核查的机制|对创新主张的约束|
|---|---|---|
|[Walker et al., AISTATS2023](https://proceedings.mlr.press/v206/walker23a.html)|Eq.8–9给出内部变分界及其额外KL；还比较了普通重参数化VI与二阶近似|近似gap、Gaussian VI和这些修复路径均已有，不能重新命名为创新|
|[RP-LDS](https://discovery.ucl.ac.uk/id/eprint/10198967/1/RP_LDS_camera_ready.pdf)|早期Gaussian recognition动态模型|广义Gaussian recognition与当前正site实现应分开|
|[RP-GSSM, ICML2026最终版](https://proceedings.mlr.press/v306/hromadka26a.html)|Sec3.1/C：full recognition = prior×Gaussian site，g=q/prior；Sec3.4明确承认近似可能偏置解；F.1用矩/MMD和矩匹配Fisher距离检查平均F接近prior|不能声称作者未讨论近似；潜在新点只能是具体条件性失败及其后果，不能是“Gaussian posterior会有误差”|
|[RAMP, ProbML2026](https://proceedings.mlr.press/v327/fox26a.html)|A.4同样使用内部界和prior替代；B.4用Cholesky precision生成有效Gaussian消息|更一般的非线性message passing已经存在；Gaussian有效site对不确定性扩张的约束仍须单独检验|
|[Discriminative Kalman Filter](https://burkh4rt.github.io/pubs/Burkhart-et-al-2020-Neural-Comput.pdf)|判别式后验与prior密度比更新；需要相应监督/分布条件|负相对精度、Gaussian density-ratio校正本身不是新方法|
|[Probabilistic contrastive learning, ICML2023](https://arxiv.org/abs/2302.02865)|已有条件下的aleatoric posterior恢复理论|概率embedding/恢复不确定性不是新研究目标|
|[UWM-JEPA](https://arxiv.org/abs/2605.25313)、[MDN-VJEPA](https://arxiv.org/abs/2610.13956)|已有belief/多模态latent未来|不能把一般“保留不确定性”作为创新|

新近邻 **Identifying Latent Neural Dynamics with Recognition-Parameterized Gaussian Process Dynamical Systems** 已出现在[作者2026-09-29论文列表](https://www.gatsby.ucl.ac.uk/~maneesh/researchli1.html)，列为NeurIPS2026。已确认题名与作者，正文当前尚未取得；不能声称已核查或已排除此近邻。OpenReview浏览出现challenge，没有绕过。当前不提出“首次非线性RP dynamics”之类主张。

完整4篇PDF、提取文本、SHA256与下载来源在outputs/literature/。最终版RP-GSSM与2025 arXiv的标题、理论与页数不同，当前判断依赖2026最终版。所有方法的理论、实际代码、我们的新实验分别记录。

DoTime的新审计直接调用固定版本方向分数、PFN batching、VAR基线。前者有prospective mask，VAR使用更长factual输入；二者信息集不能混为同一部署协议。方向分数受未知future factual噪声与共同校准偏差影响；但反向响应高分在正负剂量平衡后消失。这个协议现象不能单独构成新方法论文。
''',encoding='utf-8')
 (ROOT/'baseline_registry.md').write_text('# 第二轮底座与复现注册\n\n'+common+f'''|底座/实验|版本/配置|状态与身份|
|---|---|---|
|原先WorldTS/LatentTSF等|见上一轮source_registry与旧项目审计|已审计历史结果；本轮没有重复训练|
|PRO-DYN|733d986d349e9e71ded5d9cdeeeed8d3323d83b4|阅读实际DYN projection实现；未新复现，不能称为已训练world simulator|
|DoTime|8f227406a5b17c2040cfdcc680b493729ef47d00|实际调用metric/batching/VAR；新的受控AR1诊断，不是官方suite成绩|
|RP-GSSM|a75225d5edceb972bb872dc9a335608733567dd4|原生正精度映射与smoother已运行；1000步 {len(t0)}/6、10000步 {len(t1)}/6 个新任务训练完成|
|冻结RP + retained-F grid|训练集4096条reference；401/801 grid|初始评测 {len(initial)}/6、长训练评测 {len(extended)}/6；已有数值推断对照，不是新算法|
|普通非线性观测SSM ML|三颗种子×两个数据条件；相同10000更新预算|训练 {len(tg)}/6、评测 {len(generative)}/6；新控制任务迁移，非原论文复现|
|正确观测似然Bayes filter|真实AR/传感器/告警方程|有特权生成机制的归因oracle，不能与未知观测模型放在公平学习排行榜中|
|通用Gaussian VI|固定normalized RPM；160/320阶GH|16个已完成cell；这是标准已有方法|

原生运行使用本轮独立.venv_rpm，JAX/JAXLIB0.4.35 CPU、Flax0.10.2、Optax0.2.4、Dynamax0.1.5、NumPy1.26.4；具体源文件hash见native_rp_precision.json。GPU生成式基线使用既有只读PyTorch运行时；结果保存设备、Torch版本与checkpoint hash。旧环境与旧实验目录未修改。

原生学习固定正确prior是本机制实验的设计选择，不能当成原论文完整训练协议。所有训练参数、数据生成种子、划分、选择准则写在outputs/*plan.json。物理潜状态标签只用于冻结后的诊断读出与指标。没有测试集超参数搜索。
''',encoding='utf-8')
 (ROOT/'negative_evidence.md').write_text('# 第二轮反证与停线路径\n\n'+common+'''1. 上一轮专用反事实区间原型已被未修改CROWN在相同不确定集合上超过；该新方法路线停止，不能借更换术语继续推进。
2. DoTime中反向响应“高方向分数”依赖单向干预。平衡正负剂量后消失，否定了不加条件的普遍gaming论点。普通双臂contrast/适当oracle已能修复解释，未发现必须新架构的需求。
3. 原作者早在2023年就给出辅助界的额外KL，2026最终版明确列为限制。不能把“exact surrogate不等于true posterior”本身宣称新颖。
4. 固定正site反例中连无信息观察都可能让surrogate收缩。这是刻意挑选的参数，不代表训练必然会选择它；所以追加了原生学习与独立告警对照。
5. 初始学习结果：在prefix=1告警且无anchor时，保留F的数值推断几乎没有修复低覆盖率。否定“主因只是冻结模型推断丢掉F”的强解释；表示/目标或site容量仍是竞争解释。
6. 1000步验证损失仍下降，不能据此排除训练不足。因此在查看测试结果前，固定两种条件、所有种子统一10000步。延长预算结果必须完整保留。
7. 广义Gaussian辅助界的小TV反例需要某些full recognition方差大于prior，超出当前native正site类。不能把它直接当作native模型内反例。另建了所有full recognition方差<=prior的独立反例，后者不声称小TV。
8. Gaussian VI在固定反例上已显著减轻误差；它是已有方案。普通生成式SSM也必须作强对照。若后者解决问题，“现有world models无法解决”的宽泛缺口立即淘汰。
9. 只有一套新受控数据生成种子，没有真实观测数据验证；不能声称跨数据集/领域泛化。训练初始化重复不等于独立数据重复。
10. 潜坐标、额外维度、非线性物理读出可能绕过固定坐标下的协方差上界。当前不提出所有world model的普遍不可能性定理。
''',encoding='utf-8')
 (ROOT/'hypothesis_space.md').write_text('# 第二轮竞争假设\n\n'+common+'''|解释|可证伪实验|目前判断|
|---|---|---|
|H1：普通Gaussian家族本身无法表达rare posterior|固定严格log-concave单峰posterior，对比最优Gaussian VI与最优Gaussian辅助界|V=3稀有极限下普通VI的KL趋零而辅助界gap不消失，H1不足以解释固定构造|
|H2：全局混合F近似prior即可保证所有条件的推断准确|F=(1-eps)N1+eps*NV，eps下降，完整计算条件误差|总变差<=eps而rare条件辅助gap持续存在；只反驳无尾部/稀有度条件的uniform保证，不反驳作者的全部定理|
|H3：冻结训练模型的主要失败来自忽略F|同一checkpoint/坐标/训练reference，原生对比数值ratio posterior|prefix1的低coverage几乎未修复；初始证据不支持主要归因于该项|
|H4：学习不足|所有组统一10倍更新预算，只按验证目标选择|长训练已启动/按实际文件更新；不能预判结果|
|H5：观察的正site参数化无法表达条件方差扩张|固定正确prior的原生组件、原生类内反例；再检查训练读出与oracle条件方差|数学容量约束已确认；物理坐标/额外维度/非线性读出的限定必须保留|
|H6：常规方法已经足够|不提供latent标签的普通生成式SSM，未知告警MLP，匹配更新预算|强对照正在执行；若修复则淘汰宽泛“新方法必需”命题|
|H7：稀有条件其实不重要/统计不稳|独立数据重复，低风险/高风险决策成本、适当tail proper score及CI|尚未做完，不能用单个覆盖率替代任务价值|

下一轮判据：先读取全部10000步和普通SSM结果，做按样本配对及初始化重抽样区间，再决定是否值得增加数据种子/维度。禁止看到哪一项测试改善就改primary。当前primary预先定义为prefix16,horizon1,alarm且无anchor；prefix1是预先登记的诊断条件，不能换名为原primary。
''',encoding='utf-8')
 (ROOT/'research_ideas.md').write_text('# 候选问题与可行性\n\n'+common+'''仅保留一个需要进一步核验的候选，避免凑足三个方向。

**条件性不确定性与概率latent消息的收缩限制。** 一个传感事件可能提示状态更极端、更不确定；有效Gaussian消息的正精度却只会压缩固定prior。研究问题是：平均marginal匹配与较好均值预测，是否掩盖了罕见但可观察条件下的系统性风险低估？如何在不依赖高维观测重建的条件下保留这类信息？

已有证据包括固定概率模型的可计算反例、Gaussian家族/辅助界分离、原生组件与新任务训练初步结果。尚缺充分证据包括：长训练归因、额外潜维度与合适观测模型对照、独立数据、真实任务价值，以及与最新RP-GPDS等近邻的创新边界。

最小修复候选是允许相对prior的负精度证据并保留proper likelihood/尾部可积性。但这与DKF/EP/Gaussian VI直接相邻，**目前不能作为新机制或新方法命名**。只有发现标准方法留下的具体效率、可辨识性或校准矛盾，并通过对照排除，才进入方法设计。

计算可行：一维受控问题的原生训练在CPU执行，普通数值SSM训练用单张5090；当前无需大checkpoint或新大数据集。是否能扩展到高维decoder-free动态，是待验证问题。
''',encoding='utf-8')
 (ROOT/'final_research_direction.md').write_text('# 当前最终方向判断：尚未定案\n\n'+common+'''**Research Question：** 重建自由的概率潜状态模型在可观察的罕见条件下，是否会因Gaussian消息/学习目标而漏掉状态不确定性的扩张，从而在均值预测看似正常时低估未来尾部风险？

**Evidence：** 固定Gaussian辅助界反例、原生正site容量检查、新AR(1)+缺失测量+状态相关告警实验。初始三初始化模型在单次无anchor告警时名义95%区间约82%–84%覆盖，独立告警对照约93%；但原先primary是prefix16/horizon1，不能用此更显著诊断替换primary。全部数值与预算对照见experimental_results.md。

**Existing Methods：** 普通Gaussian VI已缓解固定反例；DKF/EP、probabilistic contrastive learning和非线性生成式SSM已有相关机制。2026 RP-GSSM最终版承认近似偏置，RAMP也已有更一般的消息传递。不能宣称现有方法尚未解决一般不确定性建模。

**Proposed Idea：** 暂不接受新架构。把“正site容量”“辅助界可积性”“marginal校准不能保证rare条件校准”拆开检验。负精度/数值ratio修正是已有对照，不能重新包装。

**Falsification：** 统一长训练、增加潜维度/非线性读出或普通已有SSM若修复效果，则淘汰相应宽泛假设；如剩余问题只是已知Gaussian VI偏差或标准模型选择，终止该Idea。

**Feasibility：** 当前服务器可执行，数据/配置/源版本/训练日志/模型hash已按文件记录。当前实验不足以断言真实数据可迁移或顶会价值。

**Next Experiment：** 完成10000步预算与普通SSM对照，读取原primary的配对置信区间；若问题仍成立，再用独立数据与潜维度对照分开验证。最新RP-GPDS正文尚未取得，是创新判断未闭合的一项。

**决策：KEEP_AS_UNVALIDATED_CANDIDATE。** 尚无达到用户最终成功标准的新颖研究方向；不强行把已知解法或局部反例提升成论文贡献。当前最值得继续的是这一个窄问题，但研究总目标保持未完成。
''',encoding='utf-8')
 if len(generative)==6:
  compared=load(OUT/'alarm_paired_comparisons.json')['records']
  def cited_difference(comparison,metric):
   row=next(r for r in compared if r['dataset']=='state_dependent_alarm' and r['prefix']==16 and r['horizon']==1 and r['comparison']==comparison+' minus native_10000' and r['metric']==metric)
   lo,hi=row['paired_hierarchical_bootstrap_95_interval'];return f"{row['mean_difference']:.5f}，95%重抽样区间[{lo:.5f},{hi:.5f}]"
  latest=f'''

## 最新强对照结论

普通生成式SSM的三颗初始化×两种告警条件已全部完成。原先登记的prefix16/horizon1/告警且无anchor条件下，其相对10000步native模型的Gaussian NLL差为{cited_difference('ordinary_generative_SSM','nll')}；尾部Brier差为{cited_difference('ordinary_generative_SSM','tail_brier')}。测试病例配对重抽样；分别训练的方法独立重抽样初始化，同一冻结checkpoint的修正才配对初始化。这不是独立数据重复或多重比较校正。

因此否定“现有world models普遍不能处理此现象”的宽泛缺口。冻结ratio校正只产生小幅改善，不能据此排除训练期间的变分偏置。下一项强对照是已有RPM模型的数值likelihood重新训练，分别保留正site类与允许full Gaussian recognition；不是新方法。
'''
  for name in ['experimental_results.md','negative_evidence.md','research_ideas.md','hypothesis_space.md']:
   path=ROOT/name;path.write_text(path.read_text(encoding='utf-8')+latest,encoding='utf-8')
  final='''# 当前方向判断：宽泛命题已被已有方法反证

'''+common+'''**Research Question：** 原生RP模型的罕见条件风险低估，究竟来自其训练时Gaussian surrogate/辅助界，还是来自Gaussian recognition概率模型本身？已有decoder-free RPM的数值似然学习能否直接修复？

**Evidence：** 原生1000与10000步共12次训练、普通生成式SSM共6次训练及各自三初始化完整评测已结束。固定模型反例、归一化推断、独立告警负对照也已保存。原登记主条件的普通SSM tail Brier相对native改善约.0226，样本/初始化配对区间不含零；单次告警诊断的coverage缺陷则被普通SSM修复到接近oracle。完整指标与适用条件见experimental_results.md。

**Existing Methods：** 不能声称现有方法仍然普遍不足。普通生成式SSM已解决当前受控问题；Gaussian VI、DKF、EP、原始RPM数值推断是必须排除的已有修复。最新RP-GPDS正文尚未取得，创新判断也未闭合。

**Proposed Idea：** 尚不接受新架构。当前最有信息量的是用已有RPM数值似然重新训练，区分训练目标偏置与proper概率模型容量。其实现与比较不作为新方法成果。

**Falsification：** 若已有decoder-free RPM重新训练即可修复，则连“必须依赖观测重建才能解决”的较窄命题也淘汰；若不能，继续排除有限batch reference、优化、维度与坐标因素。不得把某个既有变分近似失效直接命名成创新。

**Feasibility：** 单张5090已完成生成式强基线；原生模型在CPU执行。所有新实验在独立cycle2目录，保留配置、版本、数据种子、原始日志、模型与输出hash。新RPM数值训练已启动，结果按完成状态逐项保存。

**Next Experiment：** 完成outputs/rpm_exact_training_plan.json中8个已冻结的模型/初始化cell；正site与full Gaussian各有三颗state-dependent种子，独立告警control各只有seed41。随后读取同一主条件及独立held-out数据验证，不作测试集超参数搜索。

**Final decision：BROAD_CLAIM_REJECTED_NARROW_QUESTION_UNVALIDATED。** 目前没有一个满足用户最终成功标准、可据此正式立项的新颖方向。最值得继续核验的是上述窄归因问题，不能把它写成已经成立的顶会创新。
'''
  (ROOT/'final_research_direction.md').write_text(final,encoding='utf-8')
  path=ROOT/'baseline_registry.md';path.write_text(path.read_text(encoding='utf-8')+f'\n\n已有RPM数值似然对照：训练已保存 {len(tr)}/8，评测 {len(er)}/8。配置见rpm_exact_training_plan.json；通用finite-grid forward实现已与joint归一化和有限差分梯度独立核对。该项不是新的概率模型。\n',encoding='utf-8')
  path=ROOT/'literature_review.md';path.write_text(path.read_text(encoding='utf-8')+'''

补充近邻：[Biased Dreams, 2026-04](https://arxiv.org/abs/2604.25416)分析latent RSSM ensemble的epistemic uncertainty与rollout attractor；当前受控现象使用正确的ID动力学，针对观测后aleatoric条件分布，不能把二者说成同一个实验或声称首次发现latent uncertainty误导。[VRKN/TMLR2022](https://arxiv.org/abs/2210.09256)与[真实代码](https://github.com/pbecker93/vrkn)已经研究RSSM推断引起的aleatoric偏差、missing observations与sensor fusion。本文及代码已检索，尚未执行其完整任务，不能列为已复现。通用不确定性失准和Kalman式修复不新。
''',encoding='utf-8')
 if len(er)==8:
  attribution='''

## 已有decoder-free RPM对照已闭合

数值likelihood重新训练的8个cell均已完成：state-dependent条件中正site/full Gaussian各3颗种子，independent告警中各seed41。它们沿用已有RPM概率模型/目标，训练不使用潜状态真值，也没有观测decoder。原native positive-site类被重新数值训练后，主条件coverage也已恢复；full Gaussian版的尾部概率/适当评分接近普通生成式SSM。**正site proper RPM概率模型的表达能力，不能与native Kalman surrogate后验容量混为一谈。**

'''+table(er,['positive_site_exact_ML','full_gaussian_exact_ML'],points)+'''

因此，本受控诊断不再支持“需要新decoder-free方法才能修复”的命题。冻结后验校正幅度小而数值likelihood重新训练改善大，支持训练过程中近似/目标偏置这一解释；但PyTorch重实现与Flax native初始化等差异仍在，不能把全部改变量严格归于单一代码行。通用Gaussian VI/数值RPM等已有路线应被承认。

这是一项有证据的机制审计，不是已成立的新颖研究方向。暂不为其追加新架构、更多同类调参或大规模训练。所有单次告警、负对照、长训练和不利结果完整保留。
'''
  for name in ['experimental_results.md','negative_evidence.md','research_ideas.md','hypothesis_space.md']:
   path=ROOT/name;path.write_text(path.read_text(encoding='utf-8')+attribution,encoding='utf-8')
  path=ROOT/'baseline_registry.md';path.write_text(path.read_text(encoding='utf-8')+'\n\n本轮共26个训练及其评测cell全部完成（含不同预算对照），不是26套独立数据。新增RPM控制没有未结束的正式训练。\n',encoding='utf-8')
  final='''# 最终方向记录：本轮诊断路线停止推进

'''+common+'''**Research Question：** 原生RP-GSSM在可观察的罕见条件下低估未来尾部风险，是否构成需要新机制修复的研究缺口？

**Evidence：** 26个训练/评测cell已完成：native 1000/10000步各6组，普通生成式SSM6组，已有RPM数值likelihood控制8组。固定反例、真实代码组件、负对照、相同权重数值推断、预算对照、原主条件、配对区间、9项数值/概率检查均已保存。原native缺陷存在，但重新训练已有RPM也能明显修复。

**Existing Methods：** 普通SSM和不依赖观测decoder的既有RPM均可处理当前受控任务。不能把这项诊断解释成“现有方法没有解决”。原作者2023/2026论文已讨论相关近似偏置，普通VI/数值likelihood也早已存在。

**Proposed Idea：** 不推荐基于这项失败提出新架构。固定Gaussian辅助界的罕见条件推导可保留为分析材料，但其创新性与泛化价值尚不足以完成用户的研究目标，不能据此宣称首创或顶会潜力已验证。

**Falsification：** “必须新方法/必须观测重建”的修复需求已被已有方法对照反证。更高维或真实任务是否还存在独立、未被已有方案解决的困难，本轮没有证明，不能作为自动挽救该路线的理由。

**Feasibility：** 所有计算在服务器完成，GPU训练使用单张5090，原生JAX在CPU运行；配置、数据种子、代码来源、checkpoint与输出hash保留。三颗训练初始化使用同一训练/测试数据；没有虚报独立数据重复或原论文排行榜复现。

**Next Experiment：** 切换到新的问题发现，先审查其他真实模型/benchmark的结构假设与最近相关工作，再按证据决定最小实验。暂不追加这条Gaussian-surrogate路线的调参或更大训练。若调查时间相关外生噪声与反事实可识别性，必须先排除SOBI/动态因果模型等已有结果，不能把时间相关性识别混合矩阵重新命名为创新。

**最终推荐：0个已验证方向。** 当前没有满足用户要求的明确新颖、可验证且值得顶会立项的方向。研究总目标保持未完成，继续发现，不用这个局部反例凑一个Idea。
'''
  (ROOT/'final_research_direction.md').write_text(final,encoding='utf-8')
 status={'updated_utc':now,'objective_achieved':False,'current_candidate':None if len(er)==8 else 'conditional uncertainty under contractive Gaussian messages','native_initial_training_completed':len(t0),'native_initial_evaluation_completed':len(initial),'native_extended_training_completed':len(t1),'native_extended_evaluation_completed':len(extended),'generative_training_completed':len(tg),'generative_evaluation_completed':len(generative),'existing_RPM_exact_training_completed':len(tr),'existing_RPM_exact_evaluation_completed':len(er),'decision':'DIAGNOSTIC_REPAIRED_BY_EXISTING_METHODS_NO_ACCEPTED_DIRECTION' if len(er)==8 else ('BROAD_CLAIM_REJECTED_NARROW_QUESTION_UNVALIDATED' if len(generative)==6 else 'KEEP_AS_UNVALIDATED_CANDIDATE'),'next':'New literature/code failure discovery outside this repaired surrogate route; exclude existing temporal source-separation and dynamic causal-identification results before experiments'}
 (OUT/'task_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(status,indent=2))
if __name__=='__main__':main()
