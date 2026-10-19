"""Recompute compact results and research ledgers from completed server outputs."""
import datetime,hashlib,json,pathlib,platform,sys
import numpy as np
import pandas as pd
import torch
sys.stdout.reconfigure(encoding='utf-8')
ROOT=pathlib.Path(__file__).resolve().parents[1]
def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def main():
 out=ROOT/'outputs';folder=out/'epicf_baselines';status=json.loads((folder/'status.json').read_text());assert status['state']=='complete' and status['completed']==13
 rows=json.loads((folder/'results.json').read_text());assert len(rows)==13
 groups={v:[r for r in rows if r['variant']==v] for v in ['official','no_policy','no_states','no_policy_no_states']}
 metrics=['factual_rmse_zscore','cf_rmse_zscore','effect_rmse_zscore','cf_nrmse_county_range_upstream','cate_rmse_upstream']
 summaries=[]
 for variant,items in groups.items():
  assert len(items)==3 and sorted(r['seed'] for r in items)==[41,42,43]
  summaries.append({'variant':variant,'n_training_initializations':3,**{m:{'mean':float(np.mean([r[m] for r in items])),'std_initialization':float(np.std([r[m] for r in items],ddof=1))} for m in metrics}})
 protocol=json.loads((out/'epicf_protocol_audit.json').read_text())
 ids=pd.read_csv(out/'single_trajectory_identity.csv',dtype={'county':str})
 unique=np.flatnonzero(~ids['county'].duplicated());assert len(unique)==153
 data=ROOT/'data'/'single_official'
 yf=pd.read_csv(data/'y_train.csv',header=None).to_numpy().reshape(158,168)
 ycf=pd.read_csv(data/'y_test.csv',header=None).to_numpy().reshape(158,168)
 predictions={}
 checks=[]
 for variant,items in groups.items():
  predictions[variant]=[]
  for row in items:
   run=folder/f'{variant}_seed{row["seed"]}'
   assert hashlib.sha256((run/'transformer.pt').read_bytes()).hexdigest()==row['checkpoint_sha256']
   pred=np.load(run/'predictions.npz');pf=pred['factual'];pcf=pred['counterfactual'];assert pf.shape==pcf.shape==(158,168) and np.isfinite(pcf).all()
   assert abs(float(np.sqrt(np.mean((pcf-ycf)**2)))-row['cf_rmse_zscore'])<1e-5
   if variant=='no_policy_no_states':assert np.array_equal(pf,pcf),'Zero-information scenario contrast must be exactly zero'
   predictions[variant].append({'cf':(pcf-ycf)**2,'effect':((pf-pcf)-(yf-ycf))**2})
   checks.append({'run':run.name,'checkpoint_sha256_verified':True,'finite_shape_verified':True,'stored_metric_recomputed':True})
 # Descriptive bootstrap: pairs are held-out labels/scenarios within a county;
 # separately trained variants resample their three initializations independently.
 # This conditions on the training dataset and is not a new-county generalization CI.
 rng=np.random.default_rng(20261016);comparisons=[]
 for candidate in ['no_policy','no_states','no_policy_no_states']:
  for key in ['cf','effect']:
   a=np.stack([p[key] for p in predictions[candidate]])[:,unique]
   b=np.stack([p[key] for p in predictions['official']])[:,unique]
   observed=float(np.sqrt(a.mean(axis=(1,2))).mean()-np.sqrt(b.mean(axis=(1,2))).mean())
   samples=[]
   for _ in range(3000):
    c=rng.integers(0,len(unique),len(unique));ai=rng.integers(0,3,3);bi=rng.integers(0,3,3)
    samples.append(float(np.sqrt(a[ai][:,c].mean(axis=(1,2))).mean()-np.sqrt(b[bi][:,c].mean(axis=(1,2))).mean()))
   comparisons.append({'candidate':candidate,'reference':'official','metric':key+'_rmse','unique_counties':153,'candidate_minus_reference':observed,'descriptive_percentile_interval_95':np.quantile(samples,[.025,.975]).tolist()})
 dump(out/'epicf_summary.json',{'groups':summaries,'comparisons':comparisons,'bootstrap_replicates':3000,'bootstrap_scope':'Exploratory descriptive conditional-on-this-trained-dataset interval; duplicate counties removed; no independent data repetitions or multiple-testing guarantee'})
 dump(out/'validation_checks.json',{'checks':checks,'unique_counties':153,'zero_policy_zero_states_scenario_contrast_exactly_zero':True,'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()},'device':torch.cuda.get_device_name(0)})
 table='| Inputs (same native Transformer) | Factual RMSE | CF RMSE | Scenario-effect RMSE |\n|---|---:|---:|---:|\n'
 for row in summaries:
  table+='| '+row['variant']+' | '+' | '.join(f'{row[m]["mean"]:.4f} ± {row[m]["std_initialization"]:.4f}' for m in metrics[:3])+' |\n'
 reconstruction=json.loads((out/'state_reconstruction_results.json').read_text())['results']
 sr=next(r for r in reconstruction if r['setting']=='single' and r['scenario']=='sch=0-occ=CF' and r['control']=='correct_state_difference')
 text={
 'experimental_results.md':f'''# 第三轮实验结果：EpiCF 信息协议审计

服务器已完成13个配置：官方KDE一次、官方Transformer三颗初始化、三组输入消融各三颗初始化。所有神经模型均使用未修改的FluTransformer：4,743,681参数、100 epoch、Adam、lr=0.001、batch=8。消融保持18维输入，以固定置零同时作用于训练和测试；不是只在测试时删除特征。种子41/42/43只改变训练初始化，同一数据集，不是三次独立数据生成。

{table}

表内单位是按官方factual均值/标准差缩放后的病例数，±是三颗训练初始化的样本标准差。CF RMSE为所有158轨迹×168天的全局RMSE；scenario-effect定义为预测factual减预测CF与真实factual减CF的差，不将单个模拟场景差称作真实人群CATE。

官方源码的per-county range指标也已逐次计算，见`outputs/epicf_baselines/results.json`。它与原论文表格的完整归一化/历史配置未逐项恢复，**本轮是发布代码执行，未宣称论文表2数值MATCHED**。官方KDE自预测误差接近0、CF RMSE为{rows[0]['cf_rmse_zscore']:.4f}；自预测不是泛化验证。

输入审计确认测试S/E/R/M来自目标反事实场景；整段无mask attention使用同期和未来值。no_policy模型无法响应仅改变行动的查询，却在发布场景协议下优于保留行动的模型，因为场景状态也发生改变。它不是可部署的行动模拟器。no_policy_no_states的预测factual与CF逐元素完全相等，是零场景响应负对照。

153个唯一县的去重描述性bootstrap见`outputs/epicf_summary.json`；独立重采样不同模型的初始化，不把158含重复块当作158个独立县。区间只描述当前训练数据和评测样本，不声称新县泛化或数据集层稳定性。

差分重建反证：仅用每县factual拟合一个比例系数，再由CF的负ΔS重建病例，CF RMSE={sr['rmse_cases']:.4f}例、县内range归一化RMSE={sr['nrmse_county_range']:.4f}。因此“病例被S差分精确编码”的强假设被否定。未凭源码公式抹去数据中的反证。

检查：12个checkpoint SHA、预测finite/shape、已存RMSE重算、零输入场景对比恒等性均通过。全部训练日志与权重保留服务器。
''',
 'baseline_registry.md':'''# 第三轮基线与数据登记

| 对象 | 精确版本 | 本轮执行范围 |
|---|---|---|
| EpiCF published baseline | 8f7df27507dc8b01422bb373417c8d0d3efffae2 | 完整轻量源码；官方single-policy KDE、Transformer原生训练函数；新增相同架构输入消融 |
| EpiCF simulator | 824ca2a9785038eaec4e277903856d796ac4adb3 | 52个小型代码/配置/说明文件，196,094 bytes；未运行完整县ABM，未下载大型网络文件 |
| EpiCF raw data | Zenodo20762823 v1.2，CC-BY-4.0 | single10,071,252 bytes，multi15,581,482 bytes，字典3,821 bytes；官方MD5与自存SHA256均核对；两种数据均运行官方preprocessing |
| Dataset identity | 158 trajectory blocks, 153 unique county IDs | 5县各有2个完全相同的块；保留官方样本用于发布代码执行，bootstrap去重，未来划分须group by county |
| PRO-DYN | 原root已有733d986d349e9e71ded5d9cdeeeed8d3323d83b4 | 本轮重读2026-05-02 v2论文和真实源码结构；未把阅读算作复现 |

官方Transformer的train/val/printed-test全部取同一train文件及全部训练索引；本轮原样复现并明确标注。权重选择没有使用CF标签，但不具备独立验证集。KDE带宽也按自预测误差选，不是LOO。

三个消融是自主设计的诊断，不能写为原论文实验。9个消融与3个官方神经模型使用同一模型容量、优化器、训练轮数和输入信息掩码规范。目标状态存在的组属于oracle场景条件任务，不能与history-only预测公平排行。

多策略原生Transformer默认158 units不能直接reshape474个CF序列，需要显式扩展推断；本轮只做预处理/输入审计，没有伪报多策略神经训练结果。
''',
 'negative_evidence.md':'''# 第三轮负面证据

1. 上一轮RPM校准路线已被普通生成式SSM和既有RPM似然训练修复，继续关闭，不追加调参。
2. EpiCF输入信息问题是发布协议的具体证据，不等于发现新模型机制。普通causal g-computation、CRN/CT/TE-CDE和历史条件模拟已经存在；禁止把删除oracle covariates重新命名为创新。
3. “目标病例可由目标S差分精确重建”未通过。保存全部数值和时间错位/跨县负对照；不把近似源码恒等式当作数据事实。具体聚合/导出流程尚未在公开材料中完整恢复。
4. 5个完全重复县轨迹降低了独立样本数；没有证据表明它单独解释神经模型现象，不能夸大为论文结果全部失效。
5. PRO-DYN明确定义DYN为历史到未来的learnable mapping，并承认Linear是relaxed time-delay dynamics。它没有承诺完整物理系统识别；不能用“并非真实动力学”作为稻草人反驳。
6. 新检索发现Semigroup-JEPA、One-Step Next-Latent Prediction Is Not a World Model和A Control Theory of Predictability in Latent World Models。只提出“多步一致性/均值复合不闭合/低预测误差不保证规划”已经有直接近邻。完整机制核查继续；不据摘要宣称其全部已解决。
7. 时间聚合导致因果频率恢复问题至少已有Gong等UAI2017的非高斯state-space辨识结果；不能把coarse logging ambiguity本身作首创。
''',
 'hypothesis_space.md':'''# 第三轮竞争假设与反证

| 假设 | 区分实验 | 当前判断 |
|---|---|---|
| EpiCF发布输入仅含干预前历史 | 逐行跟踪preprocessing→loader→model | 否定，测试covariates取目标场景整个168天的S/E/R/M |
| 模型必须读取行动才能在发布CF指标上较好 | 相同架构/预算，训练与测试同步屏蔽policy | 否定；三颗初始化no_policy的CF/场景差误差均更低 |
| 状态只是弱无关背景 | 同步屏蔽四状态，保持容量和预算 | 当前不支持；去状态误差明显上升，但没有独立训练数据重复，不外推所有任务 |
| 直接差分恒等式几乎完全泄漏目标 | factual-only比例拟合后应用CF负ΔS | 否定强版本，误差仍大；来源/导出精确对应尚不完整 |
| 发布数据有158个独立县 | 按county及全部列检查168天块 | 否定，153唯一县+5个完全重复块 |
| 仅靠协议修订已足够形成新方法论文 | 检查g-method、已存在benchmark与最新世界模型评测 | 未获得支持，当前不推荐立项 |

下一轮不继续对上述消融加训练预算。需要寻找修正信息集后仍存在、普通强基线不能修复、并且最近文献没有覆盖的具体失败。潜在主题仅为检索线索，不当作已验证问题。
''',
 'research_ideas.md':'''# 候选研究问题：尚无接受项

本轮保留一项有证据的评测材料：EpiCF发布条件预测分数不能单独证明行动模拟能力。它可作为未来完整研究的sanity control，但不是已经通过创新核查的方法Idea。

不推荐把以下主题直接立项：删除未来状态、增加行动输入、对DMS加循环、只加semigroup consistency、JEPA加入普通重建、一般性“预测不等于规划”。这些都需要与明确已有工作进一步区分，且多项中心命题已有直接文献。

下一项检索优先级：读完整Semigroup-JEPA及最新one-step诊断的模型/信息假设，检查存在的概率、多尺度、部分观测处理；再决定是否有一个可区分的高信息量实验。当前没有承诺使用Loop Transformer或大型foundation model。
''',
 'final_research_direction.md':'''# 当前研究判断：总目标未完成

**Research Question：** EpiCF发布协议的反事实分数，能否区分读取行动的动态模拟器与只依赖目标场景状态的条件预测器？

**Evidence：** 13个已执行配置、两份公开CSV/预处理、原生模型源码和三个初始化消融。行动盲模型保留目标S/E/R/M后得分更好；数据含5组完全重复县。差分精确重建假设已被数值反证。

**Existing Methods：** 历史条件模拟、联合协变量生成、causal g-computation及合理信息集划分已有大量方法。该发现首先要求纠正任务解释，尚未证明这些方法不足。

**Proposed Idea：** 暂不推荐新架构，也不把修订数据加载或输入遮罩当作创新。保存为可复现协议审计。

**Falsification：** “需要读取行动才能获得较低CF误差”已经被行动屏蔽实验否定。“S差分精确编码病例”被差分重建实验否定。更一般的历史条件模拟缺口尚未被本轮建立。

**Feasibility：** 数据共约25.7MB、native Transformer约4.74M参数，单张5090每100 epoch约20秒；无需大型GPU或重跑ABM支持当前审计。未证明足够资源完成新的尚未定义方法。

**Next Experiment：** 先核查Semigroup-JEPA、最新one-step/规划诊断的具体假设和开源实现，再从其未覆盖的条件选一个小实验。若强普通方法修复，应再次淘汰，不为凑方向扩展同类调参。

**最终推荐：0个已验证新方向。** 这不是总任务完成报告。当前有新的实验证据和明确排除项，但没有充分支持一个新颖且值得顶会立项的Research Gap。继续自主研究，保持原成果。
''',
 'literature_review.md':'''# 第三轮文献与创新边界

检索截至2026-10-16；优先最近六个月。这里只列对决策有作用的来源，阅读深度明确区分，不把source检索计作训练。

| Primary source | 已核对部分 | 对当前判断的作用 |
|---|---|---|
| [EpiCF-Bench, KDD2026](https://arxiv.org/html/2606.05692v1) | 任务/变量/数据生成/结果/公开全套轻量基线 | 原文说明模拟state为covariate；新执行进一步确认目标场景状态整段进入测试。干预序列文字部分不一致，实际源码为逐日binary flip，不自行改成constant干预 |
| [数据v1.2](https://zenodo.org/records/20762823) | record API、字典、三个文件校验 | 相比README链接v1.1，v1.2只改字典说明，signal MD5相同；公开许可证CC-BY4.0 |
| [PRO-DYN, ICML2026](https://arxiv.org/html/2507.15774v2) | 定义、模型分类、Linear与time-delay关系、方法边界、真实仓库 | learnable point-map不保证state identification，作者亦承认；不能自动把不同定义作为新缺口 |
| [Deployable Causal Action Geometry](https://arxiv.org/html/2610.16644v1) | problem setup、anchored response target、limitations、公开数据实验描述 | 时间漂移下action ranking/response geometry已有新近方法；它明确限于one-step和persistence/overlap假设。未发现可运行官方代码，未复现其数值 |
| [Feature Selection in World Models, author analysis](https://huguesva.github.io/blog/2026/world-models/) | 线性Gaussian假设、objective排名及PushT stress test描述 | JEPA保留稳定nuisance、predictability不等于task relevance已经被作者实际实验；blog不冒充同行评审，未复现其结果 |
| [Controlled World Model Identifiability](https://arxiv.org/html/2607.22430v1) | excitation/identification条件及counterfactual均值定义 | 不能把action excitation不足或conditional mean与unit CF差异重命名 |
| [Gong et al. UAI2017](https://proceedings.mlr.press/r15/gong17a.html) | primary abstract/识别假设 | 时间聚合因果辨识已有基础，非Gaussian noise和known aggregation factor是关键条件；2026重发不是新论文 |

继续核查（尚未作完整技术评价/数值复现）：[Semigroup-JEPA](https://arxiv.org/html/2609.10464v1)、[One-Step Next-Latent Prediction Is Not a World Model](https://arxiv.org/abs/2609.36227)、[A Control Theory of Predictability in Latent World Models](https://arxiv.org/abs/2607.10362)。这些已有与均值闭合、latent memory、on/off-manifold planning有关的近邻，不能忽略。

额外screen：Dynamic OT的跨环境map monotonicity需要矩阵反例核对；其固定prior下Brenier唯一性与prior不可识别性必须区分。SOBI已有利用各源不同autocovariance辨识Gaussian混合的经典结果。当前二者都没有被立为新方向。
'''}
 position_audit=out/'lenepa_position_null_audit.json'
 if position_audit.exists():
  checked=json.loads(position_audit.read_text());assert checked['decision']=='REJECT_UNRESTRICTED_POSITION_TABLE_AS_PROOF_OF_NATIVE_LENEPA_FAILURE'
  note='''\n\n## LeNEPA位置假设的源码/组件反证\n\n固定官方源码6574eb9e123305334e09e180de67682844f2e61f；下载92个代码/配置/论文源文件883,823 bytes。默认nepa_target_layer=0、target_uses_pos_embed=false、objective覆盖num_registers=0，并在[0,8]两层施加时间正则。未修改的原生Block/SIGReg组件在三颗种子下检查：八层RoPE block对恒定时序输入的最大时间差仅约2.4e-6至2.9e-6。任意Gaussian位置表可骗过抽象time-axis statistic，但不能直接替代此实现的layer0目标；该简单构造不构成native失败证据。没有执行完整LeNEPA训练，不声称证明该方法保留所有动力学信息。详见`outputs/lenepa_position_null_audit.json`。\n'''
  for name in ['experimental_results.md','baseline_registry.md','negative_evidence.md','hypothesis_space.md','research_ideas.md']:text[name]+=note
  text['final_research_direction.md']+='\n\n补充反证：本轮还以真实LeNEPA源码和3次原生组件检查，淘汰了“任意位置表足以证明native训练目标失败”的简单假设。研究问题须从模型实际允许的函数类和配置出发，不能只依赖抽象loss反例。下一轮不重复该位置表构造。\n'
  text['literature_review.md']+='''\n\n## 新近直接竞争者\n\n[MotionJEPA, 2026-09-20](https://arxiv.org/html/2609.23881v1)已经对slow-feature suppression、time/flattened SIGReg、stop-gradient和inverse dynamics进行数值对比；[TC-LeWM, 2026-07-29](https://arxiv.org/html/2607.26924v1)已通过temporally centered residual避免跨任务cluster收缩。它们的已有机制不能重命名。MotionJEPA是视频/control，不等同于所有time-series任务；TC-LeWM的主要下游是冻结encoder的behavior cloning，不冒充latent-space planning证据。\n\n[ChronoJEPA作者仓库](https://github.com/MrRobotop/ChronoJEPA)已有time-axis collapse、position/order和下游收益的分离实验。只核查公开说明，未执行其数值；无证据将其当作顶会同行评审。它仍构成必须考虑的直接公开近邻。\n\n[LeNEPA官方实现](https://github.com/langotime/lenepa-milets-2026)已固定并执行原生组件反证；其默认input-layer target和位置配置使简单的任意位置表反例不适用。不是只读标题就声称解决或失败。\n'''
 for name,content in text.items():(ROOT/name).write_text(content,encoding='utf-8')
 dump(out/'task_status.json',{'objective_achieved':False,'current_candidate':None,'cycle3':{'epicf_configuration_cells':13,'native_kde':1,'native_transformer':3,'new_input_ablation_transformer':9,'data_settings_audited':2,'unique_counties':153,'duplicate_blocks':5,'lenepa_native_component_seeds':3 if position_audit.exists() else 0,'scientific_jobs_complete':True},'decision':'PROTOCOL_FAILURE_DEMONSTRATED_NO_ACCEPTED_METHOD_GAP','next':'Mechanism-level literature/source audit outside repaired RPM route, EpiCF oracle-state task and inadmissible position-table null','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(table)
 print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
