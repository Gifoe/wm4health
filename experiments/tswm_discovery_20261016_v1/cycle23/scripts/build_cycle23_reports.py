"""Evidence-derived internal reports and standalone diagnostic figure."""
import hashlib,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'outputs/prefix_contract/results.json').read_text())
v=json.loads((ROOT/'outputs/prefix_contract/independent_verification.json').read_text());assert v['status']=='pass'
p=json.loads((ROOT/'outputs/native_context_preflight.json').read_text());assert p['status']=='pass' and len(p['models'])==24
d=json.loads((ROOT/'outputs/external/explore_context/parallel_attempt2/download_receipt.json').read_text());assert d['published_checksum_match']
table='| Simulator pipeline | Prefix cut | Median past-output change / original std | Systems above1e-8 |\n|---|---:|---:|---:|\n'
for row in r['summary']:
    table+=f"| {row['arm']} | {row['cut']} | {100*row['relative_y_rms_median']:.4f}% | {int(round(128*row['fraction_relative_above_1e8']))}/128 |\n"
reports={
'literature_review.md':'''# 文献与创新边界：系统身份与信念更新

本轮针对通用时间序列世界模型的跨系统推理能力筛选，不以单数据集预测提升作为研究目标。以下主要是源文献/代码接口核查，不声称全部完成论文复现。

1. [Probabilistic in-context system identification](https://arxiv.org/abs/2610.13411)，2026-10-09。已有跨系统共享模型、概率读出、长上下文和合成数据用途；不能将这些能力重新命名为创新。其源码已在cycle16/17缓存。本轮只新增生成器前缀配对审计，未重做原有协方差实验，也未重训该检查点。
2. [Martingale Posterior Neural Processes](https://arxiv.org/abs/2304.09431)，ICLR2023。利用预测生成数据表达函数不确定性；通用martingale posterior并非新想法。本轮阅读原文相关机制，尚未执行其代码。
3. [From Drift to Coherence](https://proceedings.mlr.press/v306/kim26ah.html)，ICML2026。已有预测重采样和一致性微调。仅把同类正则用到时间序列，不足以成立创新。
4. [On the Conditioning Consistency Gap in Conditional Neural Processes](https://arxiv.org/abs/2604.19312)，2026-04-21；arXiv及PDF标注TMLR2026，OpenReview页面本轮被浏览器验证挡住，录用状态未独立复核。文中比较factorized joint的条件化与重新编码，给出特定均值聚合条件下的界。这个量不能直接等同于对未来证据积分的tower误差，也不覆盖所有神经世界模型。
5. [Explore the Context](https://proceedings.mlr.press/v130/achterhold21a.html)，AISTATS2021，[公开实现](https://github.com/EmbodiedVision/explorethecontext)。已有共享上下文条件动力学、潜在系统分布和主动校准。源码保持同一latent draw跨时间、跨候选控制，故“现有模型都无法共享同一系统样本”已不成立。采用修正模型选择错误的分支f0baf6262523d9c5abaf8c0381b4db97818f7c34；34个源码/许可/配置接口文件Git blob校验。24个发布检查点完成受限加载及前向接口验证，尚无新校准质量或belief-update实验结果。

原文7份已在服务器保存，正常TLS校验及SHA256见outputs/literature/manifest_requests_attempt2.json。首轮urllib证书错误原样保留；Requests CA路径恢复后全部取得，未关闭证书验证。未采用博客、AI摘要或MDPI作为论据。旧方法足够强并不自动否定所有新的任务需求，但本轮没有发现足以推荐的新机制。
''',
'baseline_registry.md':f'''# Baseline与真实执行状态

| 对象 | 固定版本/产物 | 本轮执行 | 不可声称 |
|---|---|---|---|
| 原生WH生成器 | PuraFilippo/sysid_benchmarks {r['source_revision']} | 16次原生数组精确重放；128新系统/两分支切点/四管线 | 不是新模型训练，也未复现新论文排行榜 |
| causal / fixed-reference管线 | 同参数/同噪声/独立参考校准；PROTOCOL.md | 配对模拟，512个严格前缀检查 | 不代表原checkpoint在新管线上预测更好 |
| Explore the Context | 修复分支f0baf6262523d9c5abaf8c0381b4db97818f7c34；作者最终预算validation-best | 24个权重，严格state_dict；原生前向；空/非空上下文；控制前缀 | 尚无新belief-update、主动规划或论文最终回报复现 |
| MPNP / Drift-to-Coherence / CNP理论 | primary papers | 文献边界 | 未执行，不能宣称比它们更好 |

发布修正版archive={d['bytes']}bytes，SHA1={d['sha1']}，与作者发布值一致；SHA256={d['sha256']}。Archive在服务器，未下载到本机。仅提取24个final-budget最佳检查点，共57843216bytes；没有提取大型原训练信号。817个pickle候选member包含校准结果，不能将其全部计为模型权重。

主运行时仍Python3.10.20/PyTorch2.8/RTX5090。gym0.17.2纯Python依赖单独放在本cycle；未修改既有环境。可选torch_scatter只对本编码器实际使用的非负max聚合提供进程内适配；native tensor路径与ragged路径24/24完全一致。checkpoint仅允许必要NumPy数值标量全局类型，仍weights_only=True，未开启任意pickle执行。原生数值源码未编辑。完整配置、参数数目、权重hash和作者选中step见outputs/native_context_preflight.json。
''',
'experimental_results.md':f'''# Cycle23：实测与验证

128个全新WH系统；原生burn-in200、保留930；只替换cut400或744之后的iid N(0,1)输入；参数、历史原始输入和噪声不变。独立参考激励只用于fixed-reference管线，与审计分支无关。随机种子及假设先写入PROTOCOL.md，SHA256={r['protocol_sha256']}。

{table}

结果支持内部、非线性之前的全轨迹归一化依赖未来输入；关闭最终normalize不能修复此项。外部归一化还改变历史输入坐标。普通固定参考修复了本项模拟契约，但改变了生成协议；没有重训和性能收益结论。

16次未编辑WHDataset重放（8seed×normalize两值）float32输入输出最大误差0。独立NumPy递推核对10系统所有分支，最大输出差{v['independent_numpy_output_max_error']:.4g}；独立参考常数差{v['independent_calibration_max_error']:.4g}。128文件hash核对、{v['metrics_replayed']}指标重算差{v['metric_max_error']}、{v['strict_prefix_checks']}个因果管线严格前缀检查通过。主模拟{r['elapsed_seconds']:.3f}s，独立核查{v['elapsed_seconds']:.3f}s。原始参数、信号和噪声NPZ留在服务器。

另24个老强baseline权重在服务器完成受限加载、作者best标记及预算检查；参数545038/652364/653366，ragged与tensor聚合最大差0，改变后续控制对此前输出影响0。这只是数值/接口验证，不是质量、校准或新机制实验。

工程尝试全部保留：首个文件传输慢，确认所属PID后停止，原任务4294967295；8路HTTP Range复用既有前缀，302.1s完成并通过完整发布SHA1。首个checkpoint预检因未允许NumPy标量而失败；仅加入数值类型后新job通过。没有将这些错误算作科学失败。
''',
'negative_evidence.md':'''# 被否定/尚不能成立的解释

- “只关闭最后的normalize便能保证同一物理系统的分支一致性”：128/128内部-only仍改变历史输出，已否定。
- “观察到的前缀变化来自我们未重现原生代码”：16次原生数组精确一致，独立递推通过，已排除这项工程解释。
- “这是新世界模型方法”：不是。固定参考归一化是普通协议修复，不能包装为架构创新。
- “过去的概率校准结果同时证明了真实在线系统辨识”：不成立。cycle17已披露oracle全轨迹归一化；本轮进一步定位内部依赖。旧统计结果保留，其适用范围不扩大。
- “现有模型都在每步/每个候选动作重采系统身份”：Explore-the-Context已有共享latent采样，源码直接反证。
- “条件一致性/主动系统辨识完全没人做过”：MPNP、2026条件一致性研究、2021主动上下文动力学已构成直接边界。
- “强baseline的24个权重载入即等于复现了它的最终回报”：没有新质量评测，不能这样报告。

尚未否定或证实：上述强baseline重新编码证据后的物理预测是否与其自身生成联合律一致，以及任何偏差是否损害真实预测/信息决策。不要仅凭源码结构宣布失败。
''',
'hypothesis_space.md':'''# 竞争假设及下一项反证

| 假设 | 可区分实验 | 当前状态 |
|---|---|---|
| 只有最后坐标标准化引起前缀变化 | 内部-only vs完整native | 否定 |
| 系统内部使用分支未来统计 | 固定参数/输入前缀、后缀重采；独立参考控制 | 128系统支持 |
| 自定义重放/数值误差 | 原生WHDataset精确重放、NumPy递推 | 排除 |
| 强共享latent模型的证据编码已经足够一致 | 原生物理query moments及真实后验预测 | 尚未执行 |
| 原生重新编码与其隐含联合律偏离 | no-update / 普通likelihood reweighting / exact physical posterior | 尚未执行 |
| 即便偏离也不影响用途 | 真证据proper scores/信息动作，与MC数值误差分开 | 尚未执行 |

下一工作不是再训练一个弱CNP制造缺陷。先执行NEXT_EXECUTABLE_GATE.md中六个已发布toy模型的低成本核查；其context上限10、参数维度1允许可信数值积分。两个actuation variant共享同一plant，不能算跨多个真实系统成功。未知结果不能提前写入推荐。
''',
'research_ideas.md':'''# 候选研究问题：仍在淘汰阶段

本轮只保留一个需要验证的窄线索：通用时间序列世界模型的系统级不确定性，能否在多条未来分支和新证据更新之间保持同一物理含义，并同时改善真实预测或信息决策？

这不是新架构名称。共享系统latent、Bayesian filtering、Neural Process、martingale posterior和主动辨识均已有成熟研究。不能将这些拼接或重新命名当作创新。当前实验只证明一份可用生成器不适合直接测试该附加能力，并未证明模型领域尚无解决方案。

最小可行对照：现有decoder上普通似然重加权。它是已知推断方法，作用是排查是否只是使用了不一致的近似更新。若它足够且成本可接受，不能宣称必须有新框架。若后续出现跨系统、跨状态、长上下文下的明确成本或泛化缺口，才考虑新的可学习推断状态；届时须进一步核查NP/元学习/滤波的最近工作。

先前未知输入角色、动作响应目标、低维表示线索均保留其负面证据；本轮不恢复为推荐方向。未取得新独立证据前不在已查看测试集上继续loss调参。已接受最终方向数量：0。
''',
'final_research_direction.md':'''# 当前研究决定：未确定最终创新方向

**Research Question（待验证）：** 一个共享时间序列世界模型能否维持同一未知系统的物理预测分布，并在加入新证据后给出与其自身联合模拟律一致且有用的更新？这属于通用模型能力，不是某个数据集优化。

**Evidence：** 128个全新系统的分支实验确认当前WH生成协议会因未来输入变化改变过去输出；原生精确重放及独立核查通过。24个已有强模型检查点可运行。这些结果是实验基础资格证据，尚不是未解决模型缺口的证据。

**Existing Methods：** 强上下文动力学已支持共享latent与主动校准；NP条件一致性、martingale方法及普通贝叶斯重加权均是直接边界。现阶段没有证据证明它们不足。

**Proposed Idea：** 暂不提出新架构。将推断状态更新与decoder定义的物理似然绑定，只作为已知对照进行检验；未将其列为创新贡献。

**Falsification：** 原生强模型本已满足契约；所谓偏差只来自MC/状态初值/支持范围；普通对照已足够；或者一致性差异不影响真实预测与决策。任何一项成立，都不能据此推荐新框架。

**Feasibility：** 单RTX5090；24个约0.55–0.65M参数发布模型已验证。先用6个一维系统latent模型和原生物理真值；再决定是否进入16维latent/更复杂系统。修正版archive2.21GB和原始审计数据在服务器，本机为紧凑镜像。

**Next Experiment：** NEXT_EXECUTABLE_GATE.md中固定的native/no-update/likelihood-reweight/exact-physical四方核查。尚未执行，不报告未观测指标。若低维gate没有重要残余即停止本线索，不靠增加数据集维持它。

正式推荐方向：0。任务总体ACTIVE、UNACHIEVED。本轮完成的是前置反证与强底座可执行化；没有把协议修复或旧方法改名为最终成果。
'''}
for name,text in reports.items():(ROOT/name).write_text(text,encoding='utf-8')
OUT=ROOT/'outputs/analysis';OUT.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');labels=['Raw causal','Native internal only','Native complete','Fixed reference']
x=np.arange(4);colors=['#3265a8','#e69339']
for j,cut in enumerate([400,744]):
    values=[100*next(z for z in r['summary'] if z['arm']==arm and z['cut']==cut)['relative_y_rms_median'] for arm in ['raw_causal','native_internal_only','native_complete','fixed_reference']]
    bars=ax.bar(x+(j-.5)*.31,values,.29,color=colors[j],label=f'Change inputs after t={cut}')
    for rect,val in zip(bars,values):ax.text(rect.get_x()+rect.get_width()/2,val+.22,'exact 0' if val==0 else f'{val:.2f}',ha='center',fontsize=9,rotation=90 if val==0 else 0)
ax.set_xticks(x,labels);ax.set_ylim(0,16);ax.set_ylabel('Median past-output RMS change / original std (%)');ax.set_title('Future input changes the native generated past\n128 fresh systems; original parameters and noise held fixed',loc='left',fontsize=13)
ax.legend(frameon=False,loc='upper right');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.text(.01,-.065,'Contract audit only. Fixed reference changes the data protocol; no forecasting improvement or new-method claim.',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'prefix_contract.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
print(json.dumps(dict(reports=len(reports),plots=3,accepted_directions=0)))
