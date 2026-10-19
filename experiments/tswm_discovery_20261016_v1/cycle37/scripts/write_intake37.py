"""Write transparent in-progress evidence reports from actual server state."""
import pathlib,json,datetime,hashlib,ctypes
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';now=datetime.datetime.now(datetime.timezone.utc).isoformat()
pre=json.loads((O/'preflight_v2/result.json').read_text(encoding='utf-8'))
metric=json.loads((O/'verifier_numeric_preflight_v1.json').read_text(encoding='utf-8'))
sup=json.loads((O/'supervisor_v2.json').read_text(encoding='utf-8'));state=json.loads((O/'learning_v2/status.json').read_text(encoding='utf-8'))
fits=list((O/'learning_v2').glob('*/*/fit.json'));neural=[p for p in fits if p.parent.name!='hgb']
assert pre['passed'] and metric['passed']
meta={n:json.loads((O/'preflight_v2'/f'{n}_data.json').read_text(encoding='utf-8')) for n in ['wind','solar']}
base=f'UTC {now}。C37为进行中的同信息真实预报任务学习压力筛查，非新方法、非原比赛复现。整体ACTIVE_UNACHIEVED，最终接受方向0。快照已完成neural fits={len(neural)}/12；最终结果尚未独立复核，以下不报告效果结论。'
scope='研究目标是可复用的时间序列状态学习、条件转移与未来演化框架。wind/solar仅作为条件输入和路径能力的证据载体，不把单站点提分作为成功标准；不强制action、物理识别、因果、共享零样本checkpoint、Loop或LLM。'
reports={}
reports['literature_review.md']=f'''# C37 文献与机制边界（训练中）

{base}

{scope}

本轮延续[C36文献综合](../cycle36/literature_review.md)，不重复宣布宽泛不确定输入/联合生成空白。WorldTS/Chronos-2已使用部分实际预报；DSSMF/MMFE、KITE、UKF-L、PBC与joint probabilistic models构成明确先例。当前仅进入其真实数据和公平普通对照的failure discovery。

[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)的向量场目标为平方回归。本轮固定KITE源码`3140ee824cbd80c5ec7fdf2b54666210519d0b39`，`layers/flow_loss.py`使用L1 velocity loss，且sample最终在内部平均paths。前者要求一个标准L2控制，后者通过原API num_samples=1批量repeat得到路径；并不由源码差异直接推论实际分布失败。执行资格验证原API平均与路径平均最大差{max(x['native_mean_vs_path_mean_max_abs'] or 0 for x in pre['results']):.12g}。

[Clark etal2004 Schaake shuffle，USGS原始记录](https://pubs.usgs.gov/publication/70027101)已用历史秩重排恢复时空相关。此处CAL-PIT/nearest analogue rank重排是普通控制；相同marginal样本multiset而joint path不同不构成新理论。source code/PDF而非二手摘要支持上述边界，固定协议见[LEARNING_PROTOCOL](LEARNING_PROTOCOL.md)。

[GEFCom组织者原文](https://www.pierrepinson.com/docs/Hongetal2016.pdf)§6.1/7.1给daily-midnight+24h的forecast合同。publisher原附件现已直接服务器下载并SHA固定。数据没有已识别的true future weather标签，目标轨迹残差不能被称为weather forecast error；本轮也不是逐vintage真实发布回放。

研究裁决待完整训练、独立raw-window/score/权重重放。若普通校准、rank coupling、标准目标或数值积分即可解释差异，关闭对应新机制主张。已知组件可有有限预算新组合价值，但须证明实证差异，不要求所有经典方法无限计算仍无解。
'''
reports['baseline_registry.md']=f'''# C37 baseline与来源（训练中）

{base}

| 对象 | 本轮协议 | 当前执行性质 |
|---|---|---|
| quantile MLP | 19quantiles，96×96 hidden；wind254088/solar401544参数 | 自建强直接控制，不称论文复现；相同全部history/forecast/calendar |
| KITE L1 | pin3140ee8，原train/inference/layers；64dim/2depth/4head/8rank；wind361477/solar364037参数 | 原组件迁移；缩小模型、daily loader与新probability选模不是原论文排行榜设置 |
| KITE L2 | 同initial hash/schedule/RNG，唯一velocity L1→MSE替换 | 标准目标对照，不称新方法；旧源码保留 |
| CAL-PIT rank /32-nearest analogue rank | 同一quantile fit、每lead完全相同128个值，仅改变排列 | 已知copula/Schaake类控制，额外fits0 |
| 逐lead逐q加性CAL | 所有neural模型同样可用，raw/calibrated并报 | 普通校准解释；不让单一模型独占CAL标签 |
| HGB+CAL residual bootstrap |200trees/max15leaves；joint/independent/analogue | 两个CPU fits，同全部输入，无HPO；不宣称严格参数/FLOPs匹配 |

12neural fits固定4000updates×batch64，两个seeds2026371/2026372、两个任务，FIT/CAL/VAL/TEST时间隔离及一origin embargo。VAL选模，TEST本轮不评分。末段改善/step0最佳必须标记，训练cap不等于已收敛。

18updates的6cell资格预检不计入12个formal fits。当前实际supervisor状态={sup['status']}，worker实际PID={state['worker_pid']}；进程重定向父PID不同是Windows venv正常现象，不能因此重启。独立postprocess_v1只等待原supervisor成功后运行verify_learning37.py，无另一个训练worker。
'''
counts='\n'.join('| '+n+' | '+' | '.join(str(meta[n]['counts'][k]['retained']) for k in ['fit','cal','val','test'])+' |' for n in meta)
reports['experimental_results.md']=f'''# C37 实际执行记录（训练中）

{base}

服务器直接正常TLS取得原publisher附件126360077 bytes，用时2.375秒；SHA256 `d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154`。原ZIP/CRC/member chain/raw CSV完整保留，不extractall，不执行档案内脚本。只用Task15，不拼接Task1–14重复样本。

固定wind zone1=16800小时/700日，solar zone1=18984小时/791日。solar未来predictors文件含POWER，特征白名单明确排除；标签只来自原train15。原时间0..23，不发明UTC；按组织者声明合同分origin+1..+24h。wind标签11NaN保留，严格排除涉及缺失的整窗口，不插值。

| task | FIT | CAL | VAL | TEST(未评分) |
|---|---:|---:|---:|---:|
{counts}

wind VAL从105候选origin保留66，39个因缺失历史/目标排除；solar VAL118全部保留。结果将是完整case协议的结果，不能泛化为全部原时间段。

6cell执行预检通过，18个资格updates；原生mean API与path adapter差≤9.32875e-8。独立评分器9个random/equal/duplicates fixtures，最大差{metric['max_abs']:.12g}，未使用benchmark VAL/TEST。正式训练已启动并实际产出{len(neural)}个fit.json，训练/选模曲线保留；最终效果尚未核验，不从partial log写科学结论。

实际状态在outputs/learning_v2/status.json，supervisor_v2.json管理原worker，postprocess_v1.json等待原作业；不存在可授权的重复启动。原始V1/V2数据reader和V1preflight失败保留，V3数据/V2运行器只修正hour0、NaN与numpy.str_读取问题，科学学习协议未改。
'''
reports['negative_evidence.md']=f'''# C37 负证据与限制（训练中）

{base}

当前尚无新的已验证模型失败。C36 observer-only提案仍因目标等价关闭；本轮不是改名复活该方案。

数据修正：V1错假设1..24小时，原文件0..23；V2finite检查遇wind11NaN；V3保留缺失、固定完整窗口排除。首次preflight被pandas不接受numpy.str_拦截，版本化V2仅修正转换。原源码/partial目录保留，没有重启任何已开始的formal training。

solar predictors有POWER，严格排除；从它推断metadata版本或使用其未来POWER训练均无授权。本轮没有true future weather标签，不能计算C36的physical forcing residual/q(U|F)监督，也没有建立小情景K成本压力。

L1源码、sample平均接口、边际与联合路径差异本身均已有直接普通解释，不自动产生研究gap。Schaake/coupla方法是强普通控制。容量/训练loss/FLOPs未严格一致，若训练不足或普通修复解释差距，不归因为新状态机制。wind缺失导致VAL39/105排除，必须限制总体结论。

所有最终效果需独立raw-window、selected checkpoint、schedule、sample/score重放后使用。TEST未评分；不以多个VAL结果挑出一个方法再冒充独立确认。
'''
reports['hypothesis_space.md']=f'''# C37 竞争解释（训练中）

{base}

| 解释 | 固定区分实验 | 当前证据 |
|---|---|---|
| strong direct+已有rank coupling已足够 | 同marginal quantiles的IID/历史秩/analogue秩，和HGB residual paths | 正在运行，尚未科学裁决 |
| generative缺陷来自flow训练目标 | KITE L1/L2同初始化/schedule/RNG/预算 | 正在运行，不从L1源码直接下结论 |
| 只是marginal calibration误差 | 所有模型同CAL逐q校准，raw/calibrated并报 | 已冻结，最终待复核 |
| 只是数值积分误差 | 同selected weights，10/50steps固定敏感性 | 不反选checkpoint，待结果 |
| 更复杂state机制仍有价值 | 前述健康普通方法之后的稳定跨任务残差 | 尚未建立，不能由一个低分推出 |
| 训练/容量或missingness解释差距 | 全curves/cap标记、参数/时间、各split排除数 | 必须保留；没有强称全部健康收敛 |

抽样数128仅是分布评分MC预算，不把增加paths降低估计噪声误写成外部forcing积分质量改善。一天内24点不是24个独立任务，CI按origin日历7日block；两个随机seed和origin变异分开报告。
'''
reports['research_ideas.md']=f'''# C37 候选账本（训练中）

{base}

{scope}

没有新方法被批准。当前是在预先固定的两个真实条件预报任务上，检验普通概率预测/校准/联合路径控制之后还剩什么，避免从数据接入和实现偏差直接跳到新架构。

后续若发现稳定残差，候选必须给出状态/转移学习的额外作用，并与直接分位数+copula、现有完整条件生成器和适当的state-space方法同信息比较。KITE L2、Schaake shuffle、CAL校准或换积分步数本身都不是本项目创新。C36的target-sensitive forcing压缩仍是未批准条件问题；本数据缺weather truth，不能用目标残差冒充它的pressure证据。

若已有普通控制解释现象，就关闭该具体论据，转向更有价值的能力；不靠换zone、rank、seed或重新调TEST挽救。本轮还没有资格推荐具有顶会潜力的最终框架。
'''
reports['final_research_direction.md']=f'''# C37 最终方向尚未确定，训练仍在运行

{base}

**Research Question：** {scope} 本轮检验边际预测、普通校准和已有path coupling之后是否仍有稳定的条件轨迹建模残差。

**Evidence：** 原始公开ZIP直接服务器下载并SHA固定，两个zone1数据处理完成，6cell原组件和9个独立数值fixture资格通过。12neural fits已开始，当前完成{len(neural)}；效果未独立复核，尚不能作为gap证据。

**Existing Methods：** KITE原生与标准L2控制、直接quantiles、Schaake类rank coupling、普通CAL和HGB residual paths全部明确纳入。早期不确定输入/联合模型先例继续有效。

**Proposed Idea：** 无。未把普通修复改名为新world-model框架，也未由元数据问题推导创新。

**Falsification：** 普通coupling/calibration/训练目标/数值修正解释差距、训练不足/信息不平等/完整case选择主导结果，都不足以支持新状态机制。

**Feasibility：** 单5090实际执行中，约0.25–0.40M参数组件，两个真实任务/两个seed，原数据/权重/计划/日志均独立目录保留。不是零样本共享模型要求，也不是生产部署发行保证。

**Next Experiment：** 等待当前原worker及独立verifier完成，核查全部curves、原始样本和paired score；不能重启已有作业或先改方法争取分数。完成后才决定是否有新机制值得验证。整体研究目标未完成，接受方向0。
'''
reports['CURRENT_STATE.md']=f'''# C37 正在运行

{base}

supervisor status={sup['status']}，实际worker PID={state['worker_pid']}，最近phase={state.get('phase')}，data={state.get('dataset')}，model={state.get('model')}，step={state.get('step')}。CIM已核实这些进程存在。formal training唯一；postprocess只等待同一supervisor成功后独立验证。

所有输入、split、loss、预算、seed与评分在训练前冻结，TEST未评分。见LEARNING_PROTOCOL.md。旧结果保留，不能由本文件快照判断作业终止，须检查CIM/最新supervisor和status。
'''
for name,text in reports.items():
    p=R/name;assert not p.exists(),f'Preserve previous report {p}';p.write_text(text,encoding='utf-8')
header=f'''<!-- cycle37-live-start -->
# C37真实条件轨迹学习筛查正在运行

UTC {now}。上一轮C36=PROGRESS。目标仍为可复用时间序列world-model框架，接受方向0。原GEFCom wind/solar固定zone1，未来NWP依原benchmark合同；POWER输入已排除，wind缺失完整窗口规则固定。没有未来weather truth，不冒充forcing误差实验。

6cell组件资格与9个独立数值fixture通过。原supervisor35672/实际worker{state['worker_pid']}经CIM存在，12neural fits中{len(neural)}已完成、整体仍{sup['status']}；另2个HGB和独立verifier按原流程后续执行。TEST不评分，未由partial结果作科学结论。见[冻结协议](cycle37/LEARNING_PROTOCOL.md)、[实际记录](cycle37/experimental_results.md)。不要重复启动或改训练源码；下面旧“当前/下一步”均为各轮历史。
<!-- cycle37-live-end -->

'''
p=R.parent/'CURRENT_RESEARCH_STATUS.md';old=p.read_text(encoding='utf-8');assert '<!-- cycle37-live-start -->' not in old;p.write_text(header+old,encoding='utf-8')
for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:
    p=R.parent/name;old=p.read_text(encoding='utf-8');marker='<!-- cycle37-live-index-start -->';assert marker not in old
    text=f'''{marker}
# 最新过程：C37训练正在运行

UTC {now}。两真实预报任务×两个seed×三个neural arms固定协议执行中，另有普通rank/CAL/HGB控制；最终效果尚未独立复核，接受方向0，总目标ACTIVE_UNACHIEVED。见[本项C37报告](cycle37/{name})和[总状态](CURRENT_RESEARCH_STATUS.md)。下面内容为各轮历史。
<!-- cycle37-live-index-end -->

''';p.write_text(text+old,encoding='utf-8')
print(json.dumps({'reports':len(reports),'neural_fits_complete_snapshot':len(neural),'status':sup['status'],'goal':'ACTIVE_UNACHIEVED'}))
