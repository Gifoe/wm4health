"""Close an ordinary screen without promoting it into a new-framework claim."""
from common40_v2 import *
import csv,datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=R/'outputs';S=O/'screen_v1';A=O/'analysis_v1';A.mkdir(exist_ok=False)
V=json.loads((O/'verification.json').read_text());assert V['status']=='PASS'
rows=json.loads((S/'results.json').read_text());I=json.loads((O/'intake.json').read_text());Q=json.loads((O/'qualification.json').read_text())
flat=[];arrays={}
for row in rows:
 for ev in row['evals']:
  z=np.load(S/row['key']/ev['raw']);orig=z['origins'];truth=z['truth']
  for idx,r in enumerate(REVEALS):
   sc=scores(z[f'r{r}'],truth)
   for j,e in enumerate(orig):
    flat.append({'data':row['data'],'seed':row['seed'],'family':row['family'],'steps':ev['steps'],'r':r,'origin':int(e),**{k:float(v[j]) for k,v in sc.items()}})
   for k,v in sc.items():arrays[(row['data'],row['seed'],row['family'],ev['steps'],r,k)]=(orig,v)
with (A/'all_origin_scores.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
table=[]
for name in ['ett','jena']:
 for family,steps in [('ar8mix',0)]+[('flownp',s) for s in [1,2,8,32]]:
  energies=[]
  for r in REVEALS:
   energies.append(float(np.mean([arrays[(name,sid,family,steps,r,'energy')][1].mean() for sid in [2026401,2026402]])))
  chosen=[row for row in rows if row['data']==name and row['family']==family]
  costs=[ev['single_origin_cost'] for row in chosen for ev in row['evals'] if ev['steps']==steps]
  costs=[np.mean([z.get('full_query_seconds',z.get('incremental_query_seconds')) for z in cost]) for cost in costs]
  table.append({'data':name,'family':family,'steps':steps,'r0_energy':energies[0],'r8_energy':energies[1],'r32_energy':energies[2],'mean_energy':float(np.mean(energies)),'mean_single_origin_query_seconds':float(np.mean(costs))})
with (A/'mean_quality_cost.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
# Temporal-block uncertainty for update effects, average the two seeds first.
effects=[]
for name in ['ett','jena']:
 for family,steps in [('ar8mix',0),('flownp',32)]:
  for r in [8,32]:
   origins0=arrays[(name,2026401,family,steps,0,'energy')][0]
   delta=np.mean([arrays[(name,sid,family,steps,r,'energy')][1]-arrays[(name,sid,family,steps,0,'energy')][1] for sid in [2026401,2026402]],axis=0)
   labels=origins0//168;blocks=np.unique(labels);rng=np.random.default_rng(4040);boot=[]
   groups=[delta[labels==b] for b in blocks]
   for _ in range(2000):
    chosen=[groups[i] for i in rng.integers(0,len(groups),len(groups))];boot.append(np.concatenate(chosen).mean())
   effects.append({'data':name,'family':family,'r':r,'mean_energy_change_from_r0':float(delta.mean()),'block_bootstrap90':[float(x) for x in np.quantile(boot,[.05,.95])],'temporal_blocks':len(blocks),'exploratory_not_confirmatory':True})
analysis={'table':table,'update_effects':effects,'fits':8,'updates':16000,'validation_origins':{name:len(np.load(S/f'{name}_2026401_ar8mix/paths_steps0.npz')['origins']) for name in ['ett','jena']},'selected_steps':{x['key']:x['selected_step'] for x in rows},'selected_at_cap':sum(x['selected_at_cap'] for x in rows),'last_validation_improves':sum(x['last_validation_improves_over_previous'] for x in rows),'test_scored':False,'accepted_directions':0,'goal':'ACTIVE_UNACHIEVED'}
(A/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
colors={'ar8mix':'#3366aa','flownp':'#cc7733'}
for ax,name in zip(axs,['ett','jena']):
 for family,steps in [('ar8mix',0),('flownp',32)]:
  t=next(t for t in table if t['data']==name and t['family']==family and t['steps']==steps)
  ax.plot(REVEALS,[t[f'r{r}_energy'] for r in REVEALS],marker='o',color=colors[family],label='GRU mixture' if family=='ar8mix' else 'Native FlowNP (32 steps)')
 ax.set(title=name.upper(),xlabel='Actual prefix observations revealed',ylabel='Fixed-tail fair energy / sqrt(32)');ax.set_xticks(REVEALS);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Ordinary baseline screen: two-seed exploratory VAL means',fontsize=11)
fig.savefig(A/'fixed_tail_quality.png',dpi=180);fig.savefig(A/'fixed_tail_quality.pdf');plt.close(fig)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
mdtable='| data | ordinary model | flow steps | r=0 | r=8 | r=32 | query s |\n|---|---|---:|---:|---:|---:|---:|\n'
for t in table:mdtable+=f"| {t['data']} | {t['family']} | {t['steps']} | {t['r0_energy']:.5f} | {t['r8_energy']:.5f} | {t['r32_energy']:.5f} | {t['mean_single_origin_query_seconds']:.6f} |\n"
framing='研究对象是可复用时间序列世界模型框架；数据集只作机制验证载体。本轮是普通方法筛查，不是提出了一个新框架。当前工作问题：同一学得状态能否同时支持未来推演和真实新观测下的修正。最终新机制仍未成立，接受最终方向0，整体ACTIVE_UNACHIEVED。'
limits=f"两数据、两seed、固定有限预算、VAL反复用于选择；TEST未评分。VAL origins={analysis['validation_origins']}；达到训练上限时选中 {analysis['selected_at_cap']}/8，最后一段VAL仍改善 {analysis['last_validation_improves']}/8。不能将训练不足、普通AR误差累积或原生采样器开销称为新机制缺口。SFM、适当state-space和ordinary consistency等强普通控制尚未完成。"
sources='[WorldTS 原文](https://arxiv.org/html/2609.31162v1)；[FlowNP 原文](https://arxiv.org/html/2512.23853v1)及固定源码 f3403b6479dad33a7b93ba6edcf93c53d40f8e65；[SFM 原文](https://arxiv.org/html/2602.05319v1)。完整边界沿用 [C39文献审计](../cycle39/literature_review.md)。'
reports={
 'literature_review.md':f'# C40 框架层次与创新边界\n\n{framing}\n\n{sources}\n\nWorldTS提供未来状态监督与covariate-conditioned dynamics的框架范例。它不要求我们把所有world model定义为物理可辨识或因果模型。观察修正接口与AR/SSM、FlowNP arbitrary conditioning、SFM warm-source更新有明显重叠；本轮只给共同合同下的普通模型证据，不宣称这些已有方法共同失败。PathFM/ForcingDAS等能力边界见C39，尚未作为本轮训练对照。\n\n下一机制必须说明状态监督与观测修正之间的具体交互，以及为何单独组件/普通matched repair无法解释收益。仅把已知接口连接起来没有足够创新证据。\n',
 'baseline_registry.md':f'# C40 实际代码与数据\n\n{framing}\n\n- GRU2x256 + 8 Gaussian mixture：{Q["preflight"]["ar8mix"]["parameters"]} parameters，新的共同协议普通baseline，不是官方SSM论文复现。\n- Native FlowNP 128x4, ff256：{Q["preflight"]["flownp"]["parameters"]} parameters；保留官方target-wise random time、public noisy/augmented sampler及.loc样本语义。架构缩放/数据合同迁移明确，不冒充headline reproduction。\n- 每family每数据2seed，共8fresh fits/16000updates，AdamW/batch32/2000step，全部曲线和selected/last checkpoints在服务器保留。\n- ETTh1 source commit `{I["ett_commit"]}`；OT物理单位原README未明确，不猜测。Jena原始字段T (degC)，2016两zip。原字节来源/SHA见outputs/intake.json。Jena74缺失小时槽不插值，完整窗口筛除。\n- Runtime依赖仅cycle40/vendor，旧venv/global配置和旧成果保留。\n\n{limits}\n',
 'experimental_results.md':f'# C40 固定未来目标、真实观测揭示：普通基线实测\n\nUTC {now}。{framing}\n\n192历史/96future，所有queries固定B=future65:96，r=0/8/32真实观测，剩余gap采样，train-only normalization，32 iid draws。公平U-statistic energy主指标；下表均为探索性VAL/两seed均值。query计时为single-origin、32draws、GPU同步；AR包含本次cache建立或真实观测增量更新，FlowNP包含全部原生conditioning。\n\n{mdtable}\n\n![Fixed-tail quality](outputs/analysis_v1/fixed_tail_quality.png)\n\n所有origin/seed/reveal/step的5个指标见[CSV](outputs/analysis_v1/all_origin_scores.csv)。所有曲线、cap flags、实际训练/评测成本和cache增量成本见outputs/screen_v1/results.json。逐时阻断不确定性见[analysis.json](outputs/analysis_v1/analysis.json)，只作描述，不将重叠windows或两seed当独立重复。\n\n{limits}\n\n独立核验：{V["raw_path_archives"]} raw NPZ、{V["independent_score_cells"]} score cells，max差{V["max_score_difference"]:.3g}。独立pdist实现，校对原数据/时间戳/target坐标/split/selection/checkpointSHA；未独立重新训练GPU。Native API及training loss同随机源重放见qualification.json。\n',
 'negative_evidence.md':f'# C40 反证与不能作出的结论\n\n{framing}\n\n1. 官方Jena有真实75小时gap；若拼接后假装每步一小时会构造错误动力学。V2 intake在预检停止；V3加入74 NaN槽并排除窗口，没有填补信号。\n2. 首次source replay失败来自检验脚本把native CPU初始noise/时间改成CUDA随机源。V1 qualification在任何正式fit前退出；V2精确保留native随机设备，重放后才启动。没有把这一失败解释成FlowNP的问题。\n3. 普通AR本就具备同一law上的前缀条件化与cache；不能把新增观测修正接口本身认作创新。原生FlowNP速度仅代表此API，不能代表PathFM/SFM等整个生成模型类别。\n4. {limits}\n\n没有测试集调参、没有公开上传、没有删除旧成果，也没有为新方法增加一个有利的弱高斯AR对照。\n',
 'hypothesis_space.md':f'# C40 竞争解释与下一反证\n\n{framing}\n\n| 假设 | 当前可测证据 | 不能混淆的解释 / 下一反证 |\n|---|---|---|\n| 表达/训练不足 | 完整learning curves、cap和模型容量 | 若cap仍改善，先公平延长/校准普通baseline；不称机制缺口 |\n| 逐步生成误差累积 | AR固定B、r揭示下变化 | 联合生成/适当SSM的已有解法仍需比较 |\n| 直接conditioning重算有成本 | Native FlowNP1/2/8/32曲线与cache AR计时 | SFM warm source、少步蒸馏及新高效prior可消除，不能泛化 |\n| 状态学习与修正存在非平凡冲突 | 本轮没有直接证明 | 先有known-latent可辨识fixture区分信息损失/近似误差/conditioning；再测试新交互 |\n\n{limits}\n',
 'research_ideas.md':f'# C40 框架候选，尚未接受\n\n{framing}\n\n优先工作问题是可观测修正的潜在动力学；备选是信息可用性改变时的条件状态形成。前者必须跨过AR/SSM/FlowNP/SFM完整近邻，后者必须跨过mask-aware/CAMOS等。两者目前均没有独立创新机制证据。具体scope/候选lineage/证伪见[FRAMEWORK_DIRECTION](FRAMEWORK_DIRECTION.md)。\n\n不把本轮两个数据集上的胜负命名为新架构，不继续围绕某个站点加loss追分。若普通方法已达到用途，放弃宽泛失败动机；若有稳定残差，再定位哪种state-learning interaction必要，先做最小组件交互消融。\n',
 'final_research_direction.md':f'# C40 当前研究判断\n\n{framing}\n\n**Research Question**：同一个时间序列世界模型的学得状态，怎样同时支持未来推演和新观测到来后的有效修正？\n\n**Evidence**：本轮完成了source-qualified native FlowNP与非高斯GRU状态模型的8fit真实前缀筛查，详见[全结果](experimental_results.md)。这验证了评测可执行性和普通能力/代价，不直接证明一个新机制缺口。\n\n**Existing Methods**：AR/SSM具有一致的更新接口，FlowNP直接条件生成，SFM已有旧预测到新条件分布的transport。尚不能说它们不足。\n\n**Proposed Idea**：以状态学习和观测修正的交互为开发目标；最终模块/目标函数还未被证据选定。添加普通滤波器或一致性项不算新机制。\n\n**Falsification**：普通方法以合理信息/计算达到用途，或所谓新目标只改善内部law误差而不改善真实预测，或收益依赖数据集专用结构，则当前动机/框架主张不成立。\n\n**Feasibility**：单5090实际完成本轮；原始数据/commit/config/schedules/logs/selected+last权重/raw paths/SHA保留，TEST未评分。\n\n**Next Experiment**：先根据cap/曲线排除训练不足，再完成同合同SFM/恰当state-space对照；存在稳健残差后才设计known-state机制fixture及新交互。最终论文验证需不同领域、观测条件和rollout任务，不以本站点优化结束。\n\n{limits}\n',
 'CURRENT_STATE.md':f'# C40 complete: progress only\n\nUTC {now}. {framing}\n\n8 ordinary fits,16000 updates,{V["independent_score_cells"]} independent score cells; TEST unscored. See [framework scope](FRAMEWORK_DIRECTION.md), [results](experimental_results.md), [decision](final_research_direction.md). No accepted final direction.\n'
}
for name,body in reports.items():(R/name).write_text(body,encoding='utf-8')
manifest={'utc':now,'reports':{name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in reports},'verification_sha256':hashlib.sha256((O/'verification.json').read_bytes()).hexdigest(),'analysis_sha256':hashlib.sha256((A/'analysis.json').read_bytes()).hexdigest(),'source':'Actual server execution, not literature-only idea','accepted_directions':0,'goal':'ACTIVE_UNACHIEVED'}
(R/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(analysis,indent=2))
