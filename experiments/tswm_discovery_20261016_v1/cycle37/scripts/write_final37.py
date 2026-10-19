"""Evidence-only closing reports after the original job and independent verification."""
import pathlib,json,datetime,hashlib,shutil
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs'
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
D=read(O/'analysis_v1/analysis.json');V=read(O/'verification_v1.json');T=read(O/'learning_v2/results.json')
assert V['passed'] and D['status']=='completed' and not T['test_scored']
assert D['results_sha256']==V['results_sha256']==sha(O/'learning_v2/results.json')
assert D['verification_sha256']==sha(O/'verification_v1.json')
assert read(O/'supervisor_v2.json')['status']=='completed'
names=['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md',
       'hypothesis_space.md','research_ideas.md','final_research_direction.md','CURRENT_STATE.md']
snapshot=R/'reports_in_progress_v1';assert not snapshot.exists();snapshot.mkdir()
for name in names:shutil.copy2(R/name,snapshot/name)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
base=f'UTC {now}。C37原作业完成，12个neural fits、2个HGB fits、48000个formal updates；独立复核通过。整体目标仍ACTIVE_UNACHIEVED，最终接受方向0。TEST没有评分。本轮是failure discovery与普通解释控制，不是新方法。'
scope='目标是可复用的时间序列world-model框架；两个站点仅为机制证据载体。不能把单数据集涨点、实现修正或已知copula/校准改名为最终研究方向。'
def f(x):return f'{x:.6f}'
fam=D['family_means'];pairs=D['paired_comparisons']
table='| task | model | arm | fits | hourly CRPS | daily-mean CRPS | ramp CRPS | energy | variogram | MAE | coverage90 | width90 |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for x in fam:
    table+='| '+' | '.join([x['dataset'],x['model'],x['arm'],str(x['n_fits'])]+[f(x[k]) for k in ['crps','daily_mean_crps','ramp_crps','energy','variogram','mae','coverage90','width90']])+' |\n'
pt='| task | prelisted comparison | metric | reference | candidate | improvement % | paired 95% CI, abs |\n|---|---|---|---:|---:|---:|---|\n'
for x in pairs:
    if x['seed'] not in ['mean_score_over_2_seeds','deterministic'] or x['metric'] not in ['crps','daily_mean_crps']:continue
    pct='NA' if x['improvement_pct'] is None else f"{x['improvement_pct']:.2f}"
    pt+='| '+' | '.join([x['dataset'],x['comparison'],x['metric'],f(x['reference_mean']),f(x['candidate_mean']),pct,
                        f"[{x['ci95_low_abs']:.6f}, {x['ci95_high_abs']:.6f}]"])+' |\n'
ht='| task | fit | selected step | params | fit seconds* | 128-path VAL seconds | cap flag |\n|---|---|---:|---:|---:|---:|---|\n'
for x in D['health']:
    ht+='| '+' | '.join([x['dataset'],x['fit'],str(x['selected_step']),str(x['parameter_count']),f"{x['fit_seconds']:.2f}",
                        f"{x['sample_128_val_seconds']:.4f}",str(x['last_1000_improvement_gt_1pct'])])+' |\n'
stats={n:read(O/'learning_v2'/n/'data.json') for n in ['wind','solar']}
counts='| task | FIT | CAL | VAL | TEST, unscored |\n|---|---:|---:|---:|---:|\n'
for n,m in stats.items():counts+='| '+n+' | '+' | '.join(str(m['counts'][k]['retained']) for k in ['fit','cal','val','test'])+' |\n'
verification=f'''独立raw↔canonical、window/split/scaler、schedule、initial/selected checkpoint、VAL/CAL native路径及selected64-path评分重放通过；12个neural fits和2个HGB point模型。对全部56个已存路径arms重新计算9个score，共{V['score_cells_verified']}个score cells；最大差{V['max_score_abs']:.12g}。native prediction/CAL重放最大差{V['max_prediction_abs']:.12g}（容差2e-4），不是声称逐bit为0。quantile/HGB同marginal排列恒等性通过。50-step和CAL/rank arms的score已独立重算，但不是各转换实现的全套独立重写；HGB点值已重放，不把随机residual路径生成称为第二套独立实现。'''
uncertainties=f'''两个seed、两个固定zone、一个小时级日窗口协议，全部是探索VAL结果，不是独立TEST确认。wind VAL原105候选仅66完整case，39因缺失被排除；TEST76未评分。solar VAL118/TEST119。预算cap标记{D['n_cap_improvement_flags']}/12，初始step0被选{D['n_selected_initial']}/12；无cap标记不等于已证明收敛。参数量/训练目标/CPU树与GPU网络FLOPs并未严格匹配，不能仅凭排名断言特定state机制缺失。bootstrap按7个日历日而非连续row号，2000次；CI探索性、未校正多重比较；两个seed的score平均不是forecast ensemble，也不是大量独立随机重复。'''
reports={}
reports['experimental_results.md']=f'''# C37 已验证学习结果

{base}

{scope}

## 来源与协议

完整冻结协议在[LEARNING_PROTOCOL.md](LEARNING_PROTOCOL.md)、[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md)。原publisher ZIP直接服务器正常TLS取得，126360077 bytes，SHA256 `d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154`。只用原Task15，未合并Tasks1–14重复；solar未来predictors含POWER，明确排除，标签只来自独立train15。历史168点、未来24点，daily-midnight+24h按组织者声明合同；没有虚构UTC或逐行真实发行vintage。

{counts}

{verification}

## 全部固定arms，raw/unclipped primary

{table}

各seed值、MSE、clipped共享敏感性等完整字段在[all_results.csv](outputs/analysis_v1/all_results.csv)和[analysis.json](outputs/analysis_v1/analysis.json)，没有只挑有利对照。数值单位为normalized power；coverage应接近.9，width须与coverage联读，均不是越小越好。

## 全部预列主指标对比

下表candidate改善定义reference−candidate，正数较好。neural行先在相同origin上平均两个seed的score；HGB只有一个fit。所有7个loss指标、两个独立seed与mean-score结果见[paired_comparisons.csv](outputs/analysis_v1/paired_comparisons.csv)。

{pt}

## 训练健康与成本

{ht}

*fit_seconds包含该fit训练及后续固定路径/校准/评分工作；128-path列是单独同步测量的VAL native预测时间。不能从较大batch摊销时间推断单路径成本。原worker总用时{T['elapsed_seconds']:.2f}秒，独立verifier{V['seconds']:.2f}秒。

{uncertainties}

![固定普通对照](outputs/figures_v1/conditional_path_controls.png)

![全部学习曲线](outputs/figures_v1/learning_curves.png)

本轮本身未提出或训练新的world-model方法；如何解释数值必须保留普通控制与上述限制。初步进行中报告已完整归档reports_in_progress_v1。
'''
reports['baseline_registry.md']=f'''# C37 已执行基线登记

{base}

{scope}

quantile MLP是自建直接控制，19个quantile、96×96 hidden；KITE原组件pin `3140ee824cbd80c5ec7fdf2b54666210519d0b39`，64dim/2depth/4head/rank8，原train/inference实现迁移。缩小模型、新daily loader、概率选模，明确不是原论文排行榜复现。L2 arm仅将原velocity L1换MSE；同initial hash/schedule/per-step RNG，不是本项目新方法。

两个任务×两个seed×三个neural families共12fits，各4000updates/batch64/Adam lr.001。全部同输入history target/history NWP/future NWP/calendar。每个neural都有同CAL逐lead逐q加性校准；quantile额外IID/CAL-PIT rank/32-nearest analogue rank无需另拟合，exact marginal multiset恒等。HGB两CPU fits固定200trees/15leaves/lr.05，CAL residual IID/joint/analogue；无VAL HPO，但不称FLOPs匹配。KITE50-step敏感性使用同10-step所选checkpoint，不重选。

{ht}

原selected.pt、CAL/VAL路径NPZ、schedule、canonical/raw数据留服务器。源码、固定配置、日志和JSON已按SHA本地镜像；不对外发布。{verification}

{uncertainties}
'''
reports['literature_review.md']=f'''# C37 文献与创新边界

{base}

{scope}

本轮不是新文献方向，延续[C36完整综合](../cycle36/literature_review.md)。WorldTS/Chronos-2的部分任务已有真实预报条件；DSSMF/MMFE覆盖随机外部输入/预报演化；KITE等覆盖条件路径生成。不能从一个生成器排名较低推出这些领域整体未解决。

[Flow Matching](https://arxiv.org/abs/2210.02747)的标准向量场平方回归作为L2普通控制；固定KITE原代码的L1实现与内部sample平均均已记录，但源码现象不直接证明训练分布失败。[Schaake shuffle](https://pubs.usgs.gov/publication/70027101)是历史秩重排先例，CAL/analogue rank不是新机制；逐q校准与树模型经验residual paths也都是普通解释。

[GEFCom组织者论文](https://www.pierrepinson.com/docs/Hongetal2016.pdf)给daily forecast合同；本轮原始数据已有直接服务器下载、完整archive/member/SHA证据。没有已识别true future weather标签，不能把CAL target residual covariance称为weather forecast error，也不能据此宣称完成forcing-integration K压力验证。

全部实训数字见[结果](experimental_results.md)。用户的框架目标不要求action、物理可辨识性或共享零样本checkpoint；新能力若由既有组件组成也可有价值，但需要相对于强普通完整方法的跨任务证据。
'''
reports['negative_evidence.md']=f'''# C37 负证据与未排除解释

{base}

数据/reader失败全部版本保留：hour0..23修正、wind11NaN保留并按固定完整窗口排除、pandas numpy.str_显式str转换。正式原训练没有被这些preflight修正重复启动。solar未来POWER没有进入features。

{uncertainties}

同marginal rank重排可改变daily/ramp/joint score，不能当本项目新发现；标准L2、普通校准、10/50-step变化若解释差异，就否定相应“新状态机制缺失”的归因。反过来，某个普通控制未胜也不自动证明新框架必要，仍须排除有限预算/容量/有效样本差异。

C36 observer-only提案仍因普通目标等价关闭；C35具体protected residual配置未过screen仍关闭。未重新调zone/seed/rank或TEST以复活旧方向。没有weather truth，C36 q(U|F)条件假设仍未建立；target residual结构只是目标条件分布的诊断。

{verification}
'''
reports['hypothesis_space.md']=f'''# C37 竞争解释的完整证据索引

{base}

预列比较已全部完成，详细方向、效应和探索CI见[结果表](experimental_results.md)，不得只报告其中显著行。

| 解释 | 已执行区分 | 能支持的结论边界 |
|---|---|---|
| 直接边际加经典rank coupling足够 | exact同marginal IID/shuffle/analogue；HGB residual joint/IID | 可归因对应路径变化，不能泛化所有任务 |
| 原flow loss而非state能力解释差异 | L1/L2匹配初始化、schedule、RNG和预算 | 标准目标控制；不是新方法 |
| 只是marginal calibration | 所有neural raw与同CAL校准 | 校准本身不产生机制创新 |
| 只是积分离散误差 | 同weights 10/50steps | 不重选checkpoint；不能等同外部forcing K |
| undertraining/overfit/容量 | 所有step0到4000曲线、cap标记与参数/成本 | cap无改善也不证明全局收敛 |
| 额外状态/转移机制必要 | 尚无新方法，尚无独立TEST | 本轮不能批准最终框架 |

{uncertainties}
'''
reports['research_ideas.md']=f'''# C37 候选状态

{base}

{scope}

本轮增加真实学习证据，并未把任何具体数据集的最优arm推荐为研究方向。KITE L2、CAL、Schaake/analogue秩、经验residual、改变积分步数都不能单独作为本项目新idea。此处正式接受候选仍为0；完整效应已公开于工作目录而非只保留支持预期的数字。

如存在值得深究的跨任务剩余差异，后续先设计能够区分普通direct/joint generator/state-space解释的实验；不由VAL挑一个分数最好的arm进入TEST，再冒充预先确认。C38独立文献补查异质条件/多模态状态学习是另一路证据探索，不是把本轮结果改名为多模态缺陷。
'''
reports['final_research_direction.md']=f'''# C37 当前研究裁决：未批准最终方向

{base}

**Research Question：** {scope} 本轮仅问强直接边际、普通校准与已有路径依赖方法之后还剩什么。

**Evidence：** 原publisher数据两个真实任务，12神经fits/48000updates、2树fits；全部56个评分arms与12个selected checkpoint独立复核。结果与所有预列比较完整见[experimental_results.md](experimental_results.md)。这确立了可复现的有限预算比较，不等于确立新框架的必要性。

**Existing Methods：** KITE原组件/标准L2、同信息quantile MLP、经典rank coupling、同CAL校准、HGB residual paths均已执行。不能把这些方法本身重新命名成创新。

**Proposed Idea：** 无已接受方案。本轮不为了关闭任务而强行推荐一个dataset优化。

**Falsification：** 普通目标、coupling/calibration、积分或训练/容量/缺失选择解释现象时，相关新机制归因不成立；即使对照未胜也需要额外区分实验证明必要性。

**Feasibility：** 单RTX5090实际完成；约.25–.40M参数组件，原数据/weights/paths/source保留。{uncertainties}

**Next Experiment：** 先由root解释全部固定比较并合并C38独立查新，选择能区分新机制与最强普通修复的最小试验；当前不启动第二轮调参，不评分TEST。整体自主发现目标继续ACTIVE_UNACHIEVED。
'''
reports['CURRENT_STATE.md']=f'''# C37 原作业完成且独立验证通过

{base}

原supervisor35672/worker25100已正常结束；不再用旧PID存在与否判失败。postprocess_v1仅执行独立verifier。analyze37/plot37消费verified SHA一致结果，不训练，不评分TEST。

{verification}

{uncertainties}

进行中八份报告已归档reports_in_progress_v1。研究方向仍未选定；后续从真实剩余问题出发，不把完成一个普通对照实验当总目标完成。
'''
for name,content in reports.items():(R/name).write_text(content,encoding='utf-8')
header=f'''<!-- cycle37-complete-start -->
# C37原学习作业与独立核验完成

UTC {now}。12 neural fits/48000 updates、2 HGB fits；56 path arms独立score核查，12 selected native path/checkpoint replay通过。score最大差{V['max_score_abs']:.12g}，prediction最大差{V['max_prediction_abs']:.12g}。TEST未评分，接受最终方向0，整体ACTIVE_UNACHIEVED。wind VAL66/105完整cases，非全部时间总体。所有固定比较而非只挑正面行见[C37结果](cycle37/experimental_results.md)，预列[分析计划](cycle37/ANALYSIS_PLAN.md)。下方进行中状态是历史快照，原作业已完成，不要重启。
<!-- cycle37-complete-end -->

'''
p=R.parent/'CURRENT_RESEARCH_STATUS.md';old=p.read_text(encoding='utf-8');assert '<!-- cycle37-complete-start -->' not in old;p.write_text(header+old,encoding='utf-8')
for name in names[:-1]:
    p=R.parent/name;old=p.read_text(encoding='utf-8');assert '<!-- cycle37-complete-index-start -->' not in old
    p.write_text(f'<!-- cycle37-complete-index-start -->\n# C37完整学习证据已核验\n\nUTC {now}。12 neural/2 HGB fits、56 path arms，TEST未评分。整体ACTIVE_UNACHIEVED，最终接受方向0。见[本项报告](cycle37/{name})。下面为历史。\n<!-- cycle37-complete-index-end -->\n\n'+old,encoding='utf-8')
manifest={name:{'sha256':sha(R/name),'bytes':(R/name).stat().st_size} for name in names}
(R/'REPORT_MANIFEST_FINAL.json').write_text(json.dumps({'utc':now,'reports':manifest,'analysis_sha256':sha(O/'analysis_v1/analysis.json'),
    'verification_sha256':sha(O/'verification_v1.json'),'script_sha256':sha(pathlib.Path(__file__))},indent=2),encoding='utf-8')
print(json.dumps({'reports':len(reports),'status':'completed','accepted_directions':0,'test_scored':False}))
