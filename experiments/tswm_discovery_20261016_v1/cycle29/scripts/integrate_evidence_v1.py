"""Integrate completed, independently checked evidence without modifying scientific sources."""
import csv, datetime, hashlib, json, pathlib, shutil, statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT.parent
OUT = ROOT / 'outputs/integrated_evidence_v1'
NAMES = ['literature_review.md', 'baseline_registry.md', 'experimental_results.md',
         'negative_evidence.md', 'hypothesis_space.md', 'research_ideas.md', 'final_research_direction.md']

def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))

def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()

def write(name, text):
    (ROOT / name).write_text(text.strip() + '\n', encoding='utf-8')

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])

def strip_prefix(text, start, end):
    if start in text:
        a = text.index(start); b = text.index(end, a) + len(end)
        return text[:a] + text[b:].lstrip()
    return text

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    primary = read('outputs/world_verification/model_comparison_v2.json')
    realproof = read('outputs/real_transfer/verification.json')
    secondary = read('outputs/secondary_analysis_verification.json')
    implementation = read('outputs/inference_controls_v1/results.json')
    assert all(x['status'] == 'passed' for x in [primary, realproof, secondary, implementation])
    assert primary['results_sha256'] == sha('outputs/world_evaluation_v2/results.json')
    assert primary['training_summary_sha256'] == sha('outputs/world_training_v2/summary.json')
    assert realproof['results_sha256'] == sha('outputs/real_transfer/results.json')
    assert secondary['primary_verification_sha256'] == sha('outputs/world_verification/model_comparison_v2.json')
    assert secondary['attribution_results_sha256'] == sha('outputs/world_attribution_v1/results.json')
    assert secondary['utility_results_sha256'] == sha('outputs/finite_plan_utility_v1/results.json')
    assert implementation['latency_sha256'] == sha('outputs/inference_controls_v1/latency.csv')
    assert implementation['equivalence_sha256'] == sha('outputs/inference_controls_v1/equivalence.json')
    with (ROOT / 'outputs/world_evaluation_v2/per_system_metrics.csv').open(encoding='utf-8', newline='') as f:
        metrics = list(csv.DictReader(f))
    assert len(metrics) == primary['metric_rows_rescored'] == 79872
    methods = ['conditional96', 'conditional128', 'compiled64', 'native_attention64',
               'native_attention64_chunk32', 'arx_context_selected', 'pretrained_native_icl']
    model_summary = []
    for family in ['lti', 'wh']:
        for method in methods:
            rows = [r for r in metrics if r['family'] == family and r['method'] == method and r['horizon'] == '128']
            factual = statistics.mean(float(r['clean_rmse']) for r in rows if r['plan'] == 'factual')
            effect = statistics.mean(float(r['effect_rmse']) for r in rows if r['plan'] != 'factual')
            model_summary.append(dict(family=family, method=method, factual_rmse=factual, effect_rmse=effect))
    model_table = table(['家族', '模型', 'factual RMSE', 'changed-plan effect RMSE'],
                        [(r['family'], r['method'], f"{r['factual_rmse']:.6f}", f"{r['effect_rmse']:.6f}") for r in model_summary])
    attribution = read('outputs/world_attribution_v1/paired_contrasts.json')
    contrasts = [r for r in attribution if r['method_a'] == 'compiled64' and r['method_b'] == 'conditional128'
                 and r['horizon'] == 128 and r['metric'] == 'effect']
    assert len(contrasts) == 2
    contrast_table = table(['家族', 'compiled−wide', 'ratio', '描述性跨种子/系统区间', '3个种子差值'],
                           [(r['family'], f"{r['mean_difference']:+.6f}", f"{r['ratio_of_means']:.4f}",
                             f"[{r['difference_p025']:+.6f}, {r['difference_p975']:+.6f}]",
                             ', '.join(f'{v:+.4f}' for v in r['seed_differences'])) for r in contrasts])
    real = read('outputs/real_transfer/summary.json')
    real_table = table(['物理对象', '模型', 'H128 NRMSE', '训练种子标准差'],
                      [(r['family'], r['method'], f"{r['nrmse128']:.6f}",
                        '-' if r['seed_std'] is None else f"{r['seed_std']:.6f}") for r in real])
    utility = read('outputs/finite_plan_utility_v1/summary.json')
    utility_summary = []
    for family in ['lti', 'wh']:
        for method in ['compiled64', 'conditional128', 'native_attention64_chunk32', 'arx_context_selected',
                       'pretrained_native_icl', 'always_zero', 'always_factual']:
            rows = [r for r in utility if r['family'] == family and r['method'] == method and r['horizon'] == 128]
            utility_summary.append(dict(family=family, method=method,
                regret=statistics.mean(r['mean_regret'] for r in rows),
                oracle_fraction=statistics.mean(r['oracle_choice_fraction'] for r in rows)))
    utility_table = table(['家族', '决策器', '平均 regret', '4计划内 oracle 选择率'],
                         [(r['family'], r['method'], f"{r['regret']:.7f}", f"{r['oracle_fraction']:.4f}") for r in utility_summary])
    with (ROOT / 'outputs/inference_controls_v1/latency.csv').open(encoding='utf-8', newline='') as f:
        times = list(csv.DictReader(f))
    latency = []
    for kind in ['conditional96', 'compiled64', 'conditional128', 'native_attention64']:
        for impl in dict.fromkeys(r['implementation'] for r in times if r['kind'] == kind):
            rows = [r for r in times if r['kind'] == kind and r['implementation'] == impl and r['horizon'] == '128' and r['n_plans'] == '16']
            latency.append(dict(kind=kind, implementation=impl, **{k: statistics.median(float(r[k])*1000 for r in rows)
                for k in ['context_s', 'prepare_s', 'cached_s', 'one_use_total_s', 'amortized_64_uses_s']}))
    latency_table = table(['模型', '实现', 'cached ms', '一次含准备 ms', '64次摊销 ms'],
                          [(r['kind'], r['implementation'], f"{r['cached_s']:.4f}", f"{r['one_use_total_s']:.4f}",
                            f"{r['amortized_64_uses_s']:.4f}") for r in latency])
    eq = read('outputs/inference_controls_v1/equivalence.json')
    maxeq = max(r['whole_sequence']['max_absolute'] for r in eq)
    fixed_dir = ROOT / 'outputs/fixed_update_control_v1'
    fixed_verified = (fixed_dir / 'verification.json').exists() and read('outputs/fixed_update_control_v1/verification.json')['status'] == 'passed'
    fixed_text = '等更新次数对照正在运行：两种主架构各3种子到32768次更新；相同原始batch顺序和验证选择。新增63488次更新，旧结果保留。尚无经复核的对照结论。'
    fixed_summary = []
    if fixed_verified:
        fp = read('outputs/fixed_update_control_v1/verification.json')
        assert fp['results_sha256'] == sha('outputs/fixed_update_control_v1/evaluation/results.json')
        assert fp['training_summary_sha256'] == sha('outputs/fixed_update_control_v1/training_summary.json')
        with (fixed_dir / 'evaluation/per_system_metrics.csv').open(encoding='utf-8', newline='') as f:
            fr = list(csv.DictReader(f))
        assert len(fr) == fp['metric_rows_rescored'] == 18432
        for family in ['lti', 'wh']:
            for kind in ['compiled64', 'conditional128']:
                rr = [r for r in fr if r['family'] == family and r['method'] == kind and r['horizon'] == '128']
                fixed_summary.append(dict(family=family, method=kind,
                    factual_rmse=statistics.mean(float(r['clean_rmse']) for r in rr if r['plan'] == 'factual'),
                    effect_rmse=statistics.mean(float(r['effect_rmse']) for r in rr if r['plan'] != 'factual')))
        fixed_text = ('等更新次数对照全部6次训练、18432条指标、验证最小checkpoint与神经重放复核通过。每组每种子32768次更新；不声称FLOPs/耗时完全相同。仍使用原测试世界，属于归因控制，不是新独立复现。\n\n' +
            table(['家族', '模型', 'factual RMSE', 'effect RMSE'], [(r['family'], r['method'], f"{r['factual_rmse']:.6f}", f"{r['effect_rmse']:.6f}") for r in fixed_summary]))
        attr_path=fixed_dir/'attribution.json'
        if attr_path.exists():
            attr=read('outputs/fixed_update_control_v1/attribution.json')
            assert attr['verification_sha256']==sha(fixed_dir/'verification.json')
            contrasts_fixed=[r for r in attr['contrasts'] if r['horizon']==128 and r['metric']=='effect' and r['contrast']=='compiled64_minus_conditional128']
            assert len(contrasts_fixed)==2
            fixed_text+='\n\n'+table(['家族', 'compiled−wide effect', 'ratio', '描述性跨世界/种子区间', '3种子差值'],[(r['family'],f"{r['mean_difference']:+.6f}",f"{r['ratio_of_means']:.4f}",f"[{r['difference_p025']:+.6f}, {r['difference_p975']:+.6f}]",', '.join(f'{v:+.4f}' for v in r['seed_differences'])) for r in contrasts_fixed])
            fixed_text+='\n\n等更新后LTI效应误差仍低约16.1%，WH反而高约2.4%，且WH三个种子差值均正。故原LTI优势并非只由更新次数不等解释，但没有跨家族通用优势。完整36对比、24汇总、各seed与局部cap标记见[等更新归因报告](FIXED_UPDATE_CONTROL_REPORT.md)。普通早停模型续训仍显著改善；cap局部下降标记false不能当作收敛认证。'
            falling=[f"{r['kind']}/{r['seed']}" for r in attr['curves'] if r['original_cap_flag_rule']]
            if falling:fixed_text+='\n\n当前固定cap下仍触发原局部下降规则的run为'+', '.join(falling)+'。因此此处只讨论32768更新内的有限预算实测，不将其解释为架构收敛后的能力上限。'
    backup = OUT / 'previous_living_reports'
    backup.mkdir(exist_ok=True)
    for name in NAMES:
        if (ROOT / name).exists() and not (backup / name).exists(): shutil.copy2(ROOT / name, backup / name)
    head = f'UTC {now}。C29完整证据整合；原始12/12训练和全部主/真实/次级/实现复核通过。等更新对照已复核={fixed_verified}。接受方向0，总体目标ACTIVE_UNACHIEVED。\n\n'
    scope = ('研究单位是可跨系统复用的时间序列 world model。WorldTS是问题层次参照；单数据集涨点、常规工程修复和已有模块改名均不足以立项。允许有限数据/计算下有价值的组合创新，不要求经典方法在无限资源下仍无解。')
    lit = '''# 文献与创新边界

[WorldTS](https://github.com/decisionintelligence/WorldTS)提供研究层次参照；本轮自建数据和训练不是其原论文复现。

[CLAW](https://arxiv.org/abs/2609.12278)、[DynaDiff](https://arxiv.org/abs/2505.13919v2)已覆盖上下文生成动力学；[RFFPSR](https://arxiv.org/abs/1702.03537)、[PSRNN](https://arxiv.org/abs/1705.09353)已覆盖预测状态。compiled64只是CLAW启发的自主数值迁移，不是新方法。

[In-Context Learning for System Identification](https://arxiv.org/html/2610.13411v1)已覆盖输入分布适配、不同初态、缓存与概率模型向参数模型蒸馏。普通teacher蒸馏、输入匹配和更合适的调用方式均是强已有修复。

[ALPaCA](https://arxiv.org/abs/1807.08912)、[MetaKoopman](https://arxiv.org/html/2607.26345v1)已学习特征/先验并做解析更新；[低维系统参数流形](https://arxiv.org/html/2607.18965v1)已从输入输出做摊销初始化、MAP及Laplace，并讨论多峰/重启。因此普通后验细化和弱方向回退先验尚无新意。

[Memory-Conditioned Diffusion](https://arxiv.org/html/2609.28371v1)已有指数记忆库、预测压缩与概率滚动；[AMGFNO](https://arxiv.org/html/2606.13443v1)已有条件记忆门控；[Neural Laplace Control](https://proceedings.mlr.press/v206/holt23a.html)已有未知延迟/不规则采样。定向复核发现[TIDES](https://arxiv.org/html/2605.09742v1)已让输入决定各模态实部并用真实时间ZOH；把历史编码作为固定输入即包含拟议核心更新。该物理时间记忆候选撤销，未训练。

[处理内生性的ICL](https://arxiv.org/html/2410.01265v3)及[IV-ICL](https://arxiv.org/html/2605.12924v1)排除了笼统IV+ICL创新；少样本闭环元辨识也已有Meta-ICI。新设定差异必须有强同信息比较，不能以名称划界。

[CL-DeePC/CL-SPC](https://arxiv.org/pdf/2402.14374)已分析过去噪声与未来输入相关并用顺序单步预测修复。全未来动作条件引起inverse-policy smoothing是已知目标陷阱；普通causal likelihood必须先排除，不能立为新缺口。

三份新独立研究笔记见 research/cycle30_model_opportunity.md、cycle30_inference_opportunity.md、cycle30_novelty_critic.md；论文/代码核查不等于已复现其性能。前轮三个NONE与本轮意见均不是缺口不存在的证明。
'''
    write('literature_review.md', head + scope + '\n\n' + lit)
    trains = primary['checkpoints']
    train_table = table(['模型', 'seed', '实际steps', 'best step', 'validation MSE'],
        [(r['kind'], r['seed'], r['steps'], r['best_step'], f"{r['best_validation_mse']:.6f}") for r in trains])
    write('baseline_registry.md', head + '''# 模型、数据与复现身份

数据源码sysid_benchmarks固定4c20d94ccab51abf4ebf68aa8810f4596381dd41。LTI/WH各512训练、64验证、128独立测试世界；C128/Htrain32，Htest32/64/128，每世界4个共享前缀的动作计划。SciPy+NumPy64条物理重放、全部256共同前缀及参数划分复核通过。4096步独立参考段提供每对象固定尺度，不能把C128称为完整物理校准数据预算；见CALIBRATION_SCOPE_NOTE.md。

conditional96为91361有效参数；conditional128为122433；compiled64为120705；native_attention64为230721。后者使用作者TSTransformer架构、改MSE与本轮数据/上下文，外部ICL及PCFM权重预训练预算不同。ARX仅用上下文选择阶数/正则并拟合，不把其省略。

训练FP32/TF32关、AdamW3e-4、batch32、最多32768steps、同seed相同batch顺序；独立验证选最小。原停止规则导致总更新数不相等，预算对照另做。

''' + train_table + '\n\n' + fixed_text + '''

真实数据沿用nonlinear_benchmarks官方划分与原采样率；Silverbox及Cascaded Tanks各一台物理对象，共71窗口/3记录角色。仅冻结模型rolling-origin C128/H128重放，不是官方50步初始化全段榜单、71台设备、真实反事实或神经微调。
''')
    write('experimental_results.md', head + '# 完整实验与指标\n\n' + scope + '\n\n## 独立世界的多计划模拟\n\n' + model_table + '''

每系统RMSE再平均；effect排除factual零差分。外部预训练权重预算不同，不能做公平训练效率结论。原生attention迁移的低分不等于其原论文能力上限。

''' + contrast_table + '''

2000次跨世界/训练种子bootstrap只作描述；3种子不足以证明广泛稳定性。LTI优势不能扩展到WH，且ARX在LTI更强。

## 等更新次数归因控制

''' + fixed_text + '\n\n## 独立实物记录迁移\n\n' + real_table + '''

Tanks的train-mean优于全部神经模型；Silverbox的上下文ARX明显更强。两台对象不足以证明跨领域模型优势。

## 有限计划选择诊断

''' + utility_table + '''

候选固定为factual/zero/step/pulse，目标-1/0/+1，cost=mean((y-target)^2)+.01mean(u²)。只用模型点预测选择，再由clean物理分支评分；oracle仅四计划内最优。compiled可比wide更好却仍输固定策略，说明不能把轨迹平均分改善自动解释为实际控制收益。

这些是点决策器，不是各概率模型的原生期望成本评估。平方损失必须计入边际方差，未做此控制前不得宣布“原生不确定性/规划失败”或将其立为新Gap；也未做闭环MPC或真实未执行动作。

## 普通等价实现控制

''' + latency_table + f'''

H128/16计划，12个验证case的中位数；含context/preparation、cached以及64次摊销。576时间行、288全序列等价检查通过，maxabs={maxeq:.8g}。折叠code bias/合并Δ权重后普通GRU可用cuDNN；attention缓存memory并允许SDPA。原eager耗时不能作为新编译机制动机。这些是具体实现耗时，不是所有架构的下界，未穷尽torch.compile/KV等实现。

主验证79872指标、4096独立ARX拟合；真实验证4260指标、1136ARX重拟合；次级验证120汇总/72对比/64512决策/504汇总/6utility对比全部重算通过，最大float64相对误差{secondary['max_relative_numeric_error']:.4g}。证明路径列于outputs/integrated_evidence_v1/summary.json。数学一致性不认证科学新颖性。
''')
    write('negative_evidence.md', head + '''# 负面证据及排除范围

- v1重复context归一化导致最高0.1%窗口占目标能量95.7408%；是本实验选择的混杂，不是作者代码或科学Gap。已停止并保留全部产物，v2固定参考单位，未删样本。
- “上下文生成转移权重”已有CLAW/DynaDiff；本轮LTI胜wide、WH不胜，真实对象混合。不能因此发布新架构或通用优势。
- LTI上ARX、Silverbox上ARX、Tanks上train-mean分别击败当前神经结果。不能用只含神经模型的原始图制造更强结论；新增integrated_evidence_v1/full_evidence.png明确包含这些对照。
- zero/swap code后退化只证明代码被使用，不能证明物理参数识别或状态/动力学解耦。
- eager的compiled延迟比条件GRU高；合并后两者均不到0.4ms量级（H128/16plan cached）。普通等价实现修复大部分差距，不能再以未优化时序循环推销编译方法。
- 固定四计划的点选择比固定动作差，仍未排除方差项、普通稳健规划或优化器诅咒。不得称为原生概率模型失败。
- 普通Bayesian特征求解、MAP/Laplace后验细化、IV+ICL、固定/门控记忆库均有直接先例。TIDES含有物理时间上下文衰减的核心更新，撤销原候选；CL-DeePC含有未来动作相关噪声与顺序修复，撤销其新机制归因。
- 未完成的等更新对照不填写分数；原始种子不能在新归因控制中冒充新的独立重复。没有在测试上挑rank、学习率、width或新方法。
''')
    write('hypothesis_space.md', head + '# 竞争解释与反证\n\n' + table(['解释', '已完成的区分证据', '仍未排除/下一判据'], [
        ['生成算子有跨系统优势', 'LTI effect .9279 vs wide1.1619；WH1.5572 vs1.5247', '非通用；等更新数控制；强辨识基线'],
        ['收益只是更多更新', '旧comp72192 vs wide60928总更新', '两组各3seed全部32768的控制，状态见结果报告'],
        ['解码调用导致直接模型退化', 'chunk32修复部分，但不追平', 'objective/优化/先验不匹配不能被排除'],
        ['代码识别了物理动力学', 'zero/swap固定h确实退化', '同系统不同history及可辨识性尚无证据'],
        ['新编译是速度收益来源', '普通fold+cuDNN已经接近', '旧eager对比动机削弱；不做通用下界结论'],
        ['更好轨迹必然带来更好计划', '固定动作可胜点决策器', '先做正确概率期望/强规划控制，再判断'],
        ['物理时间记忆谱是新机会', 'TIDES的同信息参数化已含核心更新', '撤销该具体机制；不宣称所有相关任务已解决'],
        ['闭环动态辨识有新机会', 'IV/Meta-ICI等强重叠；CL-DeePC已修复未来输入相关偏差', '没有剩余具体新步骤，不接受该候选']]) + '''

下一项方法必须先冻结新独立训练/验证/测试系统；用验证选择，只在封存测试上确认一次。未选定机制前不继续无目的放大C29训练。
''')
    write('research_ideas.md', head + '''# 候选研究判断

当前没有接受的新研究方向。C29是强已有组织方式的模型级筛查，结果不支持把compiled64改名发表。有限资源下的组合改进仍允许，但必须指出具体、可证伪的收益来源。

本轮定向核查后不保留正向方法候选，以下仅作撤销记录：

1. **跨系统、跨时间间隔的紧凑记忆动力学：撤销具体机制。** TIDES型time-aware selective SSM给同一历史编码后即可实现所提核心更新；没有提出额外估计器或发现有限预算优势。不同任务不自动等于新机制。
2. **短闭环日志下的动态辨识与部署模拟：未立项。** IV+ICL/Meta-ICI已有近邻；未来动作读出噪声是已知闭环目标陷阱，顺序causal likelihood是必须先排除的普通修复。C29外生输入不提供该现象证据。

后验细化路线本轮不保留：已有参数流形MAP/Laplace及多峰重启构成更直接近邻，尚未提出超过它的具体步骤。不能把检索未找到同标题当作创新证明。
''')
    write('final_research_direction.md', head + '''# 最终方向尚未确定

**Research Question：** 如何从有限历史推断可复用的状态与动力学，使同一时间序列world model在新系统上可靠模拟未来？问题层次已固定，具体机制仍需证据收敛。

**Evidence：** C29完成12个已知机制的自主训练、两个独立合成家族及两台真实对象的迁移；全量数值复核通过。LTI与非线性家族结果混合，简单辨识/常数对照仍可更强；尚无实验直接证明一个现有方法未解决的通用缺口。

**Existing Methods：** CLAW/DynaDiff、ICL sysid、ALPaCA/MetaKoopman、参数流形MAP/Laplace、记忆扩散/Neural Laplace、IV-ICL/闭环元辨识均限制宽泛创新表述。普通工程修复已解释部分速度差距。

**Proposed Idea：** 不接受compiled64为新方法。新检索提出的物理时间记忆与未来动作条件问题也有直接先例，已撤销其具体新机制主张。当前仍无接受的方法；没有凭单数据集分数推荐标题。

**Falsification：** 普通强同信息/同预算方法消除收益、效果仅一类对象或不稳定种子、额外信息/训练预算解释差距、或方法与近期论文相同，则撤销相应候选。允许有效组合创新，门槛是具体新机制与可重复有限预算收益。

**Feasibility：** 单RTX5090；远端D:/tswm_discovery_20261016_v1/cycle29保存数据、权重、optimizer、完整日志/来源；当前等更新对照复用原末状态，不重下数据、不删旧成果。尚未公开发布。

**Next Experiment：** 从现有强模型的可重复残差定位下一机制，冻结新独立实验。等更新状态及解释见experimental_results.md；已完成时不再追加本轮相同模型训练。不能重启已被TIDES或causal likelihood包含的提案，也不能把C29完成等同研究目标完成。新cycle30的精确transport诊断也只得到负面归因证据，见../cycle30/final_research_direction.md；没有接受新方法。
''')
    state = dict(updated_utc=now, stage='integrated_verified_evidence', completed_training_runs=12,
                 original_training_status='completed', primary_verified=True, real_transfer_verified=True,
                 inference_implementation_verified=True, secondary_analyses_verified=True,
                 fixed_update_verified=fixed_verified, accepted_directions=0, goal='ACTIVE_UNACHIEVED')
    (ROOT / 'CURRENT_STATE.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    summary = dict(**state, model_summary=model_summary, contrasts=contrasts, real_transfer=real,
                   utility=utility_summary, latency_ms=latency, fixed_update=fixed_summary,
                   evidence_hashes={p:sha(p) for p in ['outputs/world_verification/model_comparison_v2.json',
                     'outputs/real_transfer/verification.json', 'outputs/inference_controls_v1/results.json',
                     'outputs/secondary_analysis_verification.json']}, source_sha256=sha('scripts/integrate_evidence_v1.py'))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    marker='<!-- cycle29-current-start -->'; end='<!-- cycle29-current-end -->'
    current=BASE/'CURRENT_RESEARCH_STATUS.md'
    old=strip_prefix(current.read_text(encoding='utf-8'), marker, end)
    current.write_text(f'{marker}\n# 当前：通用时间序列world model证据筛查\n\nUTC {now}。C29原始12/12训练、主/真实/次级/等价实现复核通过；等更新对照已复核={fixed_verified}。接受方向0，目标ACTIVE_UNACHIEVED。跨家族证据混合，未把已知编译机制包装成新方向。\n\n最新材料见[完整证据](cycle29/experimental_results.md)、[研究判断](cycle29/final_research_direction.md)与[C30负面transport诊断](cycle30/final_research_direction.md)。以下历史保留，仅指当时轮次。\n{end}\n\n'+old, encoding='utf-8')
    for name in NAMES:
        path=BASE/name; a='<!-- cycle29-index-start -->'; b='<!-- cycle29-index-end -->'
        history=strip_prefix(path.read_text(encoding='utf-8'),a,b)
        path.write_text(f'{a}\n# 最新证据cycle29\n\nUTC {now}。原始12/12及全量验证完成；等更新对照已复核={fixed_verified}，接受方向0。见[cycle29/{name}](cycle29/{name})；以下历史保留。\n{b}\n\n'+history,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=['Cond.128','Compiled64','Attn. chunk32','ARX','External ICL']
    shown=['conditional128','compiled64','native_attention64_chunk32','arx_context_selected','pretrained_native_icl']
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.0),constrained_layout=True)
    colors=['#d18b30','#2c8268','#687482','#3d5697','#b45951']
    for col,family in enumerate(['lti','wh']):
        ax=axes[0,col]
        ys=[next(r['effect_rmse'] for r in model_summary if r['family']==family and r['method']==m) for m in shown]
        ax.bar(np.arange(len(ys)),ys,color=colors);ax.set_xticks(range(len(ys)),labels,rotation=20,ha='right')
        ax.set_title(f'{family.upper()}: H128 changed-plan effect')
        ax.set_ylabel('Mean per-world RMSE (log scale)' if family=='lti' else 'Mean per-world RMSE')
        if family=='lti':ax.set_yscale('log')
        for j,y in enumerate(ys):ax.annotate(f'{y:.3g}',(j,y),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
        ax.margins(y=.2);ax.grid(axis='y',alpha=.18)
        ax=axes[1,col];um=['compiled64','conditional128','always_zero','always_factual']
        ys=[next(r['regret'] for r in utility_summary if r['family']==family and r['method']==m) for m in um]
        ax.bar(range(4),ys,color=['#2c8268','#d18b30','#839f9f','#a5a8a5'])
        ax.set_xticks(range(4),['Compiled64','Cond.128','Fixed zero','Fixed factual'],rotation=20,ha='right')
        ax.set_title(f'{family.upper()}: four-plan point decision utility');ax.set_ylabel('Mean regret, H128')
        for j,y in enumerate(ys):ax.annotate(f'{y:.3g}',(j,y),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
        ax.margins(y=.2);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Original stopping protocol: classical and fixed-decision controls included',fontsize=12)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'full_evidence.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(state),flush=True)

if __name__=='__main__':main()
