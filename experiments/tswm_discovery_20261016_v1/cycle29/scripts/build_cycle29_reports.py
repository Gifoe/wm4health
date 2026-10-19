"""Living Chinese evidence reports; unknown model scores remain explicitly unknown."""
import runtime_profile
import csv,datetime,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')
def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();data=load(ROOT/'outputs/world_verification/data_replay.json');scale=load(ROOT/'outputs/world_verification/training_scale_audit.json')
    assert data['status']=='passed';training=ROOT/'outputs/world_training_v2';runs=[];progress=[]
    for folder in sorted(training.glob('*')):
        if folder.is_dir():
            if (folder/'result.json').exists():runs.append(load(folder/'result.json'))
            if (folder/'status.json').exists():
                try:progress.append(load(folder/'status.json'))
                except json.JSONDecodeError:pass
    status=load(ROOT/'outputs/jobs/world_training_v2/status.json');live=False
    if status['status']=='running':
        import psutil
        try:live=psutil.Process(status['pid']).is_running() and 'train_world_comparison_v2.py' in ' '.join(psutil.Process(status['pid']).cmdline())
        except psutil.Error:pass
    verified=ROOT/'outputs/world_verification/model_comparison_v2.json';evaluated=verified.exists() and load(verified)['status']=='passed';table='测试评估尚未完成；不填写测试分数。';contrast='尚无测试性能结果，不判断生成算子的准确率优势。'
    analysis=ROOT/'outputs/analysis';analysis.mkdir(exist_ok=True)
    if evaluated:
        evaluation=ROOT/'outputs/world_evaluation_v2';proof=load(verified);assert proof['results_sha256']==sha(evaluation/'results.json')
        with (evaluation/'per_system_metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        methods=list(dict.fromkeys(r['method'] for r in rows));summaries=[]
        lines=['| 家族 | 方法 | factual RMSE | changed-plan effect RMSE | 训练种子数 |','|---|---|---:|---:|---:|']
        for family in ['lti','wh']:
            for method in methods:
                selected=[r for r in rows if r['family']==family and r['method']==method and r['horizon']=='128'];seeds=list(dict.fromkeys(r['seed'] for r in selected));fm=[];em=[]
                for seed in seeds:
                    ss=[r for r in selected if r['seed']==seed];fm.append(float(np.mean([float(r['clean_rmse']) for r in ss if r['plan']=='factual'])));em.append(float(np.mean([float(r['effect_rmse']) for r in ss if r['plan']!='factual'])))
                summaries.append(dict(family=family,method=method,factual_rmse_mean=float(np.mean(fm)),factual_rmse_seed_std=float(np.std(fm,ddof=1)) if len(fm)>1 else None,effect_rmse_mean=float(np.mean(em)),effect_rmse_seed_std=float(np.std(em,ddof=1)) if len(em)>1 else None,n_seeds=len(seeds)))
                lines.append(f'| {family} | {method} | {np.mean(fm):.6f} | {np.mean(em):.6f} | {len(seeds)} |')
        table='\n'.join(lines)+'\n\n表中为每系统 RMSE 的均值，效应排除 factual 零差分。外部预训练 ICL 不匹配本轮预训练预算；全部原始系统/seed/计划/时域见 CSV。'
        (analysis/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
        deltas=[]
        for family in ['lti','wh']:
            a=next(r for r in summaries if r['family']==family and r['method']=='compiled64');b=next(r for r in summaries if r['family']==family and r['method']=='conditional128')
            deltas.append(dict(family=family,compiled_vs_wide_effect_relative=(a['effect_rmse_mean']/b['effect_rmse_mean']-1)))
        contrast='生成算子相对近参数量条件 GRU 的效应 RMSE 变化：'+ '；'.join(f"{r['family']}={r['compiled_vs_wide_effect_relative']:+.2%}" for r in deltas)+'。该比较本身不能成立新颖性或跨真实系统结论。'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.6));colors=['#236C8B','#D08532','#37836B','#8A6195','#65707A','#BC554A']
        shown=['conditional96','conditional128','compiled64','native_attention64','native_attention64_chunk32','pretrained_native_icl']
        for ax,family in zip(axes,['lti','wh']):
            for method,color in zip(shown,colors):
                vals=[]
                for horizon in [32,64,128]:vals.append(np.mean([float(r['effect_rmse']) for r in rows if r['family']==family and r['method']==method and r['plan']!='factual' and int(r['horizon'])==horizon]))
                ax.plot([32,64,128],vals,'o-',color=color,lw=1.3,label=method)
            ax.set(title=family.upper(),xlabel='Free simulation horizon',ylabel='Mean per-world effect RMSE');ax.grid(alpha=.2)
        axes[0].legend(fontsize=7,frameon=False);fig.suptitle('Held-out synthetic systems: known model organizations, not a new-method claim',fontsize=10);fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(analysis/f'model_comparison.{ext}',dpi=180)
        plt.close(fig)
    counts={r['kind']:r['trainable_parameters'] for r in runs}
    checks=load(training/'critical_checks.json') if (training/'critical_checks.json').exists() else {}
    for k in ['conditional96','compiled64','conditional128','native_attention64']:
        if k in checks:counts[k]=checks[k]['trainable_parameters']
    runlines=['| 模型 | seed | steps | best validation MSE | cap 下仍下降 |','|---|---:|---:|---:|---|']
    for r in runs:runlines.append(f"| {r['kind']} | {r['seed']} | {r['steps']} | {r['best_validation_mse']:.6f} | {r['underconverged_at_cap']} |")
    train_table='\n'.join(runlines) if runs else 'v2 尚无已完成模型；队列执行中。'
    stage='完整测试已独立复核' if evaluated else ('v2训练进行中，PID已实时核查' if live else f"训练状态={status['status']}；测试尚未独立复核")
    header=f'更新时间 UTC {now}。状态：{stage}。最终接受研究方向0；总体目标 ACTIVE、UNACHIEVED。\n\n'
    write('literature_review.md',header+'''# 文献与创新边界

[WorldTS](https://github.com/decisionintelligence/WorldTS) 是研究层次参照，不是必须照搬的结构。目标是可复用的状态/动态机制，不是指定数据集涨点。

[CLAW](https://arxiv.org/abs/2609.12278) 已通过上下文生成低秩 world-model 更新；因此“context→转移权重”不是本项目新意。本轮只做自主部分观测数值迁移，不宣称原论文复现。

[DynaDiff](https://arxiv.org/abs/2505.13919v2) 已有观测条件模型生成；[Explore the Context](https://proceedings.mlr.press/v130/achterhold21a.html) 已有上下文系统潜变量与多计划预测。这些最近/基础工作限制宽泛模型生成标题，不意味着单篇已解决所有部分观测时间序列。

[RFFPSR](https://arxiv.org/abs/1702.03537)、[PSRNN](https://arxiv.org/abs/1705.09353)、[Predictive-State Decoders](https://arxiv.org/abs/1709.08520) 已有可控预测状态与面向未来的状态监督。不能把它们改名为新理念；潜在新框架须说明具体训练/推理差异并取得有限资源证据。

[Distributionally robust meta-system identification](https://arxiv.org/abs/2506.18074) 已研究任务风险权重。此次普通尺度修复不作为创新。未执行以上各论文全部结果，不以缺代码推断科学缺口。
''')
    write('baseline_registry.md',header+f'''# 实际模型与数据登记

数据来自 pinned sysid_benchmarks，LTI 与 WH 两个结构家族；train1024/val128/test256 个独立系统。数据、源码、参数与种子见 outputs/world_data/manifest.json、systems.json；SciPy 独立验证通过。

| 模型 | 有效参数 | 身份 |
|---|---:|---|
| conditional96 | {counts.get('conditional96','未读取')} | 自主普通条件GRU |
| conditional128 | {counts.get('conditional128','未读取')} | 更宽容量控制 |
| compiled64 | {counts.get('compiled64','未读取')} | 已知低秩超网络思想的自主数值迁移 |
| native_attention64 | {counts.get('native_attention64','未读取')} | 作者 TSTransformer 源架构、新数据/MSE训练 |
| ARX | 按context验证选择 | 测试时仅用128步context拟合 |
| pretrained_native_icl | 外部发布权重 | 预训练预算不同，单列参考 |

v1 已因 train-only 目标尺度问题停止并完整保留。v2 不增加数据、不删窗口，不使用未来y；训练与评估采用固定参考单位，外部 ICL 的原 context-only 变换单列。v2 完成训练 {len(runs)}/12，实际验证选择如下：

{train_table}
''')
    write('experimental_results.md',header+f'''# 完整实验记录

数据独立验证：{data['unique_worlds_by_split']}；全部{data['shared_prefix_worlds']}个测试世界的分叉前缀最大差0；SciPy+NumPy独立重算{data['replay_branches']}条分支通过。

v1 训练目标审计按实际262144窗口计算：最高0.1%窗口贡献归一化目标能量的{100*scale['top_energy_fraction']['0.001']:.4f}%。这是目标能量，不能冒充已训练模型误差。修订前后验证损失单位不同，不能直接比较数值作为提升。

v2 训练完成 {len(runs)}/12，模型选择只看独立验证系统。

{train_table}

## 测试

{table}

{contrast}

测试若完成，必须与 outputs/world_verification/model_comparison_v2.json 的全量重算相对应；图是合成跨系统诊断，不是原论文排行榜、真实系统泛化或因果临床证据。
''')
    write('negative_evidence.md',header+'''# 负面证据与排除项

- 逐记录修默认参数、修归一化和补训练，不能充当通用 world-model 新贡献。
- 原生 Chronos fit 复制原基座；已知任务下冻结基座/适配分支可保留基座路径，因此宽泛的“适配必然遗忘”不作为主方向。只核查源码，没有运行其新训练。
- 短历史生成模型/低秩适配已有 CLAW、DynaDiff 等工作；本轮 compiled64 是强基线，不预先标为新方法。
- v1 重复 context 归一化导致极少训练窗口主导目标能量；已停止，所有日志/权重保留。该问题来自本实验训练选择，不归咎作者实现，也不当科学Gap。
- cap 下验证仍明显下降的模型不能用于证明架构无能；同时，有限预算实测优势仍可研究，不要求无限计算不可替代性。
''')
    write('hypothesis_space.md',header+'''# 竞争解释与反证

| 假设 | 区分实验 | 判据边界 |
|---|---|---|
| 普通条件状态已足够 | conditional96/128 对 compiled64 | 参数接近的128控制避免把容量当机制 |
| 生成算子提供复用优势 | 准确率、效应、缓存context的1/16/64计划时间 | 不能只比默认未缓存 attention |
| 直接预测的退化是调用方式造成 | native direct / chunk32 | 普通分块修复足够则不需新框架 |
| 代码只是在保存当前状态 | zero/swap operator，固定h | 依赖代码不等于恢复真实参数 |
| 低分源于训练不足 | 三种子、完整验证曲线、统一cap/停止规则 | cap仍下降明确标记 |
| 新方向只是旧机制重命名 | 对照CLAW/DynaDiff/PSR等 | 需要具体差异与新实验证据 |

所有测试计划及诊断在读取模型测试性能前固定。没有用测试结果选rank/宽度/学习率。
''')
    write('research_ideas.md',header+f'''# 候选方向

当前接受方向0，不凑三个标题。工作问题是：短历史的系统信息应如何进入可复用动态机制，使同一模型能有效模拟新系统的多种未来计划？这是模型级问题，不是单数据集优化。

{contrast}

普通条件GRU、低秩算子生成和原生直接预测均作为已有机制的控制。即便 compiled64 获胜，也需指出超出既有工作之处、跨真实系统验证与额外的最小新机制；不能仅把其命名为新框架。已知组件的有效组合仍允许作为研究路线，证据门槛是具体的新收益和清晰的相对定位。
''')
    write('final_research_direction.md',header+f'''# 最终研究方向尚未确定

**Research Question：** 如何从部分观测历史得到可复用的预测状态/系统动态，使一般时间序列 world model 在新系统上可靠进行多步、多计划模拟？当前仍需将它收敛成具体、有新颖性的研究问题。

**Evidence：** 新增两个结构家族、1408个独立世界；物理/分叉与模型关键路径检查通过，v2 已完成 {len(runs)}/12 个训练。{contrast} 这些事实尚未直接证明一个未解决的科学缺口。

**Existing Methods：** 条件状态模型、上下文识别、低秩模型生成、PSR均已有文献与可执行底座；本轮把它们当竞争者，避免重命名。

**Proposed Idea：** 尚无接受的新方法；不将重复标准化修复或 compiled64 基线包装为创新。

**Falsification：** 普通条件GRU/分块调用足够，生成算子并未被使用，或差异由容量/预算/训练目标解释，都削弱当前宽泛方法动机。

**Feasibility：** 单张 RTX5090 实际执行，源码、数据、种子、训练日志、选择记录与权重在独立 cycle29 目录；无公开发布、无删除旧结果。

**Next Experiment：** 完成当前三种子训练，再执行冻结的独立系统测试与全量复核；依据具体残差选择下一项方法实验，而不是继续逐数据集修配置。若仅重现已知编译机制优势，先完成最近工作定位与独立真实系统检验，再决定是否值得提出新的模型机制。
''')
    state=dict(updated_utc=now,stage=stage,completed_training_runs=len(runs),scheduled_runs=12,training_status=status['status'],training_pid=status['pid'],training_pid_live=live,evaluation_independently_verified=evaluated,accepted_directions=0,goal='ACTIVE_UNACHIEVED',completed_runs=[dict(kind=r['kind'],seed=r['seed'],steps=r['steps'],best_validation_mse=r['best_validation_mse']) for r in runs])
    (ROOT/'CURRENT_STATE.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    current=BASE/'CURRENT_RESEARCH_STATUS.md';old=current.read_text(encoding='utf-8');marker='<!-- cycle29-current-start -->';end='<!-- cycle29-current-end -->'
    if marker in old:old=old[old.index(end)+len(end):].lstrip()
    prefix=f'''{marker}
# 当前执行：cycle29，通用时间序列 world model

UTC {now}。{stage}。新增两个系统家族、独立参数划分与64条SciPy物理重放；已停止存在目标尺度混杂的v1并保留证据。v2采用固定参考单位，完成训练{len(runs)}/12；测试独立复核状态={evaluated}。最终接受方向0，总体目标ACTIVE、UNACHIEVED。

最新材料见 cycle29/experimental_results.md、cycle29/final_research_direction.md、cycle29/NORMALIZATION_AMENDMENT.md。以下历史结论保留，均指当时轮次。
{end}

'''
    current.write_text(prefix+old,encoding='utf-8')
    for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:
        path=BASE/name;history=path.read_text(encoding='utf-8');begin='<!-- cycle29-index-start -->';finish='<!-- cycle29-index-end -->'
        if begin in history:history=history[history.index(finish)+len(finish):].lstrip()
        index=f'{begin}\n# 最新执行 cycle29（总体目标未完成）\n\nUTC {now}。{stage}；完成训练{len(runs)}/12，接受方向0。当前材料见 [cycle29/{name}](cycle29/{name}) 与 [当前状态](CURRENT_RESEARCH_STATUS.md)。以下历史材料保留，仅代表对应轮次。\n{finish}\n\n'
        path.write_text(index+history,encoding='utf-8')
    print(json.dumps(state),flush=True)
if __name__=='__main__':main()
