"""Evidence-bound cycle reports for the completed real-data falsification study."""
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def main():
    result=read('outputs/real_comparators/precision_corrected_results.json');verify=read('outputs/real_comparators/independent_verification.json');assert verify['pass']
    controls=read('outputs/readout_path_control/results.json');assert read('outputs/readout_path_control/independent_verification.json')['pass']
    data=read('outputs/real_data/registry.json');assert read('outputs/real_data/independent_verification.json')['pass'];fits=read('outputs/real_comparators/fit_records.json')
    splits=['test','wave_matched','wave_acquisition'];methods=['persistence','ridge_arx']+result['config']['methods'];groups=[]
    def rows(split,method):return [v for v in result['rows'] if v['split']==split and v['method']==method]
    def average(split,method,key):return float(np.mean([v[key] for v in rows(split,method)]))
    table=['| Split | Method | MSE | MAE | MSE/persistence | RPM MSE | Privileged target decode MSE |','|---|---|---:|---:|---:|---:|---:|']
    for split in splits:
        for method in methods:
            rr=rows(split,method);oracle=np.mean([v['privileged_future_decode_MSE'] for v in rr]) if 'privileged_future_decode_MSE' in rr[0] else None
            table.append(f"|{split}|{method}|{average(split,method,'MSE'):.6f}|{average(split,method,'MAE'):.6f}|{average(split,method,'MSE_ratio_to_persistence'):.5f}|{np.mean([v['groups']['RPM'] for v in rr]):.6f}|{'—' if oracle is None else f'{oracle:.6f}'}|")
            groups.append(dict(split=split,method=method,initializations=len(rr) if rr[0]['seed'] is not None else 0,mean_MSE=average(split,method,'MSE'),seed_MSE=[v['MSE'] for v in rr],mean_RPM_MSE=float(np.mean([v['groups']['RPM'] for v in rr]))))
    table='\n'.join(table);path_table=['| Split | Architecture | Actual latent | TRAIN-mean future latent | TRAIN-mean both paths |','|---|---|---:|---:|---:|']
    for split in splits:
        for method in ['worldts32','worldts4','worldts4_fixed','worldts32_endtoend']:
            values=[np.mean([v['MSE'] for v in controls['rows'] if v['split']==split and v['method']==method and v['control']==kind]) for kind in ['original','mean_future','mean_both']]
            path_table.append('|'+split+'|'+method+'|'+'|'.join(f'{v:.6f}' for v in values)+'|')
    path_table='\n'.join(path_table);out=ROOT/'outputs/analysis';out.mkdir(exist_ok=True)
    boundary=[]
    for fit in fits:
        boundary.append(dict(key=fit['key'],parameters=fit['parameters'],seconds=fit['seconds'],stages=[dict(stage=v['stage'],epochs=len(v['history']),best_epoch=v['best_epoch'],updates=v['updates'],best_at_final_epoch=v['best_epoch']==len(v['history'])) for v in fit['stages']]))
    (out/'analysis.json').write_text(json.dumps(dict(groups=groups,fit_budgets=boundary,accepted_novel_directions=0,objective_achieved=False),indent=2),encoding='utf-8')
    def write(name,text):(ROOT/(name+'.md')).write_text(text.strip()+'\n',encoding='utf-8')
    write('literature_review','''# 文献与框架边界

WorldTS已提供目标序列/条件分离和两阶段训练。Pythia已把共享预测表示、数值参考引导、多模态条件与独立概率读出结合，并做直接/联合/两阶段控制。Phys-JEPA已有物理/残差状态分离与潜在转移约束。因此这些模块名称不是新贡献。原文、版本、SHA和具体边界见LITERATURE_BOUNDARY.md及outputs/literature。

本轮实际执行的是WorldTS数值核心在新真实任务上的迁移，而不是Pythia/Phys-JEPA复现或原论文排行榜复现。将低MAE解释成物理响应正确的风险，已由TSWM机制一致性等工作研究；这里的RPM分组误差只是测量变量预测，不能重新包装为新因果指标。

cycle20/21的自动目标/角色发现仍只有受控机制线索。CDL、Denoised MDPs、ACF和主动/导数子空间构成近邻。当前真实任务的角色由实验协议提供；它不能直接验证未知角色发现的新颖性，但能够检查是否有必要替代现有角色感知条件模型。
''')
    write('baseline_registry',f'''# 已执行模型与协议

同一真实风洞，16随机游走记录+10波形记录，251016行；训练记录1–10、验证11–12、测试13–16。另8份相同采集配置波形为协议变化，2份osr_mic变化波形单列采集变化。64历史/16未来、stride16，共15584个窗口全部由原CSV独立重建，误差0。未来只有5个预排定控制路径；11传感器真值、未来dt、flag/counter不作输入。

| Model | Identity |
|---|---|
| worldts32 | 原生因果patch编码器、PatchTST预测器、patch读出；两阶段；32D/token；RevIN |
| worldts4 | 同上4D/token；RevIN |
| worldts4_fixed | 同上4D/token；关窗口RevIN，保留全局训练集scaler |
| worldts32_endtoend | 同一原生架构，直接联合训练预测MSE的目标控制 |
| gru64 | 普通条件GRU状态空间，64隐状态，末次观测残差读出 |
| mlp128 | 普通128宽MLP，完整历史/计划，末次观测残差读出 |
| ridge_arx | 8历史lag+16步计划；7惩罚由验证集选择 |
| persistence | 无拟合 |

前三种两阶段模型各3seed，另三种监督模型各3seed，共18神经拟合。全部世界模型/验证选择先于测试。源代码18文件固定，训练配置见EXPERIMENT_PLAN.md；新数据配置不是原论文各数据集调优设置。报告实际参数、更新和时间，不主张精确FLOP匹配。

4D/token不是4个数的世界状态：还保留64历史tokens、条件tokens、已知未来计划及RevIN的32个统计量。不能直接与前轮4D单时刻toy作容量等价比较。

主job耗时{result['elapsed_seconds']:.2f}s，峰值CUDA分配{result['environment']['maximum_allocated_bytes']/2**20:.1f}MiB。原始数据/权重/预测/日志保存在服务器cycle22/outputs；本地为紧凑结果/报告镜像。
''')
    write('experimental_results',f'''# 真实任务结果：现有条件模型已提供有效基线

WorldTS32在随机游走测试上的MSE为0.244653，ridge0.244622、GRU0.233761，持久性0.446406。相同采集配置的波形测试上，分别为0.113769/0.105645/0.098367，均优于持久性0.193110。没有出现只有新架构才能修复的整体建模失败。

不过整体误差不能替代所有变量的动力学质量：相同配置波形上WorldTS32的RPM MSE0.050281，较持久性0.041111更差；ridge0.014701、GRU0.020799更好。这是进一步定位训练/转移目标的线索，但普通方法已经提供修复，且该现象与既有机制一致性研究接近，不能直接认定新Research Gap。

## 全部固定配置（神经模型3个初始化的均值）

{table}

数值均来自原始预测的float64重算。原始results.json保留float32聚合；最大按通道差异9.64e-5，详见precision_corrected_results.json的来源哈希和差异记录。所有原dtype聚合独立重放误差0，不是修改预测或测试后调模型。

## 潜在状态路径控制（探索性，非独立确认）

{path_table}

WorldTS32替换未来潜在状态后，随机游走MSE0.244653→0.298413，波形0.113769→0.195000。固定历史均值预测分别0.293795/0.195942。说明未来潜在状态确实参与预测，“模型完全依赖RevIN均值、预测状态没有作用”不符合这些检查点。替换造成的退化可能含离流形效应，不能当作真实世界因果效应；也不证明表示不存在其他信息。

32D原生模型的特权未来编码读出MSE在随机游走/波形为0.048892/0.033615，低于实际预测。它使用未来传感器真值，只是编码/转移读出定位，不是可部署预测。小维度、关RevIN和联合训练各有不同表现，不能从单个劣势配置推出普遍架构结论。

## 复核范围

26原CSV、5窗口文件、15584个窗口、记录划分与训练集标准化独立通过。主结果60预测文件、27阶段权重、18模型通过哈希及3416项聚合复核；全验证集预测复算最大差1.13e-8，原生潜在损失公式独立计算最大差1.44e-6；每split每神经模型前128窗口的权重/特权读出重放误差0。ridge正规方程残差1.84e-15，验证选择复算0。所有选择文件先于测试预测。

后验路径控制另对111个预测文件、训练集潜在均值和固定读出重放验证。失败审计/IO修复/float32检查过程见TECHNICAL_REPAIRS.md；没有覆盖原始失败日志。

![真实数据对照](outputs/analysis/real_conditional_comparators.png)

这是同一物理风洞的记录/协议泛化，不是多个独立物理系统；3seed是初始化重复。该实验验证条件预测，不提供未执行计划的配对干预真值，也尚未验证长期自由模拟。
''')
    write('negative_evidence','''# 负面证据和被排除解释

- 现有WorldTS、ridge与条件GRU都显著优于持久性；不能从前轮未知角色toy推导普遍缺少可用世界模型框架。
- 三个监督基线和原生WorldTS有各自优势；整体MSE最佳不是RPM最佳。不能只挑一个分组给新方法制造必要性。
- WorldTS32的未来状态替换会明显损害预测，拒绝“预测潜在状态完全没有用”的解释。
- 关RevIN/降维后的劣势同时改变训练难度、读出信息与归纳偏置，不能等同于发现新动力学机制。4D/token也不是4数状态瓶颈。
- 特权未来latent读出较好只定位可能的转移/对齐余量，普通GRU/ARX已有可行修复；一般decoder-aware/观测监督/两阶段问题已有文献。
- 两份波形改变了麦克风采集配置，已单列。其余8份仍有协议/采样速度变化，不能把全部误差解释为纯动力学OOD。
- 非神经/神经参数与实际更新预算不同；本轮是机制筛查，不是严格计算等价的新方法优越性证明。
- 部分模型在固定最大epoch附近仍取得最佳验证值，应保留有限训练预算这一替代解释，不能以其低分判定结构不可能。
- 原始float32通道聚合差异已被float64派生表修正；这不是一个新的科研方向。

最终接受的新颖方向仍为0。本轮降低前一候选作为通用框架主线的优先级；不再围绕该数据集增加loss权重搜索。
''')
    write('hypothesis_space','''# 竞争解释与当前状态

| Hypothesis | Direct evidence | Status |
|---|---|---|
| 角色感知条件模型整体不能拟合真实任务 | WorldTS32/ridge/GRU均显著胜持久性 | 本任务不支持 |
| WorldTS只是归一化均值，未来状态无用 | TRAIN-mean未来状态替换明显损害预测 | 对这些固定读出被否定 |
| 编码完全没有预测所需信息 |32D特权未来编码可读出更低误差 | 不符合32D结果；不作MI证明 |
| 潜在转移/训练目标仍有余量 | 实际预测与特权读出差、RPM分组差异 | 有定位证据；现有条件GRU/ARX已有效 |
| 所有误差由采集设置变化造成 |8份同配置波形单独评估仍有模型差异 | 不足以解释全部；时间与协议因素仍需考虑 |
| 新神经架构有必要 | 普通方法提供良好预测；成熟降维此前也能修复toy | 无直接必要性证据 |
| 自动未知角色发现构成新问题 | 当前真实任务提供角色；先前只是受控混合toy | 仍缺独立现实需求/新颖性证据，降低优先级 |

后续研究不应把当前较弱配置当作唯一对手。新候选须首先明确完整可用输入、目标能力、最近已有解法及可以改变判断的独立反证实验。
''')
    write('research_ideas','''# 研究路线调整

此前“输入状态与预测目标分开”不是可直接接受的新框架：WorldTS/Pythia已有角色/阶段分离，CDL等已有因果父变量原则。自动目标/角色发现只剩一条需要严格区分的窄线索，在未知混合toy有效，尚没有独立现实必要性。现在降低它的优先级，不强推为最终Idea。

真实任务上的RPM局部不足与潜在转移/读出对齐值得记录，但既有条件GRU/ARX提供修复，既有decoder-aware和机制一致性研究也直接相关。因此本轮不新立一个“换损失提升此数据集”的选题。

研究成功仍需一个跨系统可验证的问题和最小改进证据。下一轮应回到不同机制底座的真实开源实现，优先寻找角色已正确提供、普通条件状态模型也不能解释的残余；或者先证明未知角色需求在真实任务中普遍存在。不要重跑已完成的18个配置，不做测试集权重搜索，不把简单baseline成功删掉。

当前最终推荐方向：无。保留受控机制线索与负面证据，不满足顶会潜力要求时继续发现过程。
''')
    write('final_research_direction','''# 当前判断：尚不能接受一个新颖通用框架方向

**Research Question：** 尚无最终问题。本轮检验已有角色感知条件模型是否已经解决前轮候选所要解决的现实建模需求。

**Evidence：** 独立真实风洞26记录、18次神经训练、ridge/持久性及完整复核。WorldTS32总体有效，条件GRU/ARX也有效；未来latent路径替换明显降低预测质量。总体预测与RPM分组有差异，但已有普通方法提供有效修复。

**Existing Methods：** WorldTS/Pythia已分离条件、预测表示和读出；CDL/Denoised MDPs/ACF已有因果角色学习；主动子空间/导数降维已有压缩工具。当前证据不足以证明必须引入新框架。

**Proposed Idea：** 不把本轮结果包装成新方法。此前自动目标/角色发现保留为低优先级窄线索；尚需真实需求和精确closest-work区别。一般输入/目标分离不再作为创新主张。

**Falsification：** 本轮普通方法成功已反驳“前轮toy必然对应普遍真实世界模型缺口”的强解释。若最近工作已完整覆盖未知角色机制，或实际任务提供明确角色即可解决，则停止该路线的框架主张。

**Feasibility：** RTX5090已完整执行，数据/源码/权重/预测/日志保留；原生类未改动。新训练设置是跨任务迁移，非原论文排行榜复现。

**Next Experiment：** 不继续调当前数据集。先对不同机制的真实开源世界模型定位一个在输入角色/时间可用性/训练充分性排除后仍存在的残余，再固定新的独立反证协议。任何新方法都必须与已成功的条件GRU、系统辨识、已有表示/目标学习方法比较。

**Decision：** accepted final novel directions=0，目标ACTIVE且未完成。本轮产生否定性研究证据，不能据此宣称找到可投稿研究方向。
''')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained');labels=['Persist.','Ridge','WTS32','WTS4','WTS4 noRN','WTS e2e','GRU','MLP'];x=np.arange(8)
    for j,split in enumerate(splits):
        for ax,key,label in [(axes[0,j],'MSE','All-sensor standardized MSE'),(axes[1,j],'RPM','RPM standardized MSE')]:
            for i,method in enumerate(methods):
                rr=rows(split,method);values=[r['groups']['RPM'] if key=='RPM' else r['MSE'] for r in rr];mean=float(np.mean(values))
                ax.errorbar(i,mean,yerr=[[mean-min(values)],[max(values)-mean]],fmt='o',capsize=3,c='#3375a2')
            ax.set_yscale('log');ax.set_xticks(x,labels,rotation=50,ha='right',fontsize=8);ax.set_ylabel(label+' (log scale)')
            if key=='MSE':ax.set_title({'test':'Held-out random walks','wave_matched':'Wave protocols: matched acquisition','wave_acquisition':'Wave protocols: changed mic sampling'}[split],fontsize=11)
    fig.suptitle('Independent real task: existing conditional baselines remain effective',fontsize=14)
    fig.supxlabel('Unchanged native WorldTS components on a new task; no paper-leaderboard reproduction.\nBars:3 initialization ranges,not confidence intervals. Primary aggregates recomputed in float64.',fontsize=10)
    for ext in ['png','svg','pdf']:fig.savefig(out/('real_conditional_comparators.'+ext),dpi=170)
    plt.close(fig);(out/'result_tables.md').write_text(table+'\n\n'+path_table+'\n',encoding='utf-8')
    print(json.dumps(dict(reports=7,groups=len(groups),accepted_novel_directions=0)),flush=True)
if __name__=='__main__':main()
