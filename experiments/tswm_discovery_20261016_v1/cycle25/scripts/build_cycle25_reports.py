import json,pathlib,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/analysis';r=json.loads((OUT/'results.json').read_text());gate=json.loads((ROOT/'outputs/interface/results.json').read_text());lit=json.loads((ROOT/'outputs/literature/modern_np_manifest.json').read_text());models=r['models']
def find(collection,**kwargs):return next(a for a in r[collection] if all(a[k]==v for k,v in kwargs.items()))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in a)+' |' for a in rows])
null=[];active=[]
for m in models:
    a=find('contrasts',model=m,n=0,probe_index=0,method='native',scope='all');b=find('aggregates',model=m,n=0,probe_index=-1,method='no_update');c=find('aggregates',model=m,n=0,probe_index=0,method='native');old=find('aggregates',model=m,n=0,probe_index=0,method='is4096_full')
    null.append([m,f"{a['delta_crps']:+.6f}",f"[{a['bootstrap95'][0]:+.6f}, {a['bootstrap95'][1]:+.6f}]",f"{b['coverage95']:.4f}",f"{c['coverage95']:.4f}",f"{old['coverage95']:.4f}"])
    a=find('contrasts',model=m,n=0,probe_index=1,method='native',scope='untouched');b=find('contrasts',model=m,n=0,probe_index=1,method='native',scope='probed');active.append([m,f"{a['delta_crps']:+.6f}",f"[{a['bootstrap95'][0]:+.6f}, {a['bootstrap95'][1]:+.6f}]",f"{b['delta_crps']:+.6f}"])
fig,axes=plt.subplots(1,3,figsize=(14.4,4.6));colors={'native':'#B14D48','slr':'#427F73','is4096_full':'#476DA4'};labels={'native':'Native encoder','slr':'SLR, full covariance','is4096_full':'IS4096, full Gaussian'};y=np.arange(6)
for ax,p,scope,title in [(axes[0],0,'all','Uninformative probe'),(axes[1],1,'untouched','Sensitive probe: untouched mechanisms')]:
    for k,method in enumerate(colors):
        a=[find('contrasts',model=m,n=0,probe_index=p,method=method,scope=scope) for m in models];x=np.array([v['delta_crps'] for v in a]);lo=np.array([v['bootstrap95'][0] for v in a]);hi=np.array([v['bootstrap95'][1] for v in a]);ax.errorbar(x,y+(k-1)*.20,xerr=np.stack((x-lo,hi-x)),fmt='o',ms=4,capsize=2,color=colors[method],label=labels[method])
    ax.axvline(0,color='gray',lw=1);ax.set_title(title,fontsize=10);ax.set_xlabel('CRPS change vs no update (lower is better)');ax.set_yticks(y);ax.set_yticklabels([m.replace('unconstrained','U').replace('positive','P') for m in models],fontsize=9);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
for k,(method,label) in enumerate([('no_update','No update'),('native','Native encoder'),('is4096_full','IS4096, full Gaussian')]):
    a=[find('aggregates',model=m,n=0,probe_index=-1 if method=='no_update' else 0,method=method)['coverage95'] for m in models];axes[2].plot(a,y,'o-',label=label,ms=4,lw=1,color='#777777' if method=='no_update' else colors[method])
axes[2].axvline(.95,color='gray',ls='--',lw=1);axes[2].set_xlim(.68,1);axes[2].set_yticks(y);axes[2].set_yticklabels([m.replace('unconstrained','U').replace('positive','P') for m in models],fontsize=9);axes[2].invert_yaxis();axes[2].set_title('Marginal coverage after uninformative probe',fontsize=10);axes[2].set_xlabel('Nominal 95% interval coverage');axes[2].grid(axis='x',alpha=.2)
handles,labels2=axes[0].get_legend_handles_labels();handles.append(plt.Line2D([],[],color='#777777',marker='o',lw=1));labels2.append('No update (coverage panel)');fig.legend(handles,labels2,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.005),fontsize=9);fig.subplots_adjust(bottom=.23,wspace=.38,top=.88);fig.suptitle('Native nonlinear world model: update inconsistency does not imply uniform predictive harm',fontsize=12)
for suffix in ['png','svg','pdf']:fig.savefig(OUT/f'pendulum_information_update.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
common=f"""六个原作者 validation-best final-budget 检查点（16维系统latent，100000训练步，各变体3个seed）；64个新物理系统；历史完整前缀0/8/32；两种probe；三次独立评估MC；918行预测、306行MC聚合。没有本轮训练。同一cohort被所有检查点共享，不能算成384个独立系统。源版本`{gate['source_revision']}`。

科学指标是角速度CRPS、四个分支的联合energy/variogram、MSE和边际PIT覆盖率。四个查询共享同一个系统latent抽样。确定性cos/sin流形数据不使用full3D NLL作为科学评判；旧推断对照仍使用原decoder的full3D Gaussian似然，因此可以分开检验模型似然失配与推断失配。
"""
docs={}
docs['experimental_results.md']=f"""# 非线性多机制世界模型：实测与反证

**结论：不能把cycle24的标量失败推广为通用有害更新；本轮不支持开发新架构的必要性。** 原生更新确实改变了不该改变的真实物理预测律，但预测质量不一定恶化。必须区分这两件事。

{common}
## 冷启动，无信息probe

ΔCRPS=更新后减不更新；负值更好。区间先平均三次MC，再以64物理系统做4000次配对bootstrap；未做多重检验调整，单个checkpoint的区间不能当作6个独立数据集的确认。

{table(['checkpoint','ΔCRPS','95% bootstrap CI','prior coverage','native coverage','IS4096 full coverage'],null)}

三个unconstrained检查点CRPS点估计全部改善，其中两个区间完全低于0；三个positive版本点估计退化，其中只有seed1区间完全高于0。positive与unconstrained同时还改变stddev rescaling，不能把差异完全归因于正权约束。no-update自身覆盖率也并非恰为95%。

## 只探测一个机制后的其他机制

冷启动时，已证明物理probe只依赖quadrant1增益；其余三个增益在源prior中独立。各查询均处于原quadrant、没有速度clipping，reset时保持同一系统参数。

{table(['checkpoint','untouched ΔCRPS','95% bootstrap CI','probed ΔCRPS'],active)}

仅positive seed3的未探测机制退化区间完全高于0；并非所有模型均退化。所有六个模型在被探测机制上有明显CRPS收益。这不是“模型完全无法利用新信息”。

## 旧方法对照与计算

固定运行SLR完整协方差，以及1024/4096 defensive importance的加权粒子、对角Gaussian与完整协方差Gaussian矩投影；没有按测试指标挑预算。完整结果见[306行聚合CSV](outputs/analysis/aggregate_results.csv)、[所有原始分数](outputs/analysis/raw_scores.csv)、[配对差异](outputs/analysis/paired_contrasts.csv)、[ESS](outputs/analysis/importance_ess.csv)。三次MC不是三次新训练。

4096粒子在空历史无信息probe的ESS最小仍超过2500，信息probe下各checkpoint的ESS中位数约456–1463；并非整体粒子退化。长历史有少数低ESS，unconstrained seed3的n32敏感probe最低约1.59，不能据此断言高维Bayes失败。SLR/importance通常保持无信息probe预测接近不更新，并在敏感probe利用信息；原生摊销更新反而可能补偿prior误差，不能宣称“精确推断必定更低真实CRPS”。

主比较耗时{r['elapsed_seconds']:.2f}s；峰值GPU已分配内存{r['peak_gpu_bytes']/1024**2:.1f}MiB。此耗时包括该批预测、保存和打分，不是训练耗时或单次在线延迟。服务器D盘本轮完成后约48GB可用。全部原始大数组保留在服务器；本地保存紧凑镜像。

## 独立验证

NumPy独立物理核4480比较，最大误差1.78e-15。918预测行的15606数组/聚合复算最大误差7.60e-14；54编码器检查、144原生预测重放、72proposal/权重核查、36SLR独立NumPy求解通过。不同batch的float32网络重放最大1.79e-6；同batch严格权重误差6.11e-16，ESS2.19e-11，矩2.36e-14，SLR3.56e-15。

初版推断复核将batch改成8192，曾触发过严的零误差和重要性权重门限；保留失败日志。第二版仅将数值代数重放恢复原16384 batch；不同编码路径仍独立检查，允许3e-6语义等价误差。权重/矩/SLR代数门限反而收紧，未修改模型、数据、预测或分数。

raw results SHA256：`{r['raw_sha256']}`。统一分析JSON含两份通过的独立验证receipt及其哈希。

![固定比较](outputs/analysis/pendulum_information_update.png)
"""
docs['literature_review.md']=f"""# 文献与创新边界

[Explore the Context](https://proceedings.mlr.press/v130/achterhold21a.html)已提出跨系统latent、主动校准和共享latent多分支预测。本轮使用作者修正分支及发布权重；本审计是自主设计的机制测试，不是原文全流程复现。

[Bayesian Context Aggregation](https://openreview.net/forum?id=ufZN2-aehFa)已把context聚合建模为Bayesian推断。[Exploiting Inferential Structure in Neural Processes](https://proceedings.mlr.press/v216/tailor23a.html)，UAI2023，已有graphical-model先验、混合/Student-t和variational message passing；“换成更丰富后验”不足以构成创新。

[Flow Matching Neural Processes](https://proceedings.nips.cc/paper_files/paper/2025/hash/a92519f525c00085095fa41c5c46cdb5-Abstract-Conference.html)，NeurIPS2025，直接建模任意context/target条件律，论文3.3节明确讨论marginal/conditioning consistency，未保证其成立。[官方源码](https://github.com/danrsm/flowNP)已固定`{lit['revision']}`，51个小源码/配置/文档文件共{sum(f['bytes'] for f in lit['files'])}B，每个Git blob和SHA256核查。未声称已训练/复现FlowNP。其GP配置d_model64与正文笼统128描述不同，后续应以明确协议分别报告，不能悄悄换配置。

[CNP conditioning gap](https://arxiv.org/abs/2604.19312)，2026-04-21，讨论context重编码与条件化的KL差异及小context问题；OpenReview PDF自述TMLR04/2026，但论坛当前返回验证页，未独立确认接受状态。因factorized CNP joint与共享latent联合律不同，它不能直接替代本审计；但“更新一致性无人研究”已不能成立。

[Variational meta-learning inference for low dimensional neural system identification](https://arxiv.org/abs/2607.18965)，2026-07，已有摊销初始化+MAP/Laplace，不是本工作的创新。仅核查方法范围，未复现其JAX实现或独立认可所有数学/校准结论。

[Correct then Forecast: OSSM](https://arxiv.org/abs/2609.33566)，2026-09-27，已将传播与观测校正分开，并明确它与线性闭环SSM的关系是参数化而非扩展模型类。不能再把“分离传播和校正”本身包装为新世界模型。当前只阅读正文，没有执行其模型。

近邻方法给出了严格边界；本轮负面结果进一步削弱一般更新修复的创新必要性。不同范式的迁移必须单列，不能把Neural Process回归测试直接叫原论文world-model复现。联网下载在服务器两次connection-refused后改为本地取得相同公开源码，再逐文件哈希同步服务器；未关闭TLS或修改服务器代理设置。
"""
docs['baseline_registry.md']=f"""# 底座与复现登记

{common}
| 项目 | 固定版本/状态 | 科学角色 |
| --- | --- | --- |
| Explore-the-Context | `{gate['source_revision']}`；作者修正版发布archive已核查 | 原生模型的新机制审计 |
| unconstrained | run21/18/20；seed1/2/3；16D latent | 3个validation-best final-budget发布权重 |
| positive + rescaling | run24/23/22对应seed3/2/1 | 另3个发布权重；不是单因素ablation |
| no-update | 原先验/历史后验不加probe | 共享初态、query、模型及MC |
| SLR | 4096 prior draws，完整cov，残差+heteroscedastic noise | 成熟近似推断 |
| defensive IS | 1024及4096；各半prior/native更新proposal | 加权粒子及相同矩的diag/full投影 |
| FlowNP | `{lit['revision']}`；已核查51文件，未训练 | 后续现代范式候选，非已复现结果 |
| OSSM | 2609.33566v1；正文已读，未运行 | 排除“观测校正/传播分离”泛创新 |

物理数据：64个native context seed20262501起、独立trajectory seed21262501起，actions RNG22262501；32步完整历史；known controlled reset。保留原prior sampler行为：4096样本abs最小{gate['native_prior']['min_abs_gain']:.8f}，名义低半区[.5,1.25)实际出现率0。原作者训练环境同一sampler，未擅自修正为新分布。没有原训练parameter-ID完全去重声明。

所有seed、配置、权重hash、数据hash、原始分数、proposal/ESS和日志位于服务器cycle25。详细数值不重新训练、不覆写旧结果。大数组未拉到本地。
"""
docs['negative_evidence.md']="""# 反证与被否定的推广

1. 否定“cycle24六个scalar checkpoint退化，所以通用非线性世界模型的新观测更新必然退化”。本轮unconstrained三个checkpoint的冷启动无信息probe CRPS点估计均改善；其他variant也非全部显著退化。旧结果保留且仍成立，不把不同proper score/plant硬拼为相同实验。
2. 否定“物理上无信息的观测改变预测，所以真实预测分数一定更差”。原生近似prior有误差，无信息更新有时补偿误差。忠实性与平均分数必须分别检验，不能把低CRPS当作物理更新正确。
3. 不支持“16D非线性后验使成熟推断普遍失效”。SLR和defensive IS总体能工作，冷启动ESS充分；full/diag Gaussian与粒子均已运行。少数长历史低ESS须保留，不得夸大为普遍高维不可解。
4. 否定“positive版本比较是正权约束唯一变量”。它同时改变rescale_raw_stddev，不能做该因果归因。
5. 不把作者prior sampler的RNG共享现象当成新Research Gap。保留实际训练分布是协议要求；改成名义范围再评判会制造OOD。
6. 不将Bayes聚合、完整cov、粒子似然校正、观测校正/传播分离或flow条件采样重命名为新框架。相应近邻论文已经明确实现。
7. 保留两次远程公开源码下载失败和第一版复核的batch数值误差。没有因失败重启主比较、换checkpoint、改cohort或按测试选方法。

本轮最终接受方向0；当前工作线索不能据此声称顶会级新颖性或新方法必要性。
"""
docs['hypothesis_space.md']="""# 竞争解释与实测裁决

| 假设 | 区分实验 | 目前裁决 |
| --- | --- | --- |
| 无信息probe含隐含物理参数信息 | 源核独立递推、跨gain零控制、reset身份检查 | 排除；max差0，独立核4480检查通过 |
| 更新失真只来自1D线性toy | 原生16D latent非线性、四局部机制 | 物理预测律仍会改变，但质量损害不普遍；不能泛推新架构 |
| 受约束encoder稳定优于无约束 | 两套发布变体、三个seed，同cohort | 不支持；并有rescaling混杂 |
| 全部偏差是摊销encoder导致 | 同decoder SLR/IS与原生重编码 | 部分更新差异由encoder造成；旧方法可避免大量变化，decoder/prior也有误差 |
| 丰富后验是必须的新机制 | 粒子与相同矩diag/full Gaussian | 没有成立；简单近似足以构成强反证 |
| 现有方法完全不利用新信息 | probed quadrant与其他query分开 | 排除；所有6个原生checkpoint的probed CRPS显著改善 |
| 所有Bayes/particle结果可信 | 固定预算3MC、ESS、独立权重与SLR求解 | 总体数值通过；极少长历史低ESS仍限制局部结论 |
| 通用更新一致性是新题 | BA/structured NP/FlowNP/CNP gap/OSSM核查 | 否定该宽泛创新主张；更窄问题仍须新证据 |

不通过增加相同toy规模来维持原路线。任何继续的候选都应先给出当前成熟方法尚未解决的具体量，并在新的held-out系统/现代模型上检验。当前没有被实验证明需要的新架构。
"""
docs['research_ideas.md']="""# 候选问题与当前淘汰边界

当前没有接受的最终研究方向。不为了交付标题而包装Idea。

唯一仍登记的工作问题是：**时间序列世界模型如何在吸收新证据后，保持对其他物理未来的适当不确定性？** 它是框架层问题，数据只是判别载体。但目前只能把它列为低优先级线索：成熟近似推断已经能修复主要现象；无信息更新有时改善真实分数；更新一致性的概念已有强近邻。仅加coherence loss、Gaussian full covariance、粒子或Bayes校正不构成已证实新方法。

若继续探索，必须证明一种具体剩余能力（而非换数据集刷分）超出现有方法：同等信息/容量/预算、独立训练与检验数据、不同现代机制、真实任务或至少第二种物理机制。FlowNP源码已固定，ANP/NP/TNP等也在同仓库，可用于真正有区分力的范式迁移；目前未运行，不算证据。

本轮支持的是及时否定一般有害更新和新架构必要性的推广。该负面证据会用于后续选题，不把它勉强写成论文贡献。
"""
docs['final_research_direction.md']="""# 当前方向裁决：尚无通过证据门槛的最终Idea

**Research Question：** 世界模型应如何把新证据转化为对同一系统未来的正确更新？特别是，无信息证据及局部机制证据是否引发不应有的全局预测变化？这是跨任务状态/动力学建模问题，不是一个数据集的分数优化。

**Evidence：** cycle24有明确scalar冷启动质量退化；cycle25的6个原生非线性16D-latent检查点、64新系统、918预测行表明物理更新仍不严格忠实，但质量损害并不普遍。三种无约束checkpoint的零信息CRPS点估计改善，只有部分positive版本退化。所有模型都能利用局部敏感probe。

**Existing Methods：** SLR及1024/4096 defensive IS、diag/full矩投影均执行且通过独立数值检查，主要现象已有有效成熟对照。BA/structured NP/FlowNP及观测observer文献提供直接先例。当前没有证据证明这些既有方法仍不足以解决所观察的问题。

**Proposed Idea：** 暂不接受新的架构方案；不把普通条件化/滤波重新命名。该研究线索降优先级，不推荐现在按此撰写方法论文。

**Falsification：** “一般更新有害”及“非线性高维必需新后验机制”已被本轮反证削弱；若现代baseline在公平协议下同样满足可靠更新，放弃这条新方法主张。如果未来找到稳定残差，也必须排除模型似然误差、支持分布、MC误差及训练不足。

**Feasibility：** 单RTX5090已完成全部审计，主比较约328秒/476MiB峰值已分配显存；源码、原权重、数据、配置、日志、完整指标和复核脚本可重放。FlowNP近邻源码另已固定，无额外模型复现声明。

**Next Experiment：** 不追加相同检查点调参。先针对现代模型/最新框架筛出当前对照不能解释的具体能力，再用独立train/validation/test迁移对照检验；当前FlowNP/OSSM材料只能帮助划定问题边界，不能替代新的实验证据。若没有残差应换研究问题，而不是继续扩表。

最终接受研究方向数=0。总体自主研究目标仍ACTIVE、UNACHIEVED。WorldTS作为框架层次参照，不以某数据集优化为终点。
"""
for name,content in docs.items():(ROOT/name).write_text(content,encoding='utf-8')
receipt=dict(status='complete',files=[dict(path=k,sha256=hashlib.sha256((ROOT/k).read_bytes()).hexdigest()) for k in docs],plots=[dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in OUT.glob('pendulum_information_update.*')],analysis_sha256=hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),new_final_direction=False)
(OUT/'report_manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
