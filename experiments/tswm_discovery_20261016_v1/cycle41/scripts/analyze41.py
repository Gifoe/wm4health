"""Evidence synthesis and reports, not an accepted-idea certificate."""
import sys,pathlib,json,hashlib,datetime,csv
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'cycle29/scripts'));import runtime_profile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=R/'outputs';D=O/'controlled_v2';A=O/'analysis_v1';assert not A.exists();A.mkdir()
N=json.loads((O/'native_v1/qualification.json').read_text());C=json.loads((D/'result.json').read_text());V=json.loads((D/'verification_v2.json').read_text());assert N['status']==V['status']=='PASS' and V['cross_rows']==2304
fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
for hist in [1,8,32]:
 rows=[x for x in C['cross_summary'] if x['history']==hist];x=[r['anchors'] for r in rows]
 axes[0].plot(x,[r['alignment_entropy'] for r in rows],marker='o',label=f'Prefix length {hist}')
 axes[1].plot(x,[r['known_alignment_KL_regret'] for r in rows],marker='o',label=f'Prefix length {hist}')
axes[0].set(xlabel='Paired observation anchor prefix length',ylabel='Mean posterior alignment entropy (nats)')
axes[1].set(xlabel='Paired observation anchor prefix length',ylabel='Next-symbol KL to known-alignment oracle')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xticks([0,1,2,4,8,16])
fig.suptitle('Exact ordinary Bayesian mixture control: 128 fixed simulations',fontsize=11)
fig.savefig(A/'anchor_information.png',dpi=180);fig.savefig(A/'anchor_information.pdf');plt.close(fig)
table='| paired prefix | held-out prefix | entropy | oracle-relative KL | uniquely anchored fraction |\n|---:|---:|---:|---:|---:|\n'
for row in C['cross_summary']:table+=f"| {row['anchors']} | {row['history']} | {row['alignment_entropy']:.6f} | {row['known_alignment_KL_regret']:.6f} | {row['unique_alignment_fraction']:.4f} |\n"
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
summary={'utc':now,'native_fit_parameters':N['model_parameters'],'native_fit_loops':5,'native_minibatch_updates':15,'new_comparative_neural_fits':0,'control_rows':2304,'verified_control_cells':9216,'exact_gaussian_predictions':12,'accepted_final_directions':0,'goal':'ACTIVE_UNACHIEVED','sealed_real_test_scored':False,'synthetic_known_law_controls_scored':True,'priority_candidate':'anchored observation-interface x dynamics-module reuse','method_effect_not_yet_established':True}
(A/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
framing='本轮研究单位仍是可复用时间序列 world-model 框架；数据不是优化终点。选择一个后续开发候选，不代表已接受新颖机制或完成总目标。'
lit='''# C41 文献综合：状态的贡献必须落实到可复用能力

## 摘要与问题

本轮问：现有框架究竟证明了什么状态收益；怎样区分状态信息、转移近似和推断误差；哪种具体交互值得单卡实测。当前最值得开发的候选是有同步观测锚时，观测推断接口与动态模块的交叉复用。其创新尚未证实。

## 方法与证据分层

deep-research 的三个干净上下文分别检索主流、方法学反证、相邻建设性机制；只共享冻结 brief，不读 sibling notes，不访问服务器。主进程做源码/实验并在最后串行综合。代理原文和可核验直接来源保留在 mainstream_state41.md、critic_state41.md、constructive_state41.md。主进程另核对 PMLR/NeurIPS 原始页面和 MDN-VJEPA §8.6。论文事实、实验事实、候选假设分别陈述，未检索到完全相同组合不作为新颖证明。

## 1. 学状态、学未来 law 与学推断器是三件事

[WorldTS](https://arxiv.org/html/2609.31162v1) 与 [LatentTSF](https://proceedings.mlr.press/v306/yang26y.html) 支持 latent training 的可迁移预测配方，但 encoder 形状或预测改善本身不证明新增了关于隐藏过程的信息；[PSIM](https://proceedings.mlr.press/v48/sun16.html) 则已把未来统计量和递归过滤更新组成完整方法。因此“用未来监督一个可更新状态”不是新机制。状态坐标距离也不能跨任意 latent gauge 比较，必须以相同 observable future law 或充分性诊断作证据。

[MDN-VJEPA](https://arxiv.org/html/2610.13956v1) 明确把未来分布/轨迹/组合机制隔离于 encoder 学习，而 [LIFTS](https://arxiv.org/html/2610.03916v1) 实际联合学习生成与推断，覆盖了更完整的链条。前者 §8.6 的 learned-encoder 实证缺口不能被当作后者也没有推断能力；后者 synthetic calibration 无系统增益也不证明状态学习无用。它们共同要求把信息、模型近似和推断误差分开归因。

## 2. 复用两个接口的交互仍需要强完整对照

[NCDSSM](https://proceedings.mlr.press/v202/ansari23a.html) 已用 auxiliary observation variables 分离 recognition 与 dynamics，[modular NRI](https://papers.neurips.cc/paper_files/paper/2019/hash/b294504229c668e750dfcc4ea9617f0a-Abstract.html) 已用可重组动态模块做系统关系推断。两者不是互相覆盖：前者着重不规则/部分观测推断，后者着重动态图组成。候选需要验证有限数据下同时复用两个接口，是否在未见 view–mechanism 组合产生独立交互收益。普通 joint hierarchical SSM 可表达同一 factorization，必须给它同样锚、噪声模型和参数估计机会。若它吸收全部收益，不能把相同模型重新命名。

## 3. 界限与另一条备选

连接的训练组合图不保证目标 observable law 可确定；本轮精确置换反例和普通 Bayesian mixture 直接给出边界及正控。这个信息问题不是神经网络训练失败，也不要求恢复唯一物理状态。备选的粗路径—多层查询方向受 [mr-Diff](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d64740dd69bcc90ba225a182984b81ba-Abstract-Conference.html) 与 [generative MLMC](https://arxiv.org/html/2409.15511v2) 约束，现有粗到细生成和多层估计本身都不能冒充新算子；暂不与主线同时扩训。

## 结论：逐项回答研究问题

RQ1：已有完整方法支持预测归纳偏置、概率未来 law 和多任务推断；这些收益不能相互替代。RQ2：统一信息接口/精确 posterior/转移控制和 gauge-free observable metrics 可以分开归因，本轮已建立可执行量尺。RQ3：有锚的观测接口×动态模块复用有一个可证伪的有限数据交互问题；尚未有新方法效果，下一步必须让 ordinary joint model 先参与，而非先发明正则。
'''
reports={
'literature_review.md':lit,
'baseline_registry.md':f'''# C41 原生与控制注册

{framing}

- LIFTS固定commit e42430980e51bebf2e1b54d1beddeda7a31dee51：26个原始source文件SHA/Blob核验，C41只读组装后import/训练/采样。模型{N['model_parameters']}参数，作者minimal case1设置，5loops/15minibatch更新，80train/20interface check。执行lambda_enc=1，与论文主要实验0不同；只称作者示例资格，不称论文性能复现。
- 原生fit约{N['runtime_seconds']:.4f}s，进程peak {N['peak_allocated_GiB']:.4f}GiB。运行前其他GPU负载为 {N['gpu_context_before']}；耗时不作隔离性能benchmark。
- Imputation/forecast RNG重放、masked-value污染、forecast过去半段的future污染、observed硬替换，误差均0；评测过程权重不变。保存config/scaler/checkpoint/data/latent truth/samples/rawlogs。
- 精确MCAR Kalman与block Gaussian，有限100步burn prior，原生Cholesky有效covariance；raw float32 state arithmetic用声明的float64 LG model近似。坐标变换只比较observable law，不比较任意state坐标。
- 四状态补偿/ordinary transition修复，以及全部24 observation permutations的Bayesian mixture。真实模型参数作为privileged control给出，不冒充未知参数公平模型。
- NCDSSM/modular NRI/PSIM等本轮为文献完整对照，未执行。近期候选新方法尚未训练。
''',
'experimental_results.md':f'''# C41 实际运行结果

UTC {now}。{framing}

## 原生执行

LIFTS作者minimal例子实际训练、生成样本及接口核验通过，详见[qualification.json](outputs/native_v1/qualification.json)。本轮没有用5loops来比较准确率/收敛，也没有评分旧真实数据封存TEST。

## 控制A：精确belief与坐标规范

4record×4时间点的recursive vs block Gaussian mean/cov，以及4record×3horizon的observable gauge检查全部通过；后者独立dense24-history复算12项，max差 {V['max_dense_gaussian_abs_difference']:.3g}。这建立量尺，不证明任何学习模型状态有问题。

## 控制B：转移错误能伪装成校准收益

错误固定Q时，完美state posterior在h=1有KL regret .0320983，补偿性错误state达到约0；改回普通正确transition后全部h的regret为0。故不能从“校准state后预测变差”推出posterior inference有害。h=4/8的补偿仍不正确，全部9cells在[CSV](outputs/controlled_v2/compensation.csv)。这个经典控制不作为新贡献。

## 控制C：组合复用的信息边界及普通解

在3个训练view×dynamics cells，24个view对齐给出完全相同training law（max0）；held-out组合的transition law最大元素差 .18。普通Bayesian mixture同时使用anchor和held-out实际prefix，未丢弃新观测来制造失败。对128固定模拟：

{table}

![Ordinary Bayesian control](outputs/analysis_v1/anchor_information.png)

16个连续anchor观测只有60.94%模拟唯一确定对齐，不等于16个不同state；覆盖3个不同state时，四状态双射剩余映射即唯一，ordinary control逐cell预测差为0。不同held-out prefix长度对应不同下一目标；不可把三条曲线当同一target的单调定理。所有2304rows保留，包括有限样本非单调项。

独立permutation-matrix/product-likelihood重算{V['independent_cross_metric_cells']}指标cells，最大差{V['max_statistic_abs_difference']:.3g}；closed-form另核9compensation rows。评分的是冻结人工known-law控制，**旧真实数据TEST未评分**；不能笼统说本轮没有计算任何held-out synthetic score。未训练候选2×2，不声称方法收益。
''',
'negative_evidence.md':'''# C41 反证与排除

1. LIFTS原生实现可执行并提供随机预测/插补；不能依据某个“deterministic”注释说模型没有不确定性。
2. 精确posterior在固定错误transition下可能比补偿representation预测差；ordinary正确transition消除该效应。因此不给“校准与动力学冲突”反例冠以新机制。
3. train view–system图连通不意味着held-out observable law可确定；缺锚的对齐歧义是信息边界，不是已有模型算法失败。
4. 普通24-permutation Bayesian mixture在锚充分时能消除本例误差；候选必须越过它及joint hierarchical SSM等完整控制，而不是只胜过MAP/点估计或独立每cell模型。
5. V1 Gaussian控制因native float32返回Sigma的数值非对称在1.65e-9 mean check停止；V2按source实际Cholesky取有效对称cov，不改容差/科学效应。V1代码/failure保留。
6. 首个独立verifier把cross rows变量在Gaussian段覆盖，统计count误报5；其全部2304行数值已循环核查。V2改名并assert2304，输出9216cells；原verifier及输出保留。不把绿色status当覆盖证明。
7. 新候选没有comparative neural fit/真实多域效果，不能称发现了足以投稿的gap。旧C11/C15/C25/C32–40被关闭的动机不因本轮重新开放。
''',
'hypothesis_space.md':'''# C41 竞争假设与反证

| 假设 | 本轮证据 | 下一裁决 |
|---|---|---|
| 新环境缺的是可确定的信息 | 对齐反例完全同train law、不同held-out law | 正确的anchor/history coverage必须提供给所有模型；不以更大network解决信息不足 |
| 失败是latent坐标/弱probe | exact observable-law gauge control通过 | learned state用future law/excess proper risk；弱probe差不等于信息丢失 |
| 校准损伤来自错误transition的补偿 | exact4-state控制支持这一可能性，ordinary修复为0 | 不冻结错误Q当强baseline；共同坐标下区分posterior与transition |
| 双接口复用有有限数据交互收益 | 目前只有合法setting和ordinary positive control，无方法效果 | 同数据/推断/损失/预算2×2 + joint hierarchical SSM，检验held-out交互 |
| 所谓收益只是普通factorization | 是当前最强替代解释 | 若joint普通模型吸收全部收益，不新增模块换名 |

状态、推断和transition误差一般不独立相加。优先比较同一信息/同一generative model的inference，再检查observable law。新primitive无覆盖是压力条件，不承诺其一定泛化。
''',
'research_ideas.md':'''# C41 选择后续开发路线

优先候选A：**可更换的分布性观测接口 × 可重组动态模块**。目标是同一world model在有合理同步锚的sensor–system新组合上复用已学机制，测有限数据下的预测质量和适应成本。每个领域可分别训练同一框架；不强制万能checkpoint。当前证据只支持这是一个合法、可证伪的问题，尚未支持新架构。

备选B：共享粗路径marginal × 多层路径函数估计。已有mr-Diff/WaveletDiff/MLMC是完整近邻，须真正胜过cheap GRU/现成MLMC的真实误差—成本。暂不启动。

主流提出的law distillation×有限state预算保留为诊断控制，不优先命名新框架：ordinary teacher distillation、PSIM等重叠重，且不能重开此前被关闭的target/readout故事。三份独立notes保留开发细节与相互竞争解释；不得只挑建设性建议忽略反证。
''',
'final_research_direction.md':'''# C41 当前推荐：有锚的观测—动力学组合复用，待实测

**Research Question**：有限训练数据和同步观测锚下，一个时间序列world model如何把观测接口与动态机制分别学成可复用组件，并在未见sensor–system组合保持有用的概率推演？

**Evidence**：本轮已执行LIFTS原生例子；精确LGSSM oracle/gauge控制；四状态补偿/ordinary修复；128模拟2304cell的view–dynamics信息边界及全Bayesian普通解。直接证明的是组合coverage条件和ordinary能力，不是新方法优于现有方法。

**Existing Methods**：NCDSSM已有recognition–dynamics分离，modular NRI已有可组合transition/关系推断，LIFTS已有joint generation/inference，普通joint hierarchical SSM能表达对应factorization。当前不能宣称它们共同不足。缺少直接交互实证≠已证明全球首创。

**Proposed Idea**：以带不确定性的view-specific auxiliary likelihood对接可共享局部dynamic modules；同步anchor约束跨view接口，任务context负责module assignment/组合。待检验交互为二者交叉共享能否改善未见组合的有限样本预测，而不是增加一个latent loss。这个初始参数化可能就是ordinary structured SSM；若完全等价必须如实归类，不能另取名。

**Falsification**：strong joint SSM/NCDSSM+modular直接组合吸收所有收益；只改善seen cells；交互与0相容；优势来自更多参数/计算/锚信息；需要逐数据集换结构；或者测试组合根本不可确定。任一项否决相应创新主张。

**Feasibility**：单5090；先4维稳定系统/4views/多个primitive组合，线性层Kalman oracle，按view×system棋盘留出并给所有方法相同少量同步anchor。未知C/A只向oracle提供真值。0.3–2M initial neural budget，实际preflight后冻结。随后才考虑3个真实领域的自然sensor子集/系统划分，same implementation/rules；真实数据不声称共享物理latent语义。

**Next Experiment**：共同信息、同latent capacity、同loss/inference的2×2观测模块化×动态模块化，加入同样共享/噪声/联合估计的hierarchical SSM和强条件GRU等；先测held-out interaction与quality/cost，不能仅展示11最好。若11与ordinary joint model相同，这一阶段的输出是必要归纳偏置证据，方法创新仍需另证。不要为了保证结果正面排除ordinary control。

本轮=PROGRESS；最终接受方向0；整体ACTIVE_UNACHIEVED。当前推荐是开发优先级，完整候选训练及跨域效果尚未完成。
''',
'CURRENT_STATE.md':f'''# C41 complete, progress only

UTC {now}。LIFTS作者示例原生fit/接口资格通过，1.33M params/5loops，不称headline reproduction。精确控制和独立V2核验：2304cross rows/9216metric cells，12dense Gaussian future laws，9closed-form compensation cells。prior real TEST unscored；known-law synthetic controls scored as explicitly described。

优先候选A anchored view–dynamics composition；尚无候选方法comparative neural fit、最终方向接受数0、goal ACTIVE_UNACHIEVED。下一项是强普通联合模型参与的2×2，不是再来一轮泛泛查新，也不是给旧失败改名。所有旧模型/source/失败记录保留，没有公开发布或删除。
'''
}
for name,body in reports.items():(R/name).write_text(body,encoding='utf-8')
manifest={'utc':now,'reports':{k:hashlib.sha256((R/k).read_bytes()).hexdigest() for k in reports},'native_qualification_sha256':hashlib.sha256((O/'native_v1/qualification.json').read_bytes()).hexdigest(),'control_verification_sha256':hashlib.sha256((D/'verification_v2.json').read_bytes()).hexdigest(),'analysis_sha256':hashlib.sha256((A/'analysis.json').read_bytes()).hexdigest(),'accepted_final_directions':0,'goal':'ACTIVE_UNACHIEVED'}
(R/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
