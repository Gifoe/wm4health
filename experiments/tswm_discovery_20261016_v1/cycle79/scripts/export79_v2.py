"""Server-only evidence synthesis, complete compact publication snapshot."""
import sys,pathlib,csv,collections,hashlib,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common79_v1 import *
r=read(OUT/'PLAN_RESULTS79_V2.json');v=read(OUT/'VERIFICATION79_V2.json');pin=read(OUT/'SOURCE_PIN79_V2.json');inv=read(OUT/'PUBLIC_SOURCE_INVENTORY79_V1.json')
assert read(OUT/'run79_v2_exit.json')['exit_code']==0 and v['result_sha256']==sha(OUT/'PLAN_RESULTS79_V2.json')
assert len(r['cells'])==80 and len(r['records'])==3920 and v['records']==3920
suffix=[x for x in r['records'] if x['kind']=='suffix'];impulse=[x for x in r['records'] if x['kind']=='impulse'];assert len(suffix)==3360 and len(impulse)==560
stats=dict(suffix_rows=len(suffix),impulse_rows=len(impulse),suffix_window_edit_pairs=sum(x['n'] for x in suffix),impulse_window_edit_pairs=sum(x['n'] for x in impulse),suffix_violating_window_edits=sum(x['prefix_violating_windows'] for x in suffix),suffix_prefix_max_abs=max(x['prefix_max_abs'] for x in suffix),suffix_responding_window_edits=sum(x['edited_responding_windows'] for x in suffix),suffix_unchanged_window_edits=sum(x['unchanged_plan_windows'] for x in suffix),impulse_lagged_responding_window_edits=sum(x['prefix_violating_windows'] for x in impulse),impulse_lagged_max_abs=max(x['prefix_max_abs'] for x in impulse),impulse_lag0_responding_window_edits=sum(x['edited_responding_windows'] for x in impulse),factual_C53_max_abs=max(x['factual_vs_C53_max_abs'] for x in r['cells']),new_neural_fits=0)
def write(name,body):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(body)
sources=[
 ('LatentTSF','From Observations to States: Latent Time Series Forecasting','Yang et al.','ICML2026; arXiv2602.00297v2,2026-05-12','https://proceedings.mlr.press/v306/yang26y.html','Full paper, official proceedings and current source repository','Pointwise frozen AE and latent prediction; no general hidden-state/intervention guarantee','https://github.com/Muyiiiii/LatentTSF'),
 ('WorldTS','WorldTS: World Modeling for Multimodal Covariate-aware Time Series Forecasting','Zhu et al.','arXiv2609.31162v1,2026-09-25','https://arxiv.org/html/2609.31162v1','Full paper and current official repository independently verified by root','Multimodal conditional forecasting, two-stage dynamics/readout; future-known versus future-unknown contract','https://github.com/decisionintelligence/WorldTS'),
 ('COFE','Disentangling Coarse and Fine Latent Dynamics for Probabilistic Time Series Forecasting','Zhou et al.','IJCAI2026','https://www.ijcai.org/proceedings/2026/0602.pdf','Full primary PDF verified','Conditional block identifiability; forecast evidence does not verify physical-state assumptions','https://github.com/polars8948/COFE'),
 ('FIM-ODE','Foundation Inference Models for Ordinary Differential Equations','Hübers, Mauel, Berghaus, Seifner, Sánchez','arXiv2602.08733v2,2026-06-08; ICML2026','https://arxiv.org/html/2602.08733v2','Full paper, tutorial, public code tree and actual HF checkpoint metadata','Low-dimensional fully observed autonomous ODE; queryable vector field; prior/support limits and explicit evaluation-window checkpoint selection caveat','https://github.com/FIM4Science/OpenFIM'),
 ('ICL-SysID','In-Context Learning for System Identification: Probabilistic Formulation, Long-Sequence Handling, and Benchmark Validation','Rufolo, Piga, Pura, Forgione','arXiv2610.13411v1,2026-10-09','https://arxiv.org/html/2610.13411v1','Full paper, official source and v0.1 release asset metadata','WH prior, long input/output context and query warmup; excitation-only adaptation is parameter finetuning','https://github.com/PuraFilippo/sysid_benchmarks'),
 ('TTF','Test-Time Forecasting: Recurrent Depth Enables In-Context Dynamics Prediction','Lai, Quinn, Bao, Gilpin','arXiv2610.19552v1,2026-10-14','https://arxiv.org/html/2610.19552v1','Full primary paper incl Appendix C.2; public runnable code not established','Within-equation parameter OOD; exact EDMD eTTF comparator forbids unique recurrent-depth inference',None),
 ('OSSM','Correct then Forecast: Observer State-Space Models for Time Series Forecasting','Brachet, Clavier-Fremond, Ziani, Richard, Hudelot','arXiv2609.33566v1,2026-09-27','https://arxiv.org/html/2609.33566v1','Full primary paper; accessible supplementary implementation not independently established','Propagation/innovation split is prior art and has matched forecasting controls',None),
 ('DynaBase','A Minimal Interpretable Architecture for Zero-Shot Reconstruction of Dynamical Systems','Hemmer, Plaswig, Durstewitz','arXiv2607.14937v1,2026-07-16','https://arxiv.org/html/2607.14937v1','Full primary paper; no public executable repo verified','Two-parameter nearest-context successor control; prediction versus reconstruction objective mismatch is occupied',None),
 ('RP-GSSM','Maximum Likelihood Learning of Latent Dynamics Without Reconstruction','Hromadka, Biegun, Fox, Heald, Sahani','arXiv2505.23569v1,2025-05-29; older foundation','https://arxiv.org/abs/2505.23569','Primary paper and author repository examined by independent scouts; previous native project audit','Recognition-parametrized likelihood/latent inference; not a released general novel-observation simulator','https://github.com/gatsby-sahani/rp-gssm'),
 ('TSWM','On the Divergence of Accuracy and Mechanism Consistency in Time Series World Models','Zhang, Guo, Xu, Plotkin, Konz, Tan, Chen','arXiv2610.01842v1,2026-10-01','https://arxiv.org/html/2610.01842v1','Full paper, current commit125 Python files; C53 fixed trained models executed C79','Declared-sign consistency under plan perturbations; observational scores are not true interventional effects','https://github.com/HaochenZhang717/TSWM-Official-Repo'),
 ('DoTime','DoTime: A Synthetic Benchmark Generator for Interventional and Counterfactual Time Series','Thumm, Anthony, Chen','arXiv2607.27263v1,2026-07-29','https://arxiv.org/html/2607.27263v1','Primary full paper and package audit by critic; C10 previously executed','SCM family/noise coupling/version matters; intervention-PFN not always better than simple mean','https://github.com/thummd/dotime'),
 ('BeliefGraph-JEPA','BeliefGraph-JEPA: Structured Latent World Models for Action-Conditioned Time Series','Li, Ni, Tan, Chen','arXiv2610.05409v1,2026-10-04','https://arxiv.org/html/2610.05409v1','Full primary paper; no public implementation link established','Typed effect persistence/rollout already proposed; real-data perturbation is reliance, not causal truth',None),
 ('Response-Attenuation','Attenuated in-context identification in time-series foundation models: diagnosis under counterfactual inputs and repair by synthetic forced-system fine-tuning','Won','arXiv2610.08118v1,2026-10-06','https://arxiv.org/html/2610.08118v1','Full primary paper; code/configs on request, not a verified public implementation','Explicit input response attenuation, lag features, dither and matched synthetic-data repairs occupy generic response-deficit proposals',None),
 ('Pythia','Technical Report Pythia: Toward Foundation World Models for Multimodal Time Series','Dai et al.','arXiv2610.05240v1,2026-10-04','https://arxiv.org/html/2610.05240v1','Full primary paper; paper explicitly says code/model to be released','Separate frozen encoder/readout adaptation and large partially private multimodal pretraining; no single5090 public reproduction established',None),
 ('Meta-iLaD','Meta-iLaD: Identifiable Latent Dynamics via Meta-Learning of Dynamics Environments','Ye, Piya, Jiang, Wang','ICML2026','https://proceedings.mlr.press/v306/ye26p.html','Proceedings abstract and repository only; theorem assumptions not verified due PDF retrieval failure','State/environment/dynamics few-shot identification already a prior-art obstacle; code forthcoming','https://github.com/GoForIt-YYB/Meta-iLaD'),
 ('iCReN','Causal Representation Learning with Instantaneous and Lagged Relations via Nonstationarity','Yamada, Morioka, Kawahara','arXiv2610.03452v1,2026-10-02','https://arxiv.org/html/2610.03452v1','Full primary paper examined by adjacent scout','Invariant observation/structural functions and strong auxiliary variation assumptions; not arbitrary sensor/mechanism transfer',None),
 ('MTS3','Multi Time Scale World Models','Shaj et al.','NeurIPS2023; older foundation','https://arxiv.org/html/2310.18534v1','Full primary paper and official code checked by adjacent scout; previous project audits','Fast/slow latent inference and proprioceptive action-conditioned prediction; not demonstrated general image/control transfer','https://github.com/ALRhub/MTS3'),
 ('Video-SysID','Non-linear State-space Model Identification from Video Data using Deep Encoders','Beintema, Tóth, Schoukens','SYSID2021; arXiv2012.07721v3','https://arxiv.org/abs/2012.07721','Primary paper and pinned original code/data notebook checked by adjacent scout','History-state encoder with controlled video dynamics is established; broad changing-camera transfer not demonstrated','https://github.com/GerbenBeintema/SS-encoder-video')]
dump(OUT/'CITATION_REGISTRY79_V1.json',dict(utc=utc(),entries=[dict(key=k,title=t,authors=a,version_date=d,primary_url=u,verification=ve,supported_scope=s,code_url=c,author_results_not_our_replication=k!='TSWM') for k,t,a,d,u,ve,s,c in sources],metadata_is_not_full_theorem_verification=True,main_window=['2026-04-18','2026-10-18'],unverified_references_excluded=True))
survey='''# 时间序列世界模型：能力契约、机制证据与创新边界（C79）

## 摘要

本轮在已有实验记录之后重新开展三路独立检索，并用固定训练权重检查一个明确的时间一致性假设。审查以模型接受的信息、输出对象和外推条件为主轴。文献支持潜在预测、可查询动力学、受控辨识和结构化作用状态等不同能力；不能把它们压成单一排行榜。当前证据不足以接受一个新的机制研究方向。特别是，输入响应衰减、传播与校正分离、上下文辨识、短期误差与动力学重建的区别已有直接研究和普通修复。C79 的数值结果见独立实验报告；这里的论文结果均标为作者报告。

## 1. 问题与审查方法

RQ1：目前哪些通用能力经过了什么任务验证？RQ2：排除实现、训练、选择和信息不匹配后，哪些失败解释仍成立？RQ3：哪一个未解决机制能由单张 RTX5090 上的决定性实验支持？研究时间窗为2026-04-18至2026-10-18，并回溯必要基础工作。主流模型、批评/测量方法、相邻辨识与控制三路分别独立检索，再由主执行者核验关键原文、代码与当前发布元数据。

检索不是穷尽性系统综述，没有以文章数为成功标准。主流组沿 state/dynamics、JEPA、observer、in-context dynamics 展开；批评组从 intervention response、calibration、evaluation leakage、trajectory versus point forecasts 反向检索；相邻组从 nonlinear system identification、predictive states、meta-control、partial trajectories 补足。发现的 FIM-ODE、TTF 和最新辨识论文改变了原有“缺少动力学推断模型”的判断。文献遗漏不能证明研究缺口，方法存在也不能证明实际任务已解决。

一级证据为本项目保留源代码、配置和原始产物的实际运行；二级为核验原文的作者结果；三级为只核验摘要/仓库元数据的可用性或主张。Meta-iLaD 的定理假设未取得可核验全文，因此不以它证明任意未知观测映射的可辨识性。WorldTS 的论文没有代码链接不等于没有实现：本轮独立确认其官方仓库有454个文件，纠正检索组单靠论文链接作出的可用性遗漏。全部版本和范围保存在 citation/source registries。

## 2. 分类：按被要求回答的问题，而非架构命名

| 主契约 | 学习/推理信息 | 合格证据 | 不能自动推出 |
|---|---|---|---|
| 条件未来观测 | 历史目标、规定可用的协变量 | 同协议预测损失与校准 | 干预真值、可复用物理状态 |
| 上下文动力学辨识 | 多条状态轨迹或输入/输出上下文 | 新初值/新动作下的向量场和模拟 | 任意未知传感器、缺乏激励时的唯一识别 |
| 忠实模拟与递归推断 | 状态或含噪观测、递归状态/信念更新 | 长程统计、转移/滤波误差、真实响应 | 条件均值必须具有随机样本的粗糙度 |
| 机制或干预问题 | 已声明 SCM、行动和真值定义 | 配对干预响应、可识别结构 | 仅由未来变量打乱即可确认因果机制 |

每篇论文按主要被评测的输出归类；跨类结论只在输入和真值条件一致时转移。分类不是说同一模型不能具有多种能力。

## 3. 条件预测：潜在目标并非新颖性本身

[LatentTSF](https://proceedings.mlr.press/v306/yang26y.html) 将点态自编码表示冻结后训练预测器，而 [WorldTS](https://arxiv.org/html/2609.31162v1) 让多模态协变量参与潜在未来预测，再冻结动态模块训练读出。两者的实证目标都是预测；WorldTS 明确区分未来已知与未知协变量，不能因输入未来已知信息而指称泄漏，也不能把它称作恢复了行动因果效应。[COFE](https://www.ijcai.org/proceedings/2026/0602.pdf) 则提供概率粗/细潜在分解及有条件的块可辨识性，其生成与变化条件不能由真实预测分数反向认定已满足。

[Pythia](https://arxiv.org/html/2610.05240v1) 已研究冻结表示后的独立概率读出适配；其大规模、多设备、部分非公开语料设置不适合作为本机完整预训练复现。论文标注代码和模型待发布。因此普通读出训练必须先于“新的表示对齐机制”，也不能把其公开性边界视为技术缺口。本项目早先的 WorldTS 读出与继续训练控制已经要求同样的归因顺序。

## 4. 上下文辨识：已有真正可查询的动力学底座

[FIM-ODE](https://arxiv.org/html/2602.08733v2) 从含噪低维状态轨迹直接预测可查询向量场，超过了单纯序列续写；官方代码和约13M参数权重均可取得。其1至3维自主系统、先验与轨迹支持范围构成明确边界。它没有解决任意部分观测/行动世界模型，也已经明确讨论路径之外的欠定性。附录 D.1 将使用评估窗口选择的结果与训练损失选择结果分开披露，前者不能作为独立测试优势。本轮不把权重下载链接或参数量当作我们已运行过该模型。

与完全观测向量场不同，[Rufolo等的 ICL-SysID](https://arxiv.org/html/2610.13411v1) 使用输入/输出上下文和 query warmup；最新 CSTR 作者报告：零样本RMSE1.249，改用目标激励进行元微调后0.242，目标系统类适配后0.115。前一种“input-only”仍更新参数，不能误写成免费传感器校准，0.242也没有等于0.115。[发布的源码/权重](https://github.com/PuraFilippo/sysid_benchmarks/releases/tag/v0.1) 已在本项目早先上下文模型审计中使用，不能列为全新未碰的路线。

[TTF](https://arxiv.org/html/2610.19552v1) 把快速传播器的测试时拟合与慢特征训练联系起来；其 Duffing EDMD eTTF 明确在学习特征上闭式重估线性算子。因此“必须依靠循环深度”需要面对精确解和固定特征控制，论文标题不能替代该归因。当前没有核验到该文的可运行发布链接。三者共同反驳“目前没有从上下文推断动态规律的模型”，但没有证明跨观测/机制迁移已解决。

## 5. 模拟与递归状态：目标不同会改变胜负

[DynaBase](https://arxiv.org/html/2607.14937v1) 的双参数最近邻后继混合说明，短程误差和长程重建可能偏好不同解。它是复杂模型必须面对的普通控制，而非所有噪声、受控系统或高维观测都已解决的证明。DynaMix 的9次原生10k-step小范围拟合以及路径事件控制已在 C18/C19 执行；不能把那3个示例扩大成全论文54系统复现。

[OSSM](https://arxiv.org/html/2609.33566v1) 明确分开传播和观测校正，并有匹配预测控制；[MTS3](https://arxiv.org/html/2310.18534v1) 已结合快状态和慢任务状态；[RP-GSSM](https://arxiv.org/abs/2505.23569) 从另一条概率推断路线避免显式观测重建。这些使“加状态滤波”“区分快慢过程”本身缺乏创新性。相反，论文具体的观测、先验和实验范围仍须保留，不能仅凭经典结构存在断言实际通用迁移已经完成。

相邻领域的 [Video-SysID](https://arxiv.org/abs/2012.07721) 已从受控模拟视频联合学习历史状态编码和动力学，故“传统辨识只能处理标量”不成立；其单一视频系统也不足以证明任意视角或观测机制迁移。[Meta-iLaD](https://proceedings.mlr.press/v306/ye26p.html) 和 [iCReN](https://arxiv.org/html/2610.03452v1) 进一步占据环境适配/可辨识表示空间，但只有后者的完整假设在本轮核验：不变观测与结构、辅助变化及噪声条件，不能外推到任意同时变化的传感器和机制。

## 6. 机制评测：首先辨别真值与结构约束

[TSWM](https://arxiv.org/html/2610.01842v1) 的方向一致性是相对给定领域符号的扰动检查；[BeliefGraph-JEPA](https://arxiv.org/html/2610.05409v1) 已显式建模作用持久性和未来 rollout。真实数据里的未来 driver 扰动可说明模型依赖了输入，不能单独验证反事实响应。后者也明确指出实际学习的持久性参数不等同于物理时间尺度。普通因果时序编码是已有容量控制，不能把 Instant Gate 没有滞后响应重新包装成新机制缺口。

[DoTime](https://arxiv.org/html/2607.27263v1) 提供 SCM 干预/反事实真值，但版本和噪声耦合定义必须固定。作者结果中，干预 PFN 的优势并非跨家族普遍存在，且 Generic v1.0.0 有已披露的发散零替换。它不是真实世界因果效应的通用证书。[Won 的响应审计](https://arxiv.org/html/2610.08118v1) 已直接比较配对未来输入的真实效应，并给出滞后接口、dither、匹配强迫系统训练等修复；因此“预测好但响应弱”已被占据，仍需证明新残差超过这些控制。该文代码按需提供，不算公开可运行复现。

## 7. 跨类综合与反证

看似冲突的证据通常在信息契约不同：完整状态场推断强不意味着未知观测下的信念更新强；自主吸引子恢复强不意味着行动反事实正确；已知未来外生量带来的条件预测收益也不能归为因果识别。一个方法的理论条件明确停止适用，只能产生待测试问题；不能证明另一方法的失败原因。

本轮反证三个常见叙事。第一，通用模型没有辨识动力学：FIM-ODE 和已发布 ICL-SysID 是直接反例。第二，必须增加循环/复杂结构：TTF 自带闭式 EDMD，DynaBase 提供简约控制。第三，已测输入响应弱需要全新结构：Won 和既有因果编码给出普通数据/接口控制。反方向也不能过度：C75-C78 的标量修复没有解决高维、非线性部分观测、机制变化或持续保留的全部问题。

## 8. 开放执行问题与下一步

本轮接受的新机制方向数为0，不推荐为了交付数量硬列三个。下一项优先执行问题是：在同一观测、上下文和无未来标签选择的条件下，公开动力学推断模型的残差能否被普通局部辨识或上下文自适应消除？FIM-ODE 的已发布 base 权重、ODEBench 和原生 tutorial 可提供新底座；先复现一项原生查询/积分，再设计新初值而非已见轨迹重建的独立测试，必须分开原生复现、跨协议迁移和新设计实验。

这仍是筛选实验，不是已确立的创新。固定特征回归、足够激励、经典/神经辨识和仅用上下文的适配若解释残差，就淘汰；如果欠激励、路径外欠定或测试选择解释它，也不开发新架构。只有跨独立系统残差和普通修复后的机制证据成立，才进入方法阶段。C71 的完整原预算三种子模拟基线继续执行，在全部训练结束前不打开保留评估窗口。

## 9. 对研究问题的回答

RQ1：目前存在多种实证支持的能力，覆盖条件预测、上下文辨识、递归推断和规定 SCM 的干预；没有证据允许统一升级为无限制世界模拟。RQ2：普通修复和任务定义消除了当前多个候选解释，本轮原生检查需与之合并，不能把协议/接口现象当机制创新。RQ3：尚无达到用户标准的最终方向；下一步在可执行动态推断底座做新的反证，不延续已经修复的标量传感器线。总体研究目标仍为 ACTIVE/UNACHIEVED。

## 参考与复核

每个文内链接指向原文或作者发布。`outputs/CITATION_REGISTRY79_V1.json` 给出题名、作者、版本、核验层级和支持范围；`outputs/PUBLIC_SOURCE_INVENTORY79_V1.json` 给出当前源码 commit、文件哈希、HF revision/LFS大小和发布资产。除 C79/C53 和明确注明的项目历史外，论文数值是作者报告，不能计为本项目实验。未确认的代码可用性是检索限制，不是断言全球不存在该代码。
'''
write('literature_review.md',survey)
report=f'''# C79 原生计划时间一致性检查

当前40个固定训练模型、两个既有数据 bank 共80组回放完成。后缀动作检查{stats['suffix_rows']}行、{stats['suffix_window_edit_pairs']}个窗口/编辑对；早期预测超过容差的计数{stats['suffix_violating_window_edits']}，最大绝对变化{stats['suffix_prefix_max_abs']:.12g}。这是否定所测模型普遍存在提前响应的证据，不是模型真实因果响应已正确的证明。

## 任务与覆盖

所有 C53 的 CrossLinear/PatchTST × obs/AE-Concat/AE-Gate/AE-Gate-directional ×5种子，不替换失败臂、不新增训练。每个模型使用163 source-VAL 和803 new-crop 窗口；这些身份早已用于探索，因此不是新的独立测试。历史、未来外生量、类别行动和其他连续行动均固定；只将一个连续行动的指定后缀替为 source-TRAIN 的0.1或0.9分位数。分位数准确来自815×16训练未来行动值，不包含VAL。切点4/8/12，H16。目标未来标签不进入编辑构造。

三个非因果均值/因果累积夹具验证工具能检测提前变化并分辨滞后支持；它们不是模型结果。原生点态融合在当前时点使用未来行动，原文已有因果时序编码器。观察性内生行动的后缀也可能作为合法条件证据改变早期预测，只有假想干预计划契约才要求这里的 nonanticipation。

## 完整数值与重放

与历史 C53 已存 factual prediction 的最大差异{stats['factual_C53_max_abs']:.12g}；重复 factual 和未来目标破坏控制均保留。独立脚本重算所有3920行统计、{v['array_values_checked']}个数组值，并执行320个事先固定的原生编辑回放，最大重放误差{v['native_max_abs_replay_error']:.12g}。所有行、零动作变化和零响应均保留，无响应阈值筛除。表中的窗口对共享模型和窗口，不是独立物理系统数。

后缀被编辑区间超过数值容差的响应窗口/编辑对为{stats['suffix_responding_window_edits']}，原计划没有改变的对为{stats['suffix_unchanged_window_edits']}。单点 impulse 共{stats['impulse_rows']}行、{stats['impulse_window_edit_pairs']}个窗口/编辑对；lag0有响应{stats['impulse_lag0_responding_window_edits']}，lags1..15有响应{stats['impulse_lagged_responding_window_edits']}，最大滞后变化{stats['impulse_lagged_max_abs']:.12g}。lagged=0是当前 Instant 接口的支持结构检查，未对真实系统脉冲响应建立真值，不能称为已证实的物理记忆缺陷。

## 数值执行错误及修复

V1 在第一个 factual CPU 资格检查即停止：与旧GPU预测最大差5.835294723510742e-5，超过预置绝对/相对2e-5。该运行没有测动作编辑；其原代码、源 pin、夹具、日志和非零exit均保留。V2 在任何动作编辑分数出现前单独冻结运行时修订，使用原CUDA float32和batch128，不放宽容差、不重训、不替换模型。一次仅返回PID但未产生输出的启动尝试不计科学运行。GPU回放资格通过后才进行编辑。

## 决策

“晚期行动普遍改变早期预测”不被当前检查支持。相反，主论文明确的因果设计在所测两骨干/四臂中得到数值核验。未检查的 X-attention、其他骨干/编码器和真实干预响应不在结论范围。零滞后支持也不提供新颖性：原文已提供 causal decay/conv/attention 容量控制，最新响应衰减和 BeliefGraph-JEPA 文献也直接占据该问题。

不开发新的 Loop/编码器来修复这项未成立的缺口；不把当前结果解释为所有世界模型已解决。继续通用世界模型发现，优先核验已发布、可查询动力学底座及独立保留评估。最终接受方向0，目标 ACTIVE/UNACHIEVED。
'''
write('PLAN_CONTRACT_FINDING79_V2.md',report)
entries={
 'baseline_registry.md':'C53全40固定模型回放：2骨干×4臂×5种子，2个既有bank；无新训练。TSWM当前官方commit与保留125个Python源哈希一致。当前WorldTS公开仓库及FIM-ODE源码/权重已独立确认。FIM-ODE未在本轮执行，不能把发布元数据算成复现。ICL-SysID此前已有C28/C68/C69审计。C71原预算三种子任务继续运行，未提前打开保留评估。',
 'experimental_results.md':f"原生80组、3360后缀行和560单点行全部完成。后缀提前响应超过容差{stats['suffix_violating_window_edits']}，最大{stats['suffix_prefix_max_abs']}；C53 factual最大差{stats['factual_C53_max_abs']}。独立重算3920行和320编辑回放。完整结果见PLAN_CONTRACT_FINDING79_V2.md、per_edit_metrics79_v2.csv及哈希注册表；不是独立测试证据。",
 'negative_evidence.md':'当前两骨干/四臂不支持普遍提前响应假设。Instant接口单步作用没有滞后支持不能单独构成创新；原论文已有因果时序编码，最新文献已直接处理响应衰减。CPU与历史GPU的回放容差失败在动作扰动前被捕捉，修订只恢复原GPU环境。文献发现使通用模型无辨识能力、必须循环、普通响应衰减未被研究等叙事失效。C75-C78标量路线保持淘汰。',
 'hypothesis_space.md':'H1所测原生模型普遍提前响应：不支持。H2当前点态作用接口有有限时间支持：可由实现和单点扰动检查，但物理正确性未测。H3必须开发新结构：已有因果编码、lag特征、匹配强迫数据和EDMD为强竞争解释，未建立。H4通用迁移已解决：没有证据。下一轮只在匹配信息/上下文、充分训练和普通辨识后保留可复现残差。',
 'research_ideas.md':'最终推荐数0。优先的下一执行底座为FIM-ODE公开base模型与ODEBench，先复现官方向量场查询/积分，再做原生新初值能力及上下文-only适配的匹配控制。该计划是反证筛选，不是新Idea。不得把局部场欠定、低激励、可见状态和未知观测的差异重命名为创新。必须先同局部辨识/固定特征/普通适配比较，独立保留系统只用于一次确认。',
 'final_research_direction.md':'Research Question：哪些通用世界模型失败在匹配任务后仍需新机制？Evidence：本轮40模型时间一致性检查及三路文献核验未提供一个新正面缺口。Existing Methods：因果时序编码、输入滞后、强迫系统匹配训练、闭式EDMD、上下文辨识、observer和独立读出都已存在。Proposed Idea：无已验证的新机制。Falsification：普通辨识、充分激励、上下文适配或正确测量解释残差，就淘汰。Feasibility：13M参数FIM-ODE公开base权重、ODEBench、原生代码，单5090可先做推断；尚未执行。Next Experiment：先通过原生查询/积分资格，再冻结同信息新初值/普通辨识控制；完成C71原预算评估。ACTIVE/UNACHIEVED，不能宣称研究任务完成。'}
for name,body in entries.items():write(name,'# '+name[:-3].replace('_',' ')+'\n\n'+body+'\n\n完整边界见 literature_review.md 与 PLAN_CONTRACT_FINDING79_V2.md。\n')
with (OUT/'per_edit_metrics79_v2.csv').open('x',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(r['records'][0]));w.writeheader();w.writerows(r['records'])
dump(OUT/'RESULT_SUMMARY79_V2.json',dict(utc=utc(),**stats,cells=80,records=3920,source_train_quantiles=pin['quantiles'],raw_result_sha256=sha(OUT/'PLAN_RESULTS79_V2.json'),independent_confirmation=False,new_training_fits=0))
dump(OUT/'ARTIFACT_REGISTRY79_V2.json',dict(raw_arrays_remain_on_server_G=True,cells=r['cells'],result_sha256=sha(OUT/'PLAN_RESULTS79_V2.json')))
dump(OUT/'SEARCH_AND_ADJUDICATION79_V1.json',dict(utc=utc(),main_window=['2026-04-18','2026-10-18'],perspectives=['mainstream state/dynamics/JEPA/observer/in-context models','critic interventional response/evaluation/point functional contract','adjacent nonlinear system identification/predictive states/meta control'],independent_clean_context_scouts=3,scouts_performed_native_runs=False,root_performed_native_probe=True,queries_reconstructed_from_executed_search_themes_not_exact_export=True,new_important_prior_art=['FIM-ODE','TTF2610.19552','Won2610.08118','Rufolo2610.13411'],root_corrections=['WorldTS official repository exists, absence of paper code href is not absence of code','DynaMix/ICL-SysID/PCFM already examined in project, not untouched baselines','FIM-ODE v2 author order follows primary v2 rather than current repository bibliography','No method accepted from publication absence or explicit limitation alone'],unresolved=['No accepted final mechanism','FIM-ODE native execution not yet performed','C71 all-three-fit reserved assessment pending'],rejected_inferences=['Conditional point smoothness is not stochastic simulator failure','Observational action conditioning is not automatic causal identification','Old scalar ordinary repair is not all-world-model impossibility/solution']))
dump(OUT/'CLOSURE79_V2.json',dict(utc=utc(),status='C79_CLOSED_NO_GENERAL_ANTICIPATION_FAILURE_NO_NEW_MECHANISM_ACCEPTED',statistics=stats,result_sha256=sha(OUT/'PLAN_RESULTS79_V2.json'),verification_sha256=sha(OUT/'VERIFICATION79_V2.json'),public_source_inventory_sha256=sha(OUT/'PUBLIC_SOURCE_INVENTORY79_V1.json'),v1_cpu_qualification_failed_before_action_edits=True,unchanged_tolerance_after_runtime_amendment=True,current_TSWM_commit_matches_C53=True,independent_confirmation=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0))
status=f'''# C79 当前研究状态

40个固定TSWM模型、2个既有bank的80组原生GPU回放结束；与C53 factual最大差{stats['factual_C53_max_abs']}。3360后缀编辑行、560单点编辑行全部保留，提前响应超容差{stats['suffix_violating_window_edits']}。独立重算3920行与320原生编辑回放。结果只覆盖当前两骨干/四臂和已见身份，不是新独立系统的因果检验。

完整中文文献综述、七类台账、原生报告和公开源码/权重元数据在cycle79。WorldTS官方实现已独立确认；最新FIM-ODE、TTF、输入响应衰减审计和ICL-SysID进一步限制创新范围。没有新机制达到接受标准；总体目标ACTIVE/UNACHIEVED，接受方向0。优先下一执行为公开FIM-ODE原生查询/积分和匹配普通辨识控制，不延续已被普通方法修复的标量线。C71原预算三种子训练继续，未提前打开保留评估。

数据/权重/数组保留服务器G的授权项目子目录，代码和紧凑证据上传Gifoe/wm4health，无本地结果/模型/数据同步。旧成果和其他项目保持原样。
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V79.md').open('x',encoding='utf-8') as f:f.write(status)
files=['literature_review.md',*entries,'RESEARCH_BRIEF79_V1.md','PROBE_PROTOCOL79_V1.md','PROBE_AMENDMENT79_V2.md','PLAN_CONTRACT_FINDING79_V2.md']
files+=['scripts/'+p.name for p in sorted((ROOT/'scripts').glob('*.py'))]
files+=['outputs/'+name for name in ['SOURCE_PIN79_V1.json','FIXTURES79_V1.json','SOURCE_PIN79_V2.json','FIXTURES79_V2.json','CITATION_REGISTRY79_V1.json','PUBLIC_SOURCE_INVENTORY79_V1.json','SEARCH_AND_ADJUDICATION79_V1.json','RESULT_SUMMARY79_V2.json','ARTIFACT_REGISTRY79_V2.json','VERIFICATION79_V2.json','CLOSURE79_V2.json','per_edit_metrics79_v2.csv','run79_v1_launch.json','run79_v1_exit.json','run79_v1_stdout.log','run79_v2_launch.json','run79_v2_exit.json','run79_v2_stdout.log']]
patterns=json.loads(os.environ['C79_PRIVATE_EXPORT_PATTERNS_JSON']);dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);pub=[]
for rel in sorted(files):
 raw=(ROOT/rel).read_bytes()
 for needle in patterns:assert needle.encode() not in raw,(rel,'private export pattern')
 b=raw.decode('utf-8-sig').encode();target=dest/'files'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b);pub.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(target)))
dump(dest/'manifest.json',dict(utc=utc(),files=pub,file_count=len(pub),bytes=sum(z['bytes'] for z in pub),large_arrays_weights_raw_data_included=False,complete_compact_edit_metrics=True,result_sha256=sha(OUT/'PLAN_RESULTS79_V2.json'),verification_sha256=sha(OUT/'VERIFICATION79_V2.json'),accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0))
print(json.dumps(dict(status='C79_SNAPSHOT_READY',files=len(pub),bytes=sum(z['bytes'] for z in pub),manifest_sha256=sha(dest/'manifest.json'),statistics=stats)),flush=True)
