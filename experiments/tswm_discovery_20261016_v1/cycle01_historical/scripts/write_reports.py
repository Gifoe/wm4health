"""Generate private research ledgers from saved results; no model selection."""
import hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def write(n,s):(ROOT/n).write_text(s.strip()+'\n',encoding='utf-8')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    analysis=read('analysis_summary.json');conf=read('shared_parameter_confirmation.json');refine=read('neural_range_refinement.json')
    pins=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'upstream').glob('*/SOURCE_PIN.json'))]
    (OUT/'source_registry.json').write_text(json.dumps(pins,indent=2),encoding='utf-8')
    pinrows=[[p['repo'],p['revision'],p.get('download_scope','source archive')] for p in pins]
    screenrows=[]
    for r in analysis['synthetic_screen']:
        fmt=lambda k:f"{r[k]['mean']:.6g} ± {r[k]['sample_std']:.3g}"
        screenrows.append([r['model'],r['parameters'],fmt('factual_mse'),fmt('effect_nmse'),fmt('late_tail_gain')])
    crows=[]
    for r in conf['summaries']:
        if r['horizon']==64 and r['training_n']==16384:
            crows.append([r['method'],f"{r['point_coverage']:.2%}",f"{r['whole_family_coverage']:.2%}",f"{r['mean_width']:.5f}",f"{r['sign_certifiable_fraction']:.2%}",f"{r['wrong_signed_claim_fraction']:.2%}"])
    primary=analysis['shared_parameter_primary_cell']
    write('literature_review.md',r'''
# 文献、代码与创新边界（2026-10-16）

检索重点为 2026-04-16 至 2026-10-16；保留更早的不可识别性、系统辨识、可达性基础工作。以下是定向检索，不是系统综述或“没有找到即不存在”的证明。公开代码快照见 `outputs/source_registry.json`。论文证据、代码证据与本轮实验分别标明。

## 起点与排除范围

| 工作 | 已核查的内容 | 对当前判断的约束 |
|---|---|---|
| [LatentTSF](https://arxiv.org/abs/2602.00297) / [代码](https://github.com/Muyiiiii/LatentTSF) | 论文、README、AE/latent predictor 代码；旧复现报告 | AE 重建/潜空间预测本身不是新研究问题；本轮未重训 |
| [WorldTS](https://arxiv.org/abs/2609.31162) / [代码](https://github.com/decisionintelligence/WorldTS) | 数值 core、训练阶段、真实旧 checkpoint；本轮执行 | 本版本未来条件只经局部 encoder 与 decoder；该事实不能推广到所有 world model |
| [Accuracy and Mechanism Consistency](https://arxiv.org/abs/2610.01842) / [代码](https://github.com/HaochenZhang717/TSWM-Official-Repo) | 论文全文/实现；标题现为 On the Divergence of Accuracy and Mechanism Consistency in Time Series World Models | 已研究准确性与机制方向脱钩、方向监督、不同 plan encoder；不能把这一大命题重新命名为创新。代码默认 rho 与论文设置须显式对齐 |
| [RP-GSSM](https://arxiv.org/abs/2505.23569) / [代码](https://github.com/gatsby-sahani/rp-gssm) | 论文/README/JAX 实现 | 已研究不依赖重建的 latent dynamics；本轮未执行 JAX 训练 |
| [PRO-DYN](https://arxiv.org/abs/2507.15774) / [代码](https://github.com/ARBrachet/PRO-DYN) | 论文与代码结构 | 动态结构本身已有路线；未执行新复现 |
| [DoTime](https://arxiv.org/abs/2607.27263) / [代码](https://github.com/thummd/dotime) | prior、pairing、evaluation、PFN masking、suite 版本 | 默认 `generate_pair` 为 independent-noise interventional；显式 counterfactual 才共享噪声。新版 suite 已区分 level/effect sign，不能声称其未区分 |
| [EpiCF-Bench](https://github.com/complex-ai-lab/epi-cf-benchmark) | README/tree/数据协议 | calibrated ABM 的模拟反事实有明确结构真值；本轮未下载大型数据，未作排行榜比较 |
| [TIME](https://arxiv.org/abs/2602.12147) / [代码](https://github.com/zqiao11/TIME) | 论文/README/tree | zero-shot forecasting 不是干预或个体反事实真值 |

## 已直接淘汰的宽泛 Idea

| 先前可能的命题 | 直接近邻 | 为什么不能作创新主张 |
|---|---|---|
| 给条件预测器增加动作记忆 | [BeliefGraph-JEPA](https://arxiv.org/abs/2610.05409)；[Attenuated in-context identification](https://arxiv.org/abs/2610.08118) | 前者有持续的 latent action effect；后者已做动态响应诊断、激励对照、forced-system 微调及 ARX/N4SID/NARX 对照。本轮 FIR/GRU/ARX 也已修复 |
| 动作激励不足导致识别失败 | [On Identifiability of Controlled World Models](https://arxiv.org/abs/2607.22430) | 已有 excitation/identifiability 理论。白噪声神经网络 factorial 本轮未跑，不能把它写成验证结果 |
| 因果未来动作 mask | [Parallel Predictive World Models](https://arxiv.org/abs/2610.08627) | action-prefix parallel prediction 已有 |
| belief 而不是确定性 latent | [UWM-JEPA](https://arxiv.org/abs/2605.25313)、RP-GSSM、[MDN-VJEPA](https://arxiv.org/abs/2610.13956) | hidden-state uncertainty、multimodal/coherent rollouts 已有；后者提交日期应以原页面为准，本轮不依赖搜索摘要的异常日期 |
| 分离观测校正与自主传播 | [OSSM / Correct then Forecast](https://arxiv.org/abs/2609.33566) | observer / simulator 解耦已有 |
| 物理加残差潜状态 | [Phys-JEPA](https://arxiv.org/abs/2606.16076) | 不应重新包装为新结构 |
| 非平稳 Koopman / coarse-fine 潜动态 | [KUMA](https://proceedings.mlr.press/v306/xiong26j.html)、[COFE](https://www.ijcai.org/proceedings/2026/602) | 必须区分已有非平稳与多尺度建模 |
| 预测分布随新观测形成 martingale | [Calibrated Probability Forecast Sequences](https://arxiv.org/abs/2606.31621)、[MPNN](https://arxiv.org/abs/2506.11898) | forecast-revision consistency 已有，不是新原则 |

[Pythia](https://arxiv.org/abs/2610.05240) 与 [Meta-iLaD](https://github.com/GoForIt-YYB/Meta-iLaD) 也已筛查；后者所查公开仓库只有论文/海报而无可运行训练代码，不能写作已复现。综述条目深度不同：只有参与决策的近邻作了机制级阅读；不是所有论文都逐式复推。

## 当前反事实区间问题的最近竞争者

1. [CoMP, ICML 2022](https://proceedings.mlr.press/v162/foster22a.html) 已明确用条件相关 Gaussian latent rotation 展示相同拟合分布、不同反事实。**本轮旋转反例不是新发现**。代码固定快照已保存。
2. [Counterfactual Analysis in Dynamic Latent State Models, ICML 2023](https://proceedings.mlr.press/v202/haugh23a.html) 已作动态隐藏状态部分识别与优化界；不能宣称“首次动态反事实界”。
3. [CSM/APID, NeurIPS 2023](https://arxiv.org/abs/2306.01424) / [代码](https://github.com/Valentyn1997/CSM-APID) 已对连续结果部分识别，利用 level-set curvature 与生成模型拟合。本文主要处理二元 treatment/连续 outcome 的静态查询；代码 curvature 实现断言 dim_u=2。零曲率也不能随意解释成唯一点识别。它不是现成的冻结序列模型置信区间接口。
4. [Robust Counterfactual Inference in MDPs](https://arxiv.org/abs/2502.13731) 已对 compatible SCM 的离散动态反事实给界。
5. [Lally et al., 2026-10-14](https://arxiv.org/abs/2610.20210) / [代码](https://github.com/jessicalally/robust-cf-inference-beyond-mdps) 已联合处理 transition 统计不确定性与 causal ambiguity，构造 IMDP 并作悲观 value iteration。**联合处理两类不确定性不能作为我们的首创**。论文定义离散 state space；本轮源码矩阵/Julia IMDP 路线也吻合。未执行其完整实验，未声称超越其结果。
6. [Twin Rollouts](https://arxiv.org/abs/2608.08982) 已有共享噪声的视频分支；共享 seed 本身不是创新。
7. [Flow*](https://github.com/chenxin415/flowstar) 已有 Taylor-model flowpipe、symbolic remainder 与参数依赖的可达性计算；本轮 first-order + remainder 是这些标准思想的特例实现，**不是新验证理论**。尚未完成与 Flow*/CORA 之类成熟工具的预算匹配比较。
8. [Non-rectangular robust MDPs](https://arxiv.org/abs/2305.19004)、[其 Lp 对偶方法](https://arxiv.org/abs/2502.09432) 已研究共享模型不确定性与 rectangular approximation 的困难；参数跨时共享本身也不是创新。

## 剩余问题与创新置信度

尚值得检查的具体交集是：**对冻结的连续、非线性动态模型，在明确的噪声耦合敏感性类中，同时保留 factual abduction、跨分支/跨时间共享参数，给出既有覆盖保证又能支持非平凡效应判断的区间。**

相对所核查公开实现，尚未找到能直接完成该接口的现成方法。但“没有找到”不等于证明不存在；通用可达性工具可能已经足够。当前问题证据强于方法创新证据。不得宣称首次、不应据此直接立项写顶会论文。下一轮必须把通用可达性/静态 APID 的动态组合当作竞争解释，而不是只和松的 Lipschitz 界比较。
''')
    write('baseline_registry.md','''
# Baseline 与复现登记

所有新增文件位于独立目录。旧项目、数据与 checkpoint 只读使用，未删除或修改。以下三组不能混成排行榜。

## A. 历史原论文复现与负面结果（本轮重新查阅报告）

旧紧凑证据在 `_longhorizon_loop_dependency_audit_worktree/experiments/`，其 HEAD 为 `3849f8add3a546d4598229c0a1d19444b1a04ce8`；服务器原目录继续保留。

| 旧实验 | 状态/结论 | 本轮用途 |
|---|---|---|
| baseline_reproduction_seed2021_v1 | 24/24 历史 cells；22 MATCHED、2 CLOSE；21 fresh、3 verified reuse | 确认已有 LatentTSF/DAG/WorldTS 复现，避免重做；这些不是本轮新增结果 |
| conditional_latent_loop_seed2021_v1 | NO-GO | 不重启泛化的条件 loop 路线 |
| worldts_future_response_loop | INCONCLUSIVE | 不把不足证据改写成正面路线 |
| worldts_readout_alignment_feasibility_seed2021_v1 | NO-GO complex loss | 不重复 readout-aware loss 微调 |
| worldts_stage1_state_selection_audit_seed2021_v1 | NO_GO_NEW_STATE_SELECTION_METHOD | 不重复 epoch-1/state selection 假设 |
| neso_worldts_online_v1 | NO-GO persistent updater | 普通线性修正已解释大部分收益；真实在线 vintage 可用性仍有边界 |

## B. 本轮执行

| 对象 | 数据/信息 | 状态与公平性 |
|---|---|---|
| 4 个旧 WorldTS checkpoint：NP/PJM × H24/H360 | 各 8 个 TRAIN context，具名列恢复与训练分区标准化 | 冻结权重的有限差分/精确梯度审计；不是新增 forecast accuracy 复现，更不是电力数据上的因果干预 |
| WorldTS native two-stage / identical-core e2e / ordinary FIR | 新随机受控标量系统；history64、future actions48 | 各 3 seeds，新诊断迁移；同 train/val、验证选 ckpt；two-stage 训练阶段更多，不宣称严格 compute match |
| GRU | 同上，同输入、验证选择 | 14,881 参数，比 native 39,361 更小；普通强基线 |
| ARX / ridge | 同上训练 transition，验证选 ridge | 非神经系统辨识对照；系数从数据估计，没有使用真参数 |
| Neural Gaussian SSM | 独立 2D 非线性随机系统，12,288 train transitions | 3 seeds，验证均值 MSE 选 ckpt，3 个冻结模型用于独立 dev/confirmation 与后续 fresh cohort；自建诊断模型，不是 RP-GSSM 复现 |
| Affine Gaussian regression | 256/2,048/16,384 transitions，各 64 独立训练重复 | OLS 全局最优，排除神经训练不充分解释；两套独立实验各 192 fits |
| 区间对照 | same fitted kernel / confidence set / test trajectories | exact gauge-only、fixed-noise statistics-only、separate/paired norm tubes、shared-parameter Taylor tube、无 remainder 消融、oracle-known-kernel family |

RP-GSSM、PRO-DYN、TSWM、EpiCF、TIME、CoMP、CSM/APID、Lally IMDP 本轮只作代码/协议审查，没有伪报执行或复现成功。DoTime 的版本/配对/指标代码已查，但当前新模拟数据不是其官方 suite；不把合成系统成绩写到 DoTime 排行榜。

## C. 源码固定版本

'''+table(['Repository','Exact commit','Download scope'],pinrows)+'''

WorldTS 在本轮服务器的 `upstream/WorldTS` 中只读导入。其余近邻代码主要用于核对创新边界。权重/数据哈希见 `outputs/existing_future_support.json`、`outputs/provenance.json`。依赖版本、GPU、脚本哈希均已记录；新环境只通过只读路径复用旧依赖，没有写入旧环境。
''')
    write('experimental_results.md','''
# 实验事实与适用范围

本轮在服务器 RTX 5090 32GB 完成。数据与脚本可生成；没有下载不必要的大型信号集。所有 discovery 与 confirmation 定义见各 plan JSON。结果没有进入超参数或 checkpoint 选择。

## E1. 已训练 WorldTS 的条件可达范围

`outputs/existing_future_support.json`：4 个旧官方 core checkpoint，各 8 个训练 context。NP 输入条件支路的最大支持长度为 5；PJM 为 9。所有位置探针在理论支持之外的数值变化为精确 0，独立 autodiff 的远距梯度也为 0。保存 data/config/checkpoint hash 与权重前后 hash。该结论是本版本结构限制；输入为 observational covariates，所以不能解释为电价的真实因果效应。

## E2. 新随机动作诊断迁移，3 seeds

系统 `x[t]=.94*x[t-1]+.06*u[t]+noise(.005)`；actions AR(1) rho=.97。独立 train2048 / validation256 / discovery256 / 尚未使用的原 confirmation512；L64/H48。脉冲幅度 .5、位置0/12/24，用相同噪声的精确配对真值。effect NMSE=sum(effect error²)/sum(true effect²)；late gain 为脉冲5步之后对真值的投影系数。

下表 mean ± sample SD over 3 independent training seeds；先对3个 pulse位置平均，再跨 seed 汇总。

'''+table(['Model','Parameters','Factual MSE','Effect NMSE','Late-tail gain'],screenrows)+'''

四组 whole-plan positive-shift sign 都是100%；这是我们自定义的粗诊断，**不是 TSWM 论文的 MC 指标**。完整 pulse 原始量、training curves、峰值显存与训练秒数见 `outputs/runs/*/`。普通 FIR 只增加768参数，GRU更小；因此“增加动作记忆”作为新机制被淘汰。ARX discovery MSE≈0.0001863，effect NMSE≈1e-9。白激励 ARX 是控制，白激励神经网络没有执行，不能声称完整 excitation factorial 已完成。

## E3. 已知 coupling 不可识别性的复核

`outputs/coupling_probe.json`：2D controlled affine Gaussian，3训练样本量×3训练seed×3H，共27个描述记录。全局 action-dependent SO(2) rotation 精确保留 conditional kernel；最大 NLL 数值差约2.7e-15。N16384/seed303/H64 的 mean family width=.7580，64.84%轨迹的效应符号在此家族内不确定；4个固定 gauge 的符号分歧50%。ordinary 3-fit CF range均值=.02748，不能代表结构 coupling range。16个额外 paired simulator transitions 用普通 Procrustes 可恢复角度 .7；这是额外信息，不是免费或通常真实可获得的监督。

这复核的是 CoMP 等已知机制，不是新理论。只有一个 constant-angle 家族，不是全部 compatible SCM 的 sharp bounds。

## E4. 学得的非线性 Gaussian kernel

`outputs/neural_coupling_probe.json`：3 seeds，train12288/val2048；固定非线性 drift 的均值网络，以 val MSE 选 ckpt。一步 validation MSE=.01034/.01035/.01049，噪声基准=.01；真实均值 RMSE=.0170/.0182/.0200。dev256仅选一个全局 witness angle；confirmation1024评估。选择结果都是 -pi。kernel NLL invariance≈8.9e-16；反事实符号分歧57.6%/47.6%/36.5%。它是 sensitivity witness，不是正确 CF estimator。

`outputs/neural_range_refinement.json` 使用另一组全新1024轨迹，冻结相同 ckpt；7→256角度后，真CF覆盖率分别为59.77%/74.02%/68.36%。额外加入已知真角度的 **oracle diagnostic** 也不改变结果，排除单纯 coarse grid 解释。oracle true-kernel family含真CF100%。仍不能把 neural 结果单独归因为研究缺口；有限样本/优化 bias 是竞争解释，所以 E5/E6 使用全局最优 OLS 控制。

## E5/E6. 参数置信集与结构 coupling，独立确认

初始置信 tube screen：3样本量×64独立训练数据，各32查询轨迹，H8/32/64；已知 isotropic Gaussian noise。随后冻结 shared-parameter prototype，用**全新训练 seeds 与全新 query cohorts**重做同等规模 confirmation。总计两组各192个OLS fits，不能把每个H/查询当作独立模型重复。

下面是第二组预先指定 primary cell：N16384/H64，64拟合×32轨迹。family coverage 指完整 true global-angle effect interval 被包含；与单个真角度点覆盖不同。wrong signed claim 是断言正/负却不对整个 true family 成立的比例。

'''+table(['Method','True-point coverage','Whole-family coverage','Mean width','Sign certifiable','Wrong signed claim'],crows)+f'''

shared-parameter tube 相对 paired norm tube：宽度减少 **{primary['width_reduction']:.2%}**（按独立训练重复成对bootstrap95%CI {primary['width_reduction_paired_bootstrap95'][0]:.2%}–{primary['width_reduction_paired_bootstrap95'][1]:.2%}）；sign certification 增加 **{primary['sign_certification_increase_absolute']*100:.2f}个百分点**（CI {primary['sign_increase_paired_bootstrap95'][0]*100:.2f}–{primary['sign_increase_paired_bootstrap95'][1]*100:.2f}）。全部64拟合的32查询完整家族覆盖为100%。**这不是泛化到所有SCM/所有模型的100%保证**；理论保证只在明示假设下以训练样本置信概率成立。

**整体方法筛选失败**：预设门槛 coverage≥95%、sign增加≥5pp、width减少≥15%；前两项通过，宽度项失败（3.28%）。不修改门槛、不按测试结果换primary cell。N256/H64 joint tube宽度约13–17，相对oracle .759，说明小样本下非常保守；N2048也几乎不能可靠判断正负。线性特例可计算，但当前原型不足以解决完整问题。

## 验证与资源

`outputs/verification.json`、`verification.log`：5项关键测试通过，覆盖独立精确脉冲真值、split独立/可复现、Gaussian rotation likelihood不变、解析family极值与独立密集rollout吻合、置信集边界128个模型的外包区间检验。不是用单元测试证明全部数学性质；推导另见 `confidence_certificate.md`。

本轮12个初筛神经训练job累计约140秒；额外3个小型Gaussian均值网络训练与审计约7秒。confidence screen约5秒，shared confirmation约35秒（含区间计算）。这是小模型诊断，不是大规模benchmark训练成本估计。大权重/NPZ留服务器，紧凑结果同步本地，哈希见provenance。
''')
    write('negative_evidence.md',r'''
# 负面证据与不应继续的路线

1. **普通动作记忆增强：淘汰为新方法。** Native support 的5/9步限制确实存在，但普通 FIR、GRU、ARX 已能修复；近期 BeliefGraph-JEPA 和 attenuated-identification 工作也直接覆盖此机制。
2. **仅重训/更换 latent objective 即可修复结构零：被证伪。** 相同 core e2e 的 late impulse gain仍为0；任何权重训练都不能突破本结构的严格支持范围。该结论不推广到其他架构。
3. **Gaussian旋转不可识别性、共享seed：淘汰为创新。** CoMP、动态反事实界、Twin Rollouts已有。我们跑出的数值是复核，不是首创。
4. **首次联合统计与因果不确定性：淘汰。** Lally等2026-10-14论文已经在离散IMDP完成。不能因为自己的实验是连续变量就忽略此近邻。
5. **此前neural区间欠覆盖只是7角度网格造成：被新样本反证。** 256角度与oracle包含真角度均未消除显著欠覆盖。
6. **当前shared-parameter prototype达到预设成功标准：被反证。** primary width只降低3.28%，低于15%；完整screening gate失败。sign certification+9.28pp是真实次项收益，不能替代失败主门槛。
7. **线性prototype等同通用神经certificate：不成立。** 当前假设fully observed、affine drift、known isotropic Gaussian variance、constant SO(2) family，远比一般时间序列world model简单。
8. **普通可达性方法尚不能解决：尚未验证。** Taylor/shared-parameter标准工具是重要竞争者，Flow*/CORA预算匹配未完成。在排除前不能宣称新算法有顶会创新。
9. **历史Loop/readout/state-selection/online-updater路线：不重启。** 旧报告已NO-GO或INCONCLUSIVE；本轮不依靠调参把它们重新写成正面结果。

工程纠正：早期加载旧core缺reformer依赖，用新venv的只读pth复用已有依赖；NP/PJM是long CSV而非wide，修正为具名pivot并按真实target-first顺序后重跑；Windows脚本编码命令超长，改为上传文件后执行。失败尝试没有生成最终训练指标，未混入结果。服务器无关环境/数据均未改动。

保留的未知：非线性/隐藏状态下置信区间能否既可靠又有用；普通Taylor reachable-set工具是否已经足够；明确coupling敏感性类是否有可辩护物理解释；真实干预数据只能验证边际/平均效果，不能凭重复实验宣称获得同噪声个体反事实真值。
''')
    write('hypothesis_space.md',r'''
# 竞争假设与可证伪实验

初始训练前版本已保存为 `outputs/protocol_snapshots/hypothesis_space_initial.md`。本文件为本轮后的判断更新。

| ID | 解释/问题 | 区分实验 | 结果 | 决策 |
|---|---|---|---|---|
| H1 | Native未来条件影响支路支持有限 | 冻结旧权重；finite difference+autodiff；随机可控系统脉冲 | 支持5/9步之外严格0 | 事实成立，普通修复足够，淘汰新方法命题 |
| H2 | 只是latent训练目标不合适 | 相同core改decoded e2e | late gain仍0 | 不能解释结构零；不重训消耗预算 |
| H3 | 动作激励不足 | ordinary ARX smooth/white；若强baseline失败再做factorial | smooth ARX/GRU已经好 | 不需要发展新excitation机制；neural white实验未执行 |
| H4 | 同一准确transition kernel确定个体CF | exact action-dependent rotation | likelihood相同、CF差异大 | 证伪；但此原理已有，不作创新 |
| H5 | neural range欠覆盖只是角度网格 | frozen ckpt+全新轨迹，7/256/oracle加真角 | 仍欠覆盖 | 排除单一网格解释；需统计/模型误差控制 |
| H6 | kernel estimate的gauge range足够覆盖 | 全局最优OLS+完整解析true/estimated family | 高样本下point<95%、whole-family约16% | 显式参数不确定性不可省；非优化特有问题 |
| H7 | 普通置信tube足够有用 | 95%联合系数集+remainder | 覆盖恢复，低样本极宽 | 可靠性与效用需分别测量 |
| H8 | 跨时间共享参数prototype有足够增益 | 新64fit repetitions与预设3 gates | sign+9.28pp；width−3.28% <15% | 整体失败；不提高其结论等级 |
| H9 | 通用Taylor/非矩形robust工具已足够 | 同一confidence set、coupling family、查询、计算预算的工具对照 | 尚未执行 | 若成立则淘汰新算法方向；保留为必要下一步 |

后续独立确认必须用新的train/query seeds。不得继续在已观察的confirmation cohort上挑角度、网络、H、N或置信padding。原始scalar confirmation512至今未用于选择或结果；若继续该路线，先冻结独立新计划。
''')
    write('research_ideas.md',r'''
# 候选问题与淘汰

只保留一个有实验根据、但尚未通过完整创新性筛选的问题；不凑三个方向。

## 优先继续检查：冻结连续动态模型的可靠而有用的轨迹反事实区间

Research Question：在只用unpaired transition data、完整factual trajectory供retrospective abduction、以及明确声明的coupling敏感性类时，能否同时保留跨时共享机制和跨分支噪声约束，得到可靠且有非平凡效应判定能力的连续轨迹区间？

为何重要：同一准确conditional kernel支持可靠prospective planning，不代表单条已发生轨迹的反事实唯一。把任意seed耦合下的点估计写作确定个体结论，可能得出在compatible模型下反向的判断。纯结构range又会遗漏有限样本误差；简单补上统计置信集可能让区间几乎没有决策信息。本轮区分了这三件事。

最小机制：对冻结model保持kernel不变的coupling family，联合传播abduction与prediction对共享参数的query sensitivity；保留有界高阶remainder，按真实数据可支持的范围输出interval或abstain。first-order Taylor prototype只是已知工具的基线，不能作为独创名称。

竞争者：CoMP、CSM/APID、Haugh/Singal、Lally IMDP、Flow*/CORA、non-rectangular robust inference。潜在贡献必须发生在**连续非线性、冻结模型、abduction-conditioned query和严格覆盖/效用/成本权衡**，而不是首创不确定性或参数共享。

资源：现有脚本和全部数据生成器已在RTX5090服务器运行；核心线性确认可在CPU数十秒完成。下一轮可用小型frozen Gaussian SSM与公开DoTime连续模拟结构；无需先训练大型foundation model。RP-GSSM需单独处理JAX运行环境和控制接口，不能直接当成已支持action的baseline。

当前价值判断：**问题证据成立；创新性与通用可解决性仍不充分；尚不推荐写论文或大规模训练。** 当前prototype宽度gate失败。该题比继续Loop微调更有信息价值，但没有证据支持它已优于多数现有解决方案。

淘汰方向：条件动作记忆增强（普通基线已解决）；泛latent/readout alignment（历史NO-GO）；仅更换反事实shared seed（已知且不可识别）；首次联合两类不确定性（已有直接论文）。
''')
    write('final_research_direction.md',r'''
# 当前研究结论：一个保留问题，尚无可宣称的顶会级新方法

**Status: CONDITIONAL_RESEARCH_CANDIDATE — NOT SUBMISSION READY.**

当前证据支持继续研究下述问题，但不足以声称已经发现并验证了“明确新颖、优于现有工具”的方法。严格任务目标尚未全部闭环；不要把本文件存在误解成论文选题已经完成。负面与失败门槛全部保留。

## Research Question

**如何对冻结的连续动态模型，在明确噪声耦合假设下，联合控制有限样本模型误差和结构歧义，并给出仍有用的个体轨迹反事实区间？**

限定retrospective inference：可见完整factual path以abduct；不是给prospective forecaster未来真状态。不是估计普通平均干预效果，也不是挑战准确conditional kernel进行prospective planning的有效性。

## Evidence

1. 新非线性Gaussian kernel的一步MSE仅略高于noise floor，精确保留kernel的noise rotation仍改变36.5%–57.6%的held-out反事实效应符号。这是已知不可识别性的实证复核，不是创新证据。
2. 独立1024轨迹上把7个angle增至256个，并额外做含真angle的oracle诊断，真CF覆盖仍仅59.8%/74.0%/68.4%，排除粗网格这一单独解释。
3. 全局最优OLS控制排除神经优化不足：N16384/H64下，estimated-kernel的完整rotation range仅覆盖15.7%的true family，point coverage89.5%；因此“结构敏感性区间=可靠置信区间”不成立。
4. joint confidence tube恢复此明确家族下的覆盖；但低样本和长时域极宽。共享参数原型在新64fit确认中sign判定率4.4%→13.7%，width仅减少3.28%，**未通过预先设置的整体gate**。它展示有限的可改善空间，不是成功解决问题。

## Existing Methods

CoMP已给Gaussian rotation反例；CSM/APID已做连续静态部分识别；动态latent/离散MDP已有bounds；Lally2026-10-14已联合statistical+causal uncertainty；Flow*等已有Taylor reachability；non-rectangular robust MDP已讨论共享机制。详见带原始链接的 `literature_review.md`。

当前代码核查没有发现一个可直接用于冻结连续非线性序列模型、同时处理上述abduction/parameter/coupling约束的现成接口。这是待验证的**具体剩余缺口**，不是“所有现有方法均不足”的证明。通用可达性技术可能已经解决，因此它必须成为下一轮强对照。

## Proposed Idea

将factual abduction和counterfactual propagation作为一个共享参数计算图，用query-directed sensitivity/高阶outer bound联合传播；将kernel-preserving coupling作为明确敏感性变量，不尝试从unpaired数据中“学出唯一耦合”。输出有覆盖解释的区间及abstention。新机制必须比直接套用成熟Taylor/reachability工具更紧或更便宜；当前一阶原型还未达到这一要求。

## Falsification

- 在相同confidence/coupling set与预算下，普通Flow*/CORA或静态APID的动态组合已同样紧、同样快：淘汰新算法创新方向。
- 收益只来自低覆盖、丢弃remainder、测试集padding或更窄的无依据coupling类：淘汰。
- 真kernel/大样本下所有有意义query均先天符号不确定，且无现实可辩护的额外结构：不开发更复杂模型。
- 非线性/隐藏状态扩展只能产生几乎总是abstain的区间：不宣称实用；缩小问题或转向别处。

## Feasibility

已有native WorldTS审计、15个新小模型训练、2×192个独立OLS fits、精确对照与5项关键测试。服务器为单RTX5090 32GB。当前数据规模和置信计算只需秒到分钟；不需要大型预训练或多GPU。历史数据/ckpt保留，新增权重/数组约11MB（见provenance，后续产物可能略增）。

## Next Experiment

最小但最有决定性的下一步：在同一个2D受控系统上构造**同一联合参数confidence set与同一SO(2) coupling类**的通用validated-reachability对照，保持相同readout、factual path与时间预算；先确认当前specialized propagation是否只是弱实现。再用新的seed做一组小型非线性known-simulator确认，报告whole-family coverage、interval width、safe sign判定率、wrong-signed claim和耗时。

先跑普通成熟方法；若其已解决，就明确放弃“新证书算法”，不追加同类network调参。只有这个关口通过，才将DoTime固定版本的公开连续机制与更真实干预数据纳入。真实数据不得声称具备同噪声个体CF真值。

## 推荐

仅推荐继续检验这一个具体问题。**目前不能负责任地说已得到一个创新性闭环、可直接顶会投稿的方向。** 已确认的结论是若干方向不值得继续，以及一个有量化证据但仍需强已有工具排除的候选缺口。
''')
    write('README.md',r'''
# Autonomous TSWM research discovery — 2026-10-16 V1

Private research evidence. No public upload authorized. New standalone server directory: `D:	swm_discovery_20261016_v1`; compact mirrored copy: `D:chenyu-iclr	swm_discovery_20261016_v1`.

Start with [final_research_direction.md](final_research_direction.md). Status is **CONDITIONAL_RESEARCH_CANDIDATE**, not submission ready. Current prototype FAILED its predeclared width-improvement gate. Full evidence and negative results are included.

- `literature_review.md`: primary sources and novelty exclusions.
- `baseline_registry.md`: exact code pins, historical reuse versus new transfer.
- `experimental_results.md`: all studies, independent repetitions and limitations.
- `negative_evidence.md`: failed hypotheses and unexecuted comparisons.
- `hypothesis_space.md`: competing mechanisms and updated decisions.
- `research_ideas.md`: only one retained conditional candidate.
- `final_research_direction.md`: research question, evidence, closest methods, falsification, feasibility and next decisive experiment.
- `confidence_certificate.md`: exact restricted assumptions and algebra.
- `outputs/`: raw metrics, plans, training logs/curves, verification and SHA256 manifests. `.pt` and `.npz` retained on server; listed in provenance.

## Reproduction on the current server

Use the existing standalone `.venvScriptspython.exe`, pinned upstream archives and `config.json`. No SSH credentials are included.

```powershell
.venv\Scripts\python.exe scripts\probe_future_support.py
.venv\Scripts\python.exe scripts\run_synthetic_screen.py
.venv\Scripts\python.exe scripts\arx_control.py
.venv\Scripts\python.exe scripts\probe_coupling.py
.venv\Scripts\python.exe scripts\probe_neural_coupling.py
.venv\Scripts\python.exe scripts\refine_neural_range.py
.venv\Scripts\python.exe scripts\run_shared_parameter_confirmation.py
.venv\Scripts\python.exe scripts\run_checks.py
.venv\Scripts\python.exe scripts\analyze_results.py
.venv\Scripts\python.exe scripts\make_figures.py
.venv\Scripts\python.exe scripts\write_reports.py
```

Existing results are retained; rerunning a script may overwrite its own outputs, so copy the standalone directory for a new run. The initial `confidence_screen.json` refers to the archived v1 scripts under `outputs/code_snapshots/`, because the fixed-common-noise baseline was tightened before the *fresh* shared-parameter confirmation. Replay initial screen with that archived module/script pair and its saved plan, not the later revised module. Historical four checkpoint paths are a declared dependency; exact hashes are in `existing_future_support.json`.

For a clean environment: Python3.10, torch2.8.0+cu128, numpy1.26.4, scipy1.15.3, pandas2.3.3, matplotlib3.10.8, reformer-pytorch1.4.4, einops0.8.1. Exact local freeze subset in provenance. Windows process affinity affects only training processes. Do not rerun `fetch_sources.py` to obtain a different HEAD; reuse saved revisions/archives.
''')
    if (OUT/'feasible_extrema.json').exists():
        f=read('feasible_extrema.json')
        text=f'''

## E7. 通用可行对手：区间宽度有多少是不可省的？

另一套全新数据，8个独立OLS fits、各16条H64轨迹、N16384。普通projected Adam在同一联合系数置信球内，6 restarts×200步，分别搜索效应下/上端；每个候选对时间共享同一个W，耦合仍为同一个全局angle。优化只看factual path，不读取真实CF。对给定W的angle极值解析计算，不需要搜索angle网格。

找到的**可行内范围**平均宽度{f['mean_inner_width']:.5f}，basic sound outer范围{f['mean_basic_outer_width']:.5f}，shared-parameter outer范围{f['mean_shared_outer_width']:.5f}。因此在这组查询与**同一不确定性集合**下，任何sound算法相对basic平均宽度的改善最多为{f['maximum_possible_width_reduction_vs_basic']:.2%}；实际最优可能更小。不能从这个内范围声称已经找到了sharp外界或global optimum。

这限定了线性特例中继续做“更紧算法”的收益空间：预设15%目标已接近可行对手给出的上限。它没有证明一般非线性场景没有缺口，但进一步削弱了仅凭当前线性特例立项新算法的理由。所有记录、优化曲线在 `outputs/feasible_extrema.json`；所有端点witness参数和factual路径保存在服务器 `outputs/feasible_witnesses/*.npz`，逐个通过独立NumPy recurrence重放与置信球可行性检查。
'''
        with (ROOT/'experimental_results.md').open('a',encoding='utf-8') as stream:stream.write(text)
        for name in ['negative_evidence.md','research_ideas.md','final_research_direction.md']:
            with (ROOT/name).open('a',encoding='utf-8') as stream:
                stream.write(f'\n\n补充反证（E7）：通用可行优化已迫使同一置信集合的区间宽度至少为{f["mean_inner_width"]:.5f}，对应basic outer为{f["mean_basic_outer_width"]:.5f}。因此该线性特例中任何sound方法的最大平均宽度收益被限制在{f["maximum_possible_width_reduction_vs_basic"]:.2%}以内；这使15%筛选门槛接近可达上限。不能把不可省的不确定性宽度归咎于缺少新架构。该候选仍不足以作为已确认的顶会方向。\n')
    lirpa=[]
    for fp in sorted((OUT/'lirpa').glob('rep*_q*_H*.json')):
        if fp.name.endswith('.partial.json'):continue
        item=json.loads(fp.read_text())
        if item['horizon'] in [32,64]:lirpa.append(item)
    if lirpa:
        rows=[]
        for H in sorted({v['horizon'] for v in lirpa}):
            items=[v for v in lirpa if v['horizon']==H]
            vals=[v['results'][0]['width'] for v in items]
            rows.append([H,len(items),f'{np.mean(vals):.5f}',f"{np.mean([v['same_set_basic_width'] for v in items]):.5f}",f"{np.mean([v['same_set_shared_width'] for v in items]):.5f}",f"{np.mean([v['results'][0]['compute_seconds'] for v in items]):.2f}"])
        text='''

## E8. 强通用基线：未修改 auto_LiRPA / CROWN

固定源码 `e3930353a3589f7d10879f1e71223d1eefb57196`，独立Python3.11.15 / torch2.8.0+cpu / numpy2.2.6 / scipy1.15.3环境。将同一个regression ellipsoid写成单一L2输入球，保留跨时间/跨分支参数共享；全局角度以64个连续区间覆盖[-pi,pi]。CROWN直接包围整个factual-abduction→counterfactual计算图；没有自定义relaxation、没有更换uncertainty set。每个worker的forward用独立NumPy recurrence复核。

H8 smoke：CROWN width=.68925，alpha-CROWN=.68223，专门shared tube=.71085。alpha耗时59.0s，对比CROWN .85s，额外tightness约1%。因此保留原V1计划，另冻结V2只扩展CROWN，避免重复低信息量alpha计算。这不等于证明alpha-CROWN或GPU验证不足。

长时域下，每个H计划为2个拟合×4个查询；下面按已完成worker汇总，**不是8个独立训练重复**。时间为CPU bound computation，未将专用NumPy批量实现与通用graph实现作严格速度宣称。

'''+table(['H','Completed cases','CROWN mean width','Basic tube width','Shared Taylor width','CROWN compute seconds/case'],rows)+'''

原始结果与逐worker日志在 `outputs/lirpa/`；全局角度、共享系数球及完整factual path语义相同。CROWN在已观察的匹配case中已是重要竞争者，不能再把“共享参数能让区间变窄”当作创新。当前原型还同时失败于其先前预设宽度gate，所以**淘汰当前原型作为新方法路线**。更一般的连续非线性问题尚未由本轮证明已解决，但其潜在存在也不构成已确认的Research Gap。

环境错误单独处理：初始NumPy2.4.6继承环境的eigh触发Windows DLL exception，改用新venv内的稳定binary wheels；constant buffer的batch维警告按通用库接口修复；它们都没有被计入科学失败或算法性能结论。
'''
        with (ROOT/'experimental_results.md').open('a',encoding='utf-8') as stream:stream.write(text)
        with (ROOT/'literature_review.md').open('a',encoding='utf-8') as stream:
            stream.write('\n\n新增强近邻：[auto_LiRPA / CROWN](https://github.com/Verified-Intelligence/auto_LiRPA) 已支持一般计算图、双变量乘法与sin/cos。代码固定后执行了同一共享L2置信球和连续角度集合的计算图验证；它不是只供文字讨论的近邻。当前专用Taylor原型不能凭参数共享或微小区间改善宣称新颖。Flow*/CORA仍未执行，但当前原型不需要等待更多对照才能停止推进。\n')
        with (ROOT/'negative_evidence.md').open('a',encoding='utf-8') as stream:
            stream.write('\n\n10. **专用shared-parameter prototype优于成熟通用工具：不受证据支持。** 同集合的CROWN基线已经能直接运行，并在短/中时域给出更窄区间。当前专用方法未通过自己的宽度gate，也没有排除普通验证技术。停止将其当作顶会方法推进。\n')
        with (ROOT/'hypothesis_space.md').open('a',encoding='utf-8') as stream:
            stream.write('\n\nH9更新：已执行未修改CROWN，同一ellipsoid/coupling/factual graph。普通工具有竞争力，因此淘汰“参数共享prototype即新方法”的解释；general nonlinear scalable certification仍须独立检索/验证。\n')
        p=ROOT/'hypothesis_space.md';s=p.read_text(encoding='utf-8').replace('| 尚未执行 | 若成立则淘汰新算法创新方向；保留为必要下一步 |','| CROWN已执行；Flow*/CORA未执行 | 当前prototype不作为新方法继续 |');p.write_text(s,encoding='utf-8')
        with (ROOT/'baseline_registry.md').open('a',encoding='utf-8') as stream:
            stream.write('\n\nCROWN新增执行状态：17个worker（H8 smoke一个；H32/H64各8个matched cases），smoke另有alpha-CROWN。通用库未修改，CPU单独环境，精确同一L2系数球与SO(2)全局角度集合。若最终worker数少于计划，以 `outputs/lirpa/registry.json` 为准，不将未完成条目计为成功。\n')
        with (ROOT/'literature_review.md').open('a',encoding='utf-8') as stream:
            stream.write('\n\n附加创新排除：同一Gaussian kernel的旋转改变pathwise-gradient方差并非新原理。[ICML2018](https://proceedings.mlr.press/v80/jankowiak18a.html)和[AISTATS2019](https://proceedings.mlr.press/v89/jankowiak19a.html)已研究transport-equation的旋转自由度、控制变量与方差优化。本轮只作文献筛选，没有虚报planning-gradient新实验。\n')
        path=ROOT/'final_research_direction.md';s=path.read_text(encoding='utf-8')
        s=s.replace('# 当前研究结论：一个保留问题，尚无可宣称的顶会级新方法','# 当前研究结论：尚未找到满足最终标准的研究方向')
        s=s.replace('**Status: CONDITIONAL_RESEARCH_CANDIDATE — NOT SUBMISSION READY.**','**Status: NO_VALIDATED_NOVEL_DIRECTION — CURRENT PROTOTYPE REJECTED.**')
        s=s.replace('当前证据支持继续研究下述问题，但不足以声称已经发现并验证了“明确新颖、优于现有工具”的方法。','当前实验核实了已知局限，随后通用CROWN对照又削弱了专用prototype的创新价值；停止将其当作新方法推进。本轮尚未证明一个足够新颖且可解决的Research Gap。')
        s+='\n\n最终更新（E8）：成熟通用CROWN已在同一共享参数/全局coupling集合上完成对照。当前prototype淘汰。只把连续非线性下的可靠/有用区间保留为待检索的问题清单项，不能把它当作已经推荐的顶会选题。整体自主研究目标仍未完成，应调整范围继续探索，不能用这份报告代替目标完成。\n'
        # Rewrite the recommendation around the final evidence; do not leave a
        # rejected prototype presented as a current recommendation above a tail.
        s='''# 当前研究结论：没有通过全部关口的研究方向

**NO_VALIDATED_NOVEL_DIRECTION. 当前原型淘汰；整体自主发现目标仍未完成。**

## Research Question

本轮深入检查的问题是：冻结连续动态模型能否在明确coupling假设下，同时控制有限样本模型误差与结构歧义，并提供仍有用的个体轨迹反事实区间？它是一个具体问题清单项，当前不能作为“已确认尚未解决的新Gap”推荐。

## Evidence

- Native WorldTS未来条件支路支持只有5/9步；但普通FIR、GRU、ARX已修复受控测试，动作记忆方法路线淘汰。
- 在学得的非线性Gaussian kernel上，保持kernel不变仍可改变36.5%–57.6% held-out效应符号；这是CoMP等已知不可识别性的复核。
- 7→256个angle仍欠覆盖；全局最优OLS控制也表明拟合kernel的结构range不等于统计置信区间。
- Joint confidence tube恢复限定家族下的覆盖；shared-parameter prototype在独立确认中sign判定率4.4%→13.7%，但width仅减少3.28%，未通过预设15%gate。
- 同一集合内的可行优化把任何sound区间的最大平均宽度改善限制在约16%，说明大部分宽度由集合本身强制要求。
- 随后实际执行通用CROWN：同一共享L2系数球、同一全局连续角度集合和factual-abduction计算图，无自定义relaxation。它已构成强竞争者，短/中时域区间比当前专用prototype窄；长时域完整原始结果见 `experimental_results.md` 的E8。

## Existing Methods

CoMP已给Gaussian rotation反例；CSM/APID已做连续静态部分识别；Haugh/Singal和Lally已做动态/离散bounds，2026-10-14的Lally工作已联合统计与因果不确定性。Flow*/CROWN和non-rectangular robust inference已有参数依赖、通用计算图外界和共享机制处理。原始链接、代码pin与边界见 `literature_review.md`。

## Proposed Idea

**没有一个新机制被本轮证据批准推荐。** 已试的一阶共享参数sensitivity加有界remainder是已知Taylor/reachability思想的特例；它没有通过自身gate，也没有证明超越通用CROWN。不能给它重新命名后宣称创新。

## Falsification

本轮执行了强反证，而不只列未来检验：普通动作记忆baseline解决问题；已知文献覆盖宽泛新颖性；更密angle网格没有消除拟合误差；独立方法gate失败；通用验证工具有竞争力。这些结果足以停止当前原型的新方法路线。它们并不证明所有连续非线性world model问题均已解决。

## Feasibility

在RTX5090服务器实际完成15个新小型神经训练job、4个旧native checkpoint审计、两组各192个独立OLS fits、另8个可行对手fits，以及CPU通用验证对照。关键数据/数学不变量测试已通过。全部代码、plan、日志、原始指标、权重和witness保留于独立目录；没有公开上传，没有修改旧实验。

## Next Experiment

调整研究范围，而不继续微调本线性prototype。优先审查尚未验证的非线性、异方差/部分观测机制中，现有world model的可检验失败是否在普通完整状态估计/递归模型及成熟推断工具下仍存在。必须先给出新的竞争假设和独立数据计划；不能把本轮已淘汰的原理直接当作下一轮新Idea。检索与证据不足时继续淘汰，不强行提交论文方向。

## 推荐

当前**零个**方向满足“明确新颖、问题重要、已有方法不足、最小实验证明可改进”的完整标准。可保留上述区间问题作检索线索，但不把它冒充最终推荐选题。下一轮应继续自主发现，整体目标保持未完成。
'''
        path.write_text(s,encoding='utf-8')
        with (ROOT/'research_ideas.md').open('a',encoding='utf-8') as stream:
            stream.write('\n\nE8最终筛选：淘汰当前证书prototype的新方法路线。没有一个候选已经通过“新颖性+强基线+独立实验证据”的完整关口。下一轮应重新选取有残余缺口的机制，不继续在本线性系统上微调工具设置。\n')
    status={'research_objective_complete':False,'all_started_experiments_complete':True,
            'status':'DISCOVERY_CONTINUES_NO_VALIDATED_NOVEL_DIRECTION',
            'retained_candidate':'useful confidence/structural-sensitivity intervals for frozen continuous dynamics',
            'prototype_primary_gate_pass':False,
            'next_required_falsifier':'budget-matched mature reachability method; abandon algorithm direction if ordinary tools suffice',
            'publicly_published':False}
    if lirpa:
        status['retained_candidate']=None
        status['rejected_prototype']='shared-parameter first-order confidence tube'
        status['next_required_falsifier']='pivot to a fresh mechanism/hypothesis; do not repackage known Gaussian rotations, action memory, or Taylor bounds'
        status['all_started_experiments_complete']=len(lirpa)==16
        p=ROOT/'README.md';s=p.read_text(encoding='utf-8').replace('**CONDITIONAL_RESEARCH_CANDIDATE**','**NO_VALIDATED_NOVEL_DIRECTION**')
        s+='\n\n通用验证对照使用独立 `.venv_verifier`：Python3.11.15、torch2.8.0+cpu、numpy2.2.6、scipy1.15.3。`scripts/run_lirpa_confirmation.py` 使用保留V1及公开记录原因的V2计划；其CPU结果不能被写成GPU性能上限。总体研究目标尚未完成。\n'
        p.write_text(s,encoding='utf-8')
        p=ROOT/'research_ideas.md';s=p.read_text(encoding='utf-8').replace('只保留一个有实验根据、但尚未通过完整创新性筛选的问题；不凑三个方向。','下面记录本轮曾保留、随后在E8进一步否定当前方法创新价值的问题。没有一个候选已经通过完整创新性筛选；不凑三个方向。')
        p.write_text(s,encoding='utf-8')
    with (ROOT/'experimental_results.md').open('a',encoding='utf-8') as stream:
        stream.write('\n\n协议记录：native scalar screen的stage1实际上按latent validation loss选checkpoint，stage2和end-to-end组按factual validation MSE；初始config的统一描述过于宽泛。原config未篡改，所有差异、代码版本恢复与资源型计划变更均见 `protocol_deviations.md`。这不改变结构支持零或当前NO-GO结论，但不得声称各阶段选择目标完全相同。\n')
    (OUT/'task_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('WROTE_RESEARCH_LEDGERS',len(pins),'source pins')
if __name__=='__main__':main()
