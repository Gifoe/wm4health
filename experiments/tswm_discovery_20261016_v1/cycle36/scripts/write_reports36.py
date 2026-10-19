"""Build C36 evidence reports from completed server-side provenance outputs."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'outputs/elexon_probe_v2/result.json').read_text(encoding='utf-8'))
c=json.loads((R/'outputs/cadence_audit_v1.json').read_text(encoding='utf-8'))
assert p['status']=='qualified_asof_fixture'
assert all(x['independent_asof_selection_equal'] for x in p['asof_checks'])
assert all(x['all_eligible_hourly_timestamps_covered'] for x in c['checks'])
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
scope='目标为可复用时间序列 world-model 的状态学习、条件转移与观测生成框架。数据集只是证据载体；允许分别训练，不强制共享零样本 checkpoint、action、因果/物理识别、Loop 或 LLM。'
status=f'更新 UTC {now}。C36 是已完成的文献/源码及数据资格审计，整体目标 ACTIVE_UNACHIEVED，接受最终方向 0。本轮没有模型训练、预测误差或方法效果实验。'
source='三份独立 clean-context 原文检索见 [mainstream](mainstream_conditions36.md)、[critic](critic_conditions36.md)、[constructive](constructive_conditions36.md)。主线程进一步核验代码/卡片和原始接口；来源及整文件 SHA 见 `public_metadata/receipts.json`、`source_audit/manifest.json` 和 `outputs/elexon_probe_v2/receipts.json`。'
reports={}
reports['literature_review.md']=f'''# C36：外部预报条件与世界状态的研究边界

{status}

## 摘要与研究问题

{scope} 本轮三个冻结问题是：现有模型是否已经使用带误差的未来预报；同信息的直接预测、联合概率建模和校准积分覆盖了哪些机制；是否还存在可检验的有限预算学习残差。文献不支持“future covariates 被一律当成未来真值”的宽泛论据。数据资格有所推进，尚未证明新状态机制有必要或有效。

## 方法与证据分组

{source} 检索主窗口 2026-04-17 至 2026-10-17；更早基础工作单独标识。先宽检索再按命中方法和数据缩窄；身份与具体主张分开核验。三视角都没有执行模型或科学实验，其建议的 24-fit 设计不是完成记录。以下按机制分组，不把不同输入任务的分数混成排行榜。

## 一、已经读取预报的条件模型

[WorldTS](https://arxiv.org/html/2609.31162v1) Appendix A.1 的 NP/PJM/BE/FR/DE 使用负荷、风电或发电预报；[Chronos-2](https://arxiv.org/html/2510.15821v1) §5.3 的 EPF-DE 也含 day-ahead forecasts。它们的接口足以表达 forecast-as-input；不能由字段名 future_covariates 推导“物理未来精确已知”。反过来，读入预报也不等于已评估 ensemble/vintage 的联合概率语义。

[Cross-Unet](https://www.nature.com/articles/s41467-026-73817-3) 已针对历史与 forward weather 的异质作用设计交互；[KITE 的 ICML 2026 正式版本](https://proceedings.mlr.press/v306/cheng26t.html) 已有历史条件中心/尺度、低秩噪声流形和 conditional generation。前者是直接模型的强控制，后者是概率生成的强控制；没有证据允许把任一说成已完整解决真实 forecast-vintage uncertainty。

主流 note 中仅由作者主页未检得 WorldTS code 所产生的可用性疑问在整合时纠正：[公开 WorldTS 仓库](https://github.com/decisionintelligence/WorldTS) 早已固定为 `269426ea6217d7152a244da0e209d922acd46d4f`，此前多轮实际执行；本轮不重复下载或伪报新复现。ICML 的 KITE 也不是名字相同的 event 模型。

## 二、不确定输入、联合分布与后处理的完整先例

较早 [DSSMF，IJCAI 2019](https://www.ijcai.org/Proceedings/2019/0402.pdf) §2.4 已将未来外生变量分布经 MC 送入深状态空间模型；其误差实验为合成，不能当真 vintage 证据。[MMFE 状态空间研究，2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8182152/) §4–6 已把 forecast/revision 信息扩展为状态，但其线性理论不直接决定神经模型的有限数据效果。[UKF-L，2012](https://mlg.eng.cam.ac.uk/pub/pdf/TurRas12.pdf) §4 已按预测 marginal likelihood 学习有限 sigma points，限制了“少数任务相关情景首次可学习”的主张。

普通 joint forecasting 的 [TACTiS-2](https://arxiv.org/html/2310.01327v2) 与 [CircuITS](https://arxiv.org/html/2604.27814v1) 可表达联合变量的条件/边缘分布；它们并未在本轮被迁移到实际发行数据。[PBC](https://arxiv.org/html/2604.16238) 展示 forecast/lagged observations/climatology 的概率偏差修正与组合。已有积分、联合建模和校准机制的存在不证明所有有效组合已完成，但也不允许把这些原理改名为新世界模型。

## 三、真实误差与现有实验证据的边界

[2026-10-05 load covariate uncertainty benchmark](https://arxiv.org/html/2610.07232v1) §II-C 用 train ARIMA、validation residual scale 和独立高斯扰动生成不同噪声强度；现有 day-ahead load forecast 并非同样被扰动。作者承认 bias、temporal dependence 和 non-Gaussianity 限制。它支持输入误差值得评估，却不能证明同信息、在真实历史预报上健康训练的 strong direct predictor 仍失败。

因此公平目标首先是 p(Y_future | H_origin, F_origin)。直接模型允许学到预报不准、年龄/可靠性和非线性响应；它不必先恢复未来真实天气。新模型若使用 spread、revision 或成熟误差历史，普通基线也须获得同样信息。不能 truth-train/forecast-test 后把不匹配当创新动机。

## 四、可用数据合同而非无限强化的审计门槛

[GEFCom2014 组织者](https://www.pierrepinson.com/docs/Hongetal2016.pdf) §6.1/§7.1 给出训练和评测均提供 daily midnight、至 +24h 的 ECMWF forecast 合同，足以支持声明清楚的 benchmark 协议；本轮未下载完整数据，也未核 timezone，不添加 UTC 或独立网络接收保证。它不提供本候选所需的全部天气实况、ensemble 或 revision。

[OpenSTEF Liander2024 卡片](https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark) 有 versioned available_at，并明确部分 availability/delay 为模拟。固定的消费代码证明 as-of 选择接口存在，不能证明 weather construction 链。[Open-Meteo Single Runs](https://open-meteo.com/en/docs/single-runs-api) 区分初始化与实际可用时刻，并注明 2024 IFS 为 hindcast。hindcast 可以用于声明清楚的协议，不能冒充当年实际发行。

Elexon 一日公开原始 JSON 已在服务器验证：发布时间字段可用于 as-of 选择；原生预报为小时，实测为半小时。资格结果及范围见 [实际结果](experimental_results.md)。不要求每行历史网络抓取日志才能做研究，也不以暂缺元数据推导方法新颖性。

## 跨组综合与逐问回答

RQ1：预报条件模型已存在，且部分已原生使用实际外部预测；宽泛空白被反证。RQ2：冻结目标模型、只校准外部未来条件再积分的当前变体与普通两阶段方法在指定因子分解下目标等价，关闭其新方法主张；这不否定所有有限预算新组合。RQ3：一日发行 fixture 合格，但没有模型学习残差或新机制收益。只保留未批准的“小预算、目标相关不确定性表示与共享转移的交互”问题；压力、成本与近邻差异均待证。当前不推荐最终方向。

## 自我反审

没有把所有模型都说成泄漏，也没有把已知组件等同于所有组合都已解决。没有强加 causal/action/zero-shot 门槛。最具体提案被关闭的依据是可执行目标等价，而非单纯“先有人做过积分”；条件性压缩问题被保留为待证伪问题，不升级为创新。数据审计既没有证明方法有效，也没有成为停止研究整个问题的理由。
'''
reports['baseline_registry.md']=f'''# C36 模型、来源与执行登记

{status}

{scope}

| 对象 | 固定来源 | 本轮实际状态 |
|---|---|---|
| WorldTS | `269426ea6217d7152a244da0e209d922acd46d4f`，此前原生执行与权重保留 | 本轮读论文/旧登记，不重训；纠正“未找到 code”的视角检索遗漏 |
| KITE，ICML 2026 | `3140ee824cbd80c5ec7fdf2b54666210519d0b39` | 保存 README、requirements、7 Python source 与 repo/commit/tree；SHA/Git blob 核查。未导入、下载权重或训练 |
| Chronos-2 / Cross-Unet / DSSMF / PBC / joint models | 原文与作者代码链接见文献报告/三份 notes | 机制边界核查，不报本轮复现成绩 |
| OpenSTEF/openstef | `ca716897a5b9e5ae767a4c02e5efafb24425a654` | 保存消费 loader/as-of 版本类与示例；没有 weather 构造链的完整证据 |
| OpenSTEF Liander2024 HF | `dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044` | metadata/card/target yaml；未下载约 650MB signal files。当前 Python3.12+ source 不能冒充已在服务器3.10原样运行 |
| Elexon WINDFOR/FUELHH | 官方接口一日固定 JSON，整文件 SHA、URL、下载 UTC 保留 | 本地公共 HTTP200 下载后整文件 SHA 上传；服务器独立选择/原生频率审计执行成功 |

source_audit 共20个 source/card/config 文件、277946 bytes；它不包含模型权重，也不是20次实验。元数据与数据的文件数分开登记。服务器直接访问 Elexon swagger 的 V1 为403，保留原源码和输出目录；V2使用透明的本地公开下载传输，未绕过TLS或权限。三种研究视角建议的6arms×2tasks×2seeds没有运行。

资格 probe 无模型、无随机 seed、无 train/val/test，没有以它选超参数。全部历史成果只读保留。数据/源码整文件回读校验记录与科学结果分别位于 manifest、receipts 和 outputs；不会把传输成功当模型复现。
'''
coverage='\n'.join(f"| {x['origin']} | {x['eligible_future_half_hours']} | {x['coverage']} | {y['eligible_hourly_timestamps']} | {y['all_eligible_hourly_timestamps_covered']} |" for x,y in zip(p['asof_checks'],c['checks']))
reports['experimental_results.md']=f'''# C36 实际执行结果

{status}

## 执行与传输

服务器执行 `scripts/probe_elexon36_v2.py` 与 `scripts/audit_cadence36.py`；输出分别在 `outputs/elexon_probe_v2/result.json`、`outputs/cadence_audit_v1.json`。V1直接访问官方 swagger 为HTTP403，原版本保留；三个原始文件由本地正常TLS公共HTTP200取得，再上传服务器整文件SHA回读核验。不是远端直接下载成功，没有登录/TLS绕过。

固定日期为2026-01-01发布的预报、2026-01-02实测；日期与三个 origins 在读取信号前冻结，不按误差挑日。原始 schema/forecast/actual 文件分别为1479300、66002、8737 bytes。服务器验证数据类型、finite、时区与自身 dataset/fuelType；没有计算 forecast errors。

## 数据计数与独立 as-of 选择

预报584行、73个独立 startTime，(publishTime,startTime) 重复0；发布时间02:30..22:30Z，目标范围2025-12-31 21:00Z至2026-01-03 21:00Z。实测48行、48个半小时 startTime，(publishTime,startTime,fuelType) 重复0。实测值只作后验 label 文件资格，不参与预报版本选择。

对每个目标时刻选择 publishTime≤origin 且 startTime>origin 的最新版本；bucket-max 与独立逐目标逐行扫描完全一致。结果如下：

| origin UTC | 可用未来半小时轴点 | 精确匹配预报 | 可用原生小时轴点 | 原生小时轴完整 |
|---|---:|---:|---:|---|
{coverage}

## 必须保留的频率解释

预报73个独立时点间隔为3600秒，共72个间隔；实测48个时点间隔1800秒，共47个间隔。24个匹配点是完整次日24小时的小时预报，并非只覆盖12小时，也不是缺24条应有的小时预报。midnight origin 使用严格 startTime>origin，故00:00点排除，剩23个。

[官方页面](https://bmrs.elexon.co.uk/wind-generation)也定义小时预报。V2原输出按48个半小时轴报告coverage，原字节保留；补充审计明确原生频率。没有插值、复制、平均或评分。未来实验必须先冻结预报值与实测区间的验证定义；同timestamp不自动保证区间平均/点值语义相同。

## 可复核文件与结论范围

V2 result SHA256 `{hashlib.sha256((R/'outputs/elexon_probe_v2/result.json').read_bytes()).hexdigest()}`。原始 bytes/SHA/URL 与脚本/计划 SHA 均在输出和 receipts。补充 cadence audit 引用相同 result SHA。整文件本地镜像另外核验。

本轮没有模型 fits、预测质量、概率校准、long-horizon simulation 或新方法优势。只有一个发布日/验证日，不能证明多年接口完整性、跨领域机制或固定 fleet/curtailment/revision 口径。官方 publishTime 是发行标签，不是独立历史网络接收记录；足以构建声明清楚的 as-of 协议，不要求额外的生产部署保证。
'''
reports['negative_evidence.md']=f'''# C36 负证据与关闭事项

{status}

1. **宽泛未来真值论据关闭。** WorldTS/Chronos-2 已用负荷/发电预报；未来张量的数值已知，不等于未来真实天气已知。
2. **当前 observer-only 方法主张关闭。** 在指定模型分解下，冻结目标生成器、用外部标签校准条件分布再积分，与同信息普通两阶段校准+MC/QMC逐项等价；把积分放进 latent rollout 没产生新的学习目标。
3. **因子分解并非真实世界保证。** 写成 pθ(Y|H,U)qψ(U|F,H) 需 Y 与 F 在(H,U)下预测充分/条件独立；direct p(Y|H,F) 可保留 forecast provider/reliability 信息。不能删 F 给普通对照制造劣势。
4. **现有噪声 benchmark 不证明新残差。** train-ARIMA+Gaussian 误差不是实际 vintage，未排除训练/部署输入错配、同信息 direct 和简单校准解释。
5. **available_at 不是构造链证明。** OpenSTEF明确一些delay为模拟；这只限制真实发行叙述，既不否定公开协议价值，也不产生方法创新。
6. **本地成功不升级成服务器直下成功。** 远端V1 403，公开本地V2 bytes上传并核验；失败记录保留。
7. **24/48 不解释为半天缺失。** 原生小时forecast完整；半小时实测分辨率不同。补充审计纠正解释，未为提高分数修改数据。
8. **当前没有有限情景压力证据。** C34固定Sundial1024 paths批量采样很便宜，禁止将小batch逐path耗时外推成强制K=8动机；不同生成器的实际开销仍须测。

这些负证据关闭具体论据/目标等价提案，不证明所有神经不确定转移或有限预算组合无价值。C35保护分支配置和旧NP/PJM/NESO路线不重新调参挽救。
'''
reports['hypothesis_space.md']=f'''# C36 竞争假设与反证

{status}

| 假设 | 能区分它的证据 | 当前状态 |
|---|---|---|
| 现有模型都把外部预报当真值 | 原文实际输入合同、code、预报任务 | 宽泛版本已否证 |
| 健康训练的同信息 direct 已足够 | vintage训练/验证、同spread/age/revisions、强直接模型和校准 | 未运行；不能由旧噪声benchmark排除 |
| 只需两阶段校准+条件积分 | 同条件分布、同冻结target generator、同MC/QMC预算 | 当前observer-only提案与它目标等价，关闭该提案 |
| 未来误差的joint结构重要 | TRAIN/VAL跨lead残差，joint vs iid同输入对照 | 尚未建立真实目标相关压力 |
| 小K确有精度/实际成本矛盾 | K曲线与真实batchlatency，保持模型固定 | 尚未证明；不人为设窄预算 |
| 目标相关压缩与共享转移有额外交互 | 同rank input-PCA、learned direct、同qψ、同预算state分解对照 | 条件性未批准探针，UKF-L/KITE/任务情景压缩为近邻 |
| 改善只来自额外信息/容量/训练 | 提供全部相同信息，参数/更新/FLOPs与实际训练曲线 | 必须控制，未运行 |

框架层面允许不同数据集分别拟合同一学习规则；不以跨数据零样本作额外必要条件。测试集不用于选择机制或rank；普通方法可以胜出而不被削弱。若同信息强direct或普通压缩/校准解释收益，则不把增益归为新latent机制；若有效组合有新的有限预算交互，则可继续，不要求所有经典方法在无限算力下失败。
'''
reports['research_ideas.md']=f'''# C36 候选账本

{status}

{scope}

**已关闭候选：** 外部预报observer适应、冻结目标动力学的世界模型。它在当前分解下就是普通multilead calibration+frozen conditional generator+integration，没有可辨认的新目标；不改名为新框架。

**唯一保留的问题，未选为Idea：** 高维相关预报误差下，目标相关的有限情景表示与共享多步转移是否有样本/计算效率交互。不是声称首次不确定输入、状态、联合分布、可学习sigma points或情景压缩。最接近完整先例为DSSMF、KITE、UKF-L及任务相关scenario reduction；目标相关低秩map本身也不构成差异。

继续之前应先在TRAIN/VAL检验真实joint-error压力、strongdirect/普通calibration残差、K曲线和实际latency。没有这些压力证据，不启动notes中建议的24fits。Elexon一个GB聚合风电任务可核版本合同，单独胜利不能上升为通用world-model机制。GEFCom风/光可按原组织者合同作第二任务方向，不需要为每行捏造网络接收日志。

若压力不存在，立即转向新模型能力；若存在，先冻结最小公平比较再训练。不得通过增加模块、挑error子集、换日期/seed/rank或复用TEST来保持候选存活。这里保留一个可失败的问题，而不是推荐一个有顶会潜力已被证实的方法。
'''
reports['final_research_direction.md']=f'''# C36 研究判断：没有最终接受方向

{status}

**Research Question：** {scope} 外部预报不确定性是否存在现有同信息方法尚不能有效利用的有限预算状态学习残差，是本轮调查问题，尚未建立为论文选题。

**Evidence：** 三独立文献视角与主线程原文/源码复核；20source/card/config文件固定；服务器执行584预报/48实测的一日as-of选择及小时频率审计。两种版本选择完全一致，原生小时点覆盖24/24、24/24、23/23。没有新模型效果或科学失败证据。

**Existing Methods：** WorldTS/Chronos-2本已读取预报，Cross-Unet直接交互、KITE概率生成、DSSMF不确定外生输入、MMFE预报状态、UKF-L有限代表点、概率后处理及joint forecasting均限制宽泛创新。它们的存在不自动否定有实证差异的组合，但当前observer-only提案与普通校准积分目标等价。

**Proposed Idea：** 目前不接受任何新方法；关闭observer-only新机制主张。仅保留未批准的目标相关压缩×共享多步转移×实际小预算问题，未称创新或推荐。

**Falsification：** 同信息direct/普通calibration或输入压缩解释全部收益、没有真实K压力、优势仅来自更多输入/预算，均关闭该具体路线。原生预报是小时粒度，不把24/48对齐误读成任务困难。

**Feasibility：** RTX5090服务器及既有WorldTS/Sundial底座可用。KITE源码和OpenSTEF接口仅已审查，没有执行。Elexonfixture原始字节经本地公开HTTP下载、SHA上传再服务器执行，不伪报远端直下。一个发行日不能替代完整训练cohort。

**Next Experiment：** 只有在候选继续时，先冻结真实预报/验证值区间口径和连续时间TRAIN/VAL数据，做普通直接/误差校准与实际K成本的压力筛查；若无残差则换问题。notes的24fits没有启动，不以积累审计或大规模复现代替缺口发现。研究总目标仍未完成，最终接受方向0。
'''
reports['CURRENT_STATE.md']=f'''# C36 完成状态

{status}

{scope}

实际执行：source/card元数据固定、服务器一日as-of版本选择和native cadence审计。无模型训练/预测分数/24-fit实验。forecast584行、actual48行；hourly覆盖24/24、24/24、23/23，独立选择一致。observer-only提案因目标等价关闭，未批准新的机制。

七份报告已生成。原始来源、失败V1、成功V2、cadence补充与旧成果均保留。整体研究ACTIVE_UNACHIEVED；不把本轮文件齐全标为研究目标完成。
'''
for name,body in reports.items():
    dst=R/name
    assert not dst.exists(),f'Preserve existing report: {dst}'
    dst.write_text(body,encoding='utf-8')
manifest={'utc':now,'reports':[{'path':k,'bytes':(R/k).stat().st_size,
    'sha256':hashlib.sha256((R/k).read_bytes()).hexdigest()} for k in reports],
    'accepted_research_directions':0,'overall_goal':'ACTIVE_UNACHIEVED',
    'no_model_training_or_scores':True,'generator_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(R/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
header=f'''<!-- cycle36-complete-start -->
# C36源码/发行数据资格审计完成，尚无最终方向

UTC {now}。研究层次为可复用时间序列world-model框架。三独立原文视角核查后，当前observer-only提案与普通校准积分目标等价，关闭其新机制主张。没有模型训练或方法效果实验。

服务器一日584条预报/48条半小时实测的as-of选择独立实现相等。预报本为小时粒度，原生小时覆盖24/24、24/24、23/23；不能把24/48误读为只覆盖半天。远端直接HTTP403，原始公开JSON经本地HTTP200后SHA上传、服务器执行；原V1保留。

见[文献综合](cycle36/literature_review.md)、[实际执行](cycle36/experimental_results.md)、[研究判断](cycle36/final_research_direction.md)。所有旧成果保留，下面“当前/下一步”均是各轮历史。整体ACTIVE_UNACHIEVED，接受最终方向0。
<!-- cycle36-complete-end -->

'''
dst=R.parent/'CURRENT_RESEARCH_STATUS.md'
old=dst.read_text(encoding='utf-8');assert '<!-- cycle36-complete-start -->' not in old
dst.write_text(header+old,encoding='utf-8')
print(json.dumps({'reports':len(reports),'accepted_directions':0,'root_status_updated':True}))
