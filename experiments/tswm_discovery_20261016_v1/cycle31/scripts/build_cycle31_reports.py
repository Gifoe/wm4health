"""Integrate bounded new evidence without overwriting historical cycles."""
import pathlib,json,hashlib,datetime,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent
Q=ROOT/'outputs/normwear_qualification_v3';E=ROOT/'outputs/normwear_demo_extent_v1'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=json.loads((Q/'qualification.json').read_text(encoding='utf-8'))
v=json.loads((Q/'verification.json').read_text(encoding='utf-8'))
e=json.loads((E/'result.json').read_text(encoding='utf-8'))
assert v['status']=='passed' and v['qualification_sha256']==sha(Q/'qualification.json')
assert e['qualification_sha256']==sha(Q/'qualification.json') and e['arrays_sha256']==sha(E/'arrays.npz')
assert q['scope_correction_sha256']==sha(ROOT/'SCOPE_CORRECTION_20261017.md')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
head=f'更新 UTC {now}。整体目标 ACTIVE_UNACHIEVED，已接受最终方向 0。此为过程报告，不能冒充最终已验证创新。\n\n'
reports={}
reports['literature_review.md']=head+'''# 文献与创新边界

最新用户纠正见 [范围约束](SCOPE_CORRECTION_20261017.md)。研究层次参考 [WorldTS](https://arxiv.org/abs/2609.31162)：条件信息参与潜态预测，之后解码。其 [官方代码](https://github.com/decisionintelligence/WorldTS)包含 numerical/text/image 三种输入协议。通用框架不等于必须跨域共享一个 checkpoint；也不预设每个候选必须回答严格因果、物理状态解释或未知变化检测。

[NormWear-2](https://arxiv.org/html/2605.15465v1) 将预训练表示与非参数 latent transition 更新结合，提供公开权重。本轮已执行其 core 模型，并非只阅读摘要；下游合成输入资格检查不等于原论文生理数据表格复现。

持续变化线索的独立核查见 [主流模型](streaming_models31.md)、[相邻辨识](change_identification31.md)、[反证视角](streaming_critic31.md)。HiP-RSSM、MTS3、MOCA、APE、稳健 BOCPD 等已覆盖许多双时间尺度、参数重置和未知分段机制。AdaJEPA 的真实反馈后参数更新亦已存在。泛称“状态/机制分离”“只重置变化参数”不足以确定创新。

三个视角当前均未找到有实测余差的新机制；这是待证据的判断，不以投票替代证明。普通有限预算组合仍允许有研究价值，不能要求经典方法在无限预算下永远失败。

本轮代码级接口问题只支持固定 revision 下的实现结论，不能推断整篇论文实验无效，更不能直接命名为新的 world-model research gap。原文跨领域建模主张与源代码接口资格分开记录。
'''
reports['baseline_registry.md']=head+f'''# 模型、协议与复现状态

| 底座 | 本轮实际执行 | 不能据此声称 |
|---|---|---|
| NormWear-2 core | 公开 safetensors 严格加载全部 {q['model']['state_keys']} keys，{q['model']['parameters']:,} 参数；原 card predict、simulate、原 no-action Bayesian；action Bayesian 原错误和普通修复 | 原论文生理数据性能复现、预训练从头复现或新方法 |
| WorldTS / C29 对照 | 复用历史证据，不在本轮重复训练 | 本轮新复现 |
| AdaJEPA | 官方论文/源码/权重目录核查，未下载权重或运行 | 已在单 5090 复现 |
| HiP-RSSM / MTS3 / MOCA / APE | 本轮仅核查机制及公开代码 | 已执行或已被新方法击败 |

NormWear-2：[HF 固定 revision](https://huggingface.co/mosaic-laboratory/normwear2/tree/{q['revision']})。文件 model.safetensors 748,201,812 bytes；SHA256 `{q['weights_sha256']}`；全部权重留服务器。官方 demo 的纯代码 cells 已读取，未执行其私有数据/旧 pickle 加载 cell。

服务器 Python 3.10、torch {q['torch']}、numpy {q['numpy']}、RTX 5090。依官方 demo 的 core 构造方式，使用固定 config；未安装 transformers、未改全局环境。公开 safetensors 直接使用 core 参数名；没有参数缺失、非严格加载或随机初始化替代。

v1 runner 对参数前缀的错误假设、v2 action Bayesian 越界均原样保留。v3 的 ordinary 修复在独立运行时函数与目录，发布源码/权重保持原样。
'''
reports['experimental_results.md']=head+f'''# 已执行实验与证据

身份：原生 API 执行资格 + 自主接口算术诊断。输入为固定 seed20263101 的生成张量，没有报告科学 benchmark MAE/MSE 或模型排名。

| 检查 | 实测结果 | 解释 |
|---|---|---|
| 原 card predict，B2/C128/H64/3 channels | shape [2,64,3]，finite；新进程重放 maxabs {v['native_card_replay_max_abs']:.1g} | 原生预测可执行 |
| simulate，C128/H64，action=2 给定，target=0/1 | 后三次 encoder 的已知 action 偏差 {q['native_simulate']['known_action_encoder_errors'][1:]} | 原函数 decode 后改写已给定 action |
| ordinary target-only clamp | 四次 encoder 偏差最多 {max(q['ordinary_clamp_repair_simulate']['known_action_encoder_errors']):.9g}；输出 action 偏差 {q['ordinary_clamp_repair_simulate']['known_action_output_error']:.9g} | 保留 action 的普通修复；全通道预测时与原函数逐元素相同 |
| symbolic latent flatten，action [2]/[0]/[0,2] | 原末尾切片均不等于实际 action 特征；逆 reshape 精确 | embedding-major 交错布局不等于末尾 channel block |
| 实际 768 维 centroid 布局 | 原切片与显式 action 特征最大差 {v['centroid_channel_layout_max_abs_difference']:.8g} | 数值确认同一索引偏差；不是性能分数 |
| 原 action Bayesian，C128/H64 | IndexError，索引4但仅4个 future patches | use_cls=False 时多采一个 patch |
| 官方 demo 长度 C2048/H2048，10 states，生成输入 | 同样 IndexError，索引128但仅128 patches | 越界不只出现于短上下文 |
| ordinary ceil(H/patch) 修复 | 短长度 [2,64]，demo 长度 {e['length_repaired']['shape']}，finite | 仅已核查无 CLS、context patch 对齐合同；非通用实现声明 |
| 修复后改变全部 future target、保持 past/actions/seed | 输出差 {q['future_target_invariance']['max_abs']:.1g} | 此调用路径没有使用未来 target 真值 |
| 原 no-action Bayesian，仅 past 输入 | [3,64]，finite | 不给它任何未来目标 |

simulate 原接口返回 patch-pad 后 [1,208,3]，输入长192；记录该行为，没有静默改成原生正确长度。未在此判断 API 是否承诺裁剪。

独立 verifier 通过全部 29 个保存数组 finite 检查、NumPy action normalization/逐步输入重建、独立 centroid 轴还原、已观测 transition 计数重建（maxabs {v['independent_transition_max_abs']:.1g}）及原 card 新进程重放。见 [qualification](outputs/normwear_qualification_v3/qualification.json)、[proof](outputs/normwear_qualification_v3/verification.json)、[demo 长度检查](outputs/normwear_demo_extent_v1/result.json)。

demo 长度的普通 length 修复调用用时 {e['length_repaired']['seconds']:.3f}s，CUDA peak allocated {e['length_repaired']['peak_allocated_bytes']:,} bytes。这只是单次执行测量，不能与冷启动 card 调用或其他模型做延迟排名；reserved 包含 allocator 历史。

没有新训练、真实 counterfactual 真值、跨数据集科学比较或新模型。C29/C30 历史结论不被此次接口审计覆盖。
'''
reports['negative_evidence.md']=head+'''# 被否定或尚未支持的解释

1. **先快速做 switching 新模型**：当前没有普通强组合无法解决的实测余差。已知 state/parameter 双过滤、选择性参数重置与未知分段均有先例。本轮不启动该路线长训练。
2. **下载 checkpoint 就算复现**：否。仅 source/weight 获取不构成执行。本轮后续执行和独立重放才增加执行资格证据；仍不是论文表格复现。
3. **原 action 接口可直接作为干净强基线**：固定发布 revision 的最小调用出现已知 action 改写、feature 索引错配与 patch 越界，不能未经核查直接排名。
4. **这些实现错误就是新 Research Gap**：否。已有 target-only clamp、显式 feature index 和正确 patch count 的普通修复，不支持必须新建架构。
5. **原 Bayesian 接口失败只因 C128 太短**：同样错误在官方 demo 的 C2048/H2048 与自动10 states 上复现。数据为生成输入，不能声称完成原私有预训练数据 demo。
6. **未来 target 不变性会失败**：在 length 修复后的本次条件路径，future targets 任意改为大值，输出仍精确相同。不能据此声称存在未来目标泄漏。
7. **通用框架必须是单 checkpoint 零样本模型，或必须严格因果/物理可解释**：这不是用户要求。按数据集训练但机制可复用的 WorldTS 风格贡献也允许；依实际任务合同评价。

v1 加载前缀假设错误属于 runner 错误；v2 原接口 IndexError 属执行诊断。两者分开保留，未放宽严格加载或数值容差。
'''
reports['hypothesis_space.md']=head+'''# 竞争解释与反证

| 假设 | 能区分的证据 | 当前判断 |
|---|---|---|
| H1：未来条件在实现中被错误处理 | 捕获每次 encoder 真输入、symbolic 布局、公开 demo 长度重放 | 三处具体路径得到支持，仅限固定源码/合同 |
| H2：需要新建 world-model 架构才能修复 | 普通 clamp、显式索引和 patch-count 更正 | 目前不支持；普通修复已可运行 |
| H3：条件结果由未来 target 泄漏产生 | 固定 past/actions/seed，任意替换 future target | 此调用路径证伪，差0；不泛化到全部数据协议 |
| H4：持续学习失败来自状态冲击与机制变化混淆 | 与 dual filter/APE/IMM、稳健变化检测和训练混合事件比较，按可辨识性分层 | 未运行有判别力的科学实验，不选作最终方向 |
| H5：这些问题使原论文跨领域结论无效 | 原数据/预处理/完整配置的重放 | 没有这类证据，不能推断 |

需要从实现诊断转入模型科学问题时，必须先给出具体剩余机制、相同允许信息、总训练/调参/在线计算预算及新 held-out 系统。不能拿未修接口或未充分训练的对照制造新方法收益。
'''
reports['research_ideas.md']=head+'''# 候选研究问题

当前不推荐最终方向。主任务仍是一个可跨时间序列任务复用的 world-model 框架贡献；不是特定数据集调参，也不是一组实现 bug 修复。

当前已具备的新证据是：另一个机制不同的发布数值模型已在单5090执行，条件输入的普通修复有明确最小重现。它提供更强的后续对照，但尚无跨系统科学比较证明哪一项 world-model 能力存在未解决余差。

尚未立项的 streaming 假设不能只改名复活。有限预算、可组合方法允许成为价值，但须指明超越已有算法的具体步骤并以公平实验支持。暂不把“支持变化”“提高潜态质量”“长预测更稳定”等口号写成 proposal。

下一轮须优先围绕一个实际任务合同寻找具体差异，而不是继续扩大文献清单；对照必须包括本轮普通修复，以及任务合同匹配的强专门方法。最终可接受按数据集训练的通用框架，不额外强制 foundation 模式。
'''
reports['final_research_direction.md']=head+'''# 最终方向尚未确定

**Research Question：** 最终问题尚未选定。约束是时间序列 world model 的可复用状态/演化/条件建模机制，以 WorldTS 的研究层次为参照。

**Evidence：** 本轮只有发布 NormWear-2 的实际执行资格、三处条件接口路径的最小重现与普通修复，以及独立验证；没有新的跨系统科学失败或方法优势。C29/C30 的负证据继续保留。

**Existing Methods：** NormWear 的预训练与非参数 latent 更新、HiP-RSSM/MTS3 的状态与动态推断、APE/MOCA 等普通变化处理机制均是真实先例。不能通过更名制造新意，也不能把模型未声称的严格因果性质作为所有候选的必需门槛。

**Proposed Idea：** 尚无足够证据提出新的机制。三个普通接口修复属于 baseline qualification，不是投稿方法。

**Falsification：** 若任务匹配、输入/预算公平的已知方法解决候选失败，则撤销。若失败只来自预处理、实现、训练或任务支持不足，则不称为研究缺口。

**Feasibility：** 单 RTX5090 已严格加载约1.87亿参数模型并完成原 predict、条件路径诊断、普通修复与独立重放；没有从头预训练。源码、权重、配置、seed、数组、错误及 SHA 链均保存于本目录；权重只在服务器。

**Next Experiment：** 先完成一个任务合同明确的比较设计，再冻结新的 held-out 系统与 baseline 预算。当前尚无足够理由启动新架构长训练。后续问题选择必须回到用户认可的通用时间序列框架，不把“数据集更低分”或“有一个 bug”升格为选题。

接受最终方向数为0，整体目标未完成。此报告是持续证据记录，不是用负结果替代用户要求的最终研究发现。
'''
for name,text in reports.items():(ROOT/name).write_text(text,encoding='utf-8')
state={'updated_utc':now,'goal_status':'ACTIVE_UNACHIEVED','accepted_directions':0,
       'latest_scope':'reusable time-series world-model framework; WorldTS level; no mandatory shared zero-shot checkpoint, causality, action, Loop or LLM',
       'qualification_status':q['status'],'independent_proof':v['status'],'demo_extent':e['length_repaired']['status'],
       'new_training':False,'new_method':False,'live_scientific_jobs':0,
       'qualification_sha256':sha(Q/'qualification.json'),'verification_sha256':sha(Q/'verification.json'),
       'demo_extent_sha256':sha(E/'result.json'),'builder_sha256':sha(__file__)}
(ROOT/'CURRENT_STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
snapshot=ROOT/'outputs/report_snapshot_v1';snapshot.mkdir(exist_ok=True)
root_names=['CURRENT_RESEARCH_STATUS.md']+list(reports)
for name in root_names:
    path=BASE/name
    old=path.read_text(encoding='utf-8') if path.exists() else ''
    if not (snapshot/name).exists():(snapshot/name).write_text(old,encoding='utf-8')
    begin='<!-- cycle31-current-start -->';end='<!-- cycle31-current-end -->'
    if begin in old:
        start=old.index(begin);finish=old.index(end,start)+len(end)
        old=(old[:start]+old[finish:]).lstrip()
    prefix=f'''{begin}
# 最新 cycle31：按用户纠正研究范围

UTC {now}。通用时间序列 world-model 框架，以 WorldTS 为研究层次参照；不强制零样本共享 checkpoint、因果/物理解释或某种结构。NormWear-2 原生预测执行、三处条件接口诊断和普通修复完成，独立复核通过；不是新方法或论文性能复现。接受方向0，整体 ACTIVE_UNACHIEVED。

见[范围纠正](cycle31/SCOPE_CORRECTION_20261017.md)、[实际证据](cycle31/experimental_results.md)与[当前研究判断](cycle31/final_research_direction.md)。以下各轮历史保留，其“当前”仅指当时。
{end}

'''
    path.write_text(prefix+old,encoding='utf-8')
print(json.dumps(state,ensure_ascii=False),flush=True)
