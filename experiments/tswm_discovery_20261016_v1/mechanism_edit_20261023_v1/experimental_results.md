# DALI机制编辑候选：先排除查询构造问题

本轮已执行五个原生记录函数、18个受控输入case，并独立检查保存的context路由张量。这里的rollout、policy、decoder都是明确标记的routing spy；没有训练好的DALI checkpoint，没有测试物理预测。原生源码函数通过AST导入隔离执行，表达式未修改；JAX索引/RNG和Ninjax纯函数机制均为实际原生执行，视频排版替换为identity。

## 已确认的接口行为

| 记录入口 | 输入合约 | 现象 | 可支持的判断 |
| --- | --- | --- | --- |
| plausibility | 原生广播context，B1/2 | 30个名义编辑均未改变context | 沿时间估计sigma不能提供跨系统context尺度 |
| plausibility_obs | 原生广播context，B1/2 | 30个名义编辑均未改变context | 同上；另外其30个delta按代码相同，不是30个不同尺度 |
| plausibility_action | 原生广播context，B1/2 | 7个名义编辑均未改变context | 对比不是一个已实现的非零context编辑 |
| plausibility_obs_dataset | B10、输入65步后原生reshape为64步 | 8个维度都有非零编辑，符号随未来时间变化 | 对固定机制问题，这不是一个序列内恒定的编辑 |
| counterfactual_dreams | B1/2 | 每个维度有4个非零固定尺度编辑，轴和时间保持正确 | 不能把上述问题推广为所有released recorder都失效 |

CtxEncoder将一个context向量广播到整段时间。对这样的输出，时间维标准差在数学上为0；选择精确可表示的fixture排除了浮点噪声的影响。B2样例即使有跨样本差异，前三类入口仍取第一条序列的时间标准差，实际路由改变量为0。真实权重的浮点舍入不改变这个尺度估计的语义问题。

额外的非原生、随时间变化fixture只用于使sigma非零以隔离索引轴。在plausibility入口中，编辑只改变时间槽6的全部8个context分量，而非全部时间的第6个分量。这是当前源码索引行为；该非原生fixture不能作为学到的context或模型物理失败的证据。

## 定位到固定版本的源码

- [record_counterfactual_plausibility.py](https://github.com/frankroeder/DALI/blob/5ca393468defffe79e4e3537ba9766f6d31b510b/contextual_mbrl/dreamer/record_counterfactual_plausibility.py#L47)，wrapper位于原文件第47–112行；SHA aa37e9b67d2649fcb4513104700e7ae1597d7aa2803956844e6503c446118361。
- [record_counterfactual_plausibility_obs.py](https://github.com/frankroeder/DALI/blob/5ca393468defffe79e4e3537ba9766f6d31b510b/contextual_mbrl/dreamer/record_counterfactual_plausibility_obs.py#L56)，wrapper位于原文件第56–114行；SHA 6847970e7e64095cba056032496684f74128ecce0e46470be9a44b4f37386bf0。
- [record_counterfactual_plausibility_action.py](https://github.com/frankroeder/DALI/blob/5ca393468defffe79e4e3537ba9766f6d31b510b/contextual_mbrl/dreamer/record_counterfactual_plausibility_action.py#L47)，wrapper位于原文件第47–139行；SHA 8866e8364d996941741bf0467bd9fc603dbe81922c4979af00840644b2eb1505。
- [record_counterfactual_plausibility_obs_dataset.py](https://github.com/frankroeder/DALI/blob/5ca393468defffe79e4e3537ba9766f6d31b510b/contextual_mbrl/dreamer/record_counterfactual_plausibility_obs_dataset.py#L41)，wrapper位于原文件第41–105行；SHA 58941e33cdfceea7429a9ed6f6c5f9e4e58828961ab8fe229411d4bd75bcd0cf。
- [record_counterfactual_dreams.py](https://github.com/frankroeder/DALI/blob/5ca393468defffe79e4e3537ba9766f6d31b510b/contextual_mbrl/dreamer/record_counterfactual_dreams.py#L49)，wrapper位于原文件第49–143行；SHA a01b94de611b8d950f3e6cac61dde5404b123df7cd5344a620c8db64794bd9ef。

## 根状态、动作与episode合约

plausibility/obs/dataset入口在用一段观测推断context后选择posterior索引0作为root；旧dreams入口选择索引9。观测条件范围和root时刻不一致，需要明确是hindsight查询还是在线未来预测，不能未经定义称为泄漏。action入口手动展开prior，其root及动作时序还需要在训练好的实际模型上资格检查。

dataset入口还有固定leading-dimension10的reshape，因此本轮为其显式提供B10/T65 fixture，不把它当通用任意batch接口。没有检查实际训练数据episode是否被混合，也没有据此声称已经混合。物理机制测试必须保留episode/参数身份和明确历史/未来动作对齐。

## 普通修复控制与独立复核

提供了显式轴、外部给定尺度、每条序列固定偏移的普通函数，B1/B2/B10共9个case执行通过。这只修复context张量构造，不代表已经修好完整记录器、root/动作对齐或模型物理预测，也不是新方法。

Torch CPU独立检查2754项记录张量的轴、时间和数值属性；最大标量差9.54e-07，位于1e-6 float32语义差门槛内，轴/时间支持集合完全一致。110个原文件重验SHA。第一次AST隔离漏导入tree_map的错误、dataset batch前提修正和退出回执均保留；这是harness错误，不是源项目缺失符号。

[全部case摘要](editor_contract_recovery_v1/CASE_SUMMARY_V1.csv) · [原生编辑输出摘要](editor_contract_recovery_v1/RESULTS_V1.json) · [独立核验](editor_contract_recovery_v1/VERIFICATION_V1.json) · [普通修复控制](editor_contract_recovery_v1/ORDINARY_EDIT_CONTROL_V1.json)。

## 完整原生模型核心的CPU资格检查

原生WorldModel核心完成3次丢弃的预热更新：context Transformer、categorical RSSM、decoder、原生forward/crossmodal目标和原生optimizer均执行。原来17M变量核心并未被一个小型替代网络取代。原Config/WorldModel类通过导入隔离载入，nets/jaxutils/ninjax为完整原模块；没有执行CARL环境、Actor或Critic。

每次调用秒数为120.451256, 118.926074, 0.877289；第3次是否重新trace：False。只有没有重新trace的调用才能作此CPU配置的warm时间参考，不能据此声称GPU性能或直接外推原RL训练。非optimizer注册数组元素17052480，口径见机器记录。

首次全finite检查曾退出1。原生balance_stats明确规定空类别诊断为NaN；profiling的reward0/cont1使四个分类统计无定义。恢复只将这四个已验证空支持的统计记为N/A，模型输出、参数、optimizer状态、所有其他指标均需finite，最后一次（第3次）真实参数改变量为0.0001。首次失败没有保存逐叶名称，因此不能唯一确认原失败只由这四个统计造成。最终诊断逐项核验通过；没有将原失败改写为通过。

已返回调用累计为原失败1次加最终诊断3次，共4次丢弃的资格optimizer调用。中间还保留一次在首次返回前发生的原生process abort，完成情况未知，不冒称额外成功更新；其原因未定，后续成功也不证明由trace-counter引起。资格更新不计科学fit；本轮实际factual训练的启动/完成状态另见下方快照。输入只是C88首条TRAIN轨迹64步的16通道shape/梯度fixture，reward/cont为profiling占位标签。没有Val/EVAL读取、物理质量评估或保存成正式训练checkpoint。

[实际核心结果/状态](native_core_profile_diagnostic_v1) · [固定模型配置](native_core_profile_diagnostic_v1/CONFIG_V1.json) · [原始核心退出](native_core_profile_execution_v1/exit.json) · [诊断退出](native_core_profile_execution_diagnostic_v1/exit.json) · [NaN定义边界](CORE_PROFILE_METRIC_DOMAIN_RECOVERY_V1.md)。

## 科研判断

Research Question：同一潜在context编辑，能否在不同初始状态和行动查询下，被一个固定的真实动力学参数变化解释？

Evidence：本轮证明当前发布编辑器存在尺度/轴/时间合约问题，并建立原生模型核心可执行性证据；没有验证一个训练充分模型的全局机制失败。不能从routing spy得出模型物理结论。

Existing Methods：显式轴和固定偏移足以修复张量合约。因果abstraction/intervention-map学习已有直接先行工作；一个可辨识的显式参数模型也必须列为普通强控制。参见[文献版本补充](LITERATURE_VERSION_ADDENDUM_V1.md)。PokeWorld早期版的拖曳“功能盲区”不再作为当前失败前提。

Proposed Idea：尚无接受的新方法。正确构造查询是前提，不是world-model创新。

Falsification：若准确的原生模型在正确查询下通过跨root/action固定机制检验，或普通context投影、显式参数模型、state重新初始化足以解释差异，应淘汰相应新机制动机。若未编辑模型不准确、root物理含义不保留或语义本就不可辨识，不能计为编辑机制失败。

Feasibility：单服务器，原始源码/参数版本固定，已建立隔离JAX CPU环境；原生Windows没有官方JAX NVIDIA wheels。没有修改OS、GPU驱动或共享环境。完整数据、权重和路由张量留服务器G授权存储。

Next Experiment：在同一源核心上先执行一个独立协议的实际factual训练/rollout资格门槛，再测试全局机制意义；本轮profile并未替代该训练。若原生完整预算不具备可行性，需要另选可原生执行且明确支持机制编辑的模型，不能拿未训练或任意缩小的替代网络宣称原论文失败。

当前整体研究目标ACTIVE/UNACHIEVED，最终接受新颖方向0。三个主baseline的已归档结论保持，当前是额外候选的原生执行资格工作，不增加第四篇完整SOTA复现。


## 实际factual训练已启动

在上述丢弃的资格更新之外，已按[独立冻结协议](FACTUAL_FIT_PROTOCOL_V1.md)启动一个2000-update原生核心transfer，使用现有TRAIN/VAL和原生完整目标，不复用profile权重。它是跨任务迁移资格实验，不是原RL论文复现。发布快照状态：`RUNNING_OR_PENDING_RECEIPT`；完整fit数0。启动与逐步轨迹均保留，发布时尚未完成的结果不冒充已完成。

先达到两个动力学族的factual多步预测门槛，并另行资格核验时序，才允许后续机制归因。四个固定stochastic draw、VAL选择点0/500/1000/2000和两个普通基线门槛在启动前写入协议；未读取干预或独立EVAL。

[实际训练快照](FACTUAL_RUN_SNAPSHOT_V1.json) · [训练入口](scripts/train_factual_native_v1.py)。