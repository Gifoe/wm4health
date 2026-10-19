# C39 跨查询概率律与状态更新的文献综合

UTC 2026-10-16T23:31:05.545952+00:00。原生query诊断与普通kernel控制均完成且独立复核；最终接受方向0，整体ACTIVE_UNACHIEVED。新neural updates0；只有8个history-only AR8解析fits。

目标是可复用时间序列world-model框架；不把这些FEV片段的最优分数当idea，不强制action/causal/zero-shot、多模态或全新算子。

## 摘要与问题

本轮由mainstream、critic、constructive三个独立检索视角延续各自上下文，不读当前siblings；root合并前另核关键原文。RQ1区分同joint条件律、latent CK、history重编码和真实forecast版本；RQ2检查强普通方案后的有限质量/成本残差；RQ3寻找最多一个精确新交互。既有C23–25 system-posterior、C34 first-hit MLE、C37 marginal-rank问题不复活。

## 按概率对象分类的交叉综合

| 对象 | 最近完整方法及具体边界 | 本轮能推出什么 |
|---|---|---|
| fixed finite joint的条件化 | NDP通过joint定义conditioning；换target∪context universe须另核边缘一致性 | 不能将不同query的密度比自动当固定joint归一条件 |
| amortized arbitrary conditional | FlowNP、MissPath-FM有真正可执行接口；FlowNP §3.3不保证marginal/conditional consistency | 无理论保证不等于质量失败，必须实际强对照 |
| latent多步composition | MDN-VJEPA的trajectory mode+sample CK，SC的resampling一致性训练 | 已覆盖宽泛正则；CK下降不等于真实accuracy改善 |
| 新观测后的分布transport | SFM旧预测source+hybrid训练；ForcingDAS按schedule filtering/smoothing | 暖启动和统一conditioning均非新机制；有限成本质量仍须同合同验证 |
| 同批paths上的函数查询 | PathFM cache/PDD/CRPS | 多functional不是新观测条件化；不能用旧sampler慢概括现代生成模型 |

[FlowNP §3.3](https://arxiv.org/html/2512.23853v1)仅架构保证exchangeability，joint与任意conditional一起训练，是最近完整对照。[SFM §3](https://arxiv.org/html/2602.05319v1)实际的hybrid source/target pairs不自动成为某冻结joint的精确Bayes条件；这限定理论外推，不能据此宣布它失败。[ForcingDAS](https://arxiv.org/html/2605.14285v1)已通过noise schedule统一过滤/平滑，校准是作者承认的未系统评估项。

[MDN-VJEPA](https://arxiv.org/html/2610.13956v1)已明说一致性与质量不同，某些CK指标改善伴随probability/calibration tradeoff；其保证不是观测history重编码。From Drift to Coherence是重复同QA目标的SC微调，不能把同样正则迁来称新机制。[Forecast sequence理论](https://arxiv.org/html/2606.31621v1)还需要终端outcome校准，内部martingale本身不足。

[MissPath-FM](https://arxiv.org/html/2610.17715v1) observed-value clamping保证返回值相等，不保证各mask来自同一未知部分条件law。[PathFM](https://arxiv.org/html/2610.22358v1)已组合联合path、cache、1–2step蒸馏与proper-score校准；code/weights仍承诺将发布，未称本项目运行其权重。组件已有不排除新交互，但不能把这一整套已存在功能重新命名。

## 测量反证与实测联系

仅suffix边际矩相等仍可丢失全部prefix相关。本轮Gaussian AR正/独立重抽负控证实测量灵敏度，因此采用prefix-weighted probes和outer-group MC。反过来，旧joint错、新requery对时也会检出不兼容，故非零不能决定应蒸馏哪个teacher。完整实测见experimental_results.md。

## 对三个问题的回答

RQ1各方法约束不同概率对象，不能用“Bayes/CK/consistency”混称。RQ2当前获得了native自查询残差和普通kernel质量/成本代价，但尚未证明强完整方法不足。RQ3没有精确新机制候选；下一项高信息试验是固定同tail、真实prefix揭示的FlowNP/SFM/AR强普通screen，而非直接开发新coherence loss。

## 来源与限制

7个主文HTML和2个固定作者接口文件已正常TLS归档，SOURCE_RECEIPTS给9个SHA；旧MDN/NDP/MPNP原文及精确章节见三个独立notes和旧轮archive。所有作者reported scores仅用于原文边界，非本项目复现。未可访问来源不填造数字；MDN metadata日期不直观，不凭搜索卡片断言首发日。
