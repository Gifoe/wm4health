<!-- cycle29-index-start -->
# 最新证据cycle29

UTC 2026-10-16T19:21:51.067123+00:00。原始12/12及全量验证完成；等更新对照已复核=True，接受方向0。见[cycle29/hypothesis_space.md](cycle29/hypothesis_space.md)；以下历史保留。
<!-- cycle29-index-end -->

<!-- latest-cycle28 -->
# 最新记录：cycle28，整体研究未完成

最新完整内容见[cycle28/hypothesis_space.md](cycle28/hypothesis_space.md)和[CURRENT_RESEARCH_STATUS.md](CURRENT_RESEARCH_STATUS.md)。目标是通用时间序列world model；本轮跑通新发布PCFM，并完成真实CED同信息比较。普通glue开关已修复大量退化，已有概率ICL较强；最终接受方向仍0。以下早期历史材料中的“当前/下一步”只指当时。

---

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


H9更新：已执行未修改CROWN，同一ellipsoid/coupling/factual graph。普通工具有竞争力，因此淘汰“参数共享prototype即新方法”的解释；general nonlinear scalable certification仍须独立检索/验证。
