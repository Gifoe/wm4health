<!-- cycle29-index-start -->
# 最新证据cycle29

UTC 2026-10-16T19:21:51.067123+00:00。原始12/12及全量验证完成；等更新对照已复核=True，接受方向0。见[cycle29/negative_evidence.md](cycle29/negative_evidence.md)；以下历史保留。
<!-- cycle29-index-end -->

<!-- latest-cycle28 -->
# 最新记录：cycle28，整体研究未完成

最新完整内容见[cycle28/negative_evidence.md](cycle28/negative_evidence.md)和[CURRENT_RESEARCH_STATUS.md](CURRENT_RESEARCH_STATUS.md)。目标是通用时间序列world model；本轮跑通新发布PCFM，并完成真实CED同信息比较。普通glue开关已修复大量退化，已有概率ICL较强；最终接受方向仍0。以下早期历史材料中的“当前/下一步”只指当时。

---

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


补充反证（E7）：通用可行优化已迫使同一置信集合的区间宽度至少为0.93463，对应basic outer为1.11427。因此该线性特例中任何sound方法的最大平均宽度收益被限制在16.12%以内；这使15%筛选门槛接近可达上限。不能把不可省的不确定性宽度归咎于缺少新架构。该候选仍不足以作为已确认的顶会方向。


10. **专用shared-parameter prototype优于成熟通用工具：不受证据支持。** 同集合的CROWN基线已经能直接运行，并在短/中时域给出更窄区间。当前专用方法未通过自己的宽度gate，也没有排除普通验证技术。停止将其当作顶会方法推进。
