# C43 Current research direction

**Research Question**：可复用TS world model能否以与查询所需信息相适应的成本提供有用的随机未来推演，而不限于固定horizon逐点误差？观测/动态组合复用仍是并列开发目标。

**Evidence**：本轮PROGRESS；整体ACTIVE_UNACHIEVED，接受最终方向0。新增近邻SPF固定commit a7624ae25b47b474d5a31ca22b3ac6e6907795dd，29个source files/Gitblob+SHA确认。执行原生core/collator/scheduler与原函数体的context adapter，两次50updates preflight共100步（均不作效果）；V2原假设失败保留，V3记录而不强求跨尺度共享。独立复核9000个调度值max0，checkpoint replay max0，同尺度续接差0。此前真实TEST与本轮comparative effects均未评分。 新证据证明已有SPF确实有多尺度/续接能力；它没有证明任何新方法优于它。

**Existing Methods**：SPF直接覆盖粗尺度生成，ordinaryMLMC允许合法的分开耦合调用；cheap AR/SSM必须保留。主路线的NCDSSM/LIFTS/modular近邻不能遗漏。

**Proposed Idea**：当前只提出“查询所需尺度×采样分配”的可复用框架候选；exact shared clean marginal的必要性未知，不预先添加loss或架构。新贡献可能来自必要交互与有效训练/推断配方，不能来自组件更名。

**Falsification**：SPF+ordinarycoupling或cheap AR/SSM支配质量成本；只有平凡均值函数获益；模型bias主导；需要逐数据集换结构；以内部残差代替真实效用。出现则否决对应主张。

**Feasibility**：固定source与nativecore已在RTX5090执行并checkpointreplay。无需下载庞大ClimateSuite；后续先fittedknownprocess训练，确认价值后才扩到不同真实领域。

**Next Experiment**：新协议冻结两seed真正trained conditional SPF、同一fittedlaw highsample reference、独立process truth、nativecommonrandomcouplingMLMC与MC/cheapARSSM、非平凡函数与总成本。qualificationweights全部排除。只有ordinarycontrol有无法覆盖的具体问题时再设计shared-marginal架构。

候选A1为不同fallback，最终接受方向0；整体ACTIVE_UNACHIEVED。
