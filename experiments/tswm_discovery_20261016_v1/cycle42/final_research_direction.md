> 完成后的科学裁决：ordinary双侧factorization有人工heldout收益；新框架贡献尚未建立，seen代价与优化限制须同时保留。见[SCIENTIFIC_DECISION](SCIENTIFIC_DECISION.md)。

# C42 Research decision — framework objective retained, no accepted new method

**Research Question**：观测接口与动态机制分别学习和复用，能否改善未见组合的有限数据推演；进一步是否存在超出普通结构化SSM的必要新机制？

**Evidence**：服务器完成32个未知参数ordinary conditional HMM fits /48000正式updates；独立NumPy从selected checkpoint重建全部模型，核对131072条未来序列score和524288个next-probability cells，max NLL差4.25e-07。此前真实TEST未评分；本轮人工held-out evaluation按冻结协议评分。最终接受方向0，整体ACTIVE_UNACHIEVED。 真实数值与2×2交互完整见[results](experimental_results.md)。它们只回答受控人工环境的ordinary可学习性，不证明领域缺口。

**Existing Methods**：普通jointly fitted factorized HMM就是11；NCDSSM、modular NRI、LIFTS与CoLLAs compositional interfaces覆盖多项现有组件。目前不能断言它们共同不足。

**Proposed Idea**：当前初始参数化属于已知结构模型，本轮没有合格的新方法。允许在研究目标不变下继续发现更具体的必要交互；不以新增模块掩盖等价性。

**Falsification**：最强ordinary模型吸收全部收益，或者交互只在人工先验、特定seed和不充分优化下出现，即否决对应创新/普适性主张。新框架必须给出区别于现有模型的可证伪机制与实际用途。

**Feasibility**：本轮小型exact模型服务器CPU更快，32fits已完成，RTX5090保留给需要的神经/跨域验证。已有native LIFTS/FlowNP资格和原有数据证据可复用，无需重新下载大库。

**Next Experiment**：先用完整最近方法与代码厘清尚未覆盖的框架能力，提出区别于ordinary11的单一机制假设，再冻结一个同信息/同预算的决定性test；若无此假设，停止扩展当前合成2×2。不以继续加数据集替代新机制发现。

最终方向未达到接受标准；整体ACTIVE_UNACHIEVED。
