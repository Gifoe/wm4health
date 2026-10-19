# C41 当前推荐：有锚的观测—动力学组合复用，待实测

**Research Question**：有限训练数据和同步观测锚下，一个时间序列world model如何把观测接口与动态机制分别学成可复用组件，并在未见sensor–system组合保持有用的概率推演？

**Evidence**：本轮已执行LIFTS原生例子；精确LGSSM oracle/gauge控制；四状态补偿/ordinary修复；128模拟2304cell的view–dynamics信息边界及全Bayesian普通解。直接证明的是组合coverage条件和ordinary能力，不是新方法优于现有方法。

**Existing Methods**：NCDSSM已有recognition–dynamics分离，modular NRI已有可组合transition/关系推断，LIFTS已有joint generation/inference，普通joint hierarchical SSM能表达对应factorization。当前不能宣称它们共同不足。缺少直接交互实证≠已证明全球首创。

**Proposed Idea**：以带不确定性的view-specific auxiliary likelihood对接可共享局部dynamic modules；同步anchor约束跨view接口，任务context负责module assignment/组合。待检验交互为二者交叉共享能否改善未见组合的有限样本预测，而不是增加一个latent loss。这个初始参数化可能就是ordinary structured SSM；若完全等价必须如实归类，不能另取名。

**Falsification**：strong joint SSM/NCDSSM+modular直接组合吸收所有收益；只改善seen cells；交互与0相容；优势来自更多参数/计算/锚信息；需要逐数据集换结构；或者测试组合根本不可确定。任一项否决相应创新主张。

**Feasibility**：单5090；先4维稳定系统/4views/多个primitive组合，线性层Kalman oracle，按view×system棋盘留出并给所有方法相同少量同步anchor。未知C/A只向oracle提供真值。0.3–2M initial neural budget，实际preflight后冻结。随后才考虑3个真实领域的自然sensor子集/系统划分，same implementation/rules；真实数据不声称共享物理latent语义。

**Next Experiment**：共同信息、同latent capacity、同loss/inference的2×2观测模块化×动态模块化，加入同样共享/噪声/联合估计的hierarchical SSM和强条件GRU等；先测held-out interaction与quality/cost，不能仅展示11最好。若11与ordinary joint model相同，这一阶段的输出是必要归纳偏置证据，方法创新仍需另证。不要为了保证结果正面排除ordinary control。

本轮=PROGRESS；最终接受方向0；整体ACTIVE_UNACHIEVED。当前推荐是开发优先级，完整候选训练及跨域效果尚未完成。
