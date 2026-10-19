> 完成后的科学裁决：ordinary双侧factorization有人工heldout收益；新框架贡献尚未建立，seen代价与优化限制须同时保留。见[SCIENTIFIC_DECISION](SCIENTIFIC_DECISION.md)。

# C42 Baseline registry

服务器完成32个未知参数ordinary conditional HMM fits /48000正式updates；独立NumPy从selected checkpoint重建全部模型，核对131072条未来序列score和524288个next-probability cells，max NLL差4.25e-07。此前真实TEST未评分；本轮人工held-out evaluation按冻结协议评分。最终接受方向0，整体ACTIVE_UNACHIEVED。

四模型00/10/01/11均为本项目自主设计的ordinary neural-conditioned four-state HMM，不称原论文复现。00两conditioners均依赖view+system；10将emission的system输入置0；01将transition的view输入置0；11同时置0。每组名义1632parameters，11有效1376，单侧1504；零梯度输入列差异明确披露，不能宣称完整capacity match。exact forward filtering和joint maximum likelihood相同。

普通11也是候选初始离散形式，无独立new-method fit。Oracle只作privileged known-parameter reference，不是未知参数的公平竞争者。NCDSSM/modular NRI/LIFTS/FlowNP/GRU完整跨框架对比尚未在本轮执行。

两independent world、N=16/128、两个initializations，1500updates/fit，batch32，lr.01→.001。每250步以seen VAL选择，未用held-out combination选模。选中cap 15/32，最后interval VAL仍改善18/32；不声称收敛。
