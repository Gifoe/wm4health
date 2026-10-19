# C42 Scientific decision: known-component interaction, no accepted framework yet

研究贡献单位继续是**可复用的时间序列world model框架**。当前两个人工world不是一个要优化的数据集，而是检验组件是否确实有组合复用能力的控制载体。

32个未知参数fits证实，在这个人为构造且有静态同步锚的环境中，两侧factorization有实际的未见动态组合收益。heldout未来joint NLL每symbol（越低越好）：

|training N / seen cell|00 conditional|01 dynamics only|11 both / ordinary structured HMM|
|---:|---:|---:|---:|
|16|1.390066|1.300351|1.238231|
|128|1.385295|1.249977|1.219431|

但这个收益由**普通jointly fitted structured HMM**实现。它说明当前组合目标有可学习的结构信号，不能说明现有方法尚未解决该问题，更不能把11另命名当作新方法。它也不否定“已知组件的必要新交互”可能构成框架贡献；只是本轮没有设计/测试一个超出ordinary11的完整候选。

小数据N16时heldout交互均值0.076553，四个world/init值均正；N128时均值0.014748，其中一个为−0.007512。seen-cell交互均值是负的。分离动态还增加了seen NLL：例如N128普通11的excess约0.02467，00约0.00446。不能只展示heldout柱子，或宣称双侧复用没有代价。随着数据增多交互变弱，符合有限数据归纳偏置解释，但两个world不足证明其一般规律。

ordinary11离privileged oracle仍有0.11809/0.09929 nats每symbol的heldout excess；15/32选cap、18/32末段VAL改善。当前差距可能包括非凸学习/有限anchors/样本数，不能命名为世界模型的表示信息缺陷。C42也没有测试可组合primitive、新system ID适应、连续/非线性隐状态或真实跨领域泛化。

**裁决**：本轮PROGRESS。保留“观测接口×动态机制组合复用”作为框架开发候选，接受其受控环境中的known-component学习信号；拒绝仅凭分离结构/这张表作新方法或全球未解决问题的主张。候选的初始结构普通模型已具备，下一步应当寻找/验证具体的新增职责或必要交互，与这个强ordinary解正面对照；如果只剩换模型名字或多跑几个数据集，应停止该包装。

**下一步范围**：重新对照完整NCDSSM+modular dynamics、LIFTS及few-shot environment inference等近邻，明确一个ordinary11不自动解决、又确有实际需求的模型能力，然后冻结有竞争解释的最小测试。可以用已知组件形成有实证价值的新配方，但不能把通用structured SSM本身当作空白。没有充分新假设前，不追加本合成2×2的site/seed，也不立即扩成长训练。

最终接受方向0；整体ACTIVE_UNACHIEVED。全部32raw scores、select curves和独立验证均保留。此前真实TEST未评分。本轮没有公开发布、删除旧成果或改变服务器全局环境。
