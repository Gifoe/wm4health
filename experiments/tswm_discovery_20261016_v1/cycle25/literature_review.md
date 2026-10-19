# 文献与创新边界

[Explore the Context](https://proceedings.mlr.press/v130/achterhold21a.html)已提出跨系统latent、主动校准和共享latent多分支预测。本轮使用作者修正分支及发布权重；本审计是自主设计的机制测试，不是原文全流程复现。

[Bayesian Context Aggregation](https://openreview.net/forum?id=ufZN2-aehFa)已把context聚合建模为Bayesian推断。[Exploiting Inferential Structure in Neural Processes](https://proceedings.mlr.press/v216/tailor23a.html)，UAI2023，已有graphical-model先验、混合/Student-t和variational message passing；“换成更丰富后验”不足以构成创新。

[Flow Matching Neural Processes](https://proceedings.nips.cc/paper_files/paper/2025/hash/a92519f525c00085095fa41c5c46cdb5-Abstract-Conference.html)，NeurIPS2025，直接建模任意context/target条件律，论文3.3节明确讨论marginal/conditioning consistency，未保证其成立。[官方源码](https://github.com/danrsm/flowNP)已固定`f3403b6479dad33a7b93ba6edcf93c53d40f8e65`，51个小源码/配置/文档文件共171540B，每个Git blob和SHA256核查。未声称已训练/复现FlowNP。其GP配置d_model64与正文笼统128描述不同，后续应以明确协议分别报告，不能悄悄换配置。

[CNP conditioning gap](https://arxiv.org/abs/2604.19312)，2026-04-21，讨论context重编码与条件化的KL差异及小context问题；OpenReview PDF自述TMLR04/2026，但论坛当前返回验证页，未独立确认接受状态。因factorized CNP joint与共享latent联合律不同，它不能直接替代本审计；但“更新一致性无人研究”已不能成立。

[Variational meta-learning inference for low dimensional neural system identification](https://arxiv.org/abs/2607.18965)，2026-07，已有摊销初始化+MAP/Laplace，不是本工作的创新。仅核查方法范围，未复现其JAX实现或独立认可所有数学/校准结论。

[Correct then Forecast: OSSM](https://arxiv.org/abs/2609.33566)，2026-09-27，已将传播与观测校正分开，并明确它与线性闭环SSM的关系是参数化而非扩展模型类。不能再把“分离传播和校正”本身包装为新世界模型。当前只阅读正文，没有执行其模型。

近邻方法给出了严格边界；本轮负面结果进一步削弱一般更新修复的创新必要性。不同范式的迁移必须单列，不能把Neural Process回归测试直接叫原论文world-model复现。联网下载在服务器两次connection-refused后改为本地取得相同公开源码，再逐文件哈希同步服务器；未关闭TLS或修改服务器代理设置。
