# 底座与复现登记

六个原作者 validation-best final-budget 检查点（16维系统latent，100000训练步，各变体3个seed）；64个新物理系统；历史完整前缀0/8/32；两种probe；三次独立评估MC；918行预测、306行MC聚合。没有本轮训练。同一cohort被所有检查点共享，不能算成384个独立系统。源版本`f0baf6262523d9c5abaf8c0381b4db97818f7c34`。

科学指标是角速度CRPS、四个分支的联合energy/variogram、MSE和边际PIT覆盖率。四个查询共享同一个系统latent抽样。确定性cos/sin流形数据不使用full3D NLL作为科学评判；旧推断对照仍使用原decoder的full3D Gaussian似然，因此可以分开检验模型似然失配与推断失配。

| 项目 | 固定版本/状态 | 科学角色 |
| --- | --- | --- |
| Explore-the-Context | `f0baf6262523d9c5abaf8c0381b4db97818f7c34`；作者修正版发布archive已核查 | 原生模型的新机制审计 |
| unconstrained | run21/18/20；seed1/2/3；16D latent | 3个validation-best final-budget发布权重 |
| positive + rescaling | run24/23/22对应seed3/2/1 | 另3个发布权重；不是单因素ablation |
| no-update | 原先验/历史后验不加probe | 共享初态、query、模型及MC |
| SLR | 4096 prior draws，完整cov，残差+heteroscedastic noise | 成熟近似推断 |
| defensive IS | 1024及4096；各半prior/native更新proposal | 加权粒子及相同矩的diag/full投影 |
| FlowNP | `f3403b6479dad33a7b93ba6edcf93c53d40f8e65`；已核查51文件，未训练 | 后续现代范式候选，非已复现结果 |
| OSSM | 2609.33566v1；正文已读，未运行 | 排除“观测校正/传播分离”泛创新 |

物理数据：64个native context seed20262501起、独立trajectory seed21262501起，actions RNG22262501；32步完整历史；known controlled reset。保留原prior sampler行为：4096样本abs最小1.25005090，名义低半区[.5,1.25)实际出现率0。原作者训练环境同一sampler，未擅自修正为新分布。没有原训练parameter-ID完全去重声明。

所有seed、配置、权重hash、数据hash、原始分数、proposal/ESS和日志位于服务器cycle25。详细数值不重新训练、不覆写旧结果。大数组未拉到本地。
