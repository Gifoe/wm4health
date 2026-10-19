# 第三轮实验结果：EpiCF 信息协议审计

服务器已完成13个配置：官方KDE一次、官方Transformer三颗初始化、三组输入消融各三颗初始化。所有神经模型均使用未修改的FluTransformer：4,743,681参数、100 epoch、Adam、lr=0.001、batch=8。消融保持18维输入，以固定置零同时作用于训练和测试；不是只在测试时删除特征。种子41/42/43只改变训练初始化，同一数据集，不是三次独立数据生成。

| Inputs (same native Transformer) | Factual RMSE | CF RMSE | Scenario-effect RMSE |
|---|---:|---:|---:|
| official | 0.3300 ± 0.0058 | 0.9605 ± 0.0184 | 0.9655 ± 0.0203 |
| no_policy | 0.3417 ± 0.0057 | 0.8641 ± 0.0129 | 0.8479 ± 0.0174 |
| no_states | 0.5734 ± 0.0170 | 1.3716 ± 0.0648 | 1.3641 ± 0.0652 |
| no_policy_no_states | 0.6066 ± 0.0299 | 1.2883 ± 0.0277 | 1.2769 ± 0.0000 |


表内单位是按官方factual均值/标准差缩放后的病例数，±是三颗训练初始化的样本标准差。CF RMSE为所有158轨迹×168天的全局RMSE；scenario-effect定义为预测factual减预测CF与真实factual减CF的差，不将单个模拟场景差称作真实人群CATE。

官方源码的per-county range指标也已逐次计算，见`outputs/epicf_baselines/results.json`。它与原论文表格的完整归一化/历史配置未逐项恢复，**本轮是发布代码执行，未宣称论文表2数值MATCHED**。官方KDE自预测误差接近0、CF RMSE为1.2070；自预测不是泛化验证。

输入审计确认测试S/E/R/M来自目标反事实场景；整段无mask attention使用同期和未来值。no_policy模型无法响应仅改变行动的查询，却在发布场景协议下优于保留行动的模型，因为场景状态也发生改变。它不是可部署的行动模拟器。no_policy_no_states的预测factual与CF逐元素完全相等，是零场景响应负对照。

153个唯一县的去重描述性bootstrap见`outputs/epicf_summary.json`；独立重采样不同模型的初始化，不把158含重复块当作158个独立县。区间只描述当前训练数据和评测样本，不声称新县泛化或数据集层稳定性。

差分重建反证：仅用每县factual拟合一个比例系数，再由CF的负ΔS重建病例，CF RMSE=46.7819例、县内range归一化RMSE=0.2005。因此“病例被S差分精确编码”的强假设被否定。未凭源码公式抹去数据中的反证。

检查：12个checkpoint SHA、预测finite/shape、已存RMSE重算、零输入场景对比恒等性均通过。全部训练日志与权重保留服务器。


## LeNEPA位置假设的源码/组件反证

固定官方源码6574eb9e123305334e09e180de67682844f2e61f；下载92个代码/配置/论文源文件883,823 bytes。默认nepa_target_layer=0、target_uses_pos_embed=false、objective覆盖num_registers=0，并在[0,8]两层施加时间正则。未修改的原生Block/SIGReg组件在三颗种子下检查：八层RoPE block对恒定时序输入的最大时间差仅约2.4e-6至2.9e-6。任意Gaussian位置表可骗过抽象time-axis statistic，但不能直接替代此实现的layer0目标；该简单构造不构成native失败证据。没有执行完整LeNEPA训练，不声称证明该方法保留所有动力学信息。详见`outputs/lenepa_position_null_audit.json`。
