# 第三轮基线与数据登记

| 对象 | 精确版本 | 本轮执行范围 |
|---|---|---|
| EpiCF published baseline | 8f7df27507dc8b01422bb373417c8d0d3efffae2 | 完整轻量源码；官方single-policy KDE、Transformer原生训练函数；新增相同架构输入消融 |
| EpiCF simulator | 824ca2a9785038eaec4e277903856d796ac4adb3 | 52个小型代码/配置/说明文件，196,094 bytes；未运行完整县ABM，未下载大型网络文件 |
| EpiCF raw data | Zenodo20762823 v1.2，CC-BY-4.0 | single10,071,252 bytes，multi15,581,482 bytes，字典3,821 bytes；官方MD5与自存SHA256均核对；两种数据均运行官方preprocessing |
| Dataset identity | 158 trajectory blocks, 153 unique county IDs | 5县各有2个完全相同的块；保留官方样本用于发布代码执行，bootstrap去重，未来划分须group by county |
| PRO-DYN | 原root已有733d986d349e9e71ded5d9cdeeeed8d3323d83b4 | 本轮重读2026-05-02 v2论文和真实源码结构；未把阅读算作复现 |

官方Transformer的train/val/printed-test全部取同一train文件及全部训练索引；本轮原样复现并明确标注。权重选择没有使用CF标签，但不具备独立验证集。KDE带宽也按自预测误差选，不是LOO。

三个消融是自主设计的诊断，不能写为原论文实验。9个消融与3个官方神经模型使用同一模型容量、优化器、训练轮数和输入信息掩码规范。目标状态存在的组属于oracle场景条件任务，不能与history-only预测公平排行。

多策略原生Transformer默认158 units不能直接reshape474个CF序列，需要显式扩展推断；本轮只做预处理/输入审计，没有伪报多策略神经训练结果。


## LeNEPA位置假设的源码/组件反证

固定官方源码6574eb9e123305334e09e180de67682844f2e61f；下载92个代码/配置/论文源文件883,823 bytes。默认nepa_target_layer=0、target_uses_pos_embed=false、objective覆盖num_registers=0，并在[0,8]两层施加时间正则。未修改的原生Block/SIGReg组件在三颗种子下检查：八层RoPE block对恒定时序输入的最大时间差仅约2.4e-6至2.9e-6。任意Gaussian位置表可骗过抽象time-axis statistic，但不能直接替代此实现的layer0目标；该简单构造不构成native失败证据。没有执行完整LeNEPA训练，不声称证明该方法保留所有动力学信息。详见`outputs/lenepa_position_null_audit.json`。
