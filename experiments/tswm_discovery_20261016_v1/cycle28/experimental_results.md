# 实验结果：新发布框架与真实部分观测系统

总体目标ACTIVE、UNACHIEVED；cycle28=PROGRESS。已执行新框架，而非只给文字Idea；没有新神经训练或已接受方法方向。

## 官方发布示例

不修改 PCFM example.py 的 seed7、常数或5组main调用顺序。记录15个原生API调用，并额外执行固定单window的12个计划/glue条件和1个masked-future占位检查，共28次调用。全部原始图、输入输出、stdout/stderr保留。原生valve三计划MAE：ON0.356464°C、OFF0.231219°C、PULSE320.661514°C；ON-OFF效应MAE0.262196°C。源码原生图/历史是特权已知物理的合成例子，不是实测因果验证。

独立first-order物理递推/lfilter、seeded原始窗口及打印分数重建通过。28文件hash、1,440,768个输出finite检查、分位数顺序检查通过；28个神经调用新进程原生重放最大差6.104e-5，满足预列float32相对/绝对容差。它是same-source numerical replay，不是独立重写网络。

原生what-if为masked reconstruction。延迟前ON响应相对OFF的RMS约0.270°C（glue开）、0.220°C（关）；其他延迟方案完整保留在contract_rows.csv，最大的提前差约1.392°C。该单window显示近似提前响应，不能证明跨系统缺陷或新架构需求。masked future target换成123456后输出差严格0；没有目标占位泄漏。9边际分位数不被误当成joint路径。

原生运行8.047s，GPUpeak 496.23MiB；不是完整原论文表格复现。

## 实测CED的完整官方测试段

同一记录连续500点，前400校准；官方测试前10点只作初始化，随后90点一次free forecast；没有回灌真实输出。PCFM读取410连续历史而非拼接无关episode。ICL读取400context和10query初态，native padded30仅增加20个标准化零占位，不增加观测；direct10单列。标准化只用前400点。模型全部固定，图/glue/初始化模式预先列出，不能测试后选择最佳模式部署。

| 预列方法 | 低幅记录 RMSE | 高幅记录 RMSE | 低幅 10–90% 覆盖 | 高幅 10–90% 覆盖 |
|---|---:|---:|---:|---:|
| pcfm_CD_glue1 | 0.336438 | 0.434848 | 22.22% | 11.11% |
| pcfm_CD_glue0 | 0.188532 | 0.154891 | 90.00% | 87.78% |
| pcfm_known_direction_glue1 | 0.360608 | 0.466030 | 22.22% | 6.67% |
| pcfm_known_direction_glue0 | 0.194502 | 0.145179 | 92.22% | 92.22% |
| pcfm_CP_glue1 | 0.553822 | 0.697557 | 11.11% | 31.11% |
| pcfm_CP_glue0 | 0.377379 | 0.555143 | 73.33% | 84.44% |
| icl_native_padded30 | 0.067266 | 0.076028 | 86.67% | 90.00% |
| icl_direct10 | 0.096307 | 0.093462 | 55.56% | 58.89% |
| arx_train_selected | 0.379986 | 0.430485 | 88.89% | 66.67% |
| persistence | 0.837538 | 1.467837 | 无概率输出 | 无概率输出 |
| training_mean | 0.419359 | 0.638769 | 无概率输出 | 无概率输出 |

22行包括全部预列设置；原始单位RMSE，非百分比。ICL native padded30表现较强；PCFM公开的glue=False已大幅减轻默认退化。由此不能将默认低分当作新框架必要性。ICL更专门的系统识别预训练与PCFM一般预训练不同；参数量/前向次数不能作为公平的离线计算量比较。本轮没有分离其训练数据、任务先验和架构效应。

ARX48个训练内验证配置与两次最终refit均保留；验证是100步free-run。全部142分数独立重算、两条原始split/2000模型input值检查通过；增广最小二乘与AR状态协方差递推另验，参数/预测/分位数差小于9e-15。12次PCFM+4次ICL完整神经重放差0。最大分数重算差3.974e-8，来自float32 reduction。native数据与CED结果哈希分别724119ab9d7794df8cb81c7523dd2d83fcbf11a6289cf46008a70e952f6aae95 / 1869e6ff8d46ae963168d6125fb66d3b6b3d2ffcdfa7ad98c387771d00c2d30c。

CED运行7.859s，含加载5.859s；GPUpeak253.53MiB。记录warm-forward时间，但PCFM封装、ICL裸模型与ARX验证/拟合的计时边界不同，不作速度排行。没有神经训练。两条短实测记录不构成跨装置统计总体，没有对90时间点作IID显著性声明；边际覆盖也不能证实完整概率动力学。CED真实记录没有未执行plan真值，factual误差不是干预正确性证明。
