# C39 完整实测

UTC 2026-10-16T23:31:05.545952+00:00。原生query诊断与普通kernel控制均完成且独立复核；最终接受方向0，整体ACTIVE_UNACHIEVED。新neural updates0；只有8个history-only AR8解析fits。

目标是可复用时间序列world-model框架；不把这些FEV片段的最优分数当idea，不强制action/causal/zero-shot、多模态或全新算子。

冻结协议：[V2 native计划](NATIVE_QUERY_PLAN_V2.md)、[追加普通控制计划](ORDINARY_KERNEL_CONTROL_PLAN.md)。追加方案来自独立mainstream反证、在root读任何effect rows前写下。V1因solar前864点缺失导致有效context仅1而按原guard退出，未加载模型；原源码/日志保留。V2仅改为C34既有first-finite anchor+固定offsets，未按误差选窗口。

source-native Sundial revision3212e42564493f520593e5414af4367fc4b49226，128329680参数，50flow steps；原数据/weights沿用C34 hash，没有新下载signal/weight。8histories、2MC reps、splits96/352/360、Nouter256、Ninner8，full-growing history、native/fixed-reference两坐标。权重全程不变，9moment probes有Gaussian解析正/负控；inner先取非线性g再平均、fresh noise独立于prefix。

第一项独立检查8raw histories、48NPZ SHA、864moment rows，max probe差7.1054273576e-15、summary差7.77156117238e-16。第二项独立pairwise CRPS替代生产排序公式，16NPZ/256score cells、8AR拟合及递推，max score差4.4408920985e-16，AR replay差1.63709046319e-11，native及canonical第一prefix重放差0。两项model state同一SHA。6个same-noise API资格（长度672/768/1032 × native/reference）max relative差0；此API资格由worker执行，独立verifier没有再跑GPU API，不能夸为双实现模型全重放。

## 全query分组描述

| task | coordinates | split | median abs MC z | max abs MC z |
|---|---|---:|---:|---:|
| solar | native | 96 | 1.743 | 14.445 |
| solar | native | 352 | 2.568 | 14.078 |
| solar | native | 360 | 2.412 | 9.137 |
| solar | fixed_reference | 96 | 1.645 | 12.671 |
| solar | fixed_reference | 352 | 2.241 | 14.784 |
| solar | fixed_reference | 360 | 2.075 | 9.057 |
| jena | native | 96 | 2.002 | 6.968 |
| jena | native | 352 | 3.150 | 7.167 |
| jena | native | 360 | 3.106 | 7.226 |
| jena | fixed_reference | 96 | 2.094 | 4.305 |
| jena | fixed_reference | 352 | 1.310 | 4.687 |
| jena | fixed_reference | 360 | 1.681 | 5.526 |


原720head的短/长请求prefix本来一致；本轮测的是observed-history重编码后的law。固定原history坐标后，部分96+96 context仍有超MC误差的moment差，不能全归因RevIN或360点patch相位。具体例：solar end1536无权suffix mean两rep fixed-reference差约−.212/−.223（MC z约−12.67/−11.82）；Jena end1344 fixed-reference prefix-weighted terminal差约−.087/−.081（MC z约−3.92/−3.69）。这是必要moment的不兼容，不是已证明真实质量损害，更没有决定哪个law应当是teacher。

全部864行（包括不显著/反号）在[CSV](outputs/analysis_v1/all_query_moments.csv)。median/max z仅描述MC信噪，不作多重比较后的普遍failure结论。

## 普通kernel与真实未来质量控制

所有192点预测先用该history尺度归一。head_calls为native1、canonical962、canonical824；native一次也实际生成720维flow。AR8是history-only解析fit，跨未来不重新拟合。下表为4contexts×2MC reps均值，不是8独立随机训练。

| task | arm | marginal CRPS | whole-192 mean CRPS | terminal CRPS | max-ramp CRPS | sample seconds |
|---|---|---:|---:|---:|---:|---:|
| solar | native_joint | 0.235114 | 0.114389 | 0.998567 | 1.048485 | 0.101691 |
| solar | canonical96 | 0.241530 | 0.136043 | 0.897234 | 1.011506 | 0.260262 |
| solar | canonical8 | 0.292933 | 0.154894 | 1.065261 | 0.993896 | 3.862976 |
| solar | ar8 | 0.575703 | 0.126555 | 1.589044 | 1.687003 | 0.005610 |
| jena | native_joint | 0.455945 | 0.427539 | 1.010735 | 0.026368 | 0.100412 |
| jena | canonical96 | 0.463293 | 0.434002 | 1.028833 | 0.027473 | 0.263955 |
| jena | canonical8 | 0.669481 | 0.662734 | 1.463432 | 0.030529 | 3.874561 |
| jena | ar8 | 0.588496 | 0.565955 | 1.288280 | 0.011819 | 0.005545 |


普通canonical kernels无需训练就定义一致的自身block law，但改变原joint law。canonical96约2.6倍原single-head采样耗时；本小样本上solar mean CRPS变差、terminal却改善，Jena主分数变化小；canonical8更慢且主分数更差，AR8便宜但质量混合。不能只挑一个metric断言新框架必要；也不能称普通一致性修复没有实际代价。当前保留的是固定future-tail真实reveal下的quality/cost问题，尚非被证明未解决的机制。

原native诊断184.47秒，kernel控制74.23秒；峰值显存612.3MiB。端到端采样列含每arm必要history编码，模型加载/warmup单列未计该列。8个AR拟合时间另在原JSON。

只有每task4个早期history、2个MC repetitions；rep不是训练seed，重叠origin不是独立样本。MC CI只描述同history抽样误差，不是现实泛化区间；864个probes未校正多重比较。质量分数使用真实未来192点，但这些是先前探索过的早期载体、非独立TEST，预训练覆盖未知。未测试新增真实prefix后的固定tail条件质量，未执行FlowNP/SFM/ForcingDAS的完整训练；coherence≠accuracy。

![Query moments](outputs/analysis_v1/query_moments.png)

![Ordinary controls](outputs/analysis_v1/ordinary_kernel_controls.png)
