# C37 已验证学习结果

UTC 2026-10-16T23:04:09.714426+00:00。C37原作业完成，12个neural fits、2个HGB fits、48000个formal updates；独立复核通过。整体目标仍ACTIVE_UNACHIEVED，最终接受方向0。TEST没有评分。本轮是failure discovery与普通解释控制，不是新方法。

目标是可复用的时间序列world-model框架；两个站点仅为机制证据载体。不能把单数据集涨点、实现修正或已知copula/校准改名为最终研究方向。

## 来源与协议

完整冻结协议在[LEARNING_PROTOCOL.md](LEARNING_PROTOCOL.md)、[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md)。原publisher ZIP直接服务器正常TLS取得，126360077 bytes，SHA256 `d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154`。只用原Task15，未合并Tasks1–14重复；solar未来predictors含POWER，明确排除，标签只来自独立train15。历史168点、未来24点，daily-midnight+24h按组织者声明合同；没有虚构UTC或逐行真实发行vintage。

| task | FIT | CAL | VAL | TEST, unscored |
|---|---:|---:|---:|---:|
| wind | 383 | 88 | 66 | 76 |
| solar | 433 | 110 | 118 | 119 |


独立raw↔canonical、window/split/scaler、schedule、initial/selected checkpoint、VAL/CAL native路径及selected64-path评分重放通过；12个neural fits和2个HGB point模型。对全部56个已存路径arms重新计算9个score，共504个score cells；最大差1.21156307209e-10。native prediction/CAL重放最大差0（容差2e-4），不是声称逐bit为0。quantile/HGB同marginal排列恒等性通过。50-step和CAL/rank arms的score已独立重算，但不是各转换实现的全套独立重写；HGB点值已重放，不把随机residual路径生成称为第二套独立实现。

## 全部固定arms，raw/unclipped primary

| task | model | arm | fits | hourly CRPS | daily-mean CRPS | ramp CRPS | energy | variogram | MAE | coverage90 | width90 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wind | quantile_mlp | analogue | 2 | 0.156284 | 0.098352 | 0.085499 | 0.182625 | 0.033294 | 0.188905 | 0.383838 | 0.249586 |
| wind | quantile_mlp | analogue_calibrated | 2 | 0.135419 | 0.085257 | 0.099426 | 0.160217 | 0.035908 | 0.189595 | 0.819129 | 0.625463 |
| wind | quantile_mlp | iid | 2 | 0.156284 | 0.108149 | 0.075510 | 0.182934 | 0.033498 | 0.188905 | 0.383838 | 0.249586 |
| wind | quantile_mlp | iid_calibrated | 2 | 0.135419 | 0.098949 | 0.271567 | 0.162455 | 0.053536 | 0.189595 | 0.819129 | 0.625463 |
| wind | quantile_mlp | shuffle | 2 | 0.156284 | 0.097402 | 0.085371 | 0.182491 | 0.033421 | 0.188905 | 0.383838 | 0.249586 |
| wind | quantile_mlp | shuffle_calibrated | 2 | 0.135419 | 0.084403 | 0.093998 | 0.159816 | 0.035081 | 0.189595 | 0.819129 | 0.625463 |
| wind | kite_l1 | native | 2 | 0.173136 | 0.143709 | 0.088119 | 0.198642 | 0.033438 | 0.224690 | 0.578283 | 0.449286 |
| wind | kite_l1 | native_50steps | 2 | 0.174052 | 0.144751 | 0.088093 | 0.199704 | 0.033720 | 0.226681 | 0.593434 | 0.466738 |
| wind | kite_l1 | native_calibrated | 2 | 0.160076 | 0.132652 | 0.139709 | 0.184961 | 0.041594 | 0.224150 | 0.828914 | 0.810034 |
| wind | kite_l2 | native | 2 | 0.177218 | 0.140847 | 0.087824 | 0.204256 | 0.036832 | 0.227731 | 0.525253 | 0.412608 |
| wind | kite_l2 | native_50steps | 2 | 0.178162 | 0.141711 | 0.089142 | 0.205421 | 0.037209 | 0.229760 | 0.533460 | 0.424836 |
| wind | kite_l2 | native_calibrated | 2 | 0.161623 | 0.126204 | 0.187346 | 0.188234 | 0.049388 | 0.228131 | 0.839331 | 0.832160 |
| wind | hgb | analogue_iid | 1 | 0.109946 | 0.062399 | 0.226205 | 0.129004 | 0.043057 | 0.152904 | 0.827020 | 0.541805 |
| wind | hgb | analogue_joint | 1 | 0.109946 | 0.056696 | 0.079241 | 0.128791 | 0.030902 | 0.152904 | 0.827020 | 0.541805 |
| wind | hgb | unconditional_iid | 1 | 0.109400 | 0.059881 | 0.233662 | 0.129383 | 0.045465 | 0.152980 | 0.857323 | 0.563209 |
| wind | hgb | unconditional_joint | 1 | 0.109400 | 0.054284 | 0.080481 | 0.128273 | 0.030679 | 0.152980 | 0.857323 | 0.563209 |
| solar | quantile_mlp | analogue | 2 | 0.032755 | 0.017567 | 0.052184 | 0.056952 | 0.008310 | 0.041772 | 0.805614 | 0.096716 |
| solar | quantile_mlp | analogue_calibrated | 2 | 0.031293 | 0.016837 | 0.047616 | 0.054276 | 0.007913 | 0.043178 | 0.872352 | 0.172760 |
| solar | quantile_mlp | iid | 2 | 0.032755 | 0.018910 | 0.048420 | 0.056932 | 0.009506 | 0.041772 | 0.805614 | 0.096716 |
| solar | quantile_mlp | iid_calibrated | 2 | 0.031293 | 0.017826 | 0.062582 | 0.054765 | 0.009144 | 0.043178 | 0.872352 | 0.172760 |
| solar | quantile_mlp | shuffle | 2 | 0.032755 | 0.017442 | 0.051304 | 0.056820 | 0.008337 | 0.041772 | 0.805614 | 0.096716 |
| solar | quantile_mlp | shuffle_calibrated | 2 | 0.031293 | 0.016688 | 0.047588 | 0.054045 | 0.007906 | 0.043178 | 0.872352 | 0.172760 |
| solar | kite_l1 | native | 2 | 0.032217 | 0.018113 | 0.043732 | 0.056171 | 0.008045 | 0.041276 | 0.686088 | 0.081363 |
| solar | kite_l1 | native_50steps | 2 | 0.032196 | 0.018095 | 0.042788 | 0.056092 | 0.008286 | 0.041564 | 0.718220 | 0.086016 |
| solar | kite_l1 | native_calibrated | 2 | 0.030604 | 0.016613 | 0.039449 | 0.052400 | 0.007816 | 0.040802 | 0.786017 | 0.185654 |
| solar | kite_l2 | native | 2 | 0.033060 | 0.019776 | 0.047768 | 0.057575 | 0.008713 | 0.041780 | 0.686088 | 0.077154 |
| solar | kite_l2 | native_50steps | 2 | 0.032995 | 0.019875 | 0.046685 | 0.057511 | 0.008887 | 0.041891 | 0.705685 | 0.080738 |
| solar | kite_l2 | native_calibrated | 2 | 0.030976 | 0.017678 | 0.041785 | 0.053052 | 0.008478 | 0.041984 | 0.837394 | 0.194196 |
| solar | hgb | analogue_iid | 1 | 0.029837 | 0.014786 | 0.050454 | 0.048494 | 0.008642 | 0.038706 | 0.682556 | 0.167372 |
| solar | hgb | analogue_joint | 1 | 0.029837 | 0.015280 | 0.036194 | 0.049070 | 0.007693 | 0.038706 | 0.682556 | 0.167372 |
| solar | hgb | unconditional_iid | 1 | 0.032731 | 0.015961 | 0.047457 | 0.051197 | 0.009360 | 0.043000 | 0.646540 | 0.170612 |
| solar | hgb | unconditional_joint | 1 | 0.032731 | 0.015709 | 0.039195 | 0.050695 | 0.008244 | 0.043000 | 0.646540 | 0.170612 |


各seed值、MSE、clipped共享敏感性等完整字段在[all_results.csv](outputs/analysis_v1/all_results.csv)和[analysis.json](outputs/analysis_v1/analysis.json)，没有只挑有利对照。数值单位为normalized power；coverage应接近.9，width须与coverage联读，均不是越小越好。

## 全部预列主指标对比

下表candidate改善定义reference−candidate，正数较好。neural行先在相同origin上平均两个seed的score；HGB只有一个fit。所有7个loss指标、两个独立seed与mean-score结果见[paired_comparisons.csv](outputs/analysis_v1/paired_comparisons.csv)。

| task | prelisted comparison | metric | reference | candidate | improvement % | paired 95% CI, abs |
|---|---|---|---:|---:|---:|---|
| wind | quantile_shuffle_vs_iid | crps | 0.156284 | 0.156284 | 0.00 | [-0.000000, 0.000000] |
| wind | quantile_shuffle_vs_iid | daily_mean_crps | 0.108149 | 0.097402 | 9.94 | [0.009582, 0.011849] |
| wind | quantile_analogue_vs_iid | crps | 0.156284 | 0.156284 | 0.00 | [-0.000000, 0.000000] |
| wind | quantile_analogue_vs_iid | daily_mean_crps | 0.108149 | 0.098352 | 9.06 | [0.008599, 0.010910] |
| wind | kite_l2_vs_l1 | crps | 0.173136 | 0.177218 | -2.36 | [-0.021497, 0.014139] |
| wind | kite_l2_vs_l1 | daily_mean_crps | 0.143709 | 0.140847 | 1.99 | [-0.019594, 0.023874] |
| wind | quantile_shuffle_vs_iid_calibrated | crps | 0.135419 | 0.135419 | 0.00 | [-0.000000, 0.000000] |
| wind | quantile_shuffle_vs_iid_calibrated | daily_mean_crps | 0.098949 | 0.084403 | 14.70 | [0.011725, 0.017452] |
| wind | quantile_analogue_vs_iid_calibrated | crps | 0.135419 | 0.135419 | 0.00 | [-0.000000, 0.000000] |
| wind | quantile_analogue_vs_iid_calibrated | daily_mean_crps | 0.098949 | 0.085257 | 13.84 | [0.010674, 0.016728] |
| wind | kite_l2_vs_l1_calibrated | crps | 0.160076 | 0.161623 | -0.97 | [-0.016802, 0.014198] |
| wind | kite_l2_vs_l1_calibrated | daily_mean_crps | 0.132652 | 0.126204 | 4.86 | [-0.011488, 0.023783] |
| wind | quantile_iid_calibration | crps | 0.156284 | 0.135419 | 13.35 | [0.018342, 0.024258] |
| wind | quantile_iid_calibration | daily_mean_crps | 0.108149 | 0.098949 | 8.51 | [0.007882, 0.010822] |
| wind | quantile_shuffle_calibration | crps | 0.156284 | 0.135419 | 13.35 | [0.018271, 0.024044] |
| wind | quantile_shuffle_calibration | daily_mean_crps | 0.097402 | 0.084403 | 13.35 | [0.010154, 0.016576] |
| wind | quantile_analogue_calibration | crps | 0.156284 | 0.135419 | 13.35 | [0.018162, 0.024310] |
| wind | quantile_analogue_calibration | daily_mean_crps | 0.098352 | 0.085257 | 13.31 | [0.009775, 0.016647] |
| wind | kite_l1_calibration | crps | 0.173136 | 0.160076 | 7.54 | [0.007889, 0.018033] |
| wind | kite_l1_calibration | daily_mean_crps | 0.143709 | 0.132652 | 7.69 | [0.004688, 0.017328] |
| wind | kite_l1_50_vs_10 | crps | 0.173136 | 0.174052 | -0.53 | [-0.002101, 0.000035] |
| wind | kite_l1_50_vs_10 | daily_mean_crps | 0.143709 | 0.144751 | -0.73 | [-0.002610, 0.000210] |
| wind | kite_l2_calibration | crps | 0.177218 | 0.161623 | 8.80 | [0.011253, 0.020386] |
| wind | kite_l2_calibration | daily_mean_crps | 0.140847 | 0.126204 | 10.40 | [0.008582, 0.019889] |
| wind | kite_l2_50_vs_10 | crps | 0.177218 | 0.178162 | -0.53 | [-0.002345, 0.000071] |
| wind | kite_l2_50_vs_10 | daily_mean_crps | 0.140847 | 0.141711 | -0.61 | [-0.002423, 0.000346] |
| wind | kite_l2_native_vs_quantile_analogue_calibrated | crps | 0.135419 | 0.177218 | -30.87 | [-0.069919, -0.017785] |
| wind | kite_l2_native_vs_quantile_analogue_calibrated | daily_mean_crps | 0.085257 | 0.140847 | -65.20 | [-0.090369, -0.025082] |
| wind | kite_l2_native_vs_hgb_analogue_joint | crps | 0.109946 | 0.177218 | -61.19 | [-0.090558, -0.045738] |
| wind | kite_l2_native_vs_hgb_analogue_joint | daily_mean_crps | 0.056696 | 0.140847 | -148.42 | [-0.107829, -0.062527] |
| wind | kite_l2_native_calibrated_vs_quantile_analogue_calibrated | crps | 0.135419 | 0.161623 | -19.35 | [-0.048947, -0.005102] |
| wind | kite_l2_native_calibrated_vs_quantile_analogue_calibrated | daily_mean_crps | 0.085257 | 0.126204 | -48.03 | [-0.074007, -0.014444] |
| wind | kite_l2_native_calibrated_vs_hgb_analogue_joint | crps | 0.109946 | 0.161623 | -47.00 | [-0.072951, -0.033848] |
| wind | kite_l2_native_calibrated_vs_hgb_analogue_joint | daily_mean_crps | 0.056696 | 0.126204 | -122.60 | [-0.090328, -0.051170] |
| wind | hgb_unconditional_joint_vs_iid | crps | 0.109400 | 0.109400 | 0.00 | [-0.000000, 0.000000] |
| wind | hgb_unconditional_joint_vs_iid | daily_mean_crps | 0.059881 | 0.054284 | 9.35 | [0.002851, 0.009090] |
| wind | hgb_analogue_joint_vs_iid | crps | 0.109946 | 0.109946 | -0.00 | [-0.000000, 0.000000] |
| wind | hgb_analogue_joint_vs_iid | daily_mean_crps | 0.062399 | 0.056696 | 9.14 | [0.004012, 0.007690] |
| solar | quantile_shuffle_vs_iid | crps | 0.032755 | 0.032755 | 0.00 | [-0.000000, 0.000000] |
| solar | quantile_shuffle_vs_iid | daily_mean_crps | 0.018910 | 0.017442 | 7.76 | [0.001144, 0.001749] |
| solar | quantile_analogue_vs_iid | crps | 0.032755 | 0.032755 | 0.00 | [-0.000000, 0.000000] |
| solar | quantile_analogue_vs_iid | daily_mean_crps | 0.018910 | 0.017567 | 7.10 | [0.001046, 0.001682] |
| solar | kite_l2_vs_l1 | crps | 0.032217 | 0.033060 | -2.62 | [-0.002192, 0.000693] |
| solar | kite_l2_vs_l1 | daily_mean_crps | 0.018113 | 0.019776 | -9.18 | [-0.003705, 0.000345] |
| solar | quantile_shuffle_vs_iid_calibrated | crps | 0.031293 | 0.031293 | 0.00 | [-0.000000, 0.000000] |
| solar | quantile_shuffle_vs_iid_calibrated | daily_mean_crps | 0.017826 | 0.016688 | 6.38 | [0.000660, 0.001628] |
| solar | quantile_analogue_vs_iid_calibrated | crps | 0.031293 | 0.031293 | -0.00 | [-0.000000, 0.000000] |
| solar | quantile_analogue_vs_iid_calibrated | daily_mean_crps | 0.017826 | 0.016837 | 5.55 | [0.000501, 0.001448] |
| solar | kite_l2_vs_l1_calibrated | crps | 0.030604 | 0.030976 | -1.21 | [-0.001370, 0.000655] |
| solar | kite_l2_vs_l1_calibrated | daily_mean_crps | 0.016613 | 0.017678 | -6.41 | [-0.002753, 0.000718] |
| solar | quantile_iid_calibration | crps | 0.032755 | 0.031293 | 4.46 | [0.000209, 0.002819] |
| solar | quantile_iid_calibration | daily_mean_crps | 0.018910 | 0.017826 | 5.73 | [-0.000115, 0.002292] |
| solar | quantile_shuffle_calibration | crps | 0.032755 | 0.031293 | 4.46 | [0.000164, 0.002745] |
| solar | quantile_shuffle_calibration | daily_mean_crps | 0.017442 | 0.016688 | 4.32 | [-0.000354, 0.001825] |
| solar | quantile_analogue_calibration | crps | 0.032755 | 0.031293 | 4.46 | [0.000177, 0.002762] |
| solar | quantile_analogue_calibration | daily_mean_crps | 0.017567 | 0.016837 | 4.16 | [-0.000339, 0.001780] |
| solar | kite_l1_calibration | crps | 0.032217 | 0.030604 | 5.01 | [0.000551, 0.002597] |
| solar | kite_l1_calibration | daily_mean_crps | 0.018113 | 0.016613 | 8.28 | [0.000807, 0.002264] |
| solar | kite_l1_50_vs_10 | crps | 0.032217 | 0.032196 | 0.07 | [-0.000132, 0.000185] |
| solar | kite_l1_50_vs_10 | daily_mean_crps | 0.018113 | 0.018095 | 0.10 | [-0.000112, 0.000153] |
| solar | kite_l2_calibration | crps | 0.033060 | 0.030976 | 6.30 | [0.001228, 0.002842] |
| solar | kite_l2_calibration | daily_mean_crps | 0.019776 | 0.017678 | 10.61 | [0.001538, 0.002691] |
| solar | kite_l2_50_vs_10 | crps | 0.033060 | 0.032995 | 0.20 | [-0.000039, 0.000196] |
| solar | kite_l2_50_vs_10 | daily_mean_crps | 0.019776 | 0.019875 | -0.50 | [-0.000197, 0.000002] |
| solar | kite_l2_native_vs_quantile_analogue_calibrated | crps | 0.031293 | 0.033060 | -5.64 | [-0.004127, 0.001005] |
| solar | kite_l2_native_vs_quantile_analogue_calibrated | daily_mean_crps | 0.016837 | 0.019776 | -17.45 | [-0.005984, 0.000260] |
| solar | kite_l2_native_vs_hgb_analogue_joint | crps | 0.029837 | 0.033060 | -10.80 | [-0.005115, -0.001267] |
| solar | kite_l2_native_vs_hgb_analogue_joint | daily_mean_crps | 0.015280 | 0.019776 | -29.42 | [-0.007080, -0.001777] |
| solar | kite_l2_native_calibrated_vs_quantile_analogue_calibrated | crps | 0.031293 | 0.030976 | 1.01 | [-0.002078, 0.003430] |
| solar | kite_l2_native_calibrated_vs_quantile_analogue_calibrated | daily_mean_crps | 0.016837 | 0.017678 | -5.00 | [-0.004027, 0.002496] |
| solar | kite_l2_native_calibrated_vs_hgb_analogue_joint | crps | 0.029837 | 0.030976 | -3.82 | [-0.003067, 0.001006] |
| solar | kite_l2_native_calibrated_vs_hgb_analogue_joint | daily_mean_crps | 0.015280 | 0.017678 | -15.69 | [-0.005037, 0.000315] |
| solar | hgb_unconditional_joint_vs_iid | crps | 0.032731 | 0.032731 | 0.00 | [-0.000000, 0.000000] |
| solar | hgb_unconditional_joint_vs_iid | daily_mean_crps | 0.015961 | 0.015709 | 1.58 | [-0.000103, 0.000587] |
| solar | hgb_analogue_joint_vs_iid | crps | 0.029837 | 0.029837 | 0.00 | [-0.000000, 0.000000] |
| solar | hgb_analogue_joint_vs_iid | daily_mean_crps | 0.014786 | 0.015280 | -3.34 | [-0.000980, -0.000021] |


## 训练健康与成本

| task | fit | selected step | params | fit seconds* | 128-path VAL seconds | cap flag |
|---|---|---:|---:|---:|---:|---|
| wind | quantile_mlp_2026371 | 500 | 254088 | 16.52 | 0.0113 | False |
| wind | kite_l1_2026371 | 500 | 361477 | 155.42 | 0.6349 | False |
| wind | kite_l2_2026371 | 500 | 361477 | 155.83 | 0.6499 | False |
| wind | quantile_mlp_2026372 | 500 | 254088 | 16.11 | 0.0255 | False |
| wind | kite_l1_2026372 | 500 | 361477 | 155.95 | 0.6438 | False |
| wind | kite_l2_2026372 | 500 | 361477 | 155.67 | 0.6513 | False |
| solar | quantile_mlp_2026371 | 500 | 401544 | 17.75 | 0.0275 | False |
| solar | kite_l1_2026371 | 1500 | 364037 | 162.03 | 1.0287 | False |
| solar | kite_l2_2026371 | 1500 | 364037 | 162.45 | 1.0329 | False |
| solar | quantile_mlp_2026372 | 500 | 401544 | 17.53 | 0.0228 | False |
| solar | kite_l1_2026372 | 2500 | 364037 | 162.44 | 1.0351 | False |
| solar | kite_l2_2026372 | 2500 | 364037 | 162.11 | 1.0096 | False |


*fit_seconds包含该fit训练及后续固定路径/校准/评分工作；128-path列是单独同步测量的VAL native预测时间。不能从较大batch摊销时间推断单路径成本。原worker总用时1416.23秒，独立verifier50.78秒。

两个seed、两个固定zone、一个小时级日窗口协议，全部是探索VAL结果，不是独立TEST确认。wind VAL原105候选仅66完整case，39因缺失被排除；TEST76未评分。solar VAL118/TEST119。预算cap标记0/12，初始step0被选0/12；无cap标记不等于已证明收敛。参数量/训练目标/CPU树与GPU网络FLOPs并未严格匹配，不能仅凭排名断言特定state机制缺失。bootstrap按7个日历日而非连续row号，2000次；CI探索性、未校正多重比较；两个seed的score平均不是forecast ensemble，也不是大量独立随机重复。

![固定普通对照](outputs/figures_v1/conditional_path_controls.png)

![全部学习曲线](outputs/figures_v1/learning_curves.png)

本轮本身未提出或训练新的world-model方法；如何解释数值必须保留普通控制与上述限制。初步进行中报告已完整归档reports_in_progress_v1。
