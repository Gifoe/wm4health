# C41 反证与排除

1. LIFTS原生实现可执行并提供随机预测/插补；不能依据某个“deterministic”注释说模型没有不确定性。
2. 精确posterior在固定错误transition下可能比补偿representation预测差；ordinary正确transition消除该效应。因此不给“校准与动力学冲突”反例冠以新机制。
3. train view–system图连通不意味着held-out observable law可确定；缺锚的对齐歧义是信息边界，不是已有模型算法失败。
4. 普通24-permutation Bayesian mixture在锚充分时能消除本例误差；候选必须越过它及joint hierarchical SSM等完整控制，而不是只胜过MAP/点估计或独立每cell模型。
5. V1 Gaussian控制因native float32返回Sigma的数值非对称在1.65e-9 mean check停止；V2按source实际Cholesky取有效对称cov，不改容差/科学效应。V1代码/failure保留。
6. 首个独立verifier把cross rows变量在Gaussian段覆盖，统计count误报5；其全部2304行数值已循环核查。V2改名并assert2304，输出9216cells；原verifier及输出保留。不把绿色status当覆盖证明。
7. 新候选没有comparative neural fit/真实多域效果，不能称发现了足以投稿的gap。旧C11/C15/C25/C32–40被关闭的动机不因本轮重新开放。
