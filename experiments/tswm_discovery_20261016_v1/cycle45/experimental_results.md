# C45 Actual execution evidence

UTC 2026-10-17T11:05:22.881781+00:00。本轮PROGRESS；整体ACTIVE_UNACHIEVED，最终接受方向0。固定官方NCDSSM commit 7212bba9616d81c948f507eef9eb1f7d5cf1954e，48个选定source/config/license文件/Gitblob+SHA核查。原生线性推断与独立NumPy/dense Gaussian一致；非线性原生core完成10 discarded updates/862参数与预测回放，0正式比较fits。普通联合Markov路径控制核查了线性采样定义区别，不是新模型收益。

Native linear max absolute differences: mean6.94e-17, covariance2.78e-17, log likelihood4.44e-16; independent finite-difference gradient difference9.51e-12. A separate dense-block Gaussian conditioning implementation yields mean difference7.89e-17 and covariance difference6.25e-17. Retained sample statistics reconstruct exactly.

Both samplers match the known linear future marginal moments within predeclared MC tolerances. Their path dependence differs:

|Sampler|Cov(state1 at first,last future)|Exact joint covariance|max discrepancy in joint MC-SE units|Var(last−first), state1/state2|
|---|---:|---:|---:|---|
|native_marginal_time_samples|-0.0000247|0.0315289|59.110|0.142390 / 0.090841|
|ordinary_joint_Markov_samples|0.0310152|0.0315289|0.872|0.079771 / 0.044463|

Exact joint temporal-difference variances are0.078914/0.043891. This is a fixed-parameter numerical fixture, not a quality comparison between trained methods. The ordinary control resolves the discrepancy without changing learned dynamics.

The outer native forecast traced one auxiliary-history sample for B1 before generating16 futures; repeated-history B16/one-future calls traced16 independent auxiliary histories. This addresses the outer ensemble contract only; it does not itself turn the native marginal-time forecast into a nonlinear joint simulator.

Nonlinear qualification ran10 updates,862 parameters, all finite gradients/forecasts, checkpoint replay max0.0. Loss values on stochastic preflight draws are not convergence evidence. Original source and logs retained. Diagnostic scalar conversion emitted a nonfatal requires-grad warning; no numerical repair was made.

See [sampling-contract diagnostic](outputs/figures_v1/sampling_contract.png) and [C44 controlled quality/cost](outputs/figures_v1/controlled_quality_cost.png). The figures describe numerical/utility controls, not a new-method result.
