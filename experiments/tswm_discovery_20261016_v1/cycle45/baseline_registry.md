# C45 Native baseline registry

UTC 2026-10-17T11:05:22.881781+00:00。本轮PROGRESS；整体ACTIVE_UNACHIEVED，最终接受方向0。固定官方NCDSSM commit 7212bba9616d81c948f507eef9eb1f7d5cf1954e，48个选定source/config/license文件/Gitblob+SHA核查。原生线性推断与独立NumPy/dense Gaussian一致；非线性原生core完成10 discarded updates/862参数与预测回放，0正式比较fits。普通联合Markov路径控制核查了线性采样定义区别，不是新模型收益。

|Control|Actual execution|What it establishes|What it does not establish|
|---|---|---|---|
|Author BaseLTI|fixed known2D parameters; native matrix-exponential prediction/filter|numerical likelihood/mean/covariance/gradient contract|training quality or a new model|
|Native Base.forecast|16384 default state draws, same filtered history|timewise-marginal sampling behavior|a joint Markov trajectory sample contract|
|Ordinary joint Markov sampler|same linear process parameters/posterior,16384 draws|standard complete joint-law control|a new architecture or nonlinear repair|
|Author NCDSSMNL|z6/aux2/y2/MLP64/RK4.05;10 discarded updates; CPU|real core forward/backward/forecast and checkpoint replay|convergence, paper-headline reproduction or a trained framework benefit|

Source subset excludes data/datasets directories, media and checkpoints; core numerical files are unmodified. All loaded native module origins were checked inside this pinned cache in a fresh process. ServerPython3.10/torch2.8cu128/torchdiffeq0.2.3 differs from declaredtorch~=1.11; no global downgrade, dependency installation or numerical source patch was required. Core execution did not require the optional TensorboardX training shell. CUDA was never initialized by these tests.

Control inputs are explicitly unimplemented in this source constructor (`u_dim==0`). No action-conditioned capability is inferred. Nonlinear joint SDE trajectory sampling with independent outer auxiliary-history draws still needs an ordinary disclosed adapter and numerical convergence checks before any full-law method comparison.
