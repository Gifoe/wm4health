# Actual model and workload registry

| Work | Original source pin | Executed evidence | Comparison scope |
| --- | --- | --- | --- |
| WorldTS | 269426ea6217d7152a244da0e209d922acd46d4f | Four original NP/PJM cells previously completed | Original forecasting tasks |
| LatentTSF | 7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2 | Sixteen original ETTh1/Electricity cells previously completed; follow-up decoded-loss controls retained | Original forecasting tasks, separate from physics |
| Meta-iLaD | 3d47140f18262628d252858c3efdfe2ad1e21a86 | Original shipped MoCAP execution plus six complete paper/refined Pendulum evaluator calls from one physical checkpoint | Native qualification, then separately designed physical audit |

Three papers are not three matched baselines on one task. No current ranking or universal SOTA dominance is claimed. The newest controls use the same seed1000 Pendulum checkpoint, 20 pairs/160 roots/38 distinct reused environments. Sixty-four fixed-context draws are not 64 independently trained models. This turn adds zero weight-training fits, 200 context-latent optimizer calls, 60 query-state optimizer calls and five independent forward replays. Original full evaluation, point-estimate diagnostics and extra-information controls remain separate. [Current complete evidence](../CONTEXT_ESTIMATION_ATTRIBUTION_FINDINGS_V1.md).
