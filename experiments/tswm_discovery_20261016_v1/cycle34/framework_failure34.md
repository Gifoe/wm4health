# C34 critic/methodology note: test predictive responsibility before proposing another context gate

Date: 2026-10-17. Perspective: independent critic/methodology search under the supplied C34 brief. Status: a bounded research question and a prospective protocol; no accepted method, executed experiment, or invented result.

## Abstract

Competent numerical time-series world models already support observational forecasting with learned predictive states and external information. Their scientific value does not depend on causal interventions, a universal checkpoint, or recovering a physical state. The useful remaining question is narrower: when numerical prediction and contextual correction are trained together, does the correction add information, or partly compensate for a numerical pathway that joint training has weakened? A recent native ablation provides a reason to measure this distinction, but does not establish its cause. Existing residual supervision, numerical references, decoder-side adapters, multimodal optimization balancing, and ordinary forecast combinations cover much of the proposed solution space. Accordingly, the minimum viable investigation is a controlled interaction experiment, not a claim that residual branches or gating are new. It transfers components to a released framework on historical-only numerical covariates, compares latent and observation residuals, protects or jointly trains the numerical predictor, and includes validation-fitted combinations and a competent exogenous forecaster. A negative result closes this particular mechanism route; a positive result would justify a broader independent replication and closer novelty audit.

## 1. Frozen questions and scope

The exact intake questions are:

1. Among actual recent numerical TS world-model frameworks, what predictive/state/dynamics capabilities are claimed and empirically supported, under which input and training contracts? Distinguish source facts, reported results and untested inference.
2. Which cross-task capability tension or failure remains worth measuring after task-matched ordinary inference/training/representation controls? A source bug, mismatched inputs, one bad score or theoretical impossibility under unavailable information does not establish a new architecture problem.
3. What concrete mechanism interaction, closest existing methods, new independent public protocol and small decisive RTX5090 experiment would distinguish a defensible candidate? Supply a constructive minimum viable question or rescue route before recommending a pivot; component prior art alone is not a rejection.

The target is a reusable framework, allowing separate training on each dataset. This note imposes no zero-shot, shared-checkpoint, action, causal, physical-state, LLM, or online-switch requirement. The C33 NP/PJM decoder experiment is closed as a final-method motivation. No NP/PJM rerun, new favorable generator, or source defect is proposed here.

## 2. Search and verification

The primary window was 2026-04-17 through 2026-10-17. Search rounds covered: recent numerical world models and their native ablations; latent context corrections versus decoded utility; text-collapse/residual supervision; ordinary ensemble and gated-adapter alternatives; multimodal branch competition; and released exogenous-forecasting code. Earlier work is marked as foundation/overlap rather than presented as a recent discovery.

Identity and claim were checked separately using author arXiv abstract pages, full text, proceedings, and author code. Third-party review sites and summaries were discovery leads only. Main verified sources are WorldTS, Pythia, Phys-JEPA, REST-TS, MUSE-Bench, TimesX, and the overlap/control references below. The author WorldTS README and public numerical core were inspected at revision `269426ea6217d7152a244da0e209d922acd46d4f`; local files were read only for input-contract inspection. Pythia's document says its code and models are to be released; this note therefore proposes no immediate native Pythia reproduction. Prepared data bytes and the future experiment's runtime were not verified or downloaded in this intake.

The global three-perspective search is coordinated by the parent task. This note owns only the critic/methodology perspective; it does not claim exhaustive coverage of every time-series architecture.

## 3. Taxonomy: three explanations that must remain separate

| Explanation | Observable signature | What would discriminate it | What it would not establish |
| --- | --- | --- | --- |
| Surrogate/readout mismatch | Better latent matching without a corresponding forecast improvement | Common frozen target space; paired forecast/latent measurements; competent matched readout refit | Missing latent information, a new dynamics mechanism, or a need for physical state semantics |
| Division of predictive responsibility | Context improves the fused state while the numerical-only pathway becomes worse during joint optimization | Track the numerical pathway; isolate context gradients; compare an explicit numerical self-prediction objective | Causal identification or the usefulness of any particular covariate in the real system |
| Ordinary combination/selection effect | A context model helps some tasks or lead blocks, but a simple stack or gate removes most losses | Validation-fitted convex combinations, conditional gates, and direct residual supervision with identical inputs | Novelty of a latent architecture merely because aggregate scores improve |

WorldTS and Pythia address latent forecasting under different training contracts, whereas REST-TS and decoder-side adapters supply strong alternatives for assigning contextual responsibility. The relevant comparison is therefore between training and combination mechanisms under the same information, not between labels such as “world model” and “forecaster.”

## 4. Native evidence: useful tensions, limited causal conclusions

### 4.1 Competent forecasting is already supported

| Framework and contract | Native evidence relevant to this note | Boundary |
| --- | --- | --- |
| WorldTS: dataset-specific, regularly sampled forecasting; covariate-conditioned predictive states followed by a frozen-representation decoder stage | Reports 21 real datasets. Table 4's one-stage observation-MAE variant is worse than the full recipe on its three representative tasks; removing covariates is also worse. | These findings support the full recipe under the reported contracts. They do not isolate branch protection or establish an unavoidable information deficit. |
| Pythia: shared pretraining; historical numerical covariates and timestamp-filtered context; frozen numerical target space and separate probabilistic readout | Reports a 6.26% MASE improvement over the strongest published MUSE baseline. | Different pretraining and input contracts prevent treating it as a controlled comparison with WorldTS. |
| Phys-JEPA: multivariate forecasting with supplied physical descriptors and static/dynamic latent constraints | At H=24, Electricity favors static-only over the full objective; Traffic favors the combination. | This is native evidence of a dataset-dependent interaction. Its small initial study is not the strongest general framework motivation. |

Sources: [WorldTS full text, §§5.1–5.4 and Appendix C](https://arxiv.org/html/2609.31162v1); [Pythia full text](https://arxiv.org/html/2610.05240v1); [Phys-JEPA, Table 6](https://arxiv.org/html/2606.16076v1).

WorldTS's favorable loss ablations and Phys-JEPA's mixed constraint interaction both argue against treating an added objective as inherently helpful. They also do not justify the reverse conclusion that combinations of known objectives cannot produce a useful finite-budget framework.

### 4.2 The strongest diagnostic is a native controlled reversal

Pythia Table 14 reports:

| Small-model training variant | Numerical latent error b | Corrected latent error a | MASE | WSQL |
| --- | ---: | ---: | ---: | ---: |
| Numerical-reference guidance | 0.03614 | 0.02291 | 0.6981 | 0.4373 |
| Without reference objective | 0.03891 | 0.02236 | 0.7086 | 0.4404 |

Both use the same frozen target representation. The unguided variant obtains a lower final latent error but worse forecasts. The guided model's numerical estimate is better. Section 3.2 states that the numerical estimate has no separate first-stage forecasting loss. Do not conflate this contrast with Table 16's separately trained numerical-only comparison. [Pythia, §§3.2, F.4 and G.2](https://arxiv.org/html/2610.05240v1).

This is sufficient to reject “lower scalar latent error automatically means a more useful predictive state” as an evaluation rule. It is insufficient to identify why the reversal occurs. The latent statistic is an arithmetic group macro-average while the forecast statistics use geometric aggregation; aggregation differences can contribute to an aggregate reversal. The better numerical latent error also does not by itself establish a better decoded numerical forecast. Possible explanations include different allocation of learning to the numerical and context paths, different latent directions being emphasized, observation-loss compatibility, and ordinary generalization variance. C33 already makes broad decoder-compatibility claims a weak basis for a new method. The defensible next step is to separate these explanations experimentally.

### 4.3 Native failures elsewhere set a strong control bar

REST-TS identifies content-independent text behavior and supervises a text branch on structured numerical forecast residuals. Thus “give context its own residual target” is occupied territory. Conversely, TimesX reports that equal-weight TimesFM/Gemini averaging reaches MASE 0.619 while CodeRev reaches 0.713 and TimesFM alone 0.645. This makes ordinary combination a serious alternative, rather than a token baseline. These results concern different inputs and models; they motivate the control family, not a numerical-context performance prediction. [REST-TS author abstract and paper](https://arxiv.org/abs/2606.19413); [TimesX full text, Table 4](https://arxiv.org/html/2607.06973v1).

MUSE-Bench separates context availability from event occurrence and evaluates context removals and incorrect/misaligned context. Its scope complements TimesX's ordinary-combination warning: context relevance and access must be controlled before interpreting model differences. This protocol does not manufacture corrupted-context tasks to make a model fail. [MUSE-Bench, §3.1.2 and §4](https://arxiv.org/html/2609.15087v1).

## 5. Closest-overlap challenge

| Existing mechanism | Already covered | What a candidate would still have to show |
| --- | --- | --- |
| REST-TS, June 2026 | Independent numerical forecast plus residual-exclusive supervision for the text pathway | A measurable benefit of the latent-state interaction for numerical covariates over the direct residual counterpart. Merely changing text to numerical context is weak novelty. |
| Pythia, October 2026 | Numerical-reference guidance and contextual latent residuals, including detached-reference gradients | An effect of explicitly preserving numerical self-prediction, rather than renaming its reference objective. |
| LightGTS-Cov, February 2026 — earlier overlap | Lightweight post-decoder residual fusion for past and future covariates, typically with a frozen backbone | A sample/compute advantage beyond ordinary decoder-side fusion; no claim that frozen-backbone residual fusion is new. [Author paper](https://arxiv.org/html/2602.10412v1) |
| Gradient-Blending, CVPR 2020 — older foundation | Modality-specific losses and weights based on differing generalization/overfitting rates | A specific predictive-state interaction, not a generic claim that modality optimization is imbalanced. [Primary paper](https://conferences.computer.org/cvpr/pdfs/CVPR2020-1XMljIyuXWg2zC9bnL19Tw/716800m2692/716800m2692.pdf) |
| TimesX, ICML 2026 | Simple forecast ensembles and conservative revision controls | Gains beyond a validation-fitted convex combination and a small conditional gate. [Proceedings identity](https://proceedings.mlr.press/v306/liu26go.html) |
| Certificate-driven calibration, UAI 2026 | Gated residual heads on a fixed forecasting backbone under online shift | No novelty claim for “revert to the backbone when unreliable.” Its online objective is different from the offline experiment proposed here. [Proceedings](https://proceedings.mlr.press/v337/huang26b.html) |
| TimeXer, NeurIPS 2024 — older strong control | Task-matched numerical exogenous forecasting with released implementation | A gain over a competent ordinary model using all the same numerical history. [Author repo](https://github.com/thuml/TimeXer) |

The overlap is substantial. A residual branch, stop-gradient, a gate, and a latent loss assembled together are not yet a paper contribution. However, overlap alone does not settle whether a particular division of predictive responsibility yields a non-additive finite-data/compute advantage. The experiment below is designed to decide whether that narrower claim deserves further work.

## 6. Constructive minimum viable question

**Under historical-only, temporally resolved numerical covariates, can preserving the numerical predictor's own forecasting task while learning a contextual latent innovation improve the forecast/compute frontier beyond equally protected observation residuals and ordinary forecast combinations?**

The material setting difference is full historical numerical covariate trajectories, with no future exogenous values and no semantic encoder. Information usefulness can vary across lead times and domains. The object of study is who learns the target-history signal and who learns the incremental covariate signal during training. It is not decoder expressivity, test-time causal responses, or “robustness to noise” created after seeing results.

A conservative implementation uses a common target-state encoder E, a numerical predictor n=P(E(y_history)), a context encoder C(c_history), and a correction g. The putative interaction is between (i) a stable target-state learning space, and (ii) gradient separation that requires the numerical path to retain an independently useful forecast. Both ingredients are known. Their finite-budget interaction here is a hypothesis, with novelty still uncertain.

### 6.1 Competing hypotheses and decisive observations

| Hypothesis | Evidence expected if true | Falsifier or alternative interpretation |
| --- | --- | --- |
| H1: joint learning reallocates numerical prediction into contextual compensation | With a common E and matched readout, the jointly trained numerical-only path degrades while fused predictions improve; protection restores that path and improves fused generalization | Numerical-path error never worsens, or branch protection brings no forecast benefit. Do not call the phenomenon compensation merely from a latent norm. |
| H2: latent innovations offer a budget-specific advantage in contextual learning | Protected latent corrections beat protected output residuals at equal data/compute, especially before the learning curves converge | Direct residuals match/beat them, or the advantage disappears when input detail, regularization, or compute is matched. Denoising is not established merely by a faster learning curve. |
| H3: the effect is only an extra numerical objective or extra optimization | Joint latent training plus the identical numerical loss gives the same gains as gradient isolation | Reject the claimed isolation interaction; retain at most an ordinary auxiliary-loss result. |
| H4: the apparent gain is ordinary model selection | A validation-fitted stack or conditional gate matches the candidate across domains/lead blocks | Reject the new-framework motivation even if the candidate beats native WorldTS. |
| H5: decoder adaptation explains the difference | A competent common readout refit removes the latent-arm differences | Close the dynamics/branch-mechanism route; do not reopen C33 as a decoder innovation. |

## 7. New independent public protocol: historical-context responsibility test

This is a prospective component-transfer protocol, not a reproduction of Pythia and not a reported failure of WorldTS. No code, data download, or fit was executed for it in C34.

### 7.1 Public tasks and access contract

Use the existing **ETTh1** and **Weather** exogenous tasks from the author TimeXer repository: target column `OT`, all other numerical channels as historical covariates, lookback 96, forecast horizon 96, `features=MS`. These tasks are chosen before outcomes because they provide two different sensor settings with existing exogenous-task definitions. They are a mechanism pilot, not adequate evidence of universal framework superiority. Public preparation routes are linked in the [author README](https://github.com/thuml/TimeXer); the exact [ETTh1 script](https://raw.githubusercontent.com/thuml/TimeXer/main/scripts/forecast_exogenous/ETTh1/TimeXer.sh) and [Weather script](https://raw.githubusercontent.com/thuml/TimeXer/main/scripts/forecast_exogenous/Weather/TimeXer.sh) confirm these task settings.

Keep the source data split boundaries: ETTh1's first 12×30×24 observations for training, next 4×30×24 for validation, next 4×30×24 for test; Weather's 70/10/20 chronological split. Fit scaling only on training observations. These definitions are in the [author data loader](https://raw.githubusercontent.com/thuml/TimeXer/main/data_provider/data_loader.py). Record dataset SHA256, header, row count, timestamps and exclusions before any scoring. Actual prepared CSV availability/bytes remain a launch prerequisite, not a verified result of this note.

Every method receives exactly `(target_history, covariate_history, permissible calendar metadata)`; target futures enter only a training-loss object or the sealed scorer. **No future numerical covariates are accessible to the model.** Do not produce them and rely on a flag to ignore them. All models see the complete historical covariate trajectories; do not compare a summary-based model against a full-trajectory model and attribute the result to latent learning.

WorldTS's released numerical core has a historical covariate encoder and permits `use_future_x_condition=False`, rejecting a supplied `x_future` in that mode. This establishes interface compatibility with the proposed historical-only component transfer. It does not establish native historical-only performance on these tasks. Its public quickstart instead launches the `w_future` numerical scripts, so the new scores must be labeled as a changed input regime. [Pinned numerical core](https://raw.githubusercontent.com/decisionintelligence/WorldTS/269426ea6217d7152a244da0e209d922acd46d4f/DAG-bench/ts_benchmark/baselines/worldts/models/worldts_model.py); [pinned README](https://raw.githubusercontent.com/decisionintelligence/WorldTS/269426ea6217d7152a244da0e209d922acd46d4f/README.md).

### 7.2 Common initialization and arms

For each dataset and each of three training seeds, fit a history-only numerical teacher using the released WorldTS modules and a competent forecast readout. Freeze its target encoder E for the mechanism arms so latent errors share a coordinate system. Copy the numerical predictor and readout initialization into every relevant arm. Context-module dimensions, context payload, normalization, train examples, and batch order are paired. A fixed latent target prevents an apparent gain from being only a change in the target geometry. Throughout the formulas, D(n) is shorthand for a decoder that also receives its permitted history memory and normalization statistics; these inputs must not be silently dropped.

Use the following arms; these labels are experiment identifiers, not proposed method names:

| Arm | Training difference | Purpose |
| --- | --- | --- |
| N | Numerical teacher, no context; numerical predictor may receive the matched extra update budget | Numerical competence and extra-optimization control |
| JZ | Fused latent target loss d(n+g(n,C), E(y_future)); numerical and context paths jointly receive gradients | Context-compensation opportunity |
| JZ+N | JZ plus exactly the numerical self-forecast loss used below | Separates an extra objective from gradient isolation |
| PZ | Numerical self-forecast loss updates P only; d(sg(n)+g(sg(n),C), E(y_future)) updates context/correction only | Protected latent-innovation hypothesis |
| JY | Joint observation-space residual model, prediction D(n)+r(n,C) | Strong direct-training counterpart |
| PY | Independent numerical forecast plus a context branch supervised on sg(y_future−D(n)); the numerical task has its own loss | REST-TS-style ordinary direct residual control, adapted to numerical covariates |
| W | Full released WorldTS two-stage recipe with the same historical-only inputs | Framework-level reference; not required to share the fixed-E mechanism ablation |
| T | Released TimeXer with identical inputs and its task script settings | Competent task-matched ordinary forecaster |

For the protected arms the numerical loss is the same observation forecasting objective; choose MAE as primary and report MSE. For JZ/JZ+N/PZ use the same latent distance, weights and scale treatment. Include a Pythia-style numerical-reference loss in **either all three latent arms or a separate declared sensitivity**, not just the favored arm. Do not claim an exact Pythia implementation. The initial pilot should use no such extra loss, leaving the explicit protection comparison interpretable.

The table deliberately contains difficult controls. PY may win. JZ+N may explain everything. Either outcome is useful and closes a more ambitious mechanism claim. Match adapter capacity approximately and give PY at least the candidate's trainable-parameter and tuning budget; keep the external TimeXer control's competent task configuration rather than shrinking it to fit this match. Before claiming contextual information use, add a training-time no-context correction control with the same residual-module capacity and numerical input n. A gain that also appears there is added numerical modeling capacity, not evidence for contextual innovations. This is a predefined diagnostic control, not a corrupted-data benchmark.

### 7.3 Ordinary combinations, decoder check, and measurement

Split validation chronologically into checkpoint-selection and combination-fitting portions with a horizon-length purge at their boundary. Fit, without test access:

- Per-lead-block convex weights between N and each context model, constrained to [0,1]. Include fixed 0, 0.5, and 1 weights.
- A small regularized conditional gate using history-derived summaries, calendar metadata, lead index, and model disagreement; use a ridge/logistic gate and a two-layer MLP gate with the same input information. Gate tuning uses only its validation portion.
- Select regularization and gate type using three blocked inner folds within the combination-fitting portion, then refit on that portion. Checkpoint selection uses only the earlier portion. Do not select a different control on each test task.

Score final predictions on the same test origins. Use non-overlapping 96-step target blocks for the primary estimate and report the exact origin count. Report full-horizon MAE/MSE plus lead blocks 1–24, 25–48, and 49–96; these are prefixes/blocks of a 96-step forecast, not separately trained horizon results. Small ETTh1 test coverage limits precision. Bootstrap whole temporal blocks, retain all seed results, and never treat overlapping forecasts or seeds on the same origins as independent observations.

For each latent arm log: numerical-only forecast error using its own n; fused forecast error; numerical and corrected latent error against the common E; correction magnitude; and gradient norms reaching the numerical and contextual paths. Preserve paired per-origin and per-group values and recompute both error types with the same aggregation before discussing a metric reversal. Main inference concerns paired **forecast** errors. Latent-error/correction statistics are diagnostics and may not select checkpoints.

If a latent candidate initially passes, refit the same stronger ordinary readout on each latent arm with matched inputs and update/selection rules. Use a global aggregation readout and native patch readout as ordinary controls, not contributions. If the claimed branch effect disappears, report that outcome and close the route. This is a necessary confound check, not another C33-style decoder discovery program.

### 7.4 Executable kernel and budget

The essential gradient separation is ordinary PyTorch; the following is a specification, not a claim that a runner exists:

```python
# E and initial D come from a common numerical teacher; E is fixed.
t = E.encode_future_target(y_future, y_history).detach()
n = P(E(y_history))
numerical_forecast = D(n)
loss_n = mae(numerical_forecast, y_future)

# PZ: context cannot improve its loss by moving P or its input n.
delta = G(n.detach(), C(c_history))
loss_context = latent_distance(n.detach() + delta, t)
loss = loss_n + lambda_z * loss_context
# Optimizer parameter groups and an autograd audit verify the separation.

# PY counterpart: same historical payload and own numerical task.
residual = R(n.detach(), C(c_history))
loss_py = loss_n + mae(residual, (y_future - numerical_forecast).detach())
```

For PY make the inference prediction `numerical_forecast + residual`. For PZ decode `n + delta`. Use a common declared target normalization for prediction and E's future target. Keep normalization statistics available to all compatible readouts. Unit tests should verify actual gradient routing and the inference information boundary; these checks are not scientific results.

Pilot size: two datasets × three seeds, H=96, modest WorldTS-sized modules, plus the two released framework controls. Reuse the teacher and context features where doing so preserves the comparison; adapter fits are small. Profile 100 training steps on the RTX5090 before committing to the complete matrix. Set an aggregate 24 GPU-hour pilot cap and record trainable/total parameters, updates, examples, wall time, and peak memory for each arm. This is a resource ceiling, not a runtime prediction. Compare learning curves at common exposure and common compute. No claim is resolved if a key control remains substantially underfit at the cap; report that as an inconclusive finite-budget experiment.

Seal seeds, objectives, early stopping, combinations, source hashes, and test-origin lists before unsealing test targets. No adaptive subset or target choice is allowed after testing. A second horizon or a third domain is a separately sealed confirmation, not a way to repair the pilot's score.

### 7.5 Decision rule

For a screening decision, require PZ to improve paired MAE over **both** PY and the best validation-selected ordinary combination on both datasets, with a positive block-bootstrap lower confidence bound for the pooled paired effect; also report MSE and the worst lead-block regression. Use a predeclared practical effect threshold, for example 2% pooled MAE with no >1% domain-level MAE regression, as a resource-allocation rule rather than a theorem. A promising result must also survive JZ+N, matched compute, and the ordinary-readout check.

Failing these checks means the proposed latent/protection interaction is not supported. Beating W alone is insufficient. If PY or a gate suffices, the correct conclusion is that ordinary contextual residual forecasting explains the gain on this protocol. If only one task benefits, the evidence is a task-specific optimization, not the requested reusable framework. Even passing the pilot would justify another study, not establish an ICLR-level contribution.

## 8. Synthesis and open work

The native evidence supports a real measurement question: a context-corrected representation can score well under its own matching objective without the numerical path and observation forecast improving together. It does not yet support a new architecture gap. Existing reference guidance, direct residual learning, decoder adaptation, and simple ensembles already attack different parts of that problem. The proposed experiment is useful precisely because those alternatives are allowed to win.

The highest-priority unresolved issue is therefore whether numerical protection has an advantage **specific to latent contextual learning** at finite data/compute, rather than whether one can build another gated residual model. If it does, a broader study should vary context informativeness naturally across domains and horizons, add stronger contemporary covariate adapters, compare against multimodal optimization balancing, and quantify how the benefit changes with data and compute. If it does not, pivot away from this mechanism without reviving NP/PJM readout tuning or converting a diagnostic proxy into a deficiency claim.

## 9. Answers to the frozen questions

1. Recent frameworks support observational predictive-state forecasting under distinct contracts. WorldTS permits per-dataset training; Pythia studies shared pretraining; physical organization is a declared specialization of Phys-JEPA, not a required property of all TS world models.
2. Predictive responsibility and contextual utility are worth measuring. The native evidence makes them plausible diagnostics, while ordinary readout and forecast-combination controls remain unresolved explanations. No established architecture failure is claimed.
3. A protected numerical predictor plus latent innovation offers a constructive, falsifiable minimum viable question. The supplied independent historical-only protocol can reject it against direct residuals, joint-training controls, a validation-fitted gate/stack, released WorldTS, and TimeXer on one RTX5090. Novelty and useful effect remain unconfirmed.

## References and verification ledger

All entries below passed identity and claim checks at the indicated scope. Full text, rather than metadata alone, supports numerical/implementation details above.

1. Yuhan Zhu et al., “WorldTS: World Modeling for Multimodal Covariate-aware Time Series Forecasting,” arXiv:2609.31162, 2026. [Identity](https://arxiv.org/abs/2609.31162), [full text](https://arxiv.org/html/2609.31162v1), [pinned author README](https://raw.githubusercontent.com/decisionintelligence/WorldTS/269426ea6217d7152a244da0e209d922acd46d4f/README.md). VERIFIED. In-window primary framework.
2. Xilin Dai et al., “Pythia: Toward Foundation World Models for Multimodal Time Series,” arXiv:2610.05240, 2026. [Identity](https://arxiv.org/abs/2610.05240), [full text](https://arxiv.org/html/2610.05240v1). VERIFIED. In-window primary framework; code availability statement checked in document, no independent reproduction.
3. Weizhi Nie et al., “Phys-JEPA: Physics-Informed Latent World Models for Multivariate Time-Series Forecasting,” arXiv:2606.16076, 2026. [Identity](https://arxiv.org/abs/2606.16076), [full text](https://arxiv.org/html/2606.16076v1). VERIFIED. In-window, limited native ablation evidence.
4. Huu Hiep Nguyen et al., “Does Text Actually Help? Uncovering and Resolving Text Collapse in Multimodal Time Series Forecasting,” arXiv:2606.19413, 2026. [Identity and abstract](https://arxiv.org/abs/2606.19413), [paper](https://arxiv.org/pdf/2606.19413). VERIFIED. In-window closest residual-supervision overlap; no unverified code-release claim.
5. Peng Chen et al., “Beyond Numerical Time Series: A Unified Benchmark for Multimodal Forecasting with Heterogeneous Context,” arXiv:2609.15087, 2026. [Identity](https://arxiv.org/abs/2609.15087), [full text](https://arxiv.org/html/2609.15087v1). VERIFIED. In-window methodology; no claim its entire data package is locally ready.
6. Haoxin Liu et al., “Rethinking Multimodal Time-Series Forecasting Evaluation,” ICML, 2026. [Proceedings identity](https://proceedings.mlr.press/v306/liu26go.html), [full text](https://arxiv.org/html/2607.06973v1). VERIFIED. In-window benchmark/ordinary-combination evidence.
7. Yong Shang et al., “LightGTS-Cov: Covariate-Enhanced Time Series Forecasting,” arXiv:2602.10412, 2026. [Identity](https://arxiv.org/abs/2602.10412), [full text](https://arxiv.org/html/2602.10412v1). VERIFIED. Earlier overlap; no performance number used here.
8. Weiyao Wang, Du Tran, and Matt Feiszli, “What Makes Training Multi-Modal Classification Networks Hard?”, CVPR, 2020. [Primary paper](https://conferences.computer.org/cvpr/pdfs/CVPR2020-1XMljIyuXWg2zC9bnL19Tw/716800m2692/716800m2692.pdf), [primary supplement](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Wang_What_Makes_Training_CVPR_2020_supplemental.pdf). VERIFIED. Older optimization foundation, not TS forecasting evidence.
9. Chenfeng Huang, Zixuan Ma, and George Michailidis, “Model-Agnostic Online Certificate-Driven Calibration for Time Series Forecasting Under Distribution Shift,” UAI, 2026. [Proceedings](https://proceedings.mlr.press/v337/huang26b.html), [author abstract](https://arxiv.org/abs/2609.31960). VERIFIED. In-window gated-residual overlap; online assumptions differ.
10. Yuxuan Wang et al., “TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables,” NeurIPS, 2024. [Identity](https://arxiv.org/abs/2402.19072), [author repo](https://github.com/thuml/TimeXer). VERIFIED. Older strong control; task scripts and split code inspected, data bytes not downloaded.

Intake actions: public retrieval and read-only source inspection; wrote this note only. No server training, model download, large data download, publication, test execution, or architecture choice from scores.

## C35 pre-fit methodology supplement

Read-only audit of `cycle35/MECHANISM_PILOT_PLAN.md`, frozen 2026-10-17. This supplement evaluates the proposed formulas, not an inspected implementation or any scores. The TRAIN/VAL-only, two-task, one-seed pilot is a legitimate screen of the declared configuration. No additional architecture or prior-art requirement is introduced.

**One concrete implementation blocker if clipping is global:** clipping the combined P and g gradients to norm 1 makes context-gradient magnitude change P's update through the shared clipping multiplier, even when context losses have no autograd path to P. Use separate clipping for P and context/projection parameters, or explicitly limit the interpretation to gradient detachment with coupled optimization. Under separate clipping, identical P initialization/Adam state, frozen E/D, identical batches, and deterministic numerical forward passes, **N, PZ, PY and PZ_NC must have identical numerical-predictor trajectories**. Compare P parameters, optimizer states and numerical predictions at early steps. A discrepancy diagnoses optimizer/routing/state differences, not a scientific protection effect. Zero-initialized g means its earlier layers and PY's projection can legitimately have zero gradients at the first step; require a nonzero gradient somewhere in the correction's last layer, not every correction parameter.

**Encoding contract must be literal in the runner:** the notation `E(y_future)` must use the same history-derived target normalization as prediction. With state patch length 3, prepend the permitted two-point target-history prefix before encoding future targets, then keep the future slice, if following native WorldTS's contextual encoding. Future-derived RevIN statistics or different first-patch padding would change the target problem. This is a conditional implementation check, not an assertion that the proposed runner contains an error. Frozen D must remain differentiable with respect to its input in PZ; wrapping the fused decode in `no_grad` would remove the forecast-loss gradient to g.

The main contrasts have the following reach:

- **PZ versus JZ_N:** at the same parameter state, their forward values and scalar objectives are equal; they differ in whether fused forecast/latent losses update P, including the path through g's numerical input. This is a valid gradient-treatment contrast. A win alone does not distinguish harmful gradient direction from effective step-size changes. Recording the cosine/dot product between the numerical self-loss gradient and fused-loss gradient, alongside existing numerical/fused forecast curves, would support the interference interpretation without another fit. “Compensation” requires actual numerical forecast deterioration plus a fused improvement, not merely different latent errors.
- **PZ versus PY:** this can reject the proposed bundle at this budget, but cannot isolate latent geometry from the extra latent objective and nonlinear decoder mapping. The plan already acknowledges that limitation. PZ_NC tests incremental value of the observed covariates relative to this learned numerical representation; it does not establish conditional statistical independence from the entire raw target history. It also does not establish that JZ's correction specifically uses covariates, because no JZ_NC contrast is present.
- **Teacher selection:** selecting changing E/P checkpoints by latent validation loss does not rank a common prediction problem: the target geometry changes across checkpoints. The common selected E nevertheless preserves internal comparability among subsequent arms. Consequently, teacher Stage2 numerical forecast quality and learning curves must bound interpretation; a weak teacher cannot establish a general limitation of world models. No teacher-selection error involving TEST is apparent in the plan.
- **Selection and inference:** reusing VAL for teacher selection, arm checkpoint selection and the 2% go/no-go screen makes those selected differences exploratory. The plan's held-out confirmation requirement is necessary. A negative screen closes this precise configuration; a positive screen supports spending on matched-objective/readout/ordinary-forecaster confirmation, not accepting a reusable framework or a discovered information deficit.

Only this supplement was appended. No plan/source edits, fits, server actions or downloads were performed.
