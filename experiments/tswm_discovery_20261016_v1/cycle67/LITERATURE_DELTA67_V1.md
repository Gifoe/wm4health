# Primary-source novelty and interpretation boundaries

Verified2026-10-18. These are literature/code observations, not reproduction results for the listed papers.

## Query/candidate views already have direct prior art

[ALER-TI](https://arxiv.org/html/2607.07640), July8,2026, explicitly treats representation mismatch between incomplete queries and complete historical candidates. It contrasts asymmetric retrieval, on-the-fly masked re-encoding and a cached latent-alignment approach, then fuses retrieved information with frozen imputers. Thus neither diagnosing a conditional-view discrepancy nor adding a lightweight alignment/re-encoding module is sufficient novelty. Its task is imputation; no empirical claim about its performance on C67's context-fitted transition task is made. The paper links [author code](https://anonymous.4open.science/r/Time-series-0142/); access returned403 during this intake, so no code-execution claim is made.

## Online state/dynamics adaptation is established

[AdaJEPA](https://arxiv.org/html/2606.32026), June30,2026, adapts latent world models inside MPC using observed executed transitions and self-supervised latent prediction error. [Sandwich-Residuals](https://arxiv.org/html/2609.21740), September18,2026, instead keeps the pretrained core frozen and fits small surrounding residual corrections. Neither is executed here. Generic online adaptation or a frozen-core residual adapter cannot be presented as new without a distinct problem/mechanism and fair comparison.

## Scope of the actual released framework

[NormWear-2](https://arxiv.org/html/2605.15465) evaluates physiological conditional forecasting and context-based latent transition inference. C67 uses its exact pinned full native model, while correcting the implementation defects qualified in C66. Its [pinned demonstration](https://huggingface.co/mosaic-laboratory/normwear2/blob/3d1281c9994140056edfbbdedab57f2e6493dfaa/demo_code.ipynb) refers to a merged pretraining HDF5 absent from the released model repository. It does not specify the paper's full downstream splits/evaluator. The independently frozen protocol is therefore a same-task-family evaluation, not a numerical reproduction of the paper tables. Overlap between released CGMacros records and the unavailable pretraining corpus is unverified; no strict pretraining-disjoint generalization claim is allowed.

## What would count as progress toward the main goal

A successful ordinary view-consistency control would explain a failure and improve baseline validity. It would not itself establish a new world-model framework. Conversely, a classical model working on a few small systems is not proof that every learned reusable framework is unnecessary. The relevant standard is a clearly defined useful capability, an unresolved and reproducible failure under fair information, a method distinction from applicable prior art, and validation across independent mechanisms. Familiar components may form a valuable framework, but merely renaming an existing objective or software repair is insufficient.

The current main contrast separates feature views while retaining the same pretrained state representation, context clusters, transition counts and native sampler. Raw matching and target-domain trained analog/ridge controls prevent a false conclusion based only on a weak latent-distance baseline. Clinical causal identification, stochastic calibration and physical-state recovery are not tested by point forecast error on an observational released series.
