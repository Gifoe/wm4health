# Mechanism-reuse scope: current primary-source audit

[Modular NRI, NeurIPS2019](https://papers.neurips.cc/paper_files/paper/2019/hash/b294504229c668e750dfcc4ea9617f0a-Abstract.html) learns reusable neural interaction modules and infers assignments by simulated annealing with a learned proposal. It already discusses inferring unobserved entities, so partial observability plus modular learning is not a standalone new claim. The pinned refactored code is real and executable, but differs from the paper-era implementation. The inspected spring command uses50k tasks and20k outer epochs; each outer epoch traverses the training loader, so epochs are not individual optimizer updates. Our fixed subset experiment is separate and cannot reproduce its full-budget table.

[PRISM-WM](https://arxiv.org/html/2512.08411v1) uses context-conditioned experts, with state/action and optionally task embeddings in gating. This already blocks a generic new state-conditioned mixture proposal. Its source was not located through the inspected primary paper/project search; absence of a located release is not a research gap. Do not confuse PRISM-WM with the distinct Policy Learning with World Models work sharing the PWM abbreviation.

[Compositional continual world-model benchmark, September2026](https://arxiv.org/html/2609.22055v1), sectionsIII-V, separates action/perception composition and tests a progressive expert adaptation. Its diagnostic frozen encoder has seen all tasks; the authors explicitly identify that privilege, task-level routing and unmatched capacity as limitations. These are their reported findings, not this project's experiments. A time-series transfer would need to distinguish reuse from added capacity and altered information.

[SICSM, NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/89c61fce5a8b73871d1c4073f486b134-Abstract-Conference.html) already addresses structural inference from irregular and incomplete observations using selective state-space modeling and graph posterior inference. Its public author repository advertises code pending in the inspected version; no SICSM result is claimed here. A title/abstract does not establish that every nonlinear-sensor composition problem is solved, but it excludes a broad first-partial-observation claim.

Historical project boundary: C18 already executed native DynaMix demonstrations; C19 examined stronger path statistics. C41-C46 already tested sensor/dynamics composition with strong ordinary alignment/state-space controls. Their limits (controlled carriers, finite known views/systems and anchors) leave development questions, while their successes rule out using those same failures as new motivation. The67-cycle hashed source index is for avoiding duplicate work, not67 successful new ideas.

Current inference: a meaningful remaining candidate must specify a learning or information bottleneck surviving those complete controls. Generic modules, Gaussian observers, context inference, graph uncertainty or a new benchmark name are insufficient. Native Modular NRI training is baseline development; the test set remains closed while the mechanism question is refined.


## Primary-source refinement, C71 v3

NOVELTY_BOUNDARY71_V3.md adds primary-source boundaries for WM3C, WLA, SIHA, LeJEPA identifiability, CIPHER v2 and compositional pH models. It also qualifies the original Modular NRI hidden-node demonstration as true-edge-trained. These literature findings are distinct from our experiments.


## Runtime correction: C71 driver incident

The earlier running snapshot is historical. The first fit terminated with Windows access violation 0xc0000005 at 06:14 UTC after epoch651; WER names nvcuda64.dll. The remaining seeds and free-VAL assessment did not launch. TRAINING_INCIDENT71_V1.json records terminal receipts and checkpoint verification. This is an execution failure; no model-performance or novelty conclusion follows. Resume verification is required before restarting the campaign.


## Verified checkpoint continuation, C71

RESUME_QUALIFICATION71_V2.json reports zero differences across174 tensors and all compared optimizer/RNG/shuffle/structure/metric values for uninterrupted versus resumed qualification runs. The first formal fit resumed from epoch650 under RECOVERY_PROTOCOL71_V1.md, preserving the original loop body and checkpoint grid. The recovered campaign and its new assessment queue have new process/receipt identities. See RECOVERY_SNAPSHOT71_V1.json for the bounded live state. Root cause of the NVIDIA-driver exception remains unestablished; no final model accuracy or research-direction claim is made.


## Exact structure-orbit audit, C71

An exhaustive metadata audit supplies a local protocol boundary, not a literature novelty claim. STRUCTURE_ORBIT_FINDING71_V1.md separates graph topology from continuous initial-state generalization. The original full ModularNRI experiment is not being declared invalid by this subset audit.


## Second runtime incident and bounded continuation, C71

Runtime diagnosis uses primary Intel/Microsoft documentation only as bounded engineering context. Two different native fault modules and old firmware context do not establish a hardware or software cause. No new scientific prior-art gap follows. See BOUNDED_RUNTIME_FINDING71_V1.md for the exact claim boundaries.
