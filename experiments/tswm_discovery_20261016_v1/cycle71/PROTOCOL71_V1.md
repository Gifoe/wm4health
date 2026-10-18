# C71: native modular-relational baseline development

This is source-based subset training on the author's spring-system data, not reproduction of the full50k-task/20k-epoch paper experiment and not a new world-model method. The purpose is to instantiate a capable reusable nonlinear-dynamics baseline before judging any new mechanism-composition proposal. No final direction is accepted.

## Data fixed before model execution

Use source train rows0:2048, each49 steps/5nodes/4features; author source test rows0:256 become our development VAL, rows256:512 are reserved EVAL. This is a new split protocol: official file names do not equal our scientific split labels. Download only the contiguous state prefixes needed and the small corresponding full label files (Fortran layout); save exact source header, raw acquired prefix/full-label hashes, derived subset hashes and source identities. No guessed channel or coordinate semantics. State values retain source scaling. Original49-step official VAL files are not needed.

The native data adapter uses NPDataset's unchanged49/21/remaining split for99-step development trajectories,49-step contexts and21-step future VAL. Native training trajectories use all49steps as in the author loader. Correct container split counts explicitly; source MVAL/MTEST names/counts are swapped, which is harmless at equal sizes but invalid for our unequal subset. Reserved EVAL numeric rows must never enter a training/validation dataset, model selection or report in this phase.

## Complete baseline and training

Retain the author's1node/2edge module library, dimensions, native learned MLP state proposal512, node proposals, simulated-annealing acceptance,10-step self-regressive training, Adam moduleLR.01 and proposalSGD LR.001/momentum.9, source temperature settings(initial-9.8, acceptance-1, slope800), batch128. Actual native neural modules, compose-multiple-graphs, batched_bounce, gradient and proposal updates execute unchanged. This is a frozen source-based training protocol; no source algorithm is replaced by a simple model.

Three independent fit seeds2026711/2026712/2026713; fixed2000outer epochs(16module updates per epoch,32000 per seed). Record every epoch's actual module/proposal update counts and TRAIN/VAL losses. Checkpoint every50epochs and final, select only developmentVAL `mean_loss/V_val`; preserve last and selected weights/optimizer/scheduler/RNG states. No EVAL scoring. Single-seed evidence cannot justify a stable generalization claim. Source's full-budget paper/table claims are outside this subset screen.

## One execution qualification

First run the same executable path with a separate seed2026710,128TRAIN/128VAL, one outer epoch. This tests imports, modern-PyTorch compatibility, graph batching, actual parameter updates and logging/checkpointing. It is not performance evidence. Inspect its terminal receipt and runtime/memory before the formal3seed launch. A live or timed-out handle is not a reason to restart. Formal cap/configuration will not be altered from an observed VAL result.

## Explicit adapters and interfaces

An AST import-only loader skips the unused HDF5 dependency and replaces only the external TensorBoard logging interface with a real JSONL scalar recorder; source scientific function bodies stay unchanged. A ReduceLROnPlateau wrapper removes the obsolete verbose argument. Explicit source-data container/split adapter, deterministic seed/name, safe checkpoint/metric instrumentation and the author CLI argument namespace are outside upstream files. Hash every executed source blob and source body; do not edit the upstream checkout or shared runtime. TensorBoard artifacts, raw arrays and weights go to authorizedGstorage; small metadata goes to this newDfolder. No result/data/model files synchronize locally.

## Interpretation

Original hidden-entity inference already exists in ModularNRI, and SICSM covers partially observed structural inference; neither phrase is a new gap. PRISM-WM and its continual benchmark are distinct from Policy Learning with World Models despite the PWM abbreviation. A future candidate must demonstrate a useful remaining time-series capability with same-information/capacity/compute controls. Native training inadequacy or an API error does not establish a scientific failure. Before any new mechanism experiment, freeze its independent protocol and strongest ordinary repair. This phase keeps EVAL closed and makes no novel-method claim.
