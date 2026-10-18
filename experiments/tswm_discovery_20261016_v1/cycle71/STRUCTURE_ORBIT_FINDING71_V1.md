# Exact structure-orbit coverage in the current native baseline

## Observed result

The declared TRAIN contains 2048 trajectories with 881 distinct labeled graphs. Its graphs cover all 34 non-isomorphic simple undirected binary graph structures on five nodes. The 256-trajectory VAL contains 230 labeled graphs and 33 structure orbits. Of these trajectories, 214 have an exactly matching labeled TRAIN graph. All remaining 42 have a TRAIN graph with the same topology after a node bijection. Genuinely unseen VAL structure orbits: 0. Thus label-hash exclusion alone would not establish unseen-topology generalization for these already-trained weights.

## What was computed

`scripts/audit_graph_isomorphism71_v1.py` verifies the pinned source edge order and both canonical label-file hashes. It reads TRAIN0:2048 and development VAL0:256; reserved EVAL labels are not semantically accessed. The already-available label container is hashed for integrity only. No state trajectory, checkpoint or prediction is read. The algorithm enumerates all120 node permutations of every one of the1024 possible five-node undirected binary graphs, preserving physical edge colors. Complementation or swapping absent/present edge meanings is not an allowed equivalence. The canonical representative is the minimum ten-bit code in an orbit. A separate adjacency-matrix enumeration checks every orbit; the orbit sizes partition all1024 graphs.

The complete per-orbit counts and script/source/artifact hashes are in `outputs/GRAPH_ISOMORPHISM71_V1.json`. Per-row codes and the complete canonical lookup remain in the server artifact store. This is an exhaustive metadata result, not a sampled numerical forecast comparison.

## Consequence for the research decision

The baseline still tests prediction on new continuous trajectories and permits the already frozen free-simulation competence assessment. Structure isomorphism does not make initial conditions identical, prove the proposal network equivariant, prove that the model memorizes, or determine forecast accuracy. No claim about those behaviors follows from graph coverage. This audit does not show that the original paper's stated reproduction results are wrong; the current2048-task development protocol is distinct from its full experiment.

The proposed shortcut of discarding the214 labeled-overlap VAL trajectories and calling the remaining42 an unseen-composition test is invalid for the specific unseen-topology claim. Moving those42 rows into another partition after this model has learned all34 orbits does not fix it. Acquiring more five-node graphs from the same complete binary structural family does not create a topology absent from this training set.

If a later experiment specifically claims transfer to unseen topology, it must assign complete graph-isomorphism orbits to TRAIN/development/EVAL before fitting any shared library, representation, proposal or calibrator. If nodes have different physical roles, equivalence must preserve those roles instead of permuting all nodes. The protocol must report primitive coverage, topology coverage, initial-state and parameter distributions, observation interfaces and permitted adaptation separately. A topology shift is not automatically a new physical law. Selection must use development data and keep independent EVAL closed.

Such a protocol is necessary evidence hygiene, not an accepted new method or benchmark contribution. A new mechanism still needs a residual beyond competent ModularNRI plus observation/state inference, ordinary alignment and same-information capacity/compute controls. Current source-based relational training and its frozen forecast assessment continue unchanged. There is no new GPU fit, tuning against held-out forecasts or accepted final research direction in this audit.

## Decision and next discriminator

1. Finish all three already registered native fits and their frozen free-VAL assessment. Do not change the forecast arms or checkpoint rule from this metadata finding.
2. Do not use the current split or relabeled leftovers to claim unseen structural composition.
3. Only if the complete-baseline evidence warrants this route, design a new orbit-disjoint training cohort before a new model fit. Failure caused only by a harder split, extra sensor uncertainty, insufficient data or inference compute does not establish the sought novel gap.

Accepted final directions:0. Full research objective:ACTIVE / UNACHIEVED. Raw arrays and weights remain on the server; GitHub receives authored code, compact evidence and reports.
