# Competing hypotheses after the fixed real task

| Hypothesis | Evidence | Current interpretation |
|---|---|---|
| Wrong action-coordinate indexing is the whole problem | C66 indexing correction, retained as C67 baseline | Correct the software first; no architecture novelty follows. |
| Conditional-view mismatch dominates remaining forecast error | Indexed versus same-view transition, all other native components fixed | EVAL shows0.648% lower error, VAL slightly worse. This is not sufficient evidence of a dominant unresolved bottleneck. |
| Historical versus full-plan action-encoding length explains the effect | Past/full-view forecasts differ by only4.58e-5mg/dL max | No substantial measured output effect in these cases. |
| A simpler action distance would solve the problem | Raw-macro control | VAL improvement does not persist on EVAL. Do not tune EVAL. |
| More elaborate context transition is always better | Action-ignored, indexed, native predict/simulate and context-mean controls | Not supported by the primary point metric. This does not settle distributional dynamics or original composite metrics. |
| Ordinary supervised task reuse already gives a strong comparator | TRAIN-only analog/ridge selected on VAL | Yes in this protocol, with additional target-domain labels; not an equal-information causal comparison. |
| A new general framework has now been validated | One supported-task-family evaluation, no new method | No. Independent mechanisms, a distinct method and task-valid remaining failure are still absent. |

Separate a useful practical repair from a new research question. A reusable framework need not beat every classical model on every small system, but it must offer a specific useful capability beyond these existing components with a fair evidence package.
