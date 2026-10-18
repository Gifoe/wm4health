# C88 competing hypotheses

1. **Whole-window action abstraction causes anticipation after learning.** Measured on selected, trained checkpoints with suffix-only edits. Stock maximum early response=0.28130245; per-cell values are retained. A structural dependency at random initialization was not sufficient evidence.
2. **The effect comes from masked future observations.** Identical base histories/zeroed masked observations and a separate masked-observation perturbation test distinguish it. Replay receipts are in outputs/PAIRED_PLAN_RESULTS88_V1.json.
3. **Ordinary causal alignment solves it without a novel model.** Same initialization/capacity/batches/loss/updates for stock and lagged. Original registered rejection gate=True; response comparability=False. If those suffice, retire a method claim.
4. **An observed accuracy difference is training or representation failure.** Compare source competence, untrained/mean/persistence/ARX4 and native acRKN. One optimizer seed cannot distinguish a robust tradeoff from optimization noise. Do not infer irreducible failure from this pilot.
5. **All action-conditioned world models share the defect.** Not supported: C79's native TSWM tests found no anticipation, and causal recurrent models impose the relevant prefix structure. Do not generalize one hierarchy to the field.
