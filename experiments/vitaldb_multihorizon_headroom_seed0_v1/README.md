# VitalDB multi-horizon headroom, seed 0

This is a falsification audit of whether direct multi-horizon prediction reduces a recursive-rollout penalty in the existing prospective RSSM. The cohort, patient splits, normalization, complete-action windows and seed-0 RSSM checkpoint are reused without modifying earlier experiments. CE is never a feature or loss target.

**Outcome B (qualified):** Direct-MH's paired BIS MAE advantage over AR-RSSM grows from 0.017 at 30 seconds to 0.104 at 300 seconds (95% patient-bootstrap CI for the latter 0.034–0.187). Direct-Traj also improves at 300 seconds. Neither direct gain is clearly amplified in high-divergence or upcoming large-intervention windows. The raw Oracle-State gap is confounded by its factual later-anchor BIS/MAP residual baseline; the latent-only reset does not reveal a positive drift contribution. The existing AR-RSSM was trained on all 30 future steps, so a one-step-objective mismatch is not established. See [the report](outputs/FINAL_REPORT.md) before interpreting the outcome.

Run on the same machine that holds the Round-1 case arrays and Round-2 cached test references:

```bash
python src/train.py
python src/evaluate.py
python src/finalize.py
```

The two direct models share an identical causal future-action encoder and point decoder. Direct-MH supervises 30/60/180/300-second endpoints; Direct-Traj supervises all 30 steps. This isolates the effect of sparse horizons from the direct formulation. A no-horizon model would require a third complete training run and is omitted in this feasibility round.

The Oracle-State Transition Audit encodes factual history ending at each future step and makes one local transition. It has access to future factual history and is not a deployable forecast. Its error difference from free rollout includes the value of observing true physiology at later anchors; it cannot by itself prove latent-state drift.
