# Models, data and reproduction status

Source code: ExistentialRobotics/cross_embodiment_transfer, commit3988511598a39c8a8a6530055b72609bf1b8b373. Four pinned Python files are verified by SHA256 before use. Pure neural utility definitions are AST-extracted without their uncalled robotics dependencies; native network definitions, source consistency update and aligner update bodies remain intact.

| Entry | What was executed | Interpretation |
|---|---|---|
| Source simulator | Native BCObsAgent networks and consistency warmup, followed by explicit observable rollout training | Cross-task source training; no policy fitting or original robot-paper reproduction |
| Frozen | Selected source encoder, forward model and decoder | Baseline for the actual source capability, including no-change controls |
| Native frozen alignment + anchors | Native adversarial/cycle/dynamics alignment calls and an additional paired-anchor step | Explicit extension of an existing method; frozen-dynamics limitation retained |
| Joint output-error | Same source architecture; ordinary joint encoder/decoder/forward updates on observed calibration windows | Existing-recipe control, not a proposed method |
| Polynomial3 / RFF128 delta maps | Ridge regression with optional source-coefficient prior, selected on target VAL | Strong ordinary full-state simulator controls; different capacity/cost from neural fits |

Own controlled simulators are Duffing and Brusselator. Source, target calibration and EVAL seed namespaces are separate. Each family has two data/model seeds, four paired change conditions, fixed commands and independent EVAL initial conditions. H32 prefixes within an action pair are duplicates because the two plans first differ at step32; they are not independent observations.

The original source qualification failures, capacity repair and subsequent source-only long-rollout control are retained as separate stages. Their final counts and parameter/update budgets are generated from the actual records. No failing source is silently admitted by lowering its gate. No new hypothesis is accepted from an inadequately fitted source model.
