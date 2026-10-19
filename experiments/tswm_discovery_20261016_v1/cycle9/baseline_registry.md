# Cycle9 baseline registry

| Artifact | Revision / version | Execution status |
|---|---|---|
| LeoToso/JEPA_control |6c3163e2f597a1be75f5dfc0d2836793b821968d|Selected source/config audit only; no native training/control result|
| manantomar/agent-centric-representations |f4622fadcb1fa2b6724dbab21ee62d1ac250bfe4|Selected source audit plus normalization algebra; no RL reproduction|
| jackboyla/action-contrastive-jepa |43c96f66bb12c95208be68b0858ee658c7e1d363|Unchanged native loss function executed on a new controlled diagnostic; not a native task reproduction|
| Generic quadratic action score |cycle9 script|Existing degree-two feature/InfoNCE control;3seeds,600updates; not presented as new method|
| thummd/dotime |8f227406a5b17c2040cfdcc680b493729ef47d00|Native generator-release loader, preprocessing, PFN and CPU baseline code executed;73native dependency/protocol files retained byte-identically|
| HF thummd/do-over-time-pfn |bec6618c8b25f3a5d8cf608d4c597978d5a694ac|One original s9ho_all_causal checkpoint downloaded/verified/executed; later corrected s13 checkpoints inventoried, not executed|
| dot-Identifiability-v1 |1.2.0|Official12,150episode release; manifestMD5 and additionalSHA256 retained|
| Mean / VAR-OLS |same native source|Action-blind controls, complete suite|
| NaiveOLS |same native source|Same observed-history/do-value information as PFN; unadjusted regression, complete suite; not causal oracle|
| pfns |0.4.2|Published wheelSHA verified; no-deps installation into cycle-local vendor directory; earlier environments untouched|

Source snapshots of the3inverse-learning repositories are explicitly selected-file snapshots with complete tree inventories. No claim of complete native environment replication. No original launcher executed. Runtime profile uses process-local single-thread numeric libraries and the previously verified affinity. Source/protocol retention and package licenses are recorded. All experiment selection and metric denominators are in configs/results.
