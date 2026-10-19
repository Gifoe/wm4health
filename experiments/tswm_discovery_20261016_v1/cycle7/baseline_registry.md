# Baseline and provenance registry

| Item | Status | Exact scope |
|---|---|---|
| ActSWM source | Pinned/read/executed unchanged selected bodies | 3d2eab6dccd3034506f2cec0ee227415b291f75b;17 small source/config files, no native data/weights |
| Native ARPredictor + loss + frozen IDM | 27 completed fits | New 4D fully observed plants, smaller native architecture; not Minecraft reproduction |
| Prediction-only | Executed stop/attribution control | Same architecture/init/train data/updates; fewer auxiliary computations; already solves the pilot |
| Perfect plant + spurious force | 6 finite held-out objective witnesses | Privileged oracle diagnosis; not a trainable or fair forecasting baseline |
| Learned/probabilistic IDM, SLIM | Not executed | Required before a general inverse-dynamics failure claim |
| DoTime PFN | Metadata only | HF mirror exact revision and single checkpoint size/hash;55 checkpoints in repository, no weights downloaded |
| Earlier models/cycles | Read-only existing evidence | Preserve sealed main/cycles2–6; do not count their results as fresh execution |

Data are generated from saved explicit DGPs; compressed cohorts and SHA256 descriptors are retained. Train/validation/test use independent cohort seeds. Source pin contains per-file hashes; native source bodies are not patched. Adapter code, plans, runtime profile, raw metrics and checkpoints are retained. Python3.10.20; full runtime versions and process exits are recorded separately. No public upload is authorized for this research cycle.
