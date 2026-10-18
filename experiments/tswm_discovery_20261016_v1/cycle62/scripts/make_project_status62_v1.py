from common62 import *
s=read(OUT/'SUMMARY62_V1.json');c=read(OUT/'CLOSURE62_V1.json');assert c['status']=='CLOSED'
text=f'''# Time-series world-model discovery: C62 status

Overall research goal: ACTIVE_UNACHIEVED. Accepted final research directions: 0.

## Latest completed experiment

C62 executed nine native PRO-DYN source-model fits on measured ETTh1, totaling {s['formal_updates']} optimizer updates. Informer, Informer_PRO and Informer_DYN each used three training seeds and official ETTh1 H96 hyperparameters. The original H96 rolling point evaluation and native 192/384 block rollouts are separated. Persistence, seasonal-24 and validation-selected ridge use the same information. This is a partial code-protocol reproduction and exploratory deployment audit, not the full paper-table reproduction or a new world-model method.

All selected and terminal weights, source/data/prediction hashes, raw forecasts and independent metric replay are retained on the server. Native loop and independently assembled same-core loop agree on training-only qualification histories. Raw arrays/checkpoints remain on the authorized G disk; only code and compact evidence are published. Upstream source has no license file at the pinned revision, so this snapshot records hashes and exact retrieval links without mirroring its source.

| Arm | View | Horizon | MSE |
|---|---|---:|---:|
'''
for r in s['summary']:text+=f"| {r['arm']} | {r['view']} | {r['horizon']} | {r['mse']:.7f} |\n"
text+='''
These scores describe one already examined physical system and overlapping test windows. Inference RNG variation is approximate-attention randomness, not predictive uncertainty. A longer horizon's higher point error does not by itself demonstrate a mechanism failure. No new loss, architecture or accepted research direction is selected on this test set.

## Preserved earlier exclusions

C60's learned ordinary NSSM repaired memory-bank deficits on the measured N87 and pick-place systems. C61's ordinary learned explicit-duration HSMM met predeclared conditional-density/survival thresholds in both controlled duration families. These results exclude the tested generic recipes; they do not prove every real dynamics problem solved.

## Research boundary

The objective remains a reusable time-series world-model framework. Generic subsystem composition is already covered by compositional PHNNs, input-output subsystem identification and recent C-PHAST; none was executed in C62. Any next candidate must demonstrate a concrete residual beyond competent state-space/DSR training, with matched information and a separate mechanism/domain. The current evidence does not justify a top-conference novelty claim.

See cycle62/experimental_results.md, cycle62/PROTOCOL62_V1.md and cycle62/outputs/SUMMARY62_V1.json for the exact latest experiment. Historical cycle60/cycle61 snapshots remain immutable.
'''
p=PROJECT/'CURRENT_RESEARCH_STATUS_V62.md'
if p.exists():assert p.read_text(encoding='utf-8')==text
else:p.write_text(text,encoding='utf-8')
print(json.dumps({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'git_blob':blob(p.read_bytes()),'status':'RESEARCH_ACTIVE_UNACHIEVED'}))
