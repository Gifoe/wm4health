from common62 import *
s=read(OUT/'SUMMARY62_V1.json');c=read(OUT/'CLOSURE62_V1.json');seal=read(OUT/'MODEL_SEAL62_V2.json');assert c['status']=='CLOSED' and seal['completed_count']==8
text=f'''# Time-series world-model discovery: C62 status

Overall research goal: ACTIVE_UNACHIEVED. Accepted final research directions: 0.

## Latest result and scope correction

C62 is a partial PRO-DYN source-protocol reproduction on measured ETTh1:8 of9 planned fits completed, with{s['formal_updates']} successful-fit optimizer updates. Three formal attempts ended in native Windows access violations; at least3200 aborted-attempt updates were recorded, with exact crash-time totals unavailable. Four completed fits use native window copies and four use a cached tensor copy wrapper qualified with128 exact window comparisons, identical forward/gradients and180000 stress lookups. A per-window-copy diagnostic also failed. The wrapper did not eliminate all native faults; root cause remains unknown.

Crucially, the selected official scripts use L=H=96. PRO and DYN then have the same computation after renaming linear_PRO to linear_DYN. Same-weight, same-RNG CPU forward and gradients matched exactly in training and evaluation modes. The original choice of this equal-length cell was insufficient to isolate PRO/DYN mechanisms. There are two distinct neural architectures in the completed comparison, not three. PRO has a complete three-seed set; DYN's two fits are auxiliary duplicate executions. See cycle62/EQUIVALENCE_NOTICE62_V1.md and cycle62/outputs/PRO_DYN_EQUIVALENCE62_V1.json. This corrects the earlier pre-test note describing different modified computational paths.

All eight sealed fits were evaluated using the unchanged native CPU batch forecasting path. H96 full rolling forecasting and common-origin96/192/384 free block rollouts are separate. Persistence, seasonal-24 and validation-selected ridge receive the same past measurements. No checkpoint or method is chosen by test outcomes. Numerical loop qualification and all retained metric rows are independently replayed. This is not a full paper-table reproduction, GPU inference benchmark, causal intervention experiment or new world-model method.

| Arm | View | Horizon | MSE |
|---|---|---:|---:|
'''
for r in s['summary']:text+=f"| {r['arm']} | {r['view']} | {r['horizon']} | {r['mse']:.7f} |\n"
text+='''
ETTh1 is one previously examined measured system; test windows overlap. The model has only past seven-channel observations and known future calendar marks, with no physical action plan or future measured forcing. Growing point error cannot by itself identify a physical-dynamics failure. Approximate-attention random seeds are algorithmic randomness, not samples of the physical future. No novel framework claim is justified.

## Storage and publication

Raw measurements, selected/terminal weights and forecasts remain on the authorized server G disk. Only authored code and compact evidence are uploaded to Gifoe/wm4health; no local result-file synchronization. The unmodified upstream source is recorded by pinned Git objects and SHA256, without source mirroring because the pinned repository has no license file. All original failures, amendments and design corrections are retained.

## Preserved earlier evidence

C60: a learned ordinary NSSM repaired tested memory-bank deficits on measured N87 and pick-place systems. C61: a learned ordinary explicit-duration HSMM met conditional-density/survival thresholds in two controlled duration families. These are exclusions of tested recipes, not proofs that every real dynamics problem is solved.

The objective remains a reusable time-series world-model framework. Generic subsystem composition already has PHNN, input-output identification and C-PHAST precedents; these were scope-checked, not executed here. The next candidate needs a concrete residual beyond competent state-space/DSR learning and an independent mechanism/domain. The latest experiment establishes baseline qualification and a design boundary, not a publishable new direction.

Exact current results: cycle62/experimental_results.md, cycle62/outputs/SUMMARY62_V1.json and cycle62/PARTIAL_CLOSURE_AMENDMENT62_V1.md. Historical snapshots remain unchanged.
'''
p=PROJECT/'CURRENT_RESEARCH_STATUS_V62.md'
if p.exists():assert p.read_text(encoding='utf-8')==text
else:p.write_text(text,encoding='utf-8')
print(json.dumps({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'git_blob':blob(p.read_bytes()),'status':'RESEARCH_ACTIVE_UNACHIEVED'}))
