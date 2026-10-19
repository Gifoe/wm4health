# Current research decision

**No final direction has met the user's evidence and innovation requirements. The full goal remains ACTIVE.**

**Research Question:** no accepted final question. A provisional fresh lead concerns whether inverse-action training preserves nonlinear reachable information; the real wind-tunnel branch investigated timing versus measurement explanations and did not establish a new dynamics gap.

**Evidence:**38real recordings/965,322rows/64intervention blocks audited; ordinary ridge/HGB time/dynamics controls evaluated18recordings;128sensor-stream quantization audits performed. Low-resolution blocks contribute93–98%of HGB measured-RPM squared error. This is concrete evidence that failure attribution must account for acquisition, not direct evidence for a new dynamics method.

**Existing Methods:** TAWM, observation-intensity models, TE-CDE/TESAR-CDE, sensing-stage audits and ordinary system identification address much of the broad setting. Correctly specified measurement models have not been shown insufficient. EP-IDM/CF-JEPA are new closest works for the next search, not executed baselines here.

**Proposed Idea:** none accepted. Do not invent a new neural acquisition gate or rename a standard joint clock model. A nonlinear-reachability mechanism would need fresh empirical evidence and a precise distinction from established control theory and current inverse/multistep objectives.

**Falsification:** reject the wind-tunnel new-dynamics claim while measurement artifacts remain a sufficient competing explanation. Reject a fresh nonlinear-reachability claim if existing multistep inverse/controllability or reconstruction baselines preserve the relevant information under fair capacity/data/budget and independent held-out testing, or if literature already covers the exact mechanism.

**Feasibility:** prerecorded numeric data and pinned source are now available on the research server. This cycle's inexpensive existing controls required no new GPU model; RTX5090 remains available. Native source/code inspection for the fresh control-representation lead is feasible before deciding on any training investment.

**Next Experiment:** first inspect the exact native endpoint-inverse and factorization definitions/assumptions. If a genuinely untested residual exists, use a minimal nonlinear controlled system with known reachable effects, include strong reconstruction and existing multistep controls and avoid artificial scaling/capacity handicaps. Then require an independent non-toy task before accepting a research direction. If those conditions fail, switch search scope rather than maintain the idea.
