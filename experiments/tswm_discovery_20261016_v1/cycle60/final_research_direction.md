# C60 final direction status

**Research Question:** Does a generic latent-state model or a conventional gated control leave an unexplained conditional-simulation deficit on independent measured mechanisms?

**Evidence:** Locked raw provenance, unchanged native components, qualified output/gradient wiring, two complete fit/validation/test protocols, up to twelve fixed-budget neural fits and immutable/validation-selected ordinary banks. All outcomes are in the table and full machine-readable receipts.

| System | Control | Completed / expected | Finite rollouts | Mean test MSE / fit variance | Neural seed SD |
|---|---|---:|---:|---:|---:|
|N87|native_component_nssm|3/3|3|0.0101758|0.00598455|
|N87|ordinary_gru|3/3|3|0.0419812|0.0182712|
|N87|static|1/1|1|0.118216|N/A|
|N87|play|1/1|1|0.0516982|N/A|
|N87|ema|1/1|1|0.0633208|N/A|
|N87|hybrid|1/1|1|0.0478821|N/A|
|N87|arx|1/1|1|0.0584894|N/A|
|pick_place|native_component_nssm|3/3|3|0.00975243|0.00121676|
|pick_place|ordinary_gru|3/3|3|0.0346664|0.00908307|
|pick_place|static|1/1|1|0.396165|N/A|
|pick_place|play|1/1|1|0.646618|N/A|
|pick_place|ema|1/1|1|0.279222|N/A|
|pick_place|hybrid|1/1|1|0.512366|N/A|
|pick_place|arx|1/1|1|0.168394|N/A|

**Existing Methods:** State estimation, residual neural state-space, ordinary encoder-decoder GRU, play/relaxation and stable ARX are mature controls. Short-horizon instability and mismatched one-step evaluation cannot be repackaged as novelty.

**Proposed Idea:** None accepted. The evidence is control completion and failure attribution, not a new framework.

**Falsification:** A claimed new-memory necessity should be dropped wherever adequate ordinary controls already repair it. A surviving poor model needs initialization/horizon/optimization attribution and independent data before it counts as a gap.

**Feasibility:** Actual CPU/CUDA profiling guided execution; raw data/checkpoints/predictions remain on G:, compact evidence/code on GitHub. No environment changes.

**Next Experiment:** Use any surviving residual to define a matched, independently heldout mechanism test. Do not repeat a viewed split to tune a positive method or require capabilities the original user did not request.

Two separately fitted measured systems; the N87 split has historical C59 test exposure and remains exploratory. Pick-place contributes one chronological test record, not an independent-device population. All neural seeds are shown; record-level intervals are not seed intervals. No shared checkpoint, new causal identification, new architecture or field-wide unresolved gap is established.
