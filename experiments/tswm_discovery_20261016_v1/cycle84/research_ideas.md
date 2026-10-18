# One retained development question

**Question:** Given specified physical action ports and sufficient measurement anchors, can a reusable time-series world model be calibrated from limited target experience under coupled measurement and mechanism changes while retaining accurate observable action responses?

**Lineage:** Refine the observation/dynamics reuse candidate by removing its unidentifiable component-classification requirement. Borrow existing adaptation and controlled-identification primitives. Do not claim that combining those primitives is new.

**Current weakness:** C84 provides no failure of a competent adaptive neural baseline in this identifiable setting, and no improvement from a new mechanism. ILA, CSR, AdaJEPA, residual adaptation, ordinary joint estimation and active identification are meaningful competitors. Novelty remains uncertain.

**Minimum distinguishing experiment:** State the physical anchors and action semantics first. Separately vary observation, drift dynamics, both, and neither. Give every method identical target histories, action coverage and adaptation budgets. Evaluate independent observable free rollouts and matched action contrasts. Compare frozen, encoder-only, transition-only and joint adaptation; include competent output-space/joint-estimation controls. Check source model competence, target observability and optimization before interpreting a failure.

**Possible primitive to test, not an accepted method:** Short controlled action contrasts can constrain response calibration without requiring full source-transition matching. This is close to standard active system identification and equivariant representation learning. It needs a specific non-obvious advantage over those alternatives, rather than a new name.

**Stop rule:** If existing joint adaptation/identification matches the required action responses at the same information and compute budget, the experiment supplies no evidence for the proposed new mechanism. If the comparison is confounded by missing anchors, insufficient excitation, weak optimization or unfair model information, fix the comparison before judging the hypothesis.

This is the only retained development route in this cycle. It is not the final recommended research direction and is not evidence of publication potential.
