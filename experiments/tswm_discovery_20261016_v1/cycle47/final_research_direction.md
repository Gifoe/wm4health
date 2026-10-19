# C47 — no final research direction accepted

**Research Question:** Can a world-model learning objective preserve the coupling between stochastic physical evolution and endogenous observation timing, with useful scheduled-future simulation when observation density changes? This is a capability question; a new framework contribution is not established.

**Evidence:** Sixteen fresh fits, 21200 updates, sum of per-fit wall times 834.007s (two concurrent CPU workers; not elapsed time or GPU hours). EVAL:54 raw groups,13824 model–origin–policy scores,512 unique physical trajectories and1536 paired history contexts,64 joint futures each. Independent NumPy/SciPy verification PASS: max ES discrepancy 4.08e-07, joint NLL/65 discrepancy 2.06e-06, covariance discrepancy 1.41e-05. All frozen weight/source/data/schedule identities checked.

Deprioritize the current fixed-feature Laplace / hybrid recipe as a new world-model framework. In this qualified screen it does not improve complete-clock likelihood training: the bistable future-energy differences are adverse in both training seeds, and OU provides no reliable future-quality advantage. The mean-clock restriction loses useful dependence, but a complete ordinary stochastic-clock model already restores it.

**Existing Methods:** HyperNSDE unsplit dynamics already couple stochastic paths and clocks. Complete joint likelihood handles this dependence; PCF-GAN/SigTPP cover global transform/signature learning. Their actual applicability, not generic expressivity, bounds the current claim. See literature_review.md.

**Proposed Idea:** The tested analytic marked-clock Laplace/hybrid recipe remains unaccepted. No final new mechanism is recommended.

**Falsification:** A complete ordinary baseline restoring the capability without a new loss weakens the necessity claim; adverse hybrid future scores in both bistable seeds directly falsify the current recipe's improvement claim in this screen.

**Feasibility:** All training/evaluation ran on the authorized server. CPU was selected by actual discarded throughput; the single RTX5090 remains available for workloads that benefit. Pinned source, frozen data/configs/seeds, checkpoints, raw paths and independent verification remain saved.

**Next Experiment:** Do not tune this EVAL. A genuinely different learning interaction needs an explicitly defined reusable capability, current primary-source differentiation and a new independent protocol on stronger neural models. Any retained efficiency lead first needs a fair matched-time frontier rather than more seeds for this negative quality result.

Two separately fitted one-dimensional finite-grid Markov carriers, a known Gaussian mark/emission model, known initial prior and known exogenous density multiplier. Not continuous-SDE, learned high-dimensional latent-state, real-domain, history-dependent clock, informative-EOS, unknown observation-policy or causal-intervention validation. Actual physical EVAL paths are shared across density contexts; policy contexts and trained-model scores are not independent new trajectories.

Accepted final directions0; overall goal ACTIVE_UNACHIEVED. Completed evidence screen is progress, not completion of discovery.
