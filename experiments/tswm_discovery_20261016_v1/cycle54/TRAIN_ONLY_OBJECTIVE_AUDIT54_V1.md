# Training-only auxiliary-objective audit

Freeze this diagnostic before executing it. It must not change the formal 12-fit protocol, model selection, data split, optimizer, checkpoints, or hidden-state readout. Only completed `linear_small` RP seeds 0 and 1 are inspected. No heldout observations or latent truth are read.

Question: at the frozen parameters, does the expected native batch-32 auxiliary-objective gradient agree with the same native objective evaluated against all 200 training sequences? This separates one possible finite-reference/minibatch contribution from a generic optimization or model-capacity claim. It does not test the proper RPM likelihood and cannot by itself establish a new research gap.

For each model, evaluate the full native objective and its A gradient on the 200-sequence training bank. Compare with 32 independently sampled native batches of size 32, sampled with replacement using a fixed NumPy seed 54032. Retain every gradient and loss. Report gradient cosine, norms and Monte Carlo standard errors. A change of batch size changes the classification normalizer's additive log(B) constant; subtract that constant for descriptive scalar-loss comparisons. The derivative is unaffected.

For an interpretable direction, compute d loss/d alpha at alpha=1 for A -> alpha A. Check the full-bank autodiff directional derivative against a central finite difference with step 1e-5. This is a frozen-objective numerical check, not an optimizer update. The native stabilization projection is not applied to the derivative probe; the perturbed covariance must remain positive definite.

Also calculate the mean and covariance of the empirical mixture of full recognition distributions. Native full recognition is prior times Gaussian site: J_full=I+P_site, mean_full=J_full^{-1} eta_site. Compute mixture moments separately at each time across all training sequences, and report the KL of its moment-matched Gaussian to N(0,I). This is a moment diagnostic, not exact mixture KL or a guarantee of conditional accuracy. No physical-state coordinate or label is used.

Use the existing CPU runtime on the eight logical processors not used by the RP queue. Verify source and checkpoint hashes before/after. Retain all raw training-only arrays on server storage and compact results in the experiment directory. No result file synchronization to the workstation.

Interpretation: agreement weakens a finite-minibatch explanation at these parameters. Disagreement motivates further attribution but is already related to established contrastive-gradient and variational-approximation work. Neither result permits test-driven training changes, a claim of generative correctness, or acceptance of a research direction without independent task evidence and a useful improvement over existing controls.
