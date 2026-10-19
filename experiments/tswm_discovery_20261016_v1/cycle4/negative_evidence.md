# Cycle4 negative evidence

1. Motion-specific nuisance hypothesis failed its frozen2000-step gate:gap
0.011052<0.10. Do not replace it by a post-hoc threshold.
2. The default LeWM deficit fell from0.497659 to0.000123 with8000steps.
Training duration explains that apparent failure; no new architecture needed.
3. Ordinary training-only CCA fixes the matched-static adapter to0.000283.
4. Motion warm-start still has0.085087NMSE; a single initialization and linear
adapter do not establish native visual failure or a population optimum conflict.
5. New VJEPA source passes its native CNN input-shape check. An internal mistaken
shape suspicion was corrected by execution; it is not a baseline defect.
6. Passing future input to the old linear probe does not use it. No future leakage
was found. Target-KL has no encoder gradient in that implementation, but exp10 has
an explicit VICReg anti-collapse path; do not claim all versions lack anti-collapse.
7. Risk pilot simple attached repair did not meet its ALL-initialization gate:
False. Seed43 is retained. No hyperparameter or checkpoint selection on risk truth.
8. Full-observation learned variance is worse despite exact physical-state
availability; this is fit/regularization evidence, not nonidentifiability.
9. Ordinary predictive cross-moments recover the risk driver to NMSE
0.039036 and mean policy regret0.001368, without physical labels.
This resolves the toy's practical information-recovery deficit. A new-method route
based only on that toy is therefore stopped. Second-moment SDR is established.
10. Switching-graph basis alignment has explicit connected/spanning-tree results
in Bencherki et al.2023. That elementary central claim is covered. Finite-sample
stochastic decision guarantees would be a different, currently untested question.

Engineering only: an initially absent local staging directory caused upload failure;
created it and securely retried. CUDA convolution determinism was enabled for the
native gradient equality check. No scientific result is inferred from these errors.
