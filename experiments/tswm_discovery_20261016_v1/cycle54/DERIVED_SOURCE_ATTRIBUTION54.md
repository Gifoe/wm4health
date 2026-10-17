# Attribution for the per-target loss diagnostics

`scripts/per_target_auxiliary54_v1.py`, `v2.py` and `v3.py` adapt the algebra and native parameter operations of `rp_ssm/rpm.py::ConstrainedIVFreeEnergy.get_loss_terms` from `gatsby-sahani/rp-gssm`, commit `a75225d5edceb972bb872dc9a335608733567dd4`, by Samo Hromadka and Kai Biegun. The original source is licensed under Creative Commons Attribution-NonCommercial 4.0 International; the derived helpers retain those terms and their in-file attribution.

Original source and license: [repository](https://github.com/gatsby-sahani/rp-gssm/tree/a75225d5edceb972bb872dc9a335608733567dd4), [LICENSE](https://github.com/gatsby-sahani/rp-gssm/blob/a75225d5edceb972bb872dc9a335608733567dd4/LICENSE).

Changes expose the target-sequence axis and reference-score tensor for a one-factor unconditional model rather than immediately summing them. The diagnostic helpers are checked against the unchanged native scalar objective and gradient. They do not implement a new learning objective or replace the formal native training code. V2 changes PyTree indexing; V3 exposes the score tensor for frozen directional-derivative controls.
