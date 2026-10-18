# Existing methods and the scope of C85

[Cross-Embodiment Transfer](https://arxiv.org/html/2406.01968) supplies the pinned neural implementation. Its source/target alignment assumes reusable source dynamics. C85 changes the task, network size, source training objective and paired calibration information. A failure under changed physical dynamics is outside that invariant-dynamics premise and cannot by itself establish a gap in the paper.

[ILA](https://invariance-through-latent-alignment.github.io/) already combines latent alignment and dynamics consistency for observation changes. [MoVie](https://arxiv.org/abs/2307.00972) is another existing new-view adaptation baseline. C85 does not run either implementation.

[AdaJEPA](https://arxiv.org/html/2606.32026v1) and [Sandwich-Residuals](https://arxiv.org/html/2609.21740v1) establish online partial-model updates and lightweight input/output correction as current prior art. C85's joint output-error control is an ordinary numeric adaptation baseline, not their original implementation or benchmark reproduction. [CSR](https://arxiv.org/html/2407.20651v3) already models change factors and selective adaptation.

Dictionary regression, warm-started regularization and multi-step output-error training are established tools. C85 introduces none of them as a new method. Its contribution to the ongoing search is a falsification screen: determine whether a proposed need for a new calibration mechanism survives competent existing repairs in a clearly specified subcase.

The task is low-dimensional, fully observed up to an invertible sensor map, with known action units and eight physical calibration pairs. It does not test partial observation, a single universal cross-family pretrained model, uncertain action semantics, hidden confounding, or clinical deployment. Success here does not settle those problems; poor scores there would also require competent existing latent-state and identification controls before supporting novelty.
