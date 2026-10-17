# Multi-view temporal inference: source boundary and next-test filter

This is a source review, not a reproduction or an accepted new research direction. It extends the C46 branch notes while the final C54 fit runs. It does not alter any C54 training, selection, scoring or risk protocol.

## Direct prior art and released implementation

[Multi-View Reinforcement Learning](https://arxiv.org/html/1910.08285v1), NeurIPS2019, already formulates heterogeneous observation histories over shared dynamics. Its model-based construction includes recurrent memory, stochastic latent variables, view-specific observation models and cross-view policy transfer. Section3.2 does not impose the simultaneous availability of all views as the model-free variant does. The reported visual transformations and added-noise control tasks do not establish arbitrary partially observed sensor generalization. Generic temporal multi-view belief modeling is occupied territory; the paper is not a proof that every finite-data composition problem is solved.

The released repository is separately pinned at [bccf90fb4356cf0eecffe35fdbcab814912717fa](https://github.com/mlii/mvrl/tree/bccf90fb4356cf0eecffe35fdbcab814912717fa), committed 2020-02-09T23:02:16Z. Seven source/configuration files were inspected, with Git blob IDs recorded in MVRL_SOURCE_REVIEW54_V1.json. No native package was installed, no repository launcher ran, and no source code is copied into this project. The reviewed repository tree contains no license file; a future reproduction needs an appropriate source-use decision before redistribution.

The implementation has distinct image VAEs and a shared recurrent predictor. In train.py, one VRNN object is constructed and passed to build_rnns for the view-specific VAEs. In ops.py, build_rnns calls build_rnn_with_vae and rnn_with_vae uses reuse=True on that same RNN. process_z_with_vae pairs each encoded prefix frame and action with the next frame. The image VAE encoder itself receives an individual frame, while the RNN carries temporal information; it is not an executed general filtering-posterior interface. The wrappers implement transpose, swapping, inversion and mirroring. These are concrete distinctions between the paper's general formulation and the particular released visual experiment, not evidence of a new gap or an author error.

The source also contains distribution-alignment and cross-view reconstruction diagnostics. Their presence prevents calling shared recurrent dynamics plus separate encoders/decoders a new architecture. Their exact loss is not a certificate of arbitrary posterior calibration or success with information-destroying sensors. Dependencies target the TensorFlow1-era API, including tf.contrib. Installing that legacy runtime is not justified merely to accumulate another reproduction.

## Research decision before launching another composition sweep

The C46 rescue lead, “make the sensor noninvertible,” does not by itself justify a new method. Partial observability and temporal inference already belong to the multi-view world-model formulation. Conversely, failure on a noninvertible sensor can be unavoidable information loss, rather than deficient learning. A comparison must distinguish those explanations.

Any renewed experiment must first specify a future observation query whose conditional distribution can be learned from the allowed data. It must include a competent history-aware shared generative state-space control, not only a frame encoder or separately trained networks. A known-parameter filtering oracle must quantify irreducible ambiguity, and an ordinary adapted or joint-likelihood control must test whether the learning factorization already suffices. State-coordinate recovery is an auxiliary diagnostic; symmetry-related latent states must not be scored as a physical failure when their queried futures agree.

A potentially worthwhile residual would be a repeatable finite-data or compute advantage for learning reusable probabilistic transitions across independently held-out sensor–dynamics combinations, under matched calibration information and causal history. Merely changing the sensor map, naming a belief module, or beating a stateless baseline is insufficient. No such residual has yet been demonstrated. The immediate decision remains the C54 native RP-GSSM default/existing-EM-option comparison with corrected ordinary EM and the causal oracle. Do not start another neural sweep on the strength of this review.

Overall research objective: ACTIVE_UNACHIEVED. Accepted final directions:0. MVRL fits:0. Extra C54 optimizer updates or heldout reads from this review:0.

