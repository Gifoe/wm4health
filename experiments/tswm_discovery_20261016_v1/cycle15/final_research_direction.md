# Cycle15 decision: a real native finding, no accepted new direction

**Research Question.** Does a probabilistic time-series world model preserve uncertainty from a persistent latent mechanism while propagating future state?

**Evidence.** A pinned native MTS3 prediction operator loses state/context cross-dependence in a prescribed admissible Gaussian process. At h64 with a=1,c=1,s=.5, its nominal95% marginal interval has true coverage19.55%. Closed-form, exact augmented-state propagation and independent50,000-particle simulations agree. This is a native operator finding; no trained-model or real-data failure is claimed.

**Existing Methods.** Standard joint Gaussian propagation corrects the witness. At native latent_obs_dim30,batch32,32-step inference takes about3.3ms and adds about5.3MiB of allocated memory in this test. PETS already retains model identity along rollouts. Therefore persistent sampling or a joint filter alone does not supply innovation.

**Proposed Idea.** No new architecture accepted. The present evidence does not justify specialized covariance compression or renaming the ordinary correction. At most, a trained-model audit could test the importance of this approximation; that audit is not itself an established top-venue research direction.

**Falsification.** A negligible learned context contribution, correct held-out calibration, or an adequate ordinary joint/particle correction would kill a new-method claim. Only a consequential residual under matched information, capacity and training/inference cost would motivate further design.

**Feasibility.** Pinned native source, executable cell and cost controls, original tensors and independent verification are preserved on the single5090 server. No training, processed-data substitution or unsafe pickle loading occurred. The source-native module test fits readily; full training cost is unmeasured.

**Next Experiment.** Establish learned-model relevance before designing a mechanism: inspect a faithfully trained hierarchical latent baseline's context coupling, then compare its declared forecast distribution with ordinary joint/persistent-sampling controls and independent held-out trajectories. Original data downloads were not completed and no full reproduction is claimed. If standard methods resolve the measured issue, switch direction.

**Final recommendation: none yet. Accepted novel directions:0.** The overall autonomous research goal remains active and unachieved.
