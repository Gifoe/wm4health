# Native persistent-context uncertainty intake

Scope: general time-series world-model dynamics, not tuning one dataset. Previous cycle accepted no direction.

Inspect official MTS3 and HiP-RSSM implementations to determine whether their repeated moment propagation retains the covariance between a persistent latent mechanism/context and evolving fast state. A possible discriminator is persistent uncertain forcing: correct variance can grow quadratically over time, unlike independent fresh-noise variance. This is a source question, not an assertion that these methods fail.

Strong existing solutions include augmented-state Kalman filtering, Schmidt/consider filtering, Rao-Blackwellized particles and persistent model sampling (PETS trajectory sampling). These are mandatory novelty boundaries. Exact linear-Gaussian propagation is a control, not an invention. No new direction is accepted before checking native code, trained-model relevance, and those alternatives.

Source review first; no launcher, training, unsafe deserialization, external telemetry, or large dataset download authorized by this intake script. Ordinary subsequent research operations remain within the user's overall authorization. Preserve cycles1-14 and existing models.
