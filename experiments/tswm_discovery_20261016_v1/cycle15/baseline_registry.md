# Actual execution and provenance

## Executed native source

Official ALRhub/MTS3 revision `ab971c0ec43df7664c4b727ca69d5b8dd0cab5da`, 152 source/config/document files, no AGENTS files or detected hardcoded credentials. Import only the inspected Gaussian marginalization module. No native experiment launcher, W&B logging, downloaded pickle or pretrained checkpoint was executed.

`Predict` worker and HIPRSSM are exercised with prescribed admissible parameters. Both use the same source file and covariance formula. latent_obs_dim1 means a two-dimensional fast state and two-dimensional context; block covariance within each is exact in this experiment. This is a native implementation discriminator, not a trained model or original-paper benchmark reproduction.

## Strong ordinary controls actually executed

1. Closed-form marginal moments of a single persistent Gaussian context.
2. Closed-form marginal moments of independently redrawn contexts.
3. Standard dense augmented-state Gaussian propagation retaining fast/context cross-covariance.
4. Independent NumPy simulation, 50,000 particles per preselected case, and independent arithmetic/interval verification from saved arrays.
5. CUDA eager inference cost for native factorized vs conventional augmented propagation,6 preselected dimension/batch configurations.

## Source inspection only

Selected16-file original HiP-RSSM source at `f4adef0ce5ed4b37c9847973df320f72965bd330`. This is not a complete original checkout and none of it was executed. Other models reviewed in earlier cycles have not been retrained here.

## Dataset and training status

No dataset-based fit was run in cycle15. The publisher MTS3 processed-data Google Drive URL timed out from both hosts. The original HiP-RSSM numeric source was located and its Git blob pinned, but an optional local download did not complete; the later live request was deliberately stopped to prioritize the ordinary-control cost gate. No data was deserialized or substituted and no original-paper data reproduction is claimed. These acquisition issues did not block the native operator experiment.
