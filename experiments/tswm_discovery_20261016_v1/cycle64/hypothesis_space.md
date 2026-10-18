# Competing hypotheses and tests

| Hypothesis | Discriminating test | Current disposition |
|---|---|---|
| H1: observed whole-plan conditionals already equal interventions under sequential ignorability | Exact structural joint law vs intervention product; gain0/H1 negative controls | False for feedback H>1; this is an existing causal distinction. |
| H2: poor observational fitting fully explains the failure | Oracle observed law and its intervention mismatch; correct validation NLL | False as sole explanation; oracle conditioning already differs. |
| H3: exact policy knowledge makes ratio correction of an estimated block law safe | Known-policy precision subtraction, compare estimated-policy and exact identities | False for fitted approximations in some fixed cells. Exact generative identity remains valid. |
| H4: every frozen model requires new training | Temporal factorization of the same retained joint covariance | False for this Gaussian joint checkpoint. |
| H5: low-cost denoised-mean correction equals exact noisy causal guidance | Analytic noisy scores and identical-budget affine DDIM | False in these oracle cells. No learned generator conclusion. |
| H6: an ordinary causal model already solves this attribution problem | Correctly specified causal AR with same train data | Supported in measured cells; main reason not to accept a new framework. |
| H7: SMC operational failures prove new sampling machinery is necessary | Exact-target iid statistical floor; stronger sampler budget/sweep controls if justified | Not established. Original H64 threshold also rejects exact iid samples. |
| H8: efficient reuse of a frozen nonlinear conditional-only generator remains practically harder than existing causal generation | Train-only nonlinear generator; strongest sequential/SMC controls; sealed mechanism/query tests | Untested. Only remaining provisional question, not a supported research gap. |

Next tests must define what access is retained: full joint density, autoregressive factors, score-only generator, policy observations and total compute. Causal AR and same-checkpoint temporal factorization are compulsory whenever mathematically available. A conditional-only access restriction must be motivated by an actual released model API, not invented to disable a strong baseline.
