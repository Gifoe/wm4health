# Learned covariance diagnostic (fixed before final checkpoint evaluation)

This implements the gate in FULL_MODEL_PLAN.md. It is a validation-only mechanism diagnostic, not a proposed method or original-paper reproduction. Training files and native source remain unchanged.

Freeze the selected checkpoint after training terminates, verify its recorded SHA256, and use all pre-existing nonoverlapping validation origins. A separate smoke run at fixed step100 uses the first8 origins only to verify hooks, equations and serialization; its results cannot select a checkpoint, method or hyperparameter.

The last observed input is source position150. Capture its native worker posterior and the current slow-context posterior. At the subsequent75 predictions all future observation masks are false. The same context marginal is reused within this window. Initialize fast/context cross-covariance to zero in each control; this is an explicit assumption, not a recovered learned posterior.

Compare three covariance trajectories with identical initial marginal covariance, learned A,C,Q, context covariance S and native mean: (i) native pairwise block-diagonal projection at every step; (ii) full fast covariance with the cross-scale covariance reset to zero at every transition; (iii) ordinary augmented joint covariance, retaining fast/context cross-covariance R. The second control separates spatial projection from dependence on persistent context. Equations: P'=A P A^T+C S C^T+Q+A R C^T+C R^T A^T and R'=A R+C S. Use the same S for the entire unobserved slow window.

Verification includes reproducing the entire native projected trajectory in float64 from the captured initial factors; checking all context inputs remain constant and the future observation updates leave each prior unchanged; checking the two dense controls agree at horizon1; and an independent scalar analytical recurrence with positive, zero and negative persistence. Record discrepancies explicitly and stop on failed invariants.

Feed only each covariance's original upper/lower/paired-side diagonal summary through the unchanged native decoder. Preserve its mean input. A changed covariance summary can be out of the trained decoder's distribution. Score changes are descriptive and cannot establish a correction to the true posterior. Report latent context/noise contributions, horizon-wise covariance differences, decoder sensitivity, validation NLL/coverage and the fixed native forecast accuracy. Do not use final20% test labels.

No result from this diagnostic alone accepts an innovation. Negligible learned context contribution, negligible decoder response, or adequate ordinary inference corrections are negative evidence for an architectural claim. A remaining effect would need further attribution, appropriate trained controls and independent held-out confirmation.
