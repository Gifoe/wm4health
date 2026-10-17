# Why independent references are a diagnostic control

The C57 independent-reference adaptation develops rapidly growing recognition precision in completed train-only runs. It is not a generally safe substitute for the native normalizer. This note tests the mathematical objective without reading any formal model, state label, validation or heldout observation.

Take a stationary scalar prior with a=0, three independent time points, unit site precision, recognition mean b*x, target x=(1,1,1), and four external reference sequences equal to zero. With ideal covariance arithmetic, the per-time centered loss is exactly

    L_external(b) = 0.5*log(4/3) - b^2/12.

The target posterior is N(b/2,1/2). The positive-minus-external log kernel contributes b^2/4; local KL contributes (0.5-log(1.5))/2+b^2/24, and prior KL contributes (-0.5+log(2))/2+b^2/8. Their combination yields the expression above. The loss is unbounded below as b grows even at fixed precision. The target lies outside that finite empirical reference support. Including the positive in a four-item bank instead guarantees the log-reference term is at least -log(4), and the KL terms are nonnegative.

The script verifies the exact expression and retains the literal reporting-jitter and nonzero-a cases separately. The ideal a=0 formula is not silently asserted for those cases. The independent bank here is a mathematical adaptation, not a native option or an execution of a bounded-similarity image-learning pipeline. The counterexample alone does not prove that every population expectation is unbounded. It explains why a failed external-bank control cannot establish that competent existing global-normalizer methods are insufficient.

No model or training rule was changed, no failed attempt was removed, and no heldout score was opened. All planned controls and seed outcomes remain visible. Centered running estimates and complete-reference controls, rather than this unsafe diagnostic alone, are required to assess the candidate repair.
