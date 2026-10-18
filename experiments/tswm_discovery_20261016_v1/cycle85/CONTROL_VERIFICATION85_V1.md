# Independent verification of the ordinary controls

Reconstruct weak matrices using a polynomial test function and its analytic polynomial derivative in separate code. Check selected coefficients by augmented least squares, atol=rtol1e-5. Independently replay selected fields using RK4 and independently constructed features, atol=rtol2e-5; failed paths must retain their failure classification. Recompute all stored scores from raw arrays to1e-10. Verify validation-only selectors, data/weight/selection hashes, unchanged overlapping EVAL arrays, same-budget long/short allocations and retained noisy prefixes. Recheck every continuous physical reference with separately implemented DOP853 equations, atol=rtol1e-4. Report any numerical guard/failure without silently deleting it.

These are attribution checks. They do not validate a new method or retrospectively make first-two-seed controls confirmatory. Seed2026853 is recorded as the unopened check of the frozen ordinary-control design.
