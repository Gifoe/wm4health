# Implementation correction before any accepted pilot result

The first execution failed its independent exact O(2) cross-check on case
d=2, T=64, seed=61, before saving any scientific result row. Traceback:
`continuous_counterfactual_pilot.py:160 AssertionError` in the comparison of
the feasible local extrema with the exhaustive trigonometric critical points.

Inspection found the feasible optimiser's gradient and Armijo signs reversed:
it ascended for the requested minimum and descended for the requested maximum.
Both signs were corrected. The frozen scientific case list, data seeds, query,
SDP constraints and 1% stopping rule are unchanged. The original plan timestamp
is preserved. No failed-row metric is used as research evidence.

This is an implementation error, not model failure or scientific evidence.
