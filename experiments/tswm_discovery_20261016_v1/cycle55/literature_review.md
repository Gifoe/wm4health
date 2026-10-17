# C55 literature boundary

Source: [RP-GSSM](https://github.com/gatsby-sahani/rp-gssm/tree/a75225d5edceb972bb872dc9a335608733567dd4), checked unchanged against all21 pinned files. See PROTOCOL_V2.md and DERIVATION.md.

The [ICML2026 paper](https://proceedings.mlr.press/v306/hromadka26a.html), Sections3.1–3.4 and AppendixC, separates the limiting-model identifiability statement from the approximation used during learning. Its recognition-factor definition includes the marginal prior. The present constant-precision scalar family does not automatically satisfy the full-image assumption in that theorem. The elementary Gaussian likelihood/variational-bound argument in DERIVATION.md is a control, not a refutation of the theorem or a new world-model method.

Generic variational-parameter bias and contrastive finite-reference gradient bias already have prior art recorded in C54. The next finite-bank investigation must retain ordinary full-reference/normalizer controls and cannot present a running normalizer or stop-gradient EM as an invention.
