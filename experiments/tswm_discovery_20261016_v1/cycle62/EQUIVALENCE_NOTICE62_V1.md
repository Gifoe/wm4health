# Design limitation found before test scoring

The selected official ETTh1 scripts set L=H=96. In Informer_PRO.short_forecast the padding/truncation branches are inactive, and linear_PRO maps the same49 encoded positions to96 positions as Informer_DYN.linear_DYN. After renaming that parameter key the model computations and parameter shapes coincide. Numerically qualify identical-weight forward and gradients in evaluation and training modes, using the same RNG on training-only histories.

Thus the initial three-name comparison does not isolate PRO versus DYN mechanisms. The choice of this equal-length cell was insufficient for that purpose. Retain all original planned fits and source configuration as a partial reproduction; interpret PRO/DYN seed-pair differences as execution variability, not separate architecture evidence. Do not rescue the comparison by claiming new semantics for numerically equivalent functions.

No extra fit or hyperparameter is added on the basis of test scores. If this source family is investigated later, L unequal to H is necessary to distinguish these code paths, but doing so alone still does not establish a new world-model research gap.
