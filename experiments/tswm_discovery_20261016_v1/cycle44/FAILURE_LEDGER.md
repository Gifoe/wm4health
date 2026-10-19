# C44 implementation failure ledger

`qualify_analytic44.py` v1 failed before numerical comparisons because the returned transition matrix is a Parameter and its CPU-to-NumPy conversion omitted detach. Original source and empty attempt folder preserved. V2 changes only detached diagnostic conversion and uses a new output folder; no model or numerical formula is changed. Formal training is a separate live process and is unaffected.
