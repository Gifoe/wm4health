# Cached tensor loading qualification

The first contiguous per-window copy wrapper passed its window/forward/gradient comparisons but its stress process also terminated with a native fault, after the30000-window progress marker. It is not qualified and will not be used for training. Its diagnostic output and source are retained.

The next diagnostic tests cached float32 CPU tensors: each native DatasetForTransformer initializes values and timestamp tensors once; each sample uses only tensor slicing/cloning. This removes repeated per-window NumPy/Torch interoperability. Compare native outputs and same-core forward/gradient results, then require180000 window lookups without failure. No optimizer steps in qualification. The native model, batch generator, sample order, optimizer and validation remain unchanged. No source bytes are edited. This is runtime compatibility, not a scientific method or a demonstrated fault root cause.

Only after PASS may the five missing planned fits proceed under this cached loader; four completed native fits are preserved. Keep both failed formal attempts and the failed loader diagnostic. Report mixed loader execution paths and all known aborted-work lower bounds. No test set has been opened.
