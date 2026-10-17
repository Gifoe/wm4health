# Existing native EM switch: required falsification control

The native source already exposes `Config.em=True`, which stops gradients through the inferred posterior during the M-step. Formal C54 fits retain the public default False, as explicitly recorded before training. The manuscript's EM language and this source option must not be conflated with the default total-derivative recipe.

At the same completed small-task checkpoints, compare native False/True gradients without changing any model parameter. Use the 25 balanced batches from the previous reference control. Verify that forward scalar losses remain the same, since stop-gradient changes only the derivative. Compare mean native batch32 A gradients with all-200 A gradients separately for each switch value.

For True, independently hold the full E-step posterior and recognition sites fixed, rebuild only the prior chain as A changes, and differentiate the native M-step loss terms. Check this partial derivative against the True switch, including a central finite difference at alpha=1, step1e-5, on this explicitly fixed-posterior objective. A finite difference that recomputes the E-step would test the False total derivative and is not a valid check of an EM partial derivative.

All gradients are taken at frozen checkpoints. No optimizer step, state label, heldout observation or checkpoint selection is used. If this already-provided option removes reference dependence of the A update, the current gradient observation cannot alone justify a new world-model architecture. It does not establish that a complete training run with True has calibrated representations or better future predictions; that requires a separately declared learning comparison.
