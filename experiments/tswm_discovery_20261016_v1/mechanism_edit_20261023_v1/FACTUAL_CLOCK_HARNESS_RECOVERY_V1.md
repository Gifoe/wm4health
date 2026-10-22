# Preserve the clock-wrapper failure; qualify JAX static flags

The first clock qualification exited1 before any scanned numerical comparison: our jitted pure wrapper passed create=false/modify=false as dynamic JAX arguments. Native ninjax._prerun uses a Python conditional on create, yielding TracerBoolConversionError. This is our wrapper omission, not a numerical or physical-model failure. The training process uses a different unchanged wrapper and continues independently.

The recovery marks only create and modify as static JIT arguments. It preserves the native source, checkpoint, RNG keys, observation/action clocks, thresholds and planned checks. Original FAILURE/exit/stdout files remain immutable; recovery outputs use a separate directory. No optimizer calls or physical-intervention query is introduced.
