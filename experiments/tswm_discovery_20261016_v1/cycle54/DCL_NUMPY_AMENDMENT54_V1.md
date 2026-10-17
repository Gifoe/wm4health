# DCL preflight environment correction, before formal comparisons

The first successful numerical/GPU qualification used an unchanged53-file native core and32 discarded updates, but its separate venv inherited NumPy2.2.6 from the shared base. A direct module-path/version query confirmed this; the source requirements specify NumPy1.26.4. Preserve that run and its weights/fixtures as an initial implementation check, not an exact source-environment reproduction.

Install NumPy1.26.4 only into cycle54/.venv_dcl, retaining Torch2.8.0+cu128, typeguard2.13.3 and jaxtyping0.2.38. Do not modify the base or RP/JAX runtimes. Verify actual imported versions and module paths, rather than flattening duplicate installed-distribution metadata into a dictionary. Record the base NumPy source-file hash before/after.

Run the same numerical, native pair-boundary and32-update toy protocol again into distinct dcl_preflight_v3 outputs and artifacts. There will be64 discarded preflight optimizer updates in total across two runs,0 formal DCL fits and0 paper metrics. Do not pool the runs as independent scientific seeds or use their toy losses to select an algorithm. Neither run reads C54 heldout data or changes its models, training queues or scoring gate. The previous32-update publication snapshot remains an immutable historical snapshot; the latest report must state both runs and total count.
