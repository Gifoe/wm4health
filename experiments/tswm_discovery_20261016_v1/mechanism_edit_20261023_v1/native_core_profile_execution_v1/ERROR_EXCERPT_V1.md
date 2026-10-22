# Preserved preflight error excerpt

```text
.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\.venv_editor_cpu\lib\site-packages\jax\_src\numpy\array_methods.py:118: UserWarning: Explicitly requested dtype <class 'jax.numpy.int64'> requested in astype is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\.venv_editor_cpu\lib\site-packages\jax\_src\numpy\array_methods.py:118: UserWarning: Explicitly requested dtype <class 'jax.numpy.int64'> requested in astype is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\.venv_editor_cpu\lib\site-packages\jax\_src\numpy\array_methods.py:118: UserWarning: Explicitly requested dtype <class 'jax.numpy.int64'> requested in astype is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\.venv_editor_cpu\lib\site-packages\jax\_src\numpy\array_methods.py:118: UserWarning: Explicitly requested dtype <class 'jax.numpy.int64'> requested in astype is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\.venv_editor_cpu\lib\site-packages\jax\_src\numpy\array_methods.py:118: UserWarning: Explicitly requested dtype <class 'jax.numpy.int64'> requested in astype is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\.venv_editor_cpu\lib\site-packages\jax\_src\numpy\array_methods.py:118: UserWarning: Explicitly requested dtype <class 'jax.numpy.int64'> requested in astype is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
Optimizer model_opt has 17,052,480 variables.
Traceback (most recent call last):
  File "D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\scripts\profile_native_core_v1.py", line 34, in <module>
    leaves=jax.tree_util.tree_leaves((out,variables));assert all(np.isfinite(np.asarray(v)).all() for v in leaves if hasattr(v,'dtype'))
AssertionError

```
