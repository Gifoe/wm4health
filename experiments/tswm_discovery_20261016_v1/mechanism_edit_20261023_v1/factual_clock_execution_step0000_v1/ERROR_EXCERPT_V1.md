# Preserved actual error excerpt

```text
Encoder CNN shapes: {}
Encoder MLP shapes: {'obs': (16,)}
Decoder CNN shapes: {}
Decoder MLP shapes: {'obs': (16,)}
jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\scripts\qualify_factual_clock_v1.py", line 24, in <module>
    post,prior=run(obs_scan,embed,previous,first,initial,ctx65,key=key);stepkeys=jax.random.split(key,66)[1:];state=initial;manual=[]
  File "D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\scripts\qualify_factual_clock_v1.py", line 19, in run
    out,after=fn(params,key,*x,create=False,modify=False);jax.block_until_ready(out);assert set(after)==set(params)
  File "G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\mechanism_edit_20261023_v1\source\dreamerv3_compat\dreamerv3\ninjax.py", line 95, in purified
    out = fun(*args, **kwargs)
  File "D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\scripts\qualify_factual_clock_v1.py", line 23, in <lambda>
    obs_scan=jax.jit(nj.pure(lambda e,a,f,s,c:wm.rssm.observe(e,a,f,s,dcontext=c)));obs_step=jax.jit(nj.pure(lambda s,a,e,f,c:wm.rssm.obs_step(s,a,e,f,dcontext=c)))
  File "G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\mechanism_edit_20261023_v1\source\dreamerv3_compat\dreamerv3\ninjax.py", line 380, in wrapper
    return method(self, *args, **kwargs)
  File "G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\mechanism_edit_20261023_v1\source\dreamerv3_compat\dreamerv3\nets.py", line 287, in observe
    post, prior = jaxutils.scan(step, inputs, start, self._unroll)
  File "G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\mechanism_edit_20261023_v1\source\dreamerv3_compat\dreamerv3\jaxutils.py", line 53, in scan
    return nj.scan(fn2, start, inputs, modify=modify)[1]
  File "E:\Anaconda\envs\persist_stable_251\lib\contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\mechanism_edit_20261023_v1\source\dreamerv3_compat\dreamerv3\ninjax.py", line 271, in scan
    _prerun(fun, carry, jax.tree_util.tree_map(lambda x: x[0], xs))
  File "E:\Anaconda\envs\persist_stable_251\lib\contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\mechanism_edit_20261023_v1\source\dreamerv3_compat\dreamerv3\ninjax.py", line 295, in _prerun
    if not context().create:
jax.errors.TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
The error occurred while tracing the function purified at G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\mechanism_edit_20261023_v1\source\dreamerv3_compat\dreamerv3\ninjax.py:65 for jit. This concrete value was not available in Python because it depends on the value of the argument create.
See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError

```
