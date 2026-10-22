# Preserved preflight error excerpt

```text
{"file": "record_counterfactual_dreams.py", "fixture": "native_broadcast", "batch": 1, "calls": 40, "nonzero_edit_calls": 32, "max_abs": 10.0}
{"file": "record_counterfactual_dreams.py", "fixture": "native_broadcast", "batch": 2, "calls": 40, "nonzero_edit_calls": 32, "max_abs": 10.0}
{"file": "record_counterfactual_dreams.py", "fixture": "axis_only_non_native", "batch": 1, "calls": 40, "nonzero_edit_calls": 32, "max_abs": 10.0}
{"file": "record_counterfactual_dreams.py", "fixture": "axis_only_non_native", "batch": 2, "calls": 40, "nonzero_edit_calls": 32, "max_abs": 10.0}
{"file": "record_counterfactual_plausibility.py", "fixture": "native_broadcast", "batch": 1, "calls": 31, "nonzero_edit_calls": 0, "max_abs": 0.0}
{"file": "record_counterfactual_plausibility.py", "fixture": "native_broadcast", "batch": 2, "calls": 31, "nonzero_edit_calls": 0, "max_abs": 0.0}
{"file": "record_counterfactual_plausibility.py", "fixture": "axis_only_non_native", "batch": 1, "calls": 31, "nonzero_edit_calls": 29, "max_abs": 6.9273576736450195}
{"file": "record_counterfactual_plausibility.py", "fixture": "axis_only_non_native", "batch": 2, "calls": 31, "nonzero_edit_calls": 29, "max_abs": 6.9273576736450195}
{"file": "record_counterfactual_plausibility_action.py", "fixture": "native_broadcast", "batch": 1, "calls": 8, "nonzero_edit_calls": 0, "max_abs": 0.0}
{"file": "record_counterfactual_plausibility_action.py", "fixture": "native_broadcast", "batch": 2, "calls": 8, "nonzero_edit_calls": 0, "max_abs": 0.0}
{"file": "record_counterfactual_plausibility_action.py", "fixture": "axis_only_non_native", "batch": 1, "calls": 8, "nonzero_edit_calls": 6, "max_abs": 5.411576271057129}
{"file": "record_counterfactual_plausibility_action.py", "fixture": "axis_only_non_native", "batch": 2, "calls": 8, "nonzero_edit_calls": 6, "max_abs": 5.411576271057129}
{"file": "record_counterfactual_plausibility_obs.py", "fixture": "native_broadcast", "batch": 1, "calls": 31, "nonzero_edit_calls": 0, "max_abs": 0.0}
{"file": "record_counterfactual_plausibility_obs.py", "fixture": "native_broadcast", "batch": 2, "calls": 31, "nonzero_edit_calls": 0, "max_abs": 0.0}
{"file": "record_counterfactual_plausibility_obs.py", "fixture": "axis_only_non_native", "batch": 1, "calls": 31, "nonzero_edit_calls": 30, "max_abs": 2.309119462966919}
{"file": "record_counterfactual_plausibility_obs.py", "fixture": "axis_only_non_native", "batch": 2, "calls": 31, "nonzero_edit_calls": 30, "max_abs": 2.309119462966919}
Traceback (most recent call last):
  File "D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\scripts\run_source_editor_contract_v1.py", line 68, in <module>
    result,state=nj.pure(fn)({},jax.random.PRNGKey(SEED),data);assert not state
  File "G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\mechanism_edit_20261023_v1\source\dreamerv3_compat\dreamerv3\ninjax.py", line 95, in purified
    out = fun(*args, **kwargs)
  File "G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\mechanism_edit_20261023_v1\source\contextual_mbrl\dreamer\record_counterfactual_plausibility_obs_dataset.py", line 44, in gen_dream
    data = tree_map(lambda x: x[:, :-1].reshape([10, -1] + list(x.shape[2:])), data)
NameError: name 'tree_map' is not defined

```
