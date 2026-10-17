# Active runtime implementation

The active trainer is `train_native53_v6.py`; active supervisor `capture_native53_v6.py`; final selection file `SELECTION_SEAL_V6.json`; evaluator `score_native53_v7.py`. It uses effective batch128 /microbatch64 only for PatchTST AE concat, as numerically characterized in `RUNTIME_SUPPLEMENT_V5.md` and `microbatch_qualification_v2`.

The V5 integration attempt stopped on its first validation with `KeyError: train`: the microbatch epoch wrapper had been installed outside the read-only curve wrapper, bypassing its training-stat capture. V6 installs microbatching first and records the resulting epoch returns afterward. This is an instrumentation-order repair, with the same optimizer, data, model, loss and selection rules. The failed V5 attempt has no selected checkpoint and is retained/excluded. Completed native fits are reused; no transfer evaluation has opened and no partial attempt warm-starts a replacement.
