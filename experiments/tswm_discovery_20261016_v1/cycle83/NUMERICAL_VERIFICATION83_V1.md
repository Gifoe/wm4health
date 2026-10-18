# Canonical verifier recovery: align parameter flags, retain exact equality

The first verifier stopped at an exact native-field replay comparison and did not create a PASS artifact. Its exit/log are preserved. The numerical screen had completed normally; no simulation was restarted or changed.

The runner explicitly sets every pretrained parameter's `requires_grad` flag toFalse. The first verifier called `eval()` but omitted this setting. A32-case diagnostic replayed all16 contexts with both flag states. With trainable flags, seven Hill contexts differ slightly from stored outputs (maximum2.980232238769531e-7; maximum field-score difference2.4655766317494e-10). All repeated queries within each state are exact. With frozen flags, all16 stored-field replays are bitwise identical. This empirically isolates the flag-state difference; no claim about a particular low-level kernel is required.

Canonical `verify83_v2.py` matches the original frozen flags and retains the original exact-equality assertion and every coefficient/score/reference tolerance. It writes the first valid VERIFICATION83_V1.json. `attribute83_v2.py` and `finalize83_v3.py` use the same flags. The original versions remain for provenance. No weight, observation, hyperparameter, model coefficient, prediction array or reported numerical score is changed. NATIVE_REPLAY_DIAGNOSTIC83_V1.json records all32 comparisons.

Reproduction uses the V2 verifier, V2 attribution and V3 finalizer. The recovered queue's stage exits identify the actual executed scripts. A PASS is required before export; this note alone is not evidence of successful verification.
