# Executed interface evidence

Full weight loading and native computation succeeded. Native predict returned[1,32,2] finite values. For the160-point simulate input, native output length is176 due to the current padding expression; full outputs are retained and diagnostics use the declared first160 points. The API mutates caller input by clearing hidden future response, so each run receives a fresh clone. These are implementation facts, not predictive-quality results.

| path | output shape | known-channel output max change | second-call known-channel input max change |
|---|---|---|---|
| native_simulate | [1, 176, 2] | 0.12810114 | 0.28093562 |
| native_hidden_perturbed | [1, 176, 2] | 0.12810114 | 0.28093562 |
| clamped_simulate | [1, 176, 2] | 5.9604645e-08 | 0 |
| native_late_covariate_perturbed | [1, 176, 2] | 1.7336943 | 0.25971395 |

The native stored-tensor update changes all channels, including the known covariate, before the next encoder call. Two AST subscript changes restricting that update to hidden channels reduce the next encoder's known-covariate discrepancy from0.2809356153 to0 and output discrepancy from0.1281011403 to5.96e-8. Original source bytes remain unchanged. This is ordinary clamping, and no response accuracy improvement is inferred without ground truth.

Changing only hidden future responses by+111 produces exactly0 output change. Perturbing only the final16 known covariate points changes the first16 generated response points by0.0737522393. This is a whole-plan conditional-interface diagnostic. It is not a failure against a universal causal guarantee; conditional completion can depend on the whole known plan.

Native insight encoded observed history as[2,8,768], then flattened to[8,1536] by feature-major channel interleaving. The last768 coordinates therefore contain384 nonaction coordinates and omit384 action coordinates. Sentinel arithmetic and actual pretrained embeddings both verify that action channel1 occupies coordinates1,3,...,1535. Original unconditional/conditional and one indexing-only conditional repair each returned finite[1,16] outputs, with identical fitted3x3 transition matrices and exactly normalized rows. Indexing-only repair changes the returned conditional fixture by0.137541741; this is not an accuracy gain. It also does not prove alignment between separately encoded actions and jointly encoded state/action centroids.

Independent replay PASS. Scope: {"CPU_GPU_native_outputs": 2, "array_artifact_hashes": 2, "interface_output_shapes": 4, "interface_scalar_metrics": 6, "loaded_npz_arrays": 32, "transition_laws": 3}. Fresh CPU native predict versus GPU max absolute difference=9.83476639e-07; fresh history encoder difference=6.67572021e-06, both below fixed2e-4. Saved scalar/layout checks use at most1e-6 and are recorded individually. All32 saved arrays remain in two hashed G artifacts. No training was performed, no held-out forecasting benchmark was scored, and no causal truth was available in this fixture.

These facts justify correcting the implementation before capability comparisons. They do not establish a research gap, generalization failure, stochastic calibration failure or need for a new world-model architecture. The two sampled Bayesian branches and repaired branch share the same fitted context transitions; they are not three independent learned models.
