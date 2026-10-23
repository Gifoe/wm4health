# Same-observation ordinary state-regularization control

Frozen before this control's execution, after the rolling-origin failure discovery. This is a post-discovery attribution study, not a new method, new independent confirmation set, or AIKAE/VAIKAE reproduction. Preserve all previous results. Do not demand improvement for every individual query: finite-data estimation has a risk tradeoff.

Reuse the exact native seed1000 model, ID data, all75 environments, original11-context banks and300 queries from ROLLING_ORIGIN_AUDIT_PROTOCOL_V1. Use origins0,5,10,15,20, absolute physical time and relative future3..19. No extra observation episodes, physical labels, model-weight updates, clock changes or future query targets enter adaptation/selection.

For each environment, leave each of its11 context episodes out in turn. Infer that calibration episode's environment code from the other10 complete context episodes. Estimate its state from its own three-frame prefix at each of five origins. Thus each environment has55 calibration query-origin units from its already available completed support episodes, not additional query-future information. Final query inference uses all11 contexts. This10-to11 context-count difference is disclosed; all calibration predictions and selections are retained.

Use the native encoder's state mean mu and diagonal variance v=max(exp(logvar),1e-8). Set detached scale s=max(initial observed-prefix pixel-sum MSE,1e-6). Optimize only z:

    J(z)=prefix_pixel_sum_MSE(z) + lambda*s*0.5*sum((z-mu)^2/v)

This is an ordinary proximal/trust-region penalty around an observation-conditioned encoder estimate. It is NOT a claim of a calibrated Bayesian posterior or an independent prior; the encoder already consumed those same observations. The variance floor is a declared numerical stabilization, not a calibration result.

Candidate order is [no_update,10,1,0.1,0.01,0.001,0.0001,0]. Each finite candidate uses the original60 Adam0.01 updates, global gradient clip10, and per-example lowest J among steps0..60. No-update returns the original mu. Select one candidate **per environment**, minimizing mean calibration future MSE over that environment's11 left-out episodes and all five origins; exact ties prefer earlier candidate order. Selection uses only that environment's original11 contexts. Persist SELECTION_V1.json before evaluating any original query future or consulting old query prediction arrays.

On the300 original queries, evaluate this frozen environment-specific candidate with the same60-step inference budget, an unregularized lambda0 replay, and no-update. Report all origins, all queries, absolute future MSE, per-query harm relative to no-update, harm magnitude, state displacement and which candidate each environment selected. Retain prefix-improving future regressions. Selection is by calibration mean, not by evaluation mean, harm rate, quantiles or the cases that failed earlier. Do not change the grid/budget or add retries after seeing evaluation outcomes.

Unregularized query replay must match the previous saved complete prediction images within1e-4 maximum pixel error, and source codes/initial states within1e-5. Stop interpretation if this qualification fails; preserve the failure rather than widen tolerances. Independently reconstruct every leave-one-out bank, candidate selection, objective/argmin and full image metric. Record exact hashes, before/after model state and terminal execution/log-SHA receipts.

Save all8 candidates x5 origins x825 calibration predicted image windows on G, plus original-query predictions, state variables, posterior variances, objective traces and selection. No raw data/arrays/weights are synchronized locally or uploaded to GitHub. This control spends additional inference computation but uses the same observed11-episode library per environment; it cannot establish a fair-compute novel-method gain. A successful ordinary repair weakens a new-architecture claim. A null result alone does not establish novelty, especially because variational assimilation overfitting and uncertainty-aware latent assimilation already have direct prior art.
