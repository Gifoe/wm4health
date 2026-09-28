# VitalDB training-support geometry diagnostic

Third diagnostic in the `wm4health` series. The question is whether support of a real held-out patient and observed prospective treatment schedule predicts factual five-minute BIS rollout error. It reuses the Round 2 prospective RSSM and Round 1 cohort. TCI CE and future physiology do not enter support scores. This is a reliability audit, not a clinical safety device or a causal treatment analysis.

Run on the data host with the preceding experiments adjacent to this directory:

```bash
bash scripts/run_pipeline.sh
```

The script trains seeds 1–4 and copies the exact Round 2 seed-0 checkpoint. To resume only extraction and analysis, use `bash scripts/run_after_training.sh`. Detailed results are in `outputs/FINAL_REPORT.md`.

The fixed seed-0 representation defines a bank of **every eligible TRAIN window**. Separate state, flattened future-action, 14-feature action-summary, and block-normalized joint embeddings are indexed with FAISS IVFFlat (256 lists, 24 probes). A kNN score is the 20th neighbor's Euclidean distance. Covariance distances use 5% ridge shrinkage. Conditional support searches the 20/50/100 nearest training states and takes the closest action schedule among them; the value of k is chosen by equal-patient-weighted validation high-error average precision. Local tangent support fits PCA on 20 joint neighbors and retains 90% of their variance. Means, scales, covariance matrices, indices, and tangent fits use TRAIN only.

Dynamic support uses 50,000 randomly sampled TRAIN windows rolled out with the frozen seed-0 model. A test query at horizon *h* is compared against TRAIN rollouts at the **same** horizon, using the predicted latent and the observed prospective action at that step. A fixed TRAIN 95th-percentile within-bank distance scales each horizon. The five-seed ensemble variance is the population variance of physical BIS predictions at each step; full-horizon uncertainty is their 30-step mean. Neither score is assumed to measure true uncertainty before testing.

Primary statistical units are patients. Window weights within each patient sum to one. Correlations, high-error detection, and risk at each retained coverage use equal-patient weights. Confidence intervals and paired comparisons resample patients 1,000 times. Test error defines the top-20% and top-10% labels for descriptive evaluation only. The combined logistic risk score and selective-rollout thresholds use VALIDATION outcomes only. The random-latent control replaces latent coordinates with independent Gaussian draws; an orthogonal rotation would preserve Euclidean distances and cannot test representation specificity.

`outputs/arrays/` holds full per-seed test and validation predictions, ensemble mean/variance, latents, future-action queries, masks, and dynamic support scores on the data host. These arrays are omitted from Git because they are large and contain reconstructible patient-level trajectories; compact per-test-window support scores, summary tables, figures, checkpoints, and code are versioned. All primary analyses use real held-out outcomes. Device-computed TCI CE never enters support fitting.

`outputs/support_scores.csv` includes `BIS_MAE_full_5min` as a **post hoc evaluation column**. It is appended only after all geometric scores have been computed. It must never be used to fit support geometry or validation calibrators.
