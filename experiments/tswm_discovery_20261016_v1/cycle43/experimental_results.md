# C43 Actual execution results

本轮PROGRESS；整体ACTIVE_UNACHIEVED，接受最终方向0。新增近邻SPF固定commit a7624ae25b47b474d5a31ca22b3ac6e6907795dd，29个source files/Gitblob+SHA确认。执行原生core/collator/scheduler与原函数体的context adapter，两次50updates preflight共100步（均不作效果）；V2原假设失败保留，V3记录而不强求跨尺度共享。独立复核9000个调度值max0，checkpoint replay max0，同尺度续接差0。此前真实TEST与本轮comparative effects均未评分。

V3 GPU50step qualification runtime1.6747s, peak allocated0.0875GiB; excludes import/default parent construction and is not a runtime benchmark. Loss first0.333848/last0.360973 on different random paths/timepoints; no convergence claim. Synthetic evolving-field fixture supplies only initial field toconditioning. Native labelsdecadal/yearly/monthly are API names with no climate interpretation.

|native target|shape|same RNG replay|unchanged original full API|
|---|---|---:|---:|
|decadal|[2, 1, 1, 8, 8]|0.0|0.0|
|yearly|[2, 1, 1, 8, 8]|0.0|0.0|
|monthly|[2, 16, 1, 8, 8]|0.0|0.0|

Original standard predict and seeded modular full outputs also agree exactly for each level. Optional stop_stage=1 failed because returnedindices was not assigned; metadata-onlyrepair preserves formerly validoutputs. Within-level split/resume afterrepair agrees exactly, verified in a new process from checkpoint.

Cross-resolution first-stage difference5.03659248e-06 at these discarded preflight weights. Independent scheduler endpoints by resolution: [747.2529999999999, 747.2529999999999, 794.206]. Global time embedding depends on the whole path normalization, so identical cross-level prefix is not a valid native assumption. The difference is small and no quality consequence was tested. V2's mistakenzero assertion remains in outputs/native_v2/qualification.json; it did not invalidate native fullsampling.

All observations are API/gradient/schedule qualification. No MC/MLMC query error, model bias, realworld effect or proposedmethod benefit has yet been measured.
