# C54 causal state-risk interpretation

All16 scored models were fixed before any heldout score. This analysis reads their existing predictions; no new model is fit or selected.

The realized error decomposes exactly into realized oracle error, conditional-mean discrepancy and a finite-sample cross term. Only the cross term's conditional expectation is zero. Predictable energy and normalized excess are reported at every original horizon, together with raw errors. See outputs/causal_risk_interpretation_v1/risk_by_model_horizon.csv.

| Task | Existing-source or ordinary comparison | Horizon | Conditional-error delta | Pointwise95 sequence interval | Normalized delta |
|---|---|---:|---:|---|---:|
|linear_small|existing True minus matched False, seed0|0|0.483034|[0.346996, 0.630252]|0.449196|
|linear_small|existing True minus matched False, seed0|1|0.74496|[0.559323, 0.926358]|0.767615|
|linear_small|existing True minus matched False, seed0|5|0.872181|[0.714547, 1.03163]|1.35465|
|linear_small|existing True minus matched False, seed0|10|0.811175|[0.678436, 0.941519]|2.10426|
|linear_small|existing True minus matched False, seed0|25|1.26028|[1.10747, 1.41813]|15.2316|
|linear_small|existing True minus matched False, seed0|50|3.07007|[2.79469, 3.36859]|482.22|
|linear_small|existing True minus matched False, seed1|0|0.373564|[0.255503, 0.512892]|0.347394|
|linear_small|existing True minus matched False, seed1|1|0.462421|[0.338992, 0.59632]|0.476484|
|linear_small|existing True minus matched False, seed1|5|0.56945|[0.449998, 0.69607]|0.884458|
|linear_small|existing True minus matched False, seed1|10|0.391376|[0.311901, 0.471319]|1.01527|
|linear_small|existing True minus matched False, seed1|25|0.203027|[0.166104, 0.240208]|2.45376|
|linear_small|existing True minus matched False, seed1|50|0.435474|[0.389664, 0.478678]|68.4006|
|linear_small|mean of3 False model errors minus mean of3 corrected EM errors|0|0.0313332|[0.0247592, 0.0388161]|0.0291382|
|linear_small|mean of3 False model errors minus mean of3 corrected EM errors|1|0.0322342|[0.0257718, 0.0398462]|0.0332145|
|linear_small|mean of3 False model errors minus mean of3 corrected EM errors|5|0.0462341|[0.0378518, 0.0557279]|0.0718099|
|linear_small|mean of3 False model errors minus mean of3 corrected EM errors|10|0.0715231|[0.0587418, 0.0859101]|0.185538|
|linear_small|mean of3 False model errors minus mean of3 corrected EM errors|25|0.115781|[0.0931159, 0.139692]|1.39931|
|linear_small|mean of3 False model errors minus mean of3 corrected EM errors|50|0.073227|[0.0590617, 0.0885338]|11.5019|
|linear_small|mean of2 True model errors minus mean of2 corrected EM errors|0|0.458976|[0.339859, 0.593498]|0.426823|
|linear_small|mean of2 True model errors minus mean of2 corrected EM errors|1|0.635002|[0.491876, 0.783178]|0.654313|
|linear_small|mean of2 True model errors minus mean of2 corrected EM errors|5|0.764717|[0.627608, 0.904173]|1.18774|
|linear_small|mean of2 True model errors minus mean of2 corrected EM errors|10|0.669123|[0.566501, 0.773406]|1.73577|
|linear_small|mean of2 True model errors minus mean of2 corrected EM errors|25|0.840588|[0.752185, 0.932327]|10.1592|
|linear_small|mean of2 True model errors minus mean of2 corrected EM errors|50|1.82138|[1.66822, 1.98891]|286.087|
|linear_medium|existing True minus matched False, seed0|0|0.302918|[0.224867, 0.39114]|0.272078|
|linear_medium|existing True minus matched False, seed0|1|0.142965|[0.0949045, 0.195003]|0.142282|
|linear_medium|existing True minus matched False, seed0|5|-0.0243368|[-0.0736016, 0.0202133]|-0.0365087|
|linear_medium|existing True minus matched False, seed0|10|0.0121804|[-0.0129394, 0.0378766]|0.0305181|
|linear_medium|existing True minus matched False, seed0|25|0.0484342|[0.0286361, 0.0662495]|0.565379|
|linear_medium|existing True minus matched False, seed0|50|0.135981|[0.11494, 0.155804]|20.6294|
|linear_medium|existing True minus matched False, seed1|0|0.277179|[0.165843, 0.391101]|0.24896|
|linear_medium|existing True minus matched False, seed1|1|0.253015|[0.152027, 0.355691]|0.251807|
|linear_medium|existing True minus matched False, seed1|5|0.145537|[0.0788634, 0.213765]|0.218326|
|linear_medium|existing True minus matched False, seed1|10|0.0370148|[-0.0018582, 0.0723357]|0.0927409|
|linear_medium|existing True minus matched False, seed1|25|-0.0893915|[-0.123623, -0.0597817]|-1.04348|
|linear_medium|existing True minus matched False, seed1|50|-0.00899975|[-0.0425577, 0.0206367]|-1.36533|
|linear_medium|mean of3 False model errors minus mean of3 corrected EM errors|0|0.214478|[0.163755, 0.276459]|0.192642|
|linear_medium|mean of3 False model errors minus mean of3 corrected EM errors|1|0.247926|[0.196462, 0.308894]|0.246742|
|linear_medium|mean of3 False model errors minus mean of3 corrected EM errors|5|0.246049|[0.195327, 0.307699]|0.369109|
|linear_medium|mean of3 False model errors minus mean of3 corrected EM errors|10|0.198794|[0.165836, 0.235389]|0.49808|
|linear_medium|mean of3 False model errors minus mean of3 corrected EM errors|25|0.157827|[0.129561, 0.187784]|1.84234|
|linear_medium|mean of3 False model errors minus mean of3 corrected EM errors|50|0.0967422|[0.075967, 0.120129]|14.6765|
|linear_medium|mean of2 True model errors minus mean of2 corrected EM errors|0|0.49867|[0.404472, 0.606942]|0.4479|
|linear_medium|mean of2 True model errors minus mean of2 corrected EM errors|1|0.461087|[0.373979, 0.56055]|0.458885|
|linear_medium|mean of2 True model errors minus mean of2 corrected EM errors|5|0.32689|[0.267697, 0.39467]|0.49038|
|linear_medium|mean of2 True model errors minus mean of2 corrected EM errors|10|0.229276|[0.191436, 0.271661]|0.574453|
|linear_medium|mean of2 True model errors minus mean of2 corrected EM errors|25|0.14868|[0.133262, 0.16452]|1.73557|
|linear_medium|mean of2 True model errors minus mean of2 corrected EM errors|50|0.17015|[0.159737, 0.180617]|25.813|

Intervals condition on frozen fits and cover only complete-sequence sampling; they are pointwise and do not establish multiple-comparison significance or real-domain generalization. Mean per-model errors are compared, not prediction ensembles. Both exact source-option pairs and competent ordinary EM controls remain visible. A residual model-specific linear failure does not establish a new world-model gap, and prior-mean success at nearly unpredictable horizons does not establish useful dynamics. No direction is accepted automatically.

All existing scored artifacts/reports were hash-checked unchanged. Full risk components, normalized errors, intervals, raw resampling-index hashes and scope are in the machine-readable summary.
