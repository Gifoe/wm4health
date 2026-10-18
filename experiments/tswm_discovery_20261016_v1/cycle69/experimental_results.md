# Primary complete-trajectory results

Units are the released signal units. Do not average these systems into one raw-unit leaderboard or treat contexts/time points as independent systems. WH input/output splits follow exact official loader[5200:184000], then100000 training points and78800 test points. CED splits each500-point trajectory at400. Tanks uses its provided train/test arrays. All numeric files and split values independently verify.

The two preregistered information contracts are retained: Tanks5 versus50 initial outputs; CED4 versus10. WH uses50 in both. Only the allowed initial test outputs enter full simulation; later query prefixes are each expert's own predictions. Every remaining target is scored, including the final partial100-point block. Context windows are250 disjoint400-point windows for WH,61 overlapping windows for Tanks and1 for each CED trajectory. Float32 training-only source normalization; float64 ensemble fusion/scoring. No test hyperparameter selection.

| Record / initialization | Method | RMSE | MAE | common-suffix RMSE | scored points |
|---|---|---:|---:|---:|---:|
|WH0_initial50|meta_mean|0.0168561278|0.0115541108|0.0168561278|78750|
|WH0_initial50|meta_precision|0.0167266168|0.0113635454|0.0167266168|78750|
|WH0_initial50|dyno_author|0.00194892193|0.0015269871|0.00194892193|78750|
|WH0_initial50|context_mean|0.243897727|0.196977514|0.243897727|78750|
|Tanks0_initial5|meta_mean|0.50381765|0.34647529|0.51194763|1019|
|Tanks0_initial5|meta_precision|0.452118612|0.318409394|0.458802141|1019|
|Tanks0_initial5|dyno_author|0.405204873|0.321027311|0.410444195|1019|
|Tanks0_initial5|context_mean|2.10968802|1.80144749|2.13277168|1019|
|Tanks0_initial50|meta_mean|0.485263814|0.325442884|0.485263814|974|
|Tanks0_initial50|meta_precision|0.445228532|0.281827653|0.445228532|974|
|Tanks0_initial50|dyno_author|0.410444195|0.325753474|0.410444195|974|
|Tanks0_initial50|context_mean|2.13277168|1.81628738|2.13277168|974|
|CED0_initial4|meta_mean|0.0659805553|0.0544065069|0.0672199409|96|
|CED0_initial4|meta_precision|0.0659805553|0.0544065069|0.0672199409|96|
|CED0_initial4|dyno_author|0.0651773391|0.0521557996|0.0645752316|96|
|CED0_initial4|context_mean|0.44267169|0.388437118|0.419358533|96|
|CED0_initial10|meta_mean|0.0670938152|0.0554010798|0.0670938152|90|
|CED0_initial10|meta_precision|0.0670938152|0.0554010798|0.0670938152|90|
|CED0_initial10|dyno_author|0.0645752316|0.0508153664|0.0645752316|90|
|CED0_initial10|context_mean|0.419358533|0.367693936|0.419358533|90|
|CED1_initial4|meta_mean|0.0556973413|0.0470794208|0.0548790165|96|
|CED1_initial4|meta_precision|0.0556973413|0.0470794208|0.0548790165|96|
|CED1_initial4|dyno_author|0.0457112006|0.036776237|0.0449523294|96|
|CED1_initial4|context_mean|0.687037895|0.592847222|0.638768799|96|
|CED1_initial10|meta_mean|0.0559084879|0.0459079975|0.0559084879|90|
|CED1_initial10|meta_precision|0.0559084879|0.0459079975|0.0559084879|90|
|CED1_initial10|dyno_author|0.0449523294|0.0359257695|0.0449523294|90|
|CED1_initial10|context_mean|0.638768799|0.553034324|0.638768799|90|

dynoNet artifacts use additional target-domain training; this is a separate information group. CED dynoNet's scaler uses both training trajectories. The source ignores measured initial y and simulates from its native internal initialization; retain that behavior. The three released artifact training/selection histories were not reconstructed. Thus the table is not an equal-pretraining or equal-training-budget method ranking.

Per-context means/stds and every point forecast remain in hashed server G archives. RESULTS69_V1.json also provides early/late halves and first/last100 errors. Estimator and mixture-moment3SD fractions are descriptive: the former assumes independent expert errors and the latter is a marginal moment approximation, neither a validated trajectory probability law. No calibration parameter is fitted to test.

