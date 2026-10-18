import sys,pathlib,statistics,collections
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common68_v1 import *
assert read(OUT/'INDEPENDENT_REPLAY68_V1.json')['status']=='PASS'
v=read(OUT/'VALIDATION_RESULTS68_V1.json');e=read(OUT/'EVALUATION_RESULTS68_V1.json');sel=read(OUT/'VALIDATION_SELECTION68_V1.json');ld=read(OUT/'FORECAST_LEDGER68_V1.json');li=sel['selected']['latent']['method'];vg={r['method']:r for r in v['aggregate']};eg={r['method']:r for r in e['aggregate']}
contrasts=[]
for a,b in [(li,'bayes_indexed'),(li,'bayes_same_view'),('meta_id_author_ensemble','bayes_same_view'),('raw_arx_d3_a'+format(sel['selected']['raw']['3'],'g'),'meta_id_author_ensemble')]:
 aa={r['case']:r for r in eg[a]['per_case']};bb={r['case']:r for r in eg[b]['per_case']};delta=[aa[key]['normalized_MAE']-bb[key]['normalized_MAE'] for key in sorted(aa)];mean=float(np.mean(delta));contrasts.append(dict(a=a,b=b,mean_delta=mean,relative_percent=100*mean/eg[b]['normalized_MAE'],better_systems=sum(x<0 for x in delta),systems=len(delta),per_case_delta=dict(zip(sorted(aa),delta))))
obj=dict(utc=utc(),status='CONTROLLED_OPERATOR_COMPARISON_AND_REPLAY_CLOSED',case_count=len(ld['cases']),forecast_rows=ld['score_rows'],neural_fits=0,native_parameters=ld['native_parameters'],meta_parameters=ld['meta_parameters'],selected=sel['selected'],contrasts=contrasts,validation={r['method']:r['normalized_MAE'] for r in v['aggregate'] if r['status']=='FINITE'},evaluation={r['method']:r['normalized_MAE'] for r in e['aggregate'] if r['status']=='FINITE'},accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',interpretation_boundaries=['Two controlled transfer families, not physiological paper-table reproduction or a universal world-model failure theorem.','Linear/PCA operator failure does not identify representation information loss.','Cubic dictionary includes the true generator form; it is a solvability control.','Meta-ID pretraining and model size differ, so no equal-pretraining efficiency claim.','Known future exogenous commands are legitimate conditions; no real-world clinical causal claim.'])
dump(OUT/'ANALYSIS68_V1.json',obj)
def table(groups):
 lines=['| Method | normalized MAE | linear | cubic | MAE | RMSE |','|---|---:|---:|---:|---:|---:|']
 for r in groups:
  if r['status']!='FINITE':lines.append('|'+r['method']+'|DISQUALIFIED|||||');continue
  lines.append('|'+r['method']+'|'+ '|'.join(f'{x:.8g}' for x in [r['normalized_MAE'],r['per_family']['linear']['normalized_MAE'],r['per_family']['cubic']['normalized_MAE'],r['MAE'],r['RMSE']])+'|')
 return '\n'.join(lines)
def write(name,text):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text+'\n')
write('README.md',f'''# C68 controlled contextual-operator audit

The complete pretrained NormWear-2 and an independently pretrained author system-identification Transformer executed on24 independently seeded controlled systems. {ld['score_rows']} point forecasts and their raw arrays were retained; independent replay passed. No neural model was trained this cycle. Nine ordinary latent-operator settings and six raw-identification settings are tested on VAL; selected settings are frozen before EVAL. This is a new controlled transfer/mechanism comparison, not exact paper-table reproduction or a new method.

Read experimental_results.md for every fixed result. Raw data, weights and forecast arrays stay on server G; only authored source, compact results and reports are published. Accepted research directions0; overall goal ACTIVE / UNACHIEVED. A useful ordinary operator repair would constrain a novelty story, and failure of a linear/PCA repair would not prove information loss.
''')
write('literature_review.md',(ROOT/'LITERATURE_DELTA68_V1.md').read_text(encoding='utf-8'))
write('baseline_registry.md',f'''# Executed model registry

| Model/control | Execution | Information and boundary |
|---|---|---|
| NormWear-2 full pretrained | {ld['native_parameters']} parameters, one frozen checkpoint | Physiological pretraining; this is controlled-system transfer. History-only predict is separated from known-plan modes. |
| Native empirical transitions | Indexed and same-view action controls;3 MC seeds | Fixed10 clusters,4 fused sampled trajectories; seeds are not neural fits. |
| PCA/ridge latent operator |9 VAL configurations; selected {li} on EVAL | Same exact encoder/decoder; context-only PCA and regression. Linear model/truncation are limitations. |
| Author meta-identification Transformer | {ld['meta_parameters']} parameters, strict state load; one frozen checkpoint |8 disjoint400-point contexts,30 observed query initial pairs,100 known inputs; no future output embedding. Different pretraining. |
| Raw ARX and cubic ARX | Per-context fits;3 alpha settings each on VAL | Coefficients learned from history. Cubic dictionary includes the true functional form; structural solvability control. |
| Context mean / last output | Direct execution | Same observed history; no future inputs. |

Source revisions, bytes and SHA256 hashes are in MODEL_FILES68_V1.json and inherited C66 source receipt. Exact coefficients/bases and forecast arrays are server-only. Third-party repo has no LICENSE file in the pinned tree; source/weights are not redistributed.
''')
write('experimental_results.md',f'''# Frozen controlled-system results

24 systems:2 families x6 VAL and6 independent EVAL systems. L3200/H100, aligned scalar output/control channels, deterministic exogenous controls. Unitless signals. Primary endpoint is MAE divided by observed-context sample SD, MC-averaged within system, then equal mean over systems/families. Raw RMSE is mean per-case RMSE, not pooled RMSE. No causal clinical interpretation, uncertainty calibration claim or significance claim.

## Validation

{table(v['aggregate'])}

VAL-only selection: `{json.dumps(sel['selected'],sort_keys=True)}`. Receipt SHA256 `{sha(OUT/'VALIDATION_SELECTION68_V1.json')}`; written before EVAL scoring. No EVAL hyperparameter changes.

## Independent evaluation

{table(e['aggregate'])}

## Paired descriptive contrasts

'''+ '\n'.join(f"- {r['a']} minus {r['b']}: normalized-MAE delta {r['mean_delta']:.8g}, relative {r['relative_percent']:.4g}%; {r['better_systems']}/{r['systems']} systems lower." for r in contrasts)+'''

All per-system values and horizon halves are retained in result JSON. Native-context and selected-PCA reconstruction are descriptive diagnostics in FORECAST_LEDGER68_V1.json; low reconstruction error is not proof of predictive state sufficiency. Independent replay separately regenerates source equations, scores every retained row, refits ordinary controls, checks PCA with full SVD, redecodes operators and executes fixed fresh native calls.
''')
write('negative_evidence.md',f'''# Rejected shortcuts and retained failures

No research direction is accepted from this study. Raw polynomial success cannot prove arbitrary nonlinear dynamics already solved. Cross-domain native failure cannot prove a field-wide capability gap. An ordinary controlled operator, context identification and frozen neural adaptation all have prior art. Linear/PCA failure cannot uniquely blame representation information loss.

All launched failures remain in outputs: stage v1 import-path failure under isolated Python; stage v2 external connection refusal; first RAM checkpoint stream failed source/server SHA equality and was never executed. The failed payload is retained on G, and checked repair provenance records source-chunk comparison plus final full-file equality. Cause of byte disagreement is unknown. campaign68_v1.py was never executed; code review corrected duplicate MC_seed keyword construction in v2 before any formal scores. None of these events is scientific negative evidence against a model.

Selected latent operator {li}. Its EVAL normalized MAE is {eg[li]['normalized_MAE']:.8g}, compared with indexed native transition {eg['bayes_indexed']['normalized_MAE']:.8g}. These actual measurements constrain which explanation to pursue; they do not turn a repair into a new framework.
''')
write('hypothesis_space.md','''# Competing explanations and falsification

1. A simple fitted controlled operator explains a large native-cluster forecasting residual. Test the same frozen encoder/decoder with a VAL-selected PCA/ridge operator. Improvement supports this route only on the tested families; no improvement leaves operator nonlinearity, PCA, contextual states and decoder support unresolved.
2. Forecasting the systems is intrinsically difficult with available information. Raw learned ARX/polynomial controls and independently pretrained Meta-ID test this. A true-form dictionary is a structural solvability control, not a fair universal-model winner.
3. The pretrained representation discarded the needed state. Reconstruction/PCA diagnostics are insufficient to establish this. A claim requires a task-valid sufficiency/conditional-future test against competent nonlinear operators, without extra information for the candidate.
4. Context-fitted dynamics itself is a new contribution. Rejected by existing DMDc/SINDYc and author pretrained in-context system identification.

Do not demand physical latent recovery or causal inference from models whose task is conditional observable forecasting. Do not add test-set-specific operator degree, PCA rank, spectrum clipping or control normalization.
''')
write('research_ideas.md','''# Candidate research questions

No final direction accepted. The evidence establishes a reproducible comparison between neural priors and ordinary contextual identification; neither the familiar operator change nor its failure supplies a distinct novel mechanism.

A provisional diagnostic remains: under what changes of observation, input excitation and system regime does context uncertainty prevent reusable simulation after competent existing identification? This is not yet a recommended idea: Bayesian system identification, probabilistic meta-identification, Meta-iLaD, OSSM and RP-GSSM are close neighbors. Any follow-up must first show a specific remaining capability gap, then test a minimally distinct intervention on independently held-out mechanisms. Do not keep tuning the two C68 recurrences.
''')
write('final_research_direction.md',f'''# No accepted final direction

**Research Question:** Which reusable time-series simulation capability remains unresolved after adequate contextual identification and representation controls? This study has not yet identified a novel answer.

**Evidence:** Two full frozen pretrained models executed on24 known-law systems; {ld['score_rows']} forecasts and independent replay. See experimental_results.md and ANALYSIS68_V1.json for measured operator effects rather than an inferred state-loss story.

**Existing Methods:** Controlled regression, polynomial identification, probabilistic in-context simulation and observation-correction models already exist. The additional author pretrained checkpoint is a meaningful strong neighbor, not a newly proposed architecture.

**Proposed Idea:** None accepted. No universal model-failure claim follows from this transfer protocol.

**Falsification:** An ordinary context operator or existing meta identifier removing the residual defeats a necessity claim for representation redesign on this route. Failure of a single linear/PCA operator leaves multiple competing explanations. Reject any gain from extra labels, leaked future outputs or EVAL tuning.

**Feasibility:** One RTX5090 executes both exact models and all per-context controls; no neural retraining. Server G retains datasets, weights, operators and forecasts. Authored code/results only go to GitHub, without local result synchronization.

**Next Experiment:** Choose a distinct, independently defined context-identification or observation regime only after checking current probabilistic system-identification code and historical negative evidence. A useful question must survive mature nonlinear/state-space controls and offer a distinct mechanism. Do not optimize the C68 recurrences or rename PCA/ridge as a new method. This follow-up has not executed.

Overall goal ACTIVE / UNACHIEVED; accepted final directions0.
''')
status=f'''# Current research status: C68

C68 controlled operator comparison and independent replay are closed. {ld['score_rows']} forecasts on24 systems, two exact frozen pretrained models, zero neural fits. Primary selected-operator EVAL normalized MAE {eg[li]['normalized_MAE']:.8g}; indexed native {eg['bayes_indexed']['normalized_MAE']:.8g}; author Meta-ID {eg['meta_id_author_ensemble']['normalized_MAE']:.8g}. Details and interpretation limits: cycle68/experimental_results.md.

No distinct new framework accepted. Goal ACTIVE / UNACHIEVED. Raw arrays/weights stay server G; no local result synchronization. Preserve prior cycles and failures.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V68.md').open('x',encoding='utf-8') as f:f.write(status)
print(json.dumps(obj),flush=True)
