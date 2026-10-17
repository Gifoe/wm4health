"""Archive the running snapshot and write complete C53 evidence reports in place."""
from native53 import *
import csv
import shutil

EVAL=ART/'evaluation_v1';A=ART/'analysis_v1'
verified=json.loads((EVAL/'verification.json').read_text())
assert verified['status']=='PASS_RETAINED_FORECAST_ARITHMETIC_AND_NATIVE_SCORE_DENOMINATORS'
analysis=json.loads((A/'summary.json').read_text())
assert analysis['status']=='DESCRIPTIVE_ANALYSIS_COMPLETE_NO_RESEARCH_DIRECTION_ACCEPTED'
mechanisms=json.loads((ART/'mechanism_summary_v1/summary.json').read_text())
timing=json.loads((ART/'data_diagnostics_v2/summary.json').read_text())
plots=json.loads((R/'outputs/plot_repair_v3/qualification.json').read_text())
assert plots['status']=='PASS_PNG_SVG_PLOTS_WITH_ALL_SCORED_ARTIFACTS_UNCHANGED'
assert timing['all_accepted_windows_within_one_second_per_step']
assert verified['summary_sha256']==sha(EVAL/'summary.json')
assert len(json.loads((R/'outputs/SELECTION_SEAL_V6.json').read_text())['selected_checkpoints'])==40
OUT=R/'outputs/final_reports_v1';assert not OUT.exists();OUT.mkdir()
old_manifest=json.loads((R/'outputs/progress_snapshot_v1/report_manifest.json').read_text())
archived=[]
for item in old_manifest['files']:
    source=R/item['path'];assert sha(source)==item['sha256']
    target=R/'outputs/progress_snapshot_v1/reports'/item['path'];assert not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha(target)==item['sha256']
    archived.append({'original_path':item['path'],'archive_path':target.relative_to(R).as_posix(),'sha256':item['sha256']})
dump(OUT/'running_snapshot_archive.json',{'utc':now(),'files':archived,'original_manifest_sha256':sha(R/'outputs/progress_snapshot_v1/report_manifest.json')})

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)
aggregate=analysis['aggregate_metrics'];by={(r['bank'],r['model'],r['arm']):r for r in aggregate}
factual=[]
for r in aggregate:
    sd='' if r['n_seeds']==1 else f" +/- {r['MAE_std']:.5f}"
    factual.append([r['bank'],r['model'],r['arm'],f"{r['MAE_mean']:.6f}"+sd,f"{r['MSE_mean']:.6f}",r['n_seeds']])
main_table=table(['Bank','Backbone','Configuration','MAE mean and training-seed SD','MSE mean','Seeds'],factual)
contrasts=json.loads((A/'paired_contrasts.json').read_text())
contrast_rows=[]
for r in contrasts:
    if r['calendar_block_days']!=7 or r['baseline'] not in ['obs','ae_concat','ae_gate']:continue
    low,high=r['descriptive_percent_reduction_interval']
    contrast_rows.append([r['bank'],r['model'],r['baseline']+' -> '+r['candidate'],f"{r['percent_MAE_reduction']:.3f}",f'[{low:.3f}, {high:.3f}]'])
contrast_table=table(['Bank','Backbone','Paired contrast','MAE reduction (%)','Descriptive seven-day interval (%)'],contrast_rows)
tradeoffs=[]
for bank in ['source_val','new_crop_all']:
    persistence=by[(bank,'ordinary_control','persistence')]
    for model in ['CrossLinear','PatchTST']:
        gate=by[(bank,model,'ae_gate')]
        tradeoffs.append([bank,model,f"{100*(gate['MAE_mean']/persistence['MAE_mean']-1):.3f}",f"{100*(gate['MSE_mean']/persistence['MSE_mean']-1):.3f}"])
tradeoff_table=table(['Bank','Gate backbone','MAE change vs persistence (%)','MSE change vs persistence (%)'],tradeoffs)
with (ART/'mechanism_summary_v1/seed_summary_by_prior_role.csv').open(newline='',encoding='utf-8') as stream:mr=list(csv.DictReader(stream))
mech_rows=[]
for r in mr:
    if r['arm'] not in ['ae_gate','ae_gate_soft'] or r['metric']!='sign_agree_q0.1':continue
    mean='unavailable' if r['seed_mean']=='' else f"{float(r['seed_mean']):.5f}"
    mech_rows.append([r['bank'],r['model'],r['arm'],r['prior'].replace('greenhouse:',''),r['prior_role'],mean,r['defined_seeds'],f"{float(r['coverage_min']):.4f}-{float(r['coverage_max']):.4f}"])
mech_table=table(['Bank','Backbone','Configuration','Declared relation','Prior role','Mean q+0.10 agreement','Defined seeds','Coverage range'],mech_rows)
with (ART/'data_diagnostics_v1/native_scaling_coverage.csv').open(newline='',encoding='utf-8') as stream:dr=list(csv.DictReader(stream))
clip_rows=[]
for r in dr:
    if r['part']=='future' and r['bank']!='source_train' and (float(r['finite_below_lo_fraction'] or 0)>.01 or float(r['finite_above_hi_fraction'] or 0)>.01):
        clip_rows.append([r['bank'],r['role'],r['channel'],f"{100*float(r['finite_below_lo_fraction']):.3f}",f"{100*float(r['finite_above_hi_fraction']):.3f}"])
clip_table=table(['Bank','Role','Channel','Below fixed range (%)','Above fixed range (%)'],clip_rows)
ridge=json.loads((EVAL/'ridge_selection.json').read_text())
lead=f'''C53 completed UTC {now()}. This is a native-source subset reproduction and a separately identified new-crop transfer audit: two backbones, four configurations, five seeds, 40 predictive fits and one shared AE. Selected predictive fits contain {analysis['formal_updates']} optimizer updates and {analysis['native_fit_seconds']:.2f} native fit seconds; {analysis['hit_epoch_cap']}/40 predictive fits hit the 100-epoch cap. Codec training, qualification, failed attempts and plotting are additional cost. The native-driver failed attempt alone wasted at least 162 updates; unrecorded partial work cannot be counted exactly.

Source TRAIN/VAL has 815/163 accepted windows; the new crop has 803 windows from repeated compartment/team identities. All 40 checkpoints were fixed before transfer scoring. Five training seeds measure optimization variation, not five independent plants. Factual future action/exogenous schedules are supplied. The original paper's raw run-byte identity is unavailable, and PatchTST AE concat uses the disclosed effective-batch128/microbatch64 adaptation, so bitwise headline reproduction is not claimed.

Independent verification covered 84 retained forecast groups, 40,572 model-window predictions, 2,596,608 scalar predictions, all 40 loaded-forward/future-target-corruption checks and 80 native mechanism cells. Retained-array score arithmetic error was zero. Native mechanism functions are retained source; there is no independent intervention ground truth. Overall research goal remains ACTIVE_UNACHIEVED; accepted final directions: 0.
'''
interpretation='''The source recipe's output Gate advantage over AE concat persists on both backbones and both crops, with positive paired mean gains in all five seeds for these contrasts. AE concat alone has substantial seed dependence. Native directional supervision raises the penalized temperature-response signs but produces a CrossLinear accuracy cost; its PatchTST factual effect is small and varies by seed.

Persistence has lower aggregate MAE than every learned configuration on both crops. PatchTST Gate has lower MSE than persistence on both crops, while CrossLinear Gate does not. All predictive models were trained/selected using the native MSE protocol. The criterion-dependent comparison prevents treating the MAE ranking as evidence of general dynamical-learning failure. No loss-swap or persistence-blend is proposed as a novel world-model method.

The held-out CO2 sign remains mostly wrong, while lighting q+0.10 agreement is perfect on a restricted subset (only 2/163 source windows can be counted for that upward lighting shift). Penalized and held-out relations, positive/negative shifts and their coverage must remain separate. These are model responses against declared sign priors, without observed counterfactual effect truth. The native code and paper already distinguish supplied and held-out signs; this outcome alone supplies no new research gap.

The new crop also clips some raw exogenous values at native fixed ranges, including outdoor temperature, irradiance and outdoor PAR, and 4.024% of future indoor PAR target entries exceed its native upper range. This is a protocol/shift confound to retain in attribution, not a demonstrated cause of all errors. Physical-unit errors obtained by undoing affine scaling refer to clipped native targets. Timestamp steps are 299.980-300.067 seconds; all accepted steps are within one second of the nominal five minutes, so strict equality's timing flags are immaterial subsecond variation.
'''
sources='''Pinned implementation: [TSWM source](https://github.com/HaochenZhang717/TSWM-Official-Repo/tree/2e5e48540ff0da48e8192163969a6c58df1d5a3e), [paper](https://arxiv.org/abs/2610.01842), [original data](https://doi.org/10.4121/21960932.v1). All 125 original Python source hashes were checked. C52 records acquisition and interpolation-contract audits; required_data_manifest_v1.json specifies the 26 public XLSX loader inputs. Twelve existing caches additionally matched native reconstruction from those raw files exactly.
'''
evidence='''Server evidence paths are logical, relative to the configured artifact root: `cycle53/native_v1/evaluation_v1/`, `analysis_v1/`, `mechanism_summary_v1/`, `data_diagnostics_v1/` and `data_diagnostics_v2/`. Compact public copies and explicit path-redaction provenance are prepared for GitHub; raw signals, prediction tensors, checkpoints and environment files remain on the server. Every numerical table here derives from the retained verified files.
'''
reports={
'literature_review.md':'# C53 literature and innovation boundary\n\n'+sources+'\nThe original work already supplies frozen latent prediction, output fusion, temporal plan encoders and directional supervision. Its Appendix G tests a pre-fusion auxiliary objective and reports no stable rollout-specific benefit. C53 reproduces a subset and adds a crop audit; these existing mechanisms are not a new contribution. The native default split supplies three Greenhouse temperature signs and withholds the lighting/CO2 signs. See PRIOR_ART_BOUNDARY_V1.md for exact source links.\n\n'+interpretation+'\nThe next evidence step must include a competent stochastic latent-dynamics model, rather than extend a claim from these deterministic forecasters to all world models. RP-GSSM is a listed core contender whose full native experiment needs direct execution before further state-learning claims; citing its architecture alone is insufficient.\n',
'baseline_registry.md':'# C53 complete baseline registry\n\n'+lead+'\n'+sources+'\nThe four native configurations are observation concat, frozen-target-AE concat, frozen-target-AE Gate and Gate with the original directional objective. Native hyperparameters, data roles, group split, random seeds and selection rule remain fixed in PROTOCOL.md. The paper-native configurations have different parameter counts and observed runtimes; this is not a new-method capacity/compute-matched study.\n\nPersistence and direct ridge are original diagnostic controls with the declared inputs. Ridge alpha is '+str(ridge['selected_alpha'])+' selected exclusively by first-crop VAL equal-batch MSE from the frozen five-alpha grid. Controls have one fit, not five independent seeds. They were evaluated factually; no intervention-effect truth was added.\n\n'+main_table+'\nComplete training diagnostics, every source selection curve, all channel/horizon metrics and all compartment metrics are retained. The fresh-directory entry point passed staging/AST/raw-cache checks; its full 40-fit pipeline has not been rerun (FRESH_RUN.md).\n',
'experimental_results.md':'# C53 complete experimental results\n\n'+lead+'\n'+main_table+'\n'+tradeoff_table+'\nPositive changes in the preceding table mean larger error. The fixed recipe trains on MSE; aggregate MAE and MSE answer different questions.\n\n'+contrast_table+'\nPositive paired reductions favor the candidate. Intervals use five-seed-averaged paired window errors and synchronized seven-day blocks: seven occupied source bins and six new-crop bins, 2,000 draws. They are descriptive conditional on these crops/compartments. Three-day sensitivity and all ordinary-control contrasts are retained in paired_contrasts.json. There is no independent-domain inference.\n\n'+mech_table+'\nThis table displays q+0.10 for readability. Every prior, seed, positive/negative shift, original score total/count and unavailable entry is retained in the native JSON and companion CSVs. Lighting source upward coverage is 2/163; high agreement there is a narrow check.\n\n'+clip_table+'\n'+interpretation+'\n'+evidence,
'negative_evidence.md':'# C53 negative and limiting evidence\n\n'+interpretation+'\nA new architecture is not justified by a ranking that depends on MAE versus MSE, by a known penalized-versus-held-out sign gap, or by fixed-range clipping alone. No test-set hyperparameters, new seed selection, transfer fine-tuning or new model were used.\n\nNative runtime failures are separate engineering evidence: an incomplete memory-saturated concat attempt, an instrumentation-wrapper failure, the seed-3 CUDA-driver-module access violation and several native plotting crashes were retained. Valid selected checkpoints were never replaced by failed attempts. The retry matched all 42 initial tensors/config and the first 27 epoch objectives/RNG hashes exactly. Plot export eventually used an isolated task runtime, fixed geometry and PNG/SVG; all 84 raw forecast groups and 11 metric/input artifacts were hash-checked unchanged. Native fault modules identify where failures surfaced, not their root causes.\n\nC49\'s current conditional score recipe remains deprioritized; C52\'s constructed interpolation risk was inactive in 1,781 accepted default real windows. These earlier negatives are not revived by C53.\n',
'hypothesis_space.md':'# C53 hypothesis update\n\n'+lead+'\nH1 (native configuration advantages persist): output Gate is supported descriptively on both crops and both backbones; AE concat alone is seed dependent. This is concordance with an existing recipe.\n\nH2 (ordinary forecasting/objective choices explain an apparent failure): supported as a necessary attribution control. Persistence wins MAE, while PatchTST Gate wins MSE. This does not settle every mechanism question. Native clipping and seasonal difficulty also remain possible contributors to raw cross-crop changes.\n\nH3 (a residual novel world-model limitation is established): not supported by this experiment. A single repeated-compartment crop pair and sign-prior diagnostics do not establish an independently identified general failure or a new method.\n\nCompeting explanations for the held-out CO2 response are policy feedback/insufficient identification, omitted state and predictive-model approximation/optimization. The current observational sign check cannot distinguish them. A follow-up must specify information/identifiability assumptions and include ordinary system-identification and competent latent-dynamics controls before designing a remedy.\n',
'research_ideas.md':'# C53 research disposition\n\n'+lead+'\n'+interpretation+'\nNo new direction is recommended from C53 alone. The useful result is competent source-protocol evidence and the rejection of an easy generic-collapse narrative. Extending the native fusion/auxiliary/sign recipe, changing the loss to optimize MAE or correcting a clipping definition would not meet the requested research objective by itself.\n\nNext, inspect and execute a mechanism-different strong latent-dynamics contender (starting from the actual RP-GSSM implementation) and use it to test the still-unresolved state-learning claims from earlier cycles. Select an official experiment only after checking its task, inputs, latent truth, metrics and cost. New model effects require an independently frozen protocol and confirmation evidence. The core target remains a reusable time-series world model.\n',
'final_research_direction.md':'# Current disposition after C53\n\nOverall objective ACTIVE_UNACHIEVED. Accepted final research directions: 0.\n\nResearch Question: no new question has yet passed novelty, failure-attribution and minimal-method validation. C53 tests an existing recipe and crop transfer.\n\nEvidence: all 40 prescribed predictive fits, 15,600 updates, 84 verified forecast groups and 80 native sign-prior cells are complete. Gate transfers; MAE/MSE rankings differ; supplied signs improve while the held-out CO2 sign remains poor. These facts are insufficient for a new framework claim.\n\nExisting Methods: the native recipe and its supervised/held-out distinction already cover the central mechanism findings here. Ordinary persistence and ridge constrain interpretation. A broader conclusion still needs direct execution of competent stochastic latent-state methods.\n\nProposed Idea: none accepted; no new architecture or loss is introduced in C53.\n\nFalsification: abandon any candidate whose apparent advantage is explained by metric choice, normalization, extra information, ordinary complete controls or non-identifiability. A known sign discrepancy without intervention truth cannot certify a causal remedy.\n\nFeasibility: the RTX5090 completed this native subset; data/checkpoints/raw predictions are on the authorized large disk, and portable sources/reports are exported to wm4health. The clean entry point has staging-level validation with explicit limits.\n\nNext Experiment: audit the official RP-GSSM experiment and execute a representative native training/latent-dynamics result before making another state-representation claim. This is a targeted falsification baseline, not a claimed new method or a substitute for the ultimate independent real-domain MVP.\n',
'CURRENT_STATE.md':'# C53 complete evidence state\n\n'+lead+'\n'+evidence+'\nActive numerical stages are terminal: capture_native53_v7.py selected all 40 models; score_native53_v8.py and verify_native53.py passed; analyze_native53.py completed; mechanism-role and scaling/timing summaries completed. The old analysis supervisor records a plot crash. Plot recovery completed separately through render_native53_v4.py / repair_plots53_v3.py with five PNG/SVG figures and unchanged score hashes. Historical RUNNING/failure records must not be used as current process state.\n\nThe original running report bytes are archived under outputs/progress_snapshot_v1/reports and bound by final_reports_v1/running_snapshot_archive.json. Original server data and failed attempts were preserved. No results are synchronized to the local workstation. Public target: Gifoe/wm4health. Overall research remains active.\n'
}
for name,text in reports.items():(R/name).write_text(text,encoding='utf-8')
(R/'README.md').write_text('''# C53 native TSWM subset and new-crop audit

Read CURRENT_STATE.md and experimental_results.md for the completed 40-fit comparison. This is source-protocol subset reproduction plus a separately labeled crop transfer audit; no final novel research direction is accepted.

PROTOCOL.md, EVALUATION_SUPPLEMENT.md and ANALYSIS_SUPPLEMENT.md fix model fitting, selection and descriptive uncertainty. Checkpoints were sealed before the new crop was scored. Independent raw-array arithmetic and forward/target-corruption checks passed. Full metrics, sign-prior roles/coverage, clipping/timing diagnostics and all fit curves are preserved.

Runtime supplements disclose the concat microbatch adaptation and native-process recovery. Final plots use render_native53_v4.py, fixed layout and PNG/SVG after retained native layout/PDF failures; no models or score arrays were changed during plotting. FRESH_RUN.md describes the clean-directory entry point and its staging-level validation limit.

Public sources and compact evidence are on Gifoe/wm4health. Raw inputs, tensors, checkpoints and environment files remain on the authorized server disk. Public path aliases and their original/export hashes are recorded explicitly; path-redacted files are not claimed to have the original server byte hash. Original dated snapshots remain available in Git history and the archived report bytes.
''',encoding='utf-8')
dump(OUT/'manifest.json',{'utc':now(),'status':'C53_COMPLETE_EVIDENCE_REPORTS_RESEARCH_GOAL_ACTIVE',
    'accepted_final_directions':0,'verification_sha256':sha(EVAL/'verification.json'),
    'analysis_summary_sha256':sha(A/'summary.json'),'mechanism_summary_sha256':sha(ART/'mechanism_summary_v1/summary.json'),
    'timing_precision_sha256':sha(ART/'data_diagnostics_v2/summary.json'),'plot_qualification_sha256':sha(R/'outputs/plot_repair_v3/qualification.json'),
    'files':[{'path':name,'sha256':sha(R/name)} for name in [*reports,'README.md']]})
latest=P/'CURRENT_RESEARCH_STATUS_V53_FINAL.md';assert not latest.exists();latest.write_text(reports['CURRENT_STATE.md'],encoding='utf-8')
print(json.dumps({'status':'C53_COMPLETE_EVIDENCE_REPORTS_RESEARCH_GOAL_ACTIVE','reports':len(reports),'accepted_final_directions':0}),flush=True)
