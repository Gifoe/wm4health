"""Generate private evidence ledgers without upgrading artifacts into novelty."""
import runtime_profile
import csv,datetime,json,pathlib,platform,sys
import numpy as np,sklearn,torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def write(p,text):
 path=ROOT/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text.strip()+'\n',encoding='utf-8')
PAPERS=[
 ('causal_chambers','Causal Chambers','method + benchmark','Nature Machine Intelligence2025; archived arXiv2404.11341v2',4,4,4,'Real physical actuators, sensors and validated graphs support protocol-defined interventions. Recorded sensors are not automatically physical latent ground truth.','https://www.nature.com/articles/s42256-024-00964-x'),
 ('tawm','Time-Aware World Model','pure method','arXiv2506.08441v1,2025',3,3,3,'Chosen time-step conditioning and mixed-rate training are existing methods. They do not make a realized unknown future measurement time an available deployment input.','https://arxiv.org/html/2506.08441v1'),
 ('endoclock','EndoClock','system/tool','arXiv2608.09266v1; paper reports MICCAI Medical World Models workshop2026 acceptance',4,3,None,'Audits where task evidence survives synchronization and endogenous observation clocks. Generic clock-preservation auditing is already covered. It is not a learned controlled-simulator guarantee.','https://arxiv.org/html/2608.09266v1'),
 ('hypernsde','HyperNSDE','pure method','arXiv2610.07383v1,2026October',4,4,3,'Jointly models latent-state-dependent observation intensity, with matched-grid analyses and an open implementation. Generic joint time/value generation is not an untouched idea.','https://arxiv.org/html/2610.07383v1'),
 ('sensing_reliability','Stage-wise sensing reliability','method + benchmark','arXiv2609.07126v1; workshop-style manuscript2026',3,3,3,'Separates sensing disturbance propagation through representation, prediction, planner preference and outcome. Internal perturbation size does not certify physical outcome; no broad real-sensor timing guarantee is claimed.','https://arxiv.org/html/2609.07126v1'),
 ('cider_fm','CIDER-FM','pure method','arXiv2609.39523v1,2026September',4,4,4,'Pools observational and surrogate experimental regimes with interventional metadata; real light-chamber experiments are included. Its inspected architecture treats samples as sets, not the controlled time-series task executed here.','https://arxiv.org/html/2609.39523v1'),
 ('tesar_cde','TESAR-CDE','pure method','ICML2023; archival fallback arXiv2306.04255v1',4,4,4,'Inverse-intensity weighting for informative treatment-outcome sampling is existing work. Generic informative-sampling bias cannot be claimed as a new world-model problem.','https://proceedings.mlr.press/v202/vanderschueren23a.html'),
 ('te_cde','TE-CDE','pure method','ICML2022; archival fallback arXiv2206.08311v1',4,4,4,'Continuous-time treatment-conditioned counterfactual prediction already uses neural controlled differential equations; its limitations do not make a generic irregular-time extension novel.','https://proceedings.mlr.press/v162/seedat22b.html'),
 ('cf_jepa','CF-JEPA','pure method','arXiv2610.00727v1,2026October',3,3,3,'Uses inverse and adversarial dynamics to separate control-related and uncontrollable latents for planning. Relevant to a fresh nonlinear-reachability question, but not executed and not shown insufficient.','https://arxiv.org/html/2610.00727v1'),
 ('ep_idm','Preserving Unstable Modes through Inverse Dynamics','pure method','arXiv2610.07540v1,2026October',4,4,4,'Endpoint inverse reconstruction has linear-system reachable-subspace guarantees and nonlinear visual experiments. Nonlinear guarantees are not asserted; no theorem is refuted by this cycle.','https://arxiv.org/html/2610.07540v1')]
def main():
 inventory=read('outputs/data_audit/inventory.json');ridge=read('outputs/clock_rpm_controls/results.json');strong=read('outputs/strong_transition_controls/results.json');quant=read('outputs/timer_quantization/result.json');summary=read('outputs/real_controls_summary.json')['rows'];pin=read('sources/chamber/SOURCE_PIN.json');lit=read('outputs/literature/registry.json');history=read('outputs/fitted_training_history.json')
 assert len(inventory['rows'])==38 and sum(r['rows'] for r in inventory['rows'])==965322 and len(inventory['regimes'])==64
 assert len(ridge['rows'])==18 and len(strong['rows'])==18 and len(quant['rows'])==128 and len(history['rows'])==5
 assert len(lit['papers'])==10 and not lit['errors'] and len(pin['files'])==42
 assert all(r['nonfinite_numeric_values']==0 and r['nonpositive_timestamp_differences']==0 for r in inventory['rows'])
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();lookup={(r['group'],r['arm']):r for r in summary}
 table='| Group | Arm | Recordings | One-step MSE /3000² |16-step MSE /3000² |\n|---|---|---:|---:|---:|\n'
 for group in ['held-out walks','fast waves','slow waves','regime_jumps_single','regime_jumps_multi']:
  for arm in ['persistence','ridge_no_clock','ridge_oracle_future_dt','ridge_predicted_dt','HGB_no_clock','HGB_oracle_future_dt','HGB_predicted_dt']:
   r=lookup[(group,arm)];table+=f"| {group} | {arm} | {r['recordings']} | {r['one_step_normalized_MSE']:.7g} | {r['rollout16_normalized_MSE']:.7g} |\n"
 qtable='| Recording | Low-resolution blocks /32 | Fraction of evaluated rows | Fraction of HGB squared error |\n|---|---:|---:|---:|\n'
 for r in quant['error_contribution']:qtable+=f"| {r['recording']} | {r['low_resolution_blocks']} | {r['low_resolution_row_fraction']:.4%} | {r['low_resolution_share_of_HGB_no_clock_squared_error']:.4%} |\n"
 grids=[r for r in quant['rows'] if r['recording']=='regime_jumps_single' and r['resolution_flag']==0]
 gtable='| Regime | Sensor | Effective period grid with >99.5% coverage | Unique RPM values | Top8 atom mass |\n|---|---|---:|---:|---:|\n'
 for r in grids:gtable+=f"| {r['regime']} | {r['side']} | {r['largest_tested_grid_with_99_5_percent_coverage']} ms | {r['unique_RPM']} | {r['top8_atom_fraction']:.3%} |\n"
 scope='''This cycle executes **new real-data control experiments**, not original WorldTS, TAWM, RP-GSSM or JEPA reproductions. All numerical fits and audits run on the remote research server; ordinary CPU ridge/gradient boosting is deliberately used before expensive neural training. Native source/configuration is pinned for the data-generating protocols and inspected firmware, not claimed to be the exact flashed board version.

Train: nine actuator-walk recordings2–10. Validation:11–12. Held-out test:13–16. Truncated recording1 has1016rows and is excluded by the inventory before any model metric. Ten independently recorded wave runs and two320k regime runs are separate OOD groups and select no hyperparameters. No window crosses a recording boundary. A16-step horizon is an observation-count horizon with variable physical duration, not a fixed-wall-time simulator certificate.

Future programmed actuator/sensor commands are known and allowed by the recorded open-loop protocol. Actual future timestamps and measured sensor values are not deployable inputs. The actual-future-dt arm is a privileged conditional comparator, **not an oracle lower error bound**: finite misspecified predictors can perform worse with more information. Predicted/past-time arms share the same fitted conditional dynamics weights and use only permitted histories/commands. Standard HGB can predict intervals but is not proposed as an innovation.

Ridge uses two RPM lags, polynomial command features and train-only scaling; six ridge penalties are chosen by validation-recording increment error. HGB direct-next-state controls are more flexible, not parameter/compute matched; their default capacity is frozen before the follow-up fit and is not tuned on test/OOD. Original fitted estimator score arrays are exported for five HGB fits; ridge validation-alpha grids are retained. Seven selected estimators are retained across two model bundles; counting twelve ridge candidates and five HGB fits gives17fits, not17independent replications.

All scores are on **measured RPM** in source units, normalized by3000². Neither RPM nor pressure is privileged physical latent ground truth. Equal recording-weighted means are descriptive, without row-wise confidence intervals or cross-environment significance claims. Ridge roots include rare regime-boundary windows; the HGB follow-up excludes crossing roots. Do not interpret that small protocol difference as a model effect.'''
 write('README.md',f'''# Cycle8: real wind-tunnel qualification and controls

Evidence checkpoint; the full discovery goal remains **ACTIVE and unachieved**. No final direction or new method is accepted.

38 recordings /965,322rows /64intervention blocks audited. Completed ordinary dynamics/time controls on18recordings plus128sensor-stream quantization diagnostics. Ten primary PDFs and42pinned source/metadata files retained. Preserve sealed main/cycles2–7.

{scope}

Main decision: coarse-timer sensor blocks dominate the apparent OOD error, while a concrete quantization explanation is supported by data and inspected source. This prevents promotion of the score drop into a new latent-dynamics failure. Clock-conditioning improvements alone also do not establish a new method need; the relevant literature predates this experiment.

The seven required ledgers and diagnostic figure are saved here. Artifact manifest is produced only after completion and source/data/PDF verification. All results remain private.
''')
 write('experimental_results.md',f'''# Completed real-data experiments

## Protocol

{scope}

## Prediction controls

{table}

Full per-recording results, measured-unit endpoint errors and interval-time errors are in outputs/clock_rpm_controls/results.json and outputs/strong_transition_controls/results.json. The ridge-only past-time comparator is also retained there and in real_controls_summary.json. Validation recordings are reported separately and never pooled into the test mean.

The more flexible HGB improves held-out one-step prediction, while ordinary ridge can generalize better on wave rollouts. This is evidence against assuming greater nonlinear capacity is automatically a stronger simulator. It is not a novel comparison or an architecture population claim. Timing is not a uniform explanation of OOD failure.

## Failure localization

{qtable}

Low-resolution blocks produce a disproportionate share of measured-label squared error. Reference-voltage, oversampling and hatch changes should not be lumped into the same physical-dynamics shift. Grouped metrics are retained, not hidden by removing difficult test cases.

{gtable}

Inverted coarse-RPM periods concentrate on increasingly coarse1/4/8ms grids. The inspected source computes `float(now)-float(last_tick)` before the division. Afloat32 numeric example at70,000,000ms turns a true10ms difference into8ms; subtracting unsigned timestamps before conversion preserves10ms. This is a source-consistent explanation, **not proof that this exact firmware binary generated the archive** and not recovery of the true rotor trajectory.

## Data qualification

Both official ZIPs match publisher MD5 and full transport SHA256. All38CSVrecordings have finite numeric data and strictly increasing within-recording timestamps. The recorded targets of all64regimes match the pinned protocol target lists. There are16walk files rather than an assumed ten-seed dataset; one is short. Timestamps have a shifted origin, so no cross-file Unix-time alignment is assumed. These facts qualify permitted analysis, not a causal-identification theorem.

## Execution and uncertainty

Remote base-control runtime {ridge['seconds']:.2f}s; nonlinear-control runtime {strong['seconds']:.2f}s. One deterministic split/fitting protocol; no multiple-initialization robustness claim. Model bundles, original retained per-iteration training scores, validation grids, raw inventory and raw per-recording metrics are retained. Terminal process exits are recorded. Full stdout remains in the tool transcript; no reconstructed text is falsely presented as an original raw stdout file.
''')
 write('baseline_registry.md',f'''# Source, model and execution registry

| Object | Status | Evidence/scope |
|---|---|---|
| Causal Chambers protocols/data metadata | Pinned/read | {pin['revision']};42small source/metadata files |
| wt_walks_v1 | Downloaded/verified/audited |46,471,335compressed bytes;16walks,10wave runs,2regime runs |
| wt_test_v1 | Downloaded/verified/audited |2,610,508compressed bytes;10calibration/characterization recordings |
| Ridge ARX/polynomial | Executed ordinary control | Two fitted information arms,six validation-selected penalties each |
| HGB sample-interval predictor | Executed ordinary control | History/known commands only,150iteration maximum |
| HGB next-RPM predictor | Executed stronger existing control | Two per-sensor fits per information arm,200iteration maximum; not compute matched |
| Measured persistence | Executed | No fitted parameters; same evaluation rows/roots |
| Float32 timer diagnostic | Executed |128regime sensor streams plus calibration; source-consistent interpretation, no physical latent labels |
| TAWM/HyperNSDE/TE-CDE/TESAR-CDE | Literature/protocol only | No paper metric reproduction or modified-native execution |
| CF-JEPA/EP-IDM/Controllable-Koopman | Fresh read-only scouting | Relevant assumptions/code links; no new benchmark execution or insufficiency claim |
| Earlier main/cycles2–7 | Preserved | No completed experiment restarted or relabeled as fresh |

Native dataset software remains MIT and numeric data CC BY4.0, with notices retained. Initial metadata bundle contains41files; firmware text was added afterwards at the same pinned commit, extending SOURCE_PIN to42files. Run stage_metadata once for the original snapshot, then fetch_firmware for the documented extension; do not compare an old bundle's pin against the expanded working pin as if unchanged.

Server direct S3 was refused; public archives were fetched locally then relayed over pinned SSH, rechecked against publisher MD5 and transport SHA256, and numerically analyzed on server. No RemoteLab/hardware command, new public publication or unrelated project edit occurred.
''')
 write('negative_evidence.md','''# Rejected claims and competing explanations

1. **Large OOD measured-RPM error proves failed physical dynamics: not supported.** Low-resolution sensor regimes account for93–98%of HGB squared error while comprising16–31%of rows. Measured target quantization is a concrete competing explanation. Physical latent truth and the flashed binary are not available.
2. **Tachometer acquisition blocks until a rotor pulse: rejected for inspected source.** Interrupts maintain RPM and getters read it atomically. Correlation with sampling intervals does not establish a state-dependent clock mechanism.
3. **Sensor metadata generally changes sampling intervals enough to explain everything: not supported.** Most regime medians remain near0.30s. Actual-future-time conditioning helps some one-step comparisons but has small or mixed rollout effects, and it is not a deployable next-observation input.
4. **Better one-step predictor automatically gives better rollout generalization: rejected locally.** Flexible HGB improves held-out one-step error yet ordinary ridge is better on wave rollouts. This familiar failure alone is not a new research contribution.
5. **The clock/observation-intensity concept is novel: rejected as a broad claim.** EndoClock, HyperNSDE, TAWM, TE-CDE and TESAR-CDE already address substantial related settings. Their precise guarantees differ; no entire area is declared solved because of title overlap.
6. **A Gaussian uncertainty head or new timer-aware network is now justified: not supported.** Quantized-measurement filters, calibrated observation likelihoods and correct timing are existing necessary controls. No new latent-dynamics mechanism is accepted.
7. **The exact firmware bug caused the released data: not proven.** Period atoms and increasing grid size are consistent with float-before-difference arithmetic. A pinned public source file does not identify the flashed binary or the unobserved timing origin.
8. **No progress in research because no positive method result: false.** Actual data, protocol checks and ordinary controls eliminate a misleading failure explanation. Nevertheless they do not achieve the requested novel research direction.

Stop this route as a new-method investment. Preserve the raw difficult regimes and all metrics. Do not fix held-out data heuristically or remove low-resolution examples to claim a favorable model score. Any future physical-state evaluation must define acquisition state, timestamp availability, measurement uncertainty and an independently justified target.
''')
 write('hypothesis_space.md','''# Competing hypotheses

| Hypothesis | Falsification/control | Judgment |
|---|---|---|
| H1: irregular time is the dominant physical-simulation gap | Same dynamics information comparison, measured interval prediction, actual-time conditional arm | Not supported as dominant; improvements are conditional/mixed and literature is established |
| H2: an inferred latent state controls when RPM is read | Firmware interrupt/getter audit | Proposed blocking-read mechanism rejected; other clock mechanisms not identified |
| H3: unmodeled nonlinear plant laws explain all OOD residual | Existing flexible HGB plus per-regime diagnostics | Not established; residual is concentrated in changed sensor resolution |
| H4: acquisition quantization explains much observed score growth | Inverse-RPM period grids, error contribution, float32 source arithmetic | Strongly supported as competing measurement explanation; exact hardware causation remains unproven |
| H5: a modern predictor requires a new architecture | Existing ridge/persistence/HGB, timing and measurement controls | Not shown; no new method should be built on this evidence |
| H6: inverse-action sufficiency extends to nonlinear reachability | New primary literature/assumption audit and then matched native test | Untested fresh question; not a conclusion of wind-tunnel experiments |

No new hypothesis is accepted just because a prior paper lacks a universal nonlinear theorem. For H6, distinguish ordinary first-order controllability, endpoint action reconstruction and nonlinear reachable directions. Read native definitions and compare known reconstruction/control objectives before running another toy. A classical counterexample or a loss-minimizer construction alone is not evidence of a practical unresolved research problem.
''')
 write('research_ideas.md','''# Candidate status

**Accepted directions: zero.** Two questions remain as low-confidence watch items, not publication recommendations.

1. **Nonlinear reachability versus inverse-action sufficiency.** Recent EP-IDM has explicit linear-system guarantees; CF-JEPA structures controllable latents through inverse/adversarial objectives. A narrowly differentiated question might concern whether indirectly reachable observables survive these objectives in nonlinear systems. This is untested, and nonlinear control/Lie-bracket foundations, action-abstraction methods and multistep controllability objectives are major prior-art risks. Do not relabel classical controllability as a new theorem. Next action is native assumption/source inspection, followed by a decisive controlled test only if the residual survives that inspection.

2. **Measurement-qualified evaluation of real time-series world models.** The wind-tunnel experiment directly shows why raw sensor forecast error can conflate measurement configuration and plant behavior. The present result is a provenance/observation-model warning, with related precedents in EndoClock and existing state estimation. It does not establish a new model need or a new benchmark contribution by changing domains.

Do not rank these as two validated top-conference ideas. The first is a search lead, the second a bounded evidence/benchmark caution. Generic joint time/value modeling, adding delta-time features, replacing a damaging auxiliary term and repairing float arithmetic are not accepted innovations.
''')
 write('final_research_direction.md','''# Current research decision

**No final direction has met the user's evidence and innovation requirements. The full goal remains ACTIVE.**

**Research Question:** no accepted final question. A provisional fresh lead concerns whether inverse-action training preserves nonlinear reachable information; the real wind-tunnel branch investigated timing versus measurement explanations and did not establish a new dynamics gap.

**Evidence:**38real recordings/965,322rows/64intervention blocks audited; ordinary ridge/HGB time/dynamics controls evaluated18recordings;128sensor-stream quantization audits performed. Low-resolution blocks contribute93–98%of HGB measured-RPM squared error. This is concrete evidence that failure attribution must account for acquisition, not direct evidence for a new dynamics method.

**Existing Methods:** TAWM, observation-intensity models, TE-CDE/TESAR-CDE, sensing-stage audits and ordinary system identification address much of the broad setting. Correctly specified measurement models have not been shown insufficient. EP-IDM/CF-JEPA are new closest works for the next search, not executed baselines here.

**Proposed Idea:** none accepted. Do not invent a new neural acquisition gate or rename a standard joint clock model. A nonlinear-reachability mechanism would need fresh empirical evidence and a precise distinction from established control theory and current inverse/multistep objectives.

**Falsification:** reject the wind-tunnel new-dynamics claim while measurement artifacts remain a sufficient competing explanation. Reject a fresh nonlinear-reachability claim if existing multistep inverse/controllability or reconstruction baselines preserve the relevant information under fair capacity/data/budget and independent held-out testing, or if literature already covers the exact mechanism.

**Feasibility:** prerecorded numeric data and pinned source are now available on the research server. This cycle's inexpensive existing controls required no new GPU model; RTX5090 remains available. Native source/code inspection for the fresh control-representation lead is feasible before deciding on any training investment.

**Next Experiment:** first inspect the exact native endpoint-inverse and factorization definitions/assumptions. If a genuinely untested residual exists, use a minimal nonlinear controlled system with known reachable effects, include strong reconstruction and existing multistep controls and avoid artificial scaling/capacity handicaps. Then require an independent non-toy task before accepting a research direction. If those conditions fail, switch search scope rather than maintain the idea.
''')
 registry={r['name']:r for r in lit['papers']};search='literature-search-20261016-clock-measurement';records=[];text='# Exploratory primary-source screening\n\n10primary papers in five groups: physical evidence; temporal control; observation processes; causal treatment sampling; control representations. Scores are provisional judgments of inspected sections, not reproduction scores or venue predictions.\n\n'
 for key,title,typ,venue,ins,comp,num,claim,url in PAPERS:
  records.append({'key':key,'title':title,'type':typ,'source_status':venue,'primary_url':url,'archived_url':registry[key]['url'],'insight_1to5':ins,'completeness_1to5':comp,'numeric_evidence_1to5':num if num is not None else 'N/A tool','code_executed':False,'reading_scope':'Relevant method/protocol sections; no full proof audit','relevance':claim});text+=f"- [{title}]({url}) — {venue}; {typ}; provisional insight/completeness/numeric {ins}/{comp}/{num if num is not None else 'N/A tool'}. {claim}\n"
 write(search+'/papers.md',text);(ROOT/search).mkdir(exist_ok=True)
 with (ROOT/search/'papers.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 clusters='''

## Opportunities and boundaries

Physical evidence cluster: controlled wind/light experiments are an existing resource; the fresh contribution here is only private qualification. Residual: a physical-latent evaluation target would need separate measurement provenance.

Temporal-control cluster: chosen variable-rate control is already modeled by TAWM. Residual: realized sampling-time availability differs from a chosen action duration, but no practical unresolved method gap has been established here.

Observation-process cluster: EndoClock and HyperNSDE address evidence loss and joint observation-time generation. Residual: action/acquisition coupling could differ, but the proposed tachometer-blocking mechanism was falsified before a new model was built.

Causal-treatment cluster: TE-CDE and TESAR-CDE handle continuous-time treatment outcomes and informative observation weighting. A generic informative-observation reformulation is not novel. Precise conditional assumptions matter.

Control-representation cluster: CF-JEPA, EP-IDM and [Controllable-Koopman](https://github.com/KYMiao/Controllable-Koopman) are primary-code/theory anchors for fresh search. Their nonlinear scope and native definitions must be checked; a missing universal guarantee does not prove empirical failure. These papers were not executed in this cycle.
'''
 write('literature_review.md',text+clusters)
 write(search+'/search-notes.md','''# Scope and source discipline

Exploratory search2026-10-16. Public queries only: informative observation/sampling, world models and sensor shifts, sample intervals, semi-Markov world models, controllability/inverse dynamics/Lie brackets. No unpublished result sentence was sent to a search engine. Primary arXiv/PMLR/official repository sources drive claims; secondary indexes only locate them. MDPI/preprints.org hits were excluded from selection. No new recurring monitor was created.

Ten deduplicated papers archived and hashed. Direct PMLR download failed; constructed publisher-GitHub paths returned404. Both attempt registries are preserved. Their arXiv preprint versions were then archived successfully, while published PMLR primary pages/PDFs were inspected through browsing. Do not assume the archived preprint is byte-identical to the published proceedings.

Source quality scores are provisional fit/evidence judgments on selected sections. Not all appendices or proofs were independently checked. Current-year acceptance labels are asserted only where the primary page states them; preprint status otherwise remains explicit. The wind-tunnel firmware is pinned public text, not a recorded board binary.

No reviewed title is a kill gate for a precise residual. The current rejection follows actual measurement/confounding controls and the lack of a demonstrated new-method need, alongside established literature overlap. Native code execution for these papers is not claimed.
''')
 write(search+'/idea-grounding.md','''# Evidence-to-idea boundary

Verified observations: within-recording timestamps are ordered; all64declared regime targets match recorded settings; low-resolution blocks dominate measured-label errors; inverse-RPM periods concentrate on coarse grids; inspected float arithmetic can produce such coarsening; actual future time helps some conditional one-step estimates without a uniform rollout benefit.

Unverified: exact firmware flashed during data collection, true continuous rotor paths, the causal source of every timing change, adequacy of a calibrated observation-likelihood model, and novelty/empirical failure of nonlinear reachability representations. None should be filled in by intuition.

Mechanism primitives: event/measurement clocks; chosen control duration versus realized sample time; measurement scale/quantization; latent transition versus observation likelihood; first-order versus nonlinear reachability; inverse-action sufficiency. Protocol anchors are independent recording splits, known programmed commands, no future sensor leakage, grouped intervention diagnostics and original training-score preservation.

Potential differentiation requires evidence: a controlled-simulator problem with unknown acquisition time must beat existing joint time/value models; a nonlinear-reachability problem must survive existing multistep/inverse/controllability and reconstruction controls. Both are low-confidence search leads, not accepted ideas. No new architecture is recommended.
''')
 write('outputs/task_status.json',json.dumps({'created_utc':stamp,'cycle8':{'scientific_jobs_complete':True,'audited_recordings':38,'audited_rows':965322,'regime_blocks':64,'evaluated_recordings':18,'retained_estimators':7,'actual_fits_including_ridge_grid':17,'sensor_streams':128,'primary_pdfs':10,'accepted_research_gap':False},'goal_status':'active','objective_achieved':False,'decision':'Measurement explanation prevents new-dynamics claim; fresh nonlinear-reachability lead remains untested'},indent=2))
 write('outputs/runtime.json',json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0),'scope':'Fits on CPU under process-local affinity/thread profile; GPU available but not used as a substitute for a decisive research question'},indent=2))
 print('Generated seven evidence ledgers and exploratory search folder; goal remains active/unachieved')
if __name__=='__main__':main()
