"""Generate scoped research ledgers from completed native audit evidence."""
import runtime_profile
import datetime,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def put(name,text):
 p=ROOT/name;assert not p.exists(),f'Preserve existing ledger: {p}'
 p.write_text(text.strip()+'\n',encoding='utf-8')

def main():
 r=read(ROOT/'outputs/native_chart/results.json')
 v=read(ROOT/'outputs/native_chart_independent_verification.json')
 assert v['verified'] and len(v['rows'])==12
 table='| Dataset | H | Shift | Target relative L2 | Fixed-coordinate max error |\n|---|---:|---:|---:|---:|\n'
 for x in r['rows']:
  table+=f"| {x['dataset']} | {x['horizon']} | {x['shift']} | {x['target_current_chart_difference']['relative_L2_mean']:.6f} | {x['target_fixed_chart_control']['absolute_max']:.1f} |\n"
 times='| Dataset | H | Batch | Full forecast ms | Separate encoder+stats ms | Ratio |\n|---|---:|---:|---:|---:|---:|\n'
 import statistics
 for x in r['timings']:
  full=statistics.median(z['synchronized_wall_ms_per_call'] for z in x['full_forecast'])
  enc=statistics.median(z['synchronized_wall_ms_per_call'] for z in x['history_encoders_and_stats'])
  times+=f"| {x['dataset']} | {x['horizon']} | {x['batch']} | {full:.3f} | {enc:.3f} | {x['encoder_wall_fraction']:.3f} |\n"
 put('literature_review.md', '''# Native state reuse and mechanism uncertainty: innovation boundary

Scope: a general time-series world-model method, not a dataset-specific forecast correction. A shared pretrained foundation model is optional. Native forecasting methods are evaluated against their declared forecasting contract; probabilistic simulation and intervention claims must be explicit.

- [WorldTS](https://arxiv.org/html/2609.31162v1): covariate-aware predictive-state forecasting with two-stage training. Appendix C acknowledges open theory, temporal regularization, irregular sampling and transfer. These are broad future-work topics, not evidence for a specific new mechanism. Four previously trained native checkpoints are used here without modification.
- [Does Normalization Choice Matter for Causal Large Time-Series Models?](https://arxiv.org/html/2606.09954v1): directly discusses changing full-window statistics, causal normalization and cache reuse. Prefix and cumulative causal statistics are necessary prior controls. Generic "normalization prevents caching" is already covered; exact moving-window compatibility at useful finite cost remains a narrower question.
- [Noise or Signal?](https://arxiv.org/html/2510.04667v1): studies robust/adaptive reversible normalization and reports failed adaptation. Dataset diagnostics or a renamed RevIN variant would not establish our novelty. No claim is made that this paper evaluates our native checkpoints.
- [OneWorld](https://arxiv.org/html/2609.30946v1), Sections 3 and D: couples alternative actions through shared latent physical-mechanism compatibility, using same-world and mismatched-world reset groups. Its learned compatibility score is explicitly not claimed to be a calibrated Bayes factor. This is direct overlap with generic shared-mechanism multi-future proposals. A single factual trajectory without reset groups is a distinct data regime, but not itself a new solution. Native source is promised in the paper; no native OneWorld run is claimed here.
- [MDN-VJEPA](https://arxiv.org/html/2610.13956v1): mixture transitions, trajectory coupling and transition composition already address broad stochastic-rollout concerns. The manuscript distinguishes its uncertainty construction from a general epistemic/aleatoric decomposition. We have not reproduced it.
- [Bayesian Context Aggregation for Neural Processes](https://github.com/boschresearch/bayesian-context-aggregation): ICLR 2021 author's official archived implementation, with Bayesian latent aggregation and predictive likelihood variants. A persistent global latent or additive evidence update is not new. Its native experiment is a GP regression suite, not a partially observed controlled-dynamics result. Source license is AGPL-3.0; no installation/execution is performed in this cycle.
- [Deconfounding Reinforcement Learning in Observational Settings](https://arxiv.org/pdf/1812.10576), Sections 3.2-3.4: includes a policy/action likelihood conditional on latent state/confounders, controlled latent transitions and intervention-time adjustment. It explicitly discusses treatment choices carrying information about latent state. Therefore an action-likelihood branch plus do-conditioning is not a new mechanism. Its assumptions and identifiability qualifications must be retained; this citation does not establish identification from arbitrary confounded data.

Overlap was checked from primary papers and author repositories. Abstract-only leads (PULSE, DynaMITE, newer recognition-parameterized GP dynamics) are not treated as reproduced methods or definitive exclusions. Search-result publication dates can conflict with manuscript metadata; no venue status is inferred from a search snippet. Exact archived document bytes are recorded under outputs/literature/manifest.json. The versionless DRL PDF is pinned by content hash, not assigned a guessed arXiv revision.
''')
 put('baseline_registry.md', '''# Baseline registry — cycle 14

## Native frozen models actually executed

- WorldTS source: `269426ea6217d7152a244da0e209d922acd46d4f`.
- Existing `NP` and `PJM`, H=24 and H=360, seed 2021; stage2_decoder_mix checkpoints, original resolved configs and original data SHA256 checked against the earlier sealed provenance registry.
- Native constructor, target/condition encoders and complete native forecast used. No fitting, adaptation, edited weights or new test-driven model selection.
- Inputs: 32 training-context origins per checkpoint; target OT first, six historical covariates, actual corresponding future covariates for native forecast timing. This follows the historical forecasting interface and is not a controlled causal experiment.
- NP state patch 3, latent 128, condition 64; PJM state patch 5, latent/condition 224. Raw target patches contain only 3 or 5 scalars: a generic latent transport is not automatically cheaper than re-encoding.
- Executed Python source snapshot: `outputs/native_worldts_executed_source.zip`, 67 hash-checked files. Runtime/config/checkpoint/data hashes are recorded in results.json. Large pre-existing inputs remain read-only in their original folders; no checkpoint duplication is required for the arithmetic verification.

## Controls actually executed

- Newly computed moving-window reference statistics.
- The original reference statistics applied to the shifted window; identical overlapping patches compared after excluding replicate-padding-dependent positions.
- Independent NumPy float64 arithmetic over the saved GPU float32 arrays.
- Complete native forecast vs separately timed stats plus history target/condition encoders, batch 1 and 8. These are descriptive separate timings, not an additive partition or realized caching speedup.

## Literature controls not executed this cycle

Prefix/cumulative causal normalization, Bayesian neural processes, hierarchical/GP state-space models, UWM-JEPA/BJEPA, MDN-VJEPA, OneWorld, and proximal/deconfounding methods. Do not mark them as reproduced. Existing cycle 1-13 experiments remain separate evidence with their original task definitions.

The official Bayesian Context Aggregation repository was pinned at `20cfb5230e7be2f980365a992dab138ddd027070` (36 files, AGPL-3.0) and archived without executing its code. Source was fetched locally because the server refused GitHub connections, then transferred and hash-verified on the server. The native aggregator already supports incremental Gaussian evidence updates; its sampled decoder broadcasts a latent draw across target queries. Neither feature is a proposed invention. This is source inspection, not a trained dynamics-model result.
''')
 put('experimental_results.md',f'''# Completed native discriminator, not a new-method result

All scientific jobs in this bounded audit terminated successfully. Research discovery remains incomplete.

## Protocol

Four unchanged trained checkpoints; 32 origins drawn entirely from the corresponding training region; shifts 1, 8, 24. Exact shared raw patches are verified. Comparing current versus original window statistics isolates a change of representation coordinates. It does not establish lost physical information, forecast degradation, a source bug or a causal failure.

{table}
The condition encoders are audited separately in the machine-readable results. All 24 target/condition fixed-reference comparisons are exactly zero for the saved floating-point tensors. Independent verification covers 12 rows, 124 saved arrays and 128 training origins. No scientific fit was rerun.

## Timing

Eight warm-up calls, then three repetitions of 40 calls; CUDA events and synchronized wall time both saved.

{times}
The separately timed encoder+stats component is approximately 19%-24% of full wall time. This establishes limited current component cost, not a measured encoder-removal speedup. The helper omits native finite-value validation synchronization, and separate calls are not an additive execution decomposition. Consequently the raw legacy field `encoder_elimination_ideal_speedup_upper_bound` must not be interpreted as a rigorous bound. No actual cached forecast or transport method was benchmarked. All full forecasts are about 4.5-5.7 ms in this specific environment; these measurements are not general hardware guarantees.

## Evidence integrity and limitations

Checkpoint/config/data SHA256 and complete before/after tensor fingerprints match the prior registry. Executed source bytes are archived and checked. NumPy independently reconstructs chart differences and statistics with metric tolerance 2e-6. This verifier does not independently replay checkpoint forecasts on CPU. There is no new held-out forecast score, trained method, multi-seed efficacy comparison, causal benchmark, or mechanism-uncertainty experiment in cycle 14.
''')
 put('negative_evidence.md', '''# Negative evidence and rejected overclaims

1. Overlapping raw patches do change latent coordinates under moving-window statistics, but fixed-reference encoding gives exact equality. Treating this as information collapse or a failure of the official native forecasting task is unsupported.
2. In these four native checkpoints the historical encoders are shallow and strongly expand small target patches. A learned latent chart transport could be more expensive than ordinary re-encoding. Separate timing suggests a limited component opportunity, but does not prove every caching method valueless.
3. Causal normalization and cache reuse are already explicit prior art. Renaming this as gauge/state consistency would not establish novelty.
4. Persistent shared physical mechanisms across alternative futures are already addressed by OneWorld, and latent function/mechanism posteriors by neural processes and Bayesian system identification. Lack of paired reset supervision is a possible regime distinction, not sufficient novelty or proof of failure.
5. The proposed two-role view of actions—evidence about hidden state and a physical input—overlaps explicit action likelihoods and do-adjustment in earlier deconfounding latent models. It is rejected in this broad form before launching another predictable toy demonstration. Arbitrary observational hidden confounding must not be claimed identifiable.
6. Mixture trajectories and transition composition are already covered by stochastic JEPA literature. We have no new empirical failure of these controls.

No method has been selected using a test set. No native model has been blamed for an interface it never claimed to provide. No accepted research direction is manufactured to fill the report.
''')
 put('hypothesis_space.md', '''# Competitive hypotheses and decisions

| ID | Precise hypothesis | Discriminator | Current status |
|---|---|---|---|
| A0 | Identical overlapping raw patches encode identically despite changing window statistics | Original-vs-moving chart, excluding padding | Rejected in the current chart; fixed-chart control exact |
| A1 | Native representation drift proves worse forecasting or lost state information | Held-out native forecast controls with declared interface | Not tested; must not infer from A0 |
| A2 | Cheap transport offers important native end-to-end benefit beyond ordinary causal normalization/re-encoding | Actual speed/error frontier against compiled/recomputed and causal-stat controls | Not established; low priority given direct overlap and small shallow encoders |
| B0 | A shared latent mechanism for alternative futures is absent from prior art | OneWorld plus latent neural-process/Bayesian literature | Rejected as a broad novelty claim |
| B1 | Without paired reset supervision, finite-budget latent belief models can retain calibrated mechanism uncertainty across condition queries under partial observation | Correct ordinary Bayesian/particle and neural-process controls, common data/compute, separately scored within-path and between-query distributions | Unresolved regime-specific question; no new mechanism accepted |
| C0 | Joint action likelihood and controlled dynamics is a new mechanism for two-role actions | DRL Sections 3.2-3.4, proximal causal dynamics literature | Rejected as a broad novelty claim |

For B1, an existing correctly conditioned hierarchical/particle model matching the proposed method's quality and cost falsifies a claimed new advantage. A Gaussian approximation failure alone does not establish a new gap; mixture and particle controls are mandatory. Single-factual training cannot identify arbitrary cross-intervention couplings without declared structural assumptions. Ordinary forecast-only models are not required to output such couplings.

The next useful step is a source-and-protocol comparison of existing amortized mechanism-posterior controls before any new architecture. Training is justified only after specifying an identifiable data regime, an existing method's actual contract and a concrete residual prediction that a strong ordinary control does not trivially resolve.
''')
 put('research_ideas.md', '''# Candidate evolution — no accepted direction

The lineage and original three exploratory cards are preserved in IDEA_EVOLUTION.md. These are candidates, not final recommendations.

1. **Coordinate-compatible reusable predictive states.** Revised from the broad normalization/caching idea after direct prior-art review. A real contribution would require measured cost/accuracy benefits under an explicit persistent-state interface, beyond simple prefix/cumulative statistics and ordinary recomputation. Current native evidence weakens its priority. No transport architecture is justified yet.
2. **Reusable uncertainty about persistent dynamics from factual histories.** This remains a different fallback, narrowed after OneWorld and Bayesian neural-process review. A possible residual involves partially observed systems, no reset-group supervision, calibrated uncertainty and finite inference cost across multiple future conditions. A global latent, an ensemble or a posterior aggregator is not an invention. There is no established empirical failure or accepted new method yet.
3. **Conditional transition reuse.** The earlier generic future-memory hypothesis was repaired by ordinary FIR/GRU controls; it must not be repackaged. A fresh proposal needs an advantage beyond controlled SSM/CDE/Koopman and ordinary recurrent models. No such residual is demonstrated this cycle. The subsequent two-role action variant is rejected in its broad form because explicit policy likelihood plus intervention adjustment already exists.

No dataset-specific optimization is recommended. Observational forecasting, prospective conditional prediction and identified intervention simulation are distinct experimental contracts and will be reported separately.
''')
 put('final_research_direction.md', '''# Research decision — objective still active

**Final recommended research direction: not yet established. Accepted directions: 0.**

**Research Question.** Can a general time-series world model maintain reusable predictive state with useful conditional responses and uncertainty? This is an umbrella objective, not an innovation claim. The narrower remaining intake question is whether existing amortized mechanism-posterior methods already satisfy calibrated multi-query prediction under partial observation without reset-group supervision at practical finite cost.

**Evidence.** Four unchanged trained native WorldTS checkpoints show normalization-coordinate drift on shared patches; fixed-reference controls give exactly equal states. Separate component timings show approximately 19%-24% of native forecast wall time for history encoders and statistics. These results do not demonstrate a forecast failure or a useful new transport method.

**Existing Methods.** Causal normalization directly addresses cache reuse; neural processes/Bayesian identification already represent persistent latent mechanisms; OneWorld already couples branches by shared mechanisms; probabilistic JEPA already handles coherent stochastic transitions. Earlier deconfounding models already separate action evidence and controlled transitions. No claim that these methods are universally sufficient or mutually interchangeable is made.

**Proposed Idea.** None accepted. The surviving posterior-reuse question requires a concrete residual beyond those methods before proposing a new mechanism. Combining standard components without such a residual is insufficient.

**Falsification.** An ordinary existing posterior model that matches a candidate's quality, calibration, information access and compute falsifies its claimed advantage. A different input contract or an underfit Gaussian strawman is not acceptable evidence. Unidentified observational counterfactuals are excluded.

**Feasibility.** The RTX 5090 server and frozen native WorldTS checkpoints are available. Small controlled linear/nonlinear partially observed families could validate inference correctness before a broader cross-domain study; they would be newly designed experiments, not original-paper reproductions. No expensive run should start merely to generate more result rows.

**Next Experiment.** First freeze the exact data/inference contract and reproduce an existing Bayesian/latent neural-process control's conditional path construction. Only if a finite-resource residual remains should the next discriminator compare that control with matched recurrent/particle alternatives on independent held-out systems. These findings alone do not justify new long training or a dataset-specific fix. Ordinary research remains authorized; no further confirmation is requested.

This file is a cycle decision record, not a claim that the user's research objective has been completed.
''')
 status={'cycle':14,'native_checkpoints_audited':4,'independent_chart_rows_verified':12,'saved_arrays_checked':124,
         'new_training_fits':0,'native_source_files_archived':67,'scientific_audit_complete':True,
         'actual_cache_method_measured':False,'novel_direction_accepted':False,'objective_achieved':False,
         'decision':'Lower priority of chart caching; reject broad shared-mechanism and action-evidence novelty claims; refine identifiable finite-resource posterior-reuse question',
         'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (ROOT/'outputs/task_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
 print(json.dumps(status))
if __name__=='__main__':main()
