"""Generate evidence ledgers from completed remote results, without gap inflation."""
import runtime_profile
import datetime,json,pathlib,platform,subprocess,sys
import numpy as np
from native_child import ROOT

def read(relative):return json.loads((ROOT/relative).read_text(encoding='utf-8'))
def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')
def main():
 native=read('outputs/child_native_A_r1/result.json');assert native['state']=='complete'
 density=read('outputs/conditional_density/results.json');assert len(density)==6
 contracts=read('outputs/native_contracts/results.json');noise=read('outputs/generator_noise_audit/results.json')
 recover=read('outputs/signal_recoverability/results.json')['rows'];prior=read('outputs/trained_prior_audit/results.json')['rows']
 train=[json.loads(s) for s in (ROOT/'outputs/child_native_A_r1/train.jsonl').read_text().splitlines()]
 assert native['updates']==15470 and len(train)==20
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 summary=[]
 for checkpoint in ['initial','best','final']:
  rows=[r for r in prior if r['checkpoint']==checkpoint]
  summary.append({'checkpoint':checkpoint,'scalar_conditionals':len(rows),'max_absolute_mass_error':max(r['absolute_mass_error'] for r in rows),'minimum_reachable_base_mass':min(r['base_noise_mass_with_preimage'] for r in rows),'conditionals_with_derivative_sign_changes':sum(r['derivative_sign_changes']>0 for r in rows)})
 (ROOT/'outputs/trained_prior_audit/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 header='''# Cycle 6: hierarchical stochastic-model audit

Evidence checkpoint. The autonomous research-discovery objective remains ACTIVE;
no direction has yet passed both the novelty and empirical gates. This folder is
independent of the sealed main experiment and cycles2–5.
'''
 write('README.md',header+f'''
Completed on the remote RTX5090: full native-architecture CHiLD A configuration,
10 epochs /15,470 updates /462,921 parameters /{native['seconds']:.1f} seconds;
six controlled conditional-density fits; three native real-world component
audits; original-versus-independent-noise generator audit; six supervised linear
recoverability certificates;63 scalar conditional density checks across initial,
best and final priors. The model run uses disclosed manual Torch orchestration
and an nn.Module container adapter, so it is CLOSE, not exact launcher replication.

The source implementation has concrete parameter-registration and generator
issues. Neither is renamed as a research contribution. Training does not support
the proposed general density-failure explanation. Source pins, data hashes,
frozen plans, raw logs, failed attempts and checkpoints are retained privately.
See the seven research ledgers and IMPLEMENTATION_CORRECTIONS.md.

No formal long-horizon/causal/density theorem is claimed, no public content was
published, and no older project or sealed cycle was modified.
''')
 write('baseline_registry.md','''# Baseline and protocol registry

| Object | Pin / data / execution | Classification |
|---|---|---|
| CHiLD source | MinghaoFu/CHiLD,28cc8a8a702018c9c54bd02e26cbd8ccf756ab99;110 small source/config files,474898 bytes | Private pinned source, not the real-world dataset zip |
| Native synthetic A | Native stationary_z_link,seed772,z=[1,2,4],100000 seven-step trajectories; X=(100000,7,4),Y=(100000,7,7) | Original released generator, source issues disclosed |
| Full CHiLD model | A.yaml architecture/loss/AdamW;train98976,val1024,seed770,B64,10 full epochs,15470 updates; no batch cap | CLOSE source/config execution; manual Torch orchestration/container adapter, different data-loader RNG/environment; not exact paper-table replication |
| Real-world CHiLD components | Three seeds81/82/83; unchanged model/prior classes, native finite forward/backward/Adam step and eval ablation | Component audit only, no full real-data reproduction |
| Conditional-density experiment | Native 33537-param inverse MLP vs ordinary33538-param normalized Gaussian, same hidden128/depth3;3 seeds;2000 updates each | New controlled component experiment, not CHiLD latent-recovery reproduction |
| Independent-noise A | Same seed772,shape/graph/mixing; only hierarchy-global innovation indexing repaired | Explicitly modified benchmark control; not original data and not a certificate of all theorem assumptions |
| Linear recoverability | Train-only supervised OLS to true factors, context widths1/3/5;native fixed train/val split | Signal-recoverability certificate, not an input/supervision-matched unsupervised baseline |
| Piecewise conditional integration | Full-real-line affine-region enumeration of scalar inverse MLP at fixed parents; analytic identity/abs/shifted-abs checks | Numerical scalar diagnostics, not a global joint-density certificate |

The real-world transition priors sit in an ordinary outer Python list. In this
path none of40 prior tensors belongs to the model optimizer parameter set. The
synthetic implementation uses registered modules and is a separate code path;
do not transfer that finding to its optimizer. External launchers/logging were
bypassed; no third-party logging credential was used.
''')
 lines=['# Experimental results','',f'Generated UTC: {stamp}','',
 'All claims below refer to completed remote execution. Distinct protocols are not combined into a leaderboard.','',
 '## Full native A configuration','',
 f"- Complete:10 epochs,15470 updates,462921 parameters,{native['seconds']:.1f}s.",
 f"- Best validation Spearman permutation MCC:{native['best_validation_mcc']:.9f},epoch{native['best_epoch']}.",
 f"- End-of-training MCC:{train[-1]['validation']['mcc_spearman']:.9f};the best checkpoint remains preserved.",
 '- Validation latent labels select the checkpoint, matching the synthetic validation convention. No independent test or paper-exact replication is claimed.',
 '- Reconstruction is stochastic; rechecking the same best model can change reconstruction MSE while posterior-mean MCC stays fixed.','',
 '## Native real-world parameter registration','',
 '| Seed | Prior parameter tensors | Registered in full model | Max prior weight change after native optimizer step | Eval change after prior zeroing |',
 '|---|---:|---:|---:|---:|']
 for r in contracts:lines.append(f"|{r['seed']}|{r['real_prior_parameter_tensors']}|{r['real_prior_tensors_registered']}|{r['real_prior_parameter_max_update']:.1f}|{r['real_eval_max_change_after_prior_zero']:.1f}|")
 lines+=['','Nonzero prior gradients exist, but the native optimizer does not contain those tensors. Evaluation reconstructs supplied windows; this is not evidence of blind prior rollout. This is an implementation/protocol result, not a theorem refutation.','',
 '## Generator audit','',
 f"Original innovations:rank{noise['native']['innovation_cov_rank_at_1e-8']},maximum off-diagonal absolute correlation{noise['native']['maximum_absolute_offdiag_correlation']:.9f}.",
 f"Independent-noise control:rank{noise['independent_noise_control']['innovation_cov_rank_at_1e-8']},maximum off-diagonal absolute correlation{noise['independent_noise_control']['maximum_absolute_offdiag_correlation']:.9f}.",
 'The pinned helper named leaky_ReLU returns its input unchanged. Active transitions/mixing therefore use linear operations. The hierarchy reuses noise indices across layers. Fixing noise indexing does not establish sufficient variability, invertibility, independent-view or every other theorem premise.','',
 '## Supervised recoverability certificates','',
 '| Dataset | Context width | Validation mean Spearman to true factors | Validation mean R2 |','|---|---:|---:|---:|']
 for r in recover:lines.append(f"|{r['variant']}|{r['context_width']}|{r['mean_spearman']:.9f}|{r['mean_R2']:.9f}|")
 lines+=['','The certificate uses factor labels during fitting. It shows signal accessibility for these finite data, not unsupervised identifiability or a superior fair model. Independent innovations leave nonzero residual error for hidden factors at finite context.','',
 '## Conditional-density isolation','',
 '| Model | Seed | Selected step | Mean proper test NLL over3 fixed contexts | Maximum scalar mass error |','|---|---:|---:|---:|---:|']
 for r in density:
  mean=np.mean([t['proper_normalized_test_NLL'] for t in r['test']]);err=max(abs(t['integrated_surrogate_mass']-1) for t in r['test'])
  lines.append(f"|{r['kind']}|{r['seed']}|{r['selected_step']}|{mean:.9f}|{err:.3g}|")
 lines+=['','All9 trained native scalar conditionals tested are monotonic and normalized to numerical precision. The ordinary normalized Gaussian fits the Gaussian DGP better at this fixed budget. That is an existing control, not a new proposed method. Three model initializations share a fixed data cohort; they are not3 independent datasets.','',
 '## Native trained-prior conditional diagnostics','',
 '| Checkpoint | Scalar conditionals | Maximum mass error | Minimum reachable base-noise mass | Conditionals with derivative sign changes |','|---|---:|---:|---:|---:|']
 for r in summary:lines.append(f"|{r['checkpoint']}|{r['scalar_conditionals']}|{r['max_absolute_mass_error']:.9g}|{r['minimum_reachable_base_mass']:.9g}|{r['conditionals_with_derivative_sign_changes']}|")
 lines+=['','These are3 fixed training trajectories ×7 scalar factors, not held-out model selection or a global joint certificate. A folded inverse can have an integral near1 while failing to cover half the base Gaussian; normalization and invertibility are distinct. The final prior has normalized, unfolded tested conditionals. The generic claim of systematic trained likelihood cheating is unsupported.']
 write('experimental_results.md','\n'.join(lines))
 write('negative_evidence.md','''# Negative evidence and rejected interpretations

1. Non-bijective architecture alone does not establish a trained density failure.
   All9 tested trained component conditionals normalized and were monotonic;
   the final full-model21 tested factors normalized to numerical precision.
   The best checkpoint has a small maximum scalar mass error,0.0043505, not
   evidence for an unrestricted global failure or paper theorem counterexample.
2. Folded initialization may accidentally integrate to1 while exposing only half
   of the Gaussian base. Keep invertibility and density mass tests separate.
3. The native hierarchy's shared noise and identity activation helper invalidate
   treating this generator as a generic nonlinear independent-innovation test.
   Repaired innovation independence alone does not certify theorem assumptions.
4. High supervised OLS recovery is not a fair unsupervised model comparison.
   Finite-context residual uncertainty on the independent-noise control also
   means distributional identifiability cannot be equated to exact per-sample
   hidden-state recovery.
5. Real-world prior registration and reconstruction-path defects are concrete
   code/protocol issues. Ordinary ModuleList registration is an existing repair;
   this does not establish a new research problem or novelty.
6. Training MCC degradation after epoch1 is descriptive for one run. Data-family
   and objective/optimization explanations remain; no new state-selection loss
   or architecture is justified by this trace.
7. Generic probabilistic JEPA, multimodal trajectory prediction, Bayesian
   self-consistency, compositional mechanisms and prediction-versus-control
   distinctions already have close prior work. Latest versions matter: the
   current Branch-JEPA v3 includes full-set proper scoring, and ABI v6 includes
   real time series. Older versions do not support claims those are missing.

Implementation mistakes in our adapter/auditor were caught before accepting the
affected result and are recorded in IMPLEMENTATION_CORRECTIONS.md. No failed
attempt was silently replaced or counted as successful execution.
''')
 write('hypothesis_space.md','''# Competing hypotheses and falsification

| Hypothesis | Decisive comparison | Result / decision |
|---|---|---|
| Hierarchical temporal prior explains native real-world evaluation gains | Parameter registration, native optimizer update, zero-prior eval ablation | Prior has nonzero gradients but zero optimizer update and zero eval effect in all3 audited seeds. Source-path explanation; no new-method route. |
| Native synthetic recovery failure is only missing observable information | Train-only supervised OLS certificates, larger context | Original finite-data mean Spearman .9959–.9995. Information-only explanation insufficient there; supervised vs unsupervised distinction remains. |
| Non-bijective inverse necessarily corrupts trained density | Exact scalar full-line mass and root tests on trained components and full native checkpoints | Strong form falsified in tested cases; no expansion of this route. |
| Generator implements independent hierarchical nonlinear innovations | Reconstruct innovations from actual source; only noise-index repair | Native rank4/shared innovations; repaired rank7. Identity helper remains. Source issue, not novel dynamics defect. |
| A learned deterministic context exactly equals hidden stochastic state | Same OLS certificate on independent-noise control | Finite-context hidden-factor residual persists. Exact realization recovery is not a consequence of distributional identifiability. No theorem refuted. |
| A new belief/mixture/consistency architecture is needed | Closest full-paper/source checks | UWM-JEPA,Branch-JEPA,ABI-SC,COMET and ordinary normalized models cover broad mechanisms. No precise surviving residual yet. |

Stop gates: ordinary known repair explains the effect; task/input definition is
incompatible; information is unavailable; no healthy native execution; closest
existing method already supplies the proposed mechanism. None of these can be
overridden by renaming the effect or retuning the test set.
''')
 write('research_ideas.md','''# Candidate directions

**Accepted directions:0.** No new architecture is proposed on the current evidence.

The CHiLD density/identifiability route is stopped at this checkpoint: native
data/protocol issues and negative trained-density tests block a clean novelty
claim. More seeds of that same flawed generator would have low information gain.

One exploratory screening question remains, not a recommended paper direction:
**Which decision-relevant stochastic feature remains lost after applying strong
existing probabilistic belief models, properly scored mixtures and ordinary
state-estimation controls under the same causal input contract?**

This question must be narrowed against current UWM-JEPA,Branch-JEPA v3,ABI-SC v6,
MDN-VJEPA,ActTime and existing POMDP/causal-sequence literature before a new long
run. Merely showing point forecasts miss multimodality, or forecast accuracy
differs from intervention/decision accuracy, is already covered and inadmissible.

No result here establishes that this residual exists. A fresh executable
benchmark and one precise failure signature are prerequisites, not deliverables
that can be assumed from related-work titles.
''')
 write('final_research_direction.md','''# Current research decision

**No final research direction is justified yet.** Cycle6 is a completed evidence
checkpoint; the overall autonomous discovery objective remains ACTIVE.

**Research Question.** Which failure survives correct data generation, a healthy
full native model, a common causal input contract, and strong existing stochastic
inference/prediction controls? No precise unsolved residual is accepted.

**Evidence.** Full CHiLD A executes; source parameter registration and generator
issues are real. Six matched component fits and63 scalar prior diagnostics do
not support a general trained density-failure mechanism. Supervised certificates
separate finite-data accessibility from unsupervised identifiability.

**Existing Methods.** Module registration, normalized conditional models,
Bayesian filtering, proper full-distribution scores, mixture world models and
amortized posterior self-consistency are established controls. They cannot be
renamed as the proposed innovation.

**Proposed Idea.** None endorsed. Stop density-head tuning and additional runs on
the same native generator until a distinct residual is identified.

**Falsification.** Reject a candidate when the known baseline solves it at matched
information/budget, the model does not claim the tested task, source errors
explain it, or the data cannot identify it. No test-set search for a positive row.

**Feasibility.** One RTX5090 completed the462921-param model and six component
fits. The next screening experiment should stay small and include analytic or
simulator truth plus a known-method control before expanding seeds/datasets.

**Next Experiment.** First verify current executable protocols of recent belief,
intervention and trajectory-distribution methods; freeze one remaining precise
failure signature and its existing-control comparison. If none survives,
change problem family instead of increasing this generator's training budget.
''')
 write('literature_review.md','''# Literature and innovation boundary

Exploratory search, not exhaustive coverage. Public primary papers are archived
with exact version and SHA256 in outputs/literature/registry.json. A title/abstract
screen is not a reproduction. Older cycles retain earlier verified boundaries.

| Primary work | Boundary relevant to this audit | Verification scope |
|---|---|---|
| [CHiLD,NeurIPS2025](https://arxiv.org/html/2510.18310v1) | Hierarchical temporal latent distribution identifiability needs stated independent-view/operator/variability assumptions. It does not imply exact deterministic hidden-state recovery from a finite context. | Paper, pinned code, native full configuration and component audits; no theorem refutation |
| [Ada-Diffuser,2026](https://arxiv.org/html/2605.16054v1) | Future context is used in identifiable dynamic generation and refinement. Joint posterior/Gibbs refinement can be legitimate; self-generated context alone is not evidence of leakage. | Paper read; public executable code not verified |
| [CaRiNG,ICML2024](https://proceedings.mlr.press/v235/chen24ai.html) | Non-invertible observation maps differ from a non-bijective inverse innovation network. Do not conflate these assumptions. | Primary paper/appendix screening; no execution |
| [SurVAE,NeurIPS2020](https://arxiv.org/abs/2007.02731) | Proper surjective generative modeling and branch accounting are established. | Primary paper archive, scoped mathematical boundary |
| [Exploding inverses,AISTATS2021](https://proceedings.mlr.press/v130/behrmann21a.html) | Numerical inverse instability is established even for bijections; it is distinct from structural folding. | Primary PDF read; no execution |
| [Bayesian self-consistency,ICLR2026,v6](https://arxiv.org/html/2501.13483v6) | Uses prior/likelihood/posterior identities on unlabeled data, including real time series and images. Generic posterior consistency is not new. The earlier v1 toy scope is outdated. | Latest primary version verified; code link located, no execution |
| [Branch-JEPA,v3](https://arxiv.org/html/2607.05238v3) | Finite weighted latent trajectories with full-set Energy Score and decoded distribution scoring; a generic multimodal/proper-score JEPA proposal is covered. v1 MoP-JEPA does not define the current boundary. | Latest primary paper read; no native execution |
| [UWM-JEPA](https://arxiv.org/html/2605.25313v1) | Belief-structured blind rollout with joint unitary-spectrum preservation; paper distinguishes full-state invariants from reduced-state readout and limits long-horizon evidence. Generic belief preservation is not new. | Primary paper read; public GitHub link located, no execution |
| [COMET,2024](https://arxiv.org/html/2404.15109v1) | Reusable mechanisms and composition across dynamics environments predate a new compositional-world-model claim. | Primary paper read; no execution |
| [Mechanistic World Models,2026](https://arxiv.org/abs/2607.12474) | Mechanistic modularity is an existing conceptual program. | Abstract screening and primary PDF archive, not a full implementation audit |
| [World Models in Pieces,2026](https://arxiv.org/html/2606.24842v1) | Local transition certification from agent decisions uses discrete/anchor assumptions; neither generic certification nor unrestricted nonlinear guarantees can be assumed. | Primary method read; no execution |
| [ThermoForce,2026](https://arxiv.org/html/2607.03942v1) | Frozen passive forecast plus physics-structured forced response already targets intervention-sign/magnitude failures in HVAC. | Primary paper read; release states code/data forthcoming, no executed baseline |
| [Rethinking World Models for Biosignals,2026](https://arxiv.org/html/2610.17510v1) | Action latency, external drivers and endogenous dynamics already motivate intervention-validation tests. | Primary perspective read; not empirical method reproduction |

New search leads:ActTime (anonymous repository availability unresolved),
What Must a World Model Distinguish for Planning? (2609.33030), and MDN-VJEPA.
Do not cite secondary search summaries as executed evidence or accepted novelty
boundaries. Several reviewed papers are preprints; venue status is asserted only
where verified. Future-dated journal issue metadata is not proof of availability
at this cutoff. MDPI and preprints.org leads are excluded from evidence selection.
''')
 environment={'created_utc':stamp,'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'profile':'runtime_profile.py: process-local affinity0xffff0000; numerical threads1; MKL sequential; external logging disabled','source_pin':'28cc8a8a702018c9c54bd02e26cbd8ccf756ab99'}
 import torch,scipy
 environment.update(torch_version=torch.__version__,torch_file=torch.__file__,numpy_version=np.__version__,numpy_file=np.__file__,scipy_version=scipy.__version__,gpu=torch.cuda.get_device_name())
 (ROOT/'outputs/environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
 status={'created_utc':stamp,'objective_achieved':False,'goal_status':'active','accepted_research_directions':0,'cycle6':{'scientific_jobs_complete':True,'native_full_runs':1,'conditional_density_fits':6,'component_audit_seeds':3,'scalar_prior_diagnostics':63,'supervised_recoverability_certificates':6,'decision':'STOP general density-failure and native-generator expansion; no accepted novel gap'},'safety':'Private outputs only; no public publication or old-project mutation'}
 (ROOT/'outputs/task_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
 print(json.dumps(status))
if __name__=='__main__':main()
