"""Generate raw research ledgers from completed server evidence, no publication."""
import datetime,json,pathlib,re,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')
def main():
 cofe=read('outputs/cofe_etth1_affinity1/results.json');contracts=read('outputs/cofe_contracts/results.json');trained=read('outputs/trained_cofe_audit/results.json');risk=read('outputs/continuous_counterfactual/results.json')
 assert read('outputs/cofe_etth1_affinity1/status.json')['state']=='complete'
 assert read('outputs/kuma_native_pems03_affinity1/status.json')['exit_code']==0
 assert read('outputs/continuous_counterfactual/status.json')['state']=='complete'
 kuma=[json.loads(s) for s in (ROOT/'outputs/kuma_native_pems03_affinity1/results.jsonl').read_text().splitlines() if s.strip()];assert len(kuma)==1;kuma=kuma[0]
 log=(ROOT/'outputs/kuma_native_pems03_affinity1/stdout.log').read_text();epochs=re.findall(r'Epoch (\d+) \| train MSE ([.\d]+) \| val MSE ([.\d]+) \| val MAE ([.\d]+)',log)
 oldlog=(ROOT/'outputs/kuma_native_pems03/stdout.log').read_text();old_epochs=re.findall(r'Epoch (\d+) \| train MSE ([.\d]+) \| val MSE ([.\d]+) \| val MAE ([.\d]+)',oldlog)
 assert epochs[:len(old_epochs)]==old_epochs
 maximum_gap=max(e['relative_gap'] for row in risk['rows'] for e in row['extrema'])
 maximum_abs=max(e['absolute_gap'] for row in risk['rows'] for e in row['extrema'])
 maximum_residual=max(e['solver']['max_constraint_violation'] for row in risk['rows'] for e in row['extrema'])
 signed=[(1 if e['kind']=='min' else -1)*(e['feasible']-e['SDP_bound']) for row in risk['rows'] for e in row['extrema']]
 summary={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kuma':kuma,'kuma_epochs_completed':len(epochs),'kuma_best_epoch_by_logged_val':int(min(epochs,key=lambda t:float(t[2]))[0]),'kuma_matching_failed_attempt_prefix_epochs':len(old_epochs),'cofe_trained_models':3,'cofe_parameters':cofe[0]['parameters'],'risk_cases':len(risk['rows']),'risk_generic_SDP_max_absolute_gap':maximum_abs,'risk_generic_SDP_max_relative_gap':maximum_gap,'risk_max_constraint_violation':maximum_residual,'risk_smallest_signed_gap_numerical_tolerance':min(signed),'risk_max_dense_check_error':max(row['dense_feature_max_error'] for row in risk['rows']),'risk_max_twin_check_error':max(row['direct_twin_max_error_20'] for row in risk['rows']),'risk_decision':risk['decision'],'scientific_jobs_complete':True,'objective_achieved':False,'accepted_research_directions':[]}
 (ROOT/'outputs/cycle5_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 pins=[read(str(p.relative_to(ROOT))) for p in sorted((ROOT/'sources').glob('*/SOURCE_PIN.json'))]
 (ROOT/'outputs/combined_source_registry.json').write_text(json.dumps(pins,indent=2),encoding='utf-8')
 write('README.md',f'''# Cycle 5: native-model attribution and computational falsification

Research objective remains **ACTIVE**. No final direction is accepted.
Server directory: `D:\\tswm_discovery_20261016_v1\\cycle5`.
All previous cycles are preserved. These are private research records.

Completed evidence: one full released-code KUMA PEMS03 run, three full COFE-model
ETTh1 training/control runs, three native COFE component seeds, three trained
checkpoint objective checks, and 18 controlled Gaussian counterfactual cases.
Interrupted first attempts remain separate and are excluded from completed fits.

- KUMA: test MSE {kuma['mse']:.9f}, MAE {kuma['mae']:.9f}, {len(epochs)} epochs,
  validation-selected checkpoint; unmodified released engine, no batch caps.
- COFE: a free noise-estimator head can be zeroed after training without changing
  predictive distribution parameters. This lowers its native objective.
  Correcting a dispersion mapping removes synthetic level-shift sensitivity.
  These establish objective/implementation limitations, not a novel method.
- Counterfactual pilot: ordinary Gaussian conditioning and generic SDP achieve
  the frozen 1% gap criterion in all 18 cases; maximum relative gap
  {maximum_gap:.6g}. Stop custom solver development in this restricted regime.

Read `experimental_results.md` for numbers and protocol boundaries,
`negative_evidence.md` for stopped claims, and `final_research_direction.md` for
the current decision. No paper-table exact replication or top-conference-ready
research direction is claimed.
''')
 write('baseline_registry.md',f'''# Baseline and data registry

| Model/source | Pinned revision | Execution status and interpretation |
|---|---|---|
| COFE, polars8948/COFE | a53409aa6d97c65b5b36b7f6b27fc5fbdb5e3df1 | Full {cofe[0]['parameters']}-parameter native model/loss; three 50x100-update ETTh1 runs. CLOSE source/config reproduction plus two private implementation controls. Manual uniform-origin sampler replaces the GluonTS iterable; not paper-table MATCHED. |
| KUMA, SuperBearcafedo/Review, KUMA-Official | 8ed21081366309e313f821f3dc760f6ac945ac4b | Complete disclosed PEMS03 96-to-12 full configuration; unmodified engine/model/data/evaluation, seed2025, width256, max20, early stop14. Worker0 replaces launcher10; pinned single-thread runtime. Source-faithful released-configuration reproduction. |
| Meta-iLaD | 0bc016a51b7ddb5f23b143a3fc01d2e34a1f92cd | Public repository has README/papers, zero Python source; explicitly says code forthcoming. No reproduction claimed. |
| TIME | 912291c69045d369ac45fcd5e96ecf60398c092a | Code availability archived (34 Python files). Zero-shot forecasting benchmark; no new model execution here. |

PEMS03 comes from the pinned KUMA repository, 15,800,415 bytes, SHA256
34859517425eded66a0014db63de54b95d0428c38890741a1710ed4d3b043de8.
Native data shape (26208,358,1); first feature, chronological60/20/20,
per-sensor scaler fitted on training only. Windows train15617, val5135, test5135.

ETTh1 comes from zhouhaoyi/ETDataset revision
1d16c8f4f943005d613b5bc962e9eeb06058cf07, 2,589,657 bytes, SHA256
f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066.
The source Git blob hashes and paths are in `outputs/dataset_registry.json`.
Both files were fetched locally after server raw-GitHub refusal, then uploaded
with SHA verification; no synthetic substitution or TLS bypass.

The KUMA README reference is MSE0.063/MAE0.165. Our result is close to those
rounded repository values. The PMLR landing page was verified, but its linked
paper PDF was not successfully archived; this is not independent verification
of every paper-table setting. [Primary landing page](https://proceedings.mlr.press/v306/xiong26j.html).

The continuous counterfactual pilot is newly designed with known kernels;
it is neither an existing benchmark score nor a learned-world-model comparison.
''')
 table='\n'.join(f"| {r['variant']} | {r['selected_epoch']} | {r['best_val_native_raw_CRPS']:.6f} | {r['test']['0.0']['normalized_MSE']:.6f} | {r['test']['0.0']['empirical_CRPS_normalized']:.6f} | {r['test']['0.0']['coverage90']:.6f} |" for r in cofe)
 stress='\n'.join(f"| {r['variant']} | {offset} | {r['test'][str(offset)]['coverage90']:.6f} | {r['test'][str(offset)]['interval90_width_normalized']:.6f} | {r['test'][str(offset)]['empirical_CRPS_normalized']:.6f} |" for r in cofe for offset in [-3.,0.,3.])
 write('experimental_results.md',f'''# Executed experiments and metrics

## KUMA: complete released-code configuration

PEMS03 96-to-12, seed2025, full width256/FF256/one layer, batch32, Adam0.002,
learning rate halves each epoch, patience3/max20, workers0, no batch caps.
Training completed {len(epochs)} epochs, best logged validation at epoch11.
The native engine reloaded its validation-selected checkpoint before testing.
Test MSE **{kuma['mse']:.12f}**, MAE **{kuma['mae']:.12f}**, 5135 windows.
One seed; a healthy reproduction, not a new mechanism result.
`outputs/kuma_native_pems03_affinity1/` stores raw log, argv, checkpoint, native
unrounded `results.jsonl`, process metadata and successful exit status.

## COFE: full native-model reproduction and implementation controls

All three use 638917 parameters, ETTh1, context24/horizon24, seed1, Adam0.001,
50 epochs x100 updates, batch48, clip0.5. Training-only global scaler; fixed
train/val/test endpoints8640/11520/14400; 30 validation and30 test windows,
native rolling stride96. Origins are sampled uniformly on native support192..8616,
a declared difference from the GluonTS sampler. Native loss bodies, native
100-sample standard-normal latent-prior generation and quantile scores retained.
Validation raw-unit19-quantile score alone selects the checkpoint; no test tuning.

| Variant | Selected epoch | Best validation native CRPS | Test normalised MSE | Test empirical CRPS | Test90% coverage |
|---|---:|---:|---:|---:|---:|
{table}

`scale_fix` applies softplus in normalized coordinates then multiplies the
context standard deviation and divides by absolute affine scale; it never adds
a context mean to dispersion. `scale_fix_noaux` additionally sets the two
auxiliary weights to zero, with unchanged architecture/data/update budget.
One seed and one dataset do not support a stable improvement or an ablation
claim that the auxiliary term never helps. MSE and probabilistic scores trade off.

The following stress adds a fixed normalized level offset to both history and
target after checkpoint selection. This is synthetic real-series stress, not a
causal intervention or full-pipeline unit-conversion claim. Original-unit
normalized CRPS has a changing absolute-target denominator under this stress;
use standardized empirical CRPS and coverage/width for location comparisons.

| Variant | Offset | Coverage90 | Interval90 width | Empirical CRPS |
|---|---:|---:|---:|---:|
{stress}

Three component seeds41/42/43 passed finite native forward/loss/backward checks.
Zeroing the unconstrained noise-estimator parameters leaves forecast loc/scale
exactly unchanged and reaches the auxiliary regularizer minimum. The core
gradient equals its no-auxiliary gradient after zeroing. This is also confirmed
on all three trained checkpoints using training samples only; released loss
1.020407438 ->1.020308495, distribution-parameter change exactly0. It undermines
mechanism attribution through this auxiliary objective; it is not a new method.
Native num_samples10 returns component locations, unlike100 observation draws.
The native default100 used above is unaffected by that special10 branch.

## Continuous partially observed counterfactual computation

18 frozen cases: d2/3/5 x T64/256 x independent system seeds61/62/63.
The two horizons are nested within each system, not independent replications.
Known two-action linear Gaussian transition/measurement model; one global
time-invariant R in O(d), both determinant components, identical factual and
interventional transition kernels across the restricted SCM family. Factual
observations are noisy/partial. Fixed open-loop alternative action plan; query
is conditional terminal squared error. This is not a bound over all SCMs.

The terminal query is affine in vec(R) with Gaussian random coefficients.
An ordinary augmented Kalman filter obtains their posterior mean/covariance.
Independent dense Gaussian conditioning agrees within {summary['risk_max_dense_check_error']:.3g};
20 direct twin-state Kalman checks/case agree within {summary['risk_max_twin_check_error']:.3g}.
Both O(2) components have an independent all-critical-roots cross-check.
Existing Shor SDP with row/column orthogonality constraints plus feasible
multistart/polar rounding gives maximum absolute gap {maximum_abs:.9g}, maximum
relative gap {maximum_gap:.9g}; all36 extrema pass the frozen1% gate.
Maximum numerical constraint violation {maximum_residual:.3g}. Tiny signed
gap reversals down to {min(signed):.3g} are solver tolerance effects, not formal
certificates. The SDP numbers are **not interval-arithmetic-validated bounds**.

Decision: **{risk['decision']}**. Existing tools already resolve this small
regime. No learned-world-model, nonlinear-system or globally sharp
nonparametric counterfactual claim is supported.

## Runtime and implementation failures

The first unpinned KUMA attempt exited3221225477 (Windows0xC0000005) after epoch12.
The first COFE attempt disappeared after epoch27; its precise exit code was not
captured. Neither incomplete attempt contributes to the completed comparisons.
Their logs/checkpoints are preserved. No contemporary exhaustion event was found;
host application-error events include native-library faults, without proving
the physical cause. No unrelated job or global system setting was changed.

Process-local affinity0xffff0000, one CPU thread, MKL sequential were used for
new independent retries. A45-second arithmetic diagnostic passed15672 loops.
The complete four fits then exited normally. The first {len(old_epochs)} KUMA epochs
match the interrupted attempt's printed training/validation metrics exactly.
This is bounded evidence of a working profile, not proof of permanent stability.

The first counterfactual pilot failed its independent O(2) assertion because its
feasible optimiser's ascent/descent signs were reversed. Corrected before any
accepted row; scientific plan/seeds/stopping gate stayed fixed. The failure is
retained in `continuous_counterfactual_implementation_failure.md`.
''')
 write('negative_evidence.md',f'''# Rejected explanations and stopped directions

1. **COFE forecasting improvement demonstrates successful scale disentanglement.**
   Not established. Native auxiliary noise estimates can be set to zero while
   leaving predictive distributions unchanged, both at random initialization
   and trained checkpoints. A decoder dispersion error is an alternative
   explanation for calibration under level shifts. Do not treat these source
   findings as a new architecture contribution or universal identifiability result.
2. **The source variance repair is a novel research mechanism.** Rejected.
   Location/scale equivariance is ordinary probability calculus. The single-seed
   real-data comparison demonstrates a concrete implementation repair only.
3. **COFE has an invalid posterior/prior mismatch.** Rejected as a diagnosis.
   Context-conditioned variational latents and standard-normal generative priors
   can be valid. The native sampler was retained; no unsupported prior replacement.
4. **The special10-sample branch invalidates native100-sample results.** Rejected.
   We checked the branch, but evaluated100 samples as configured.
5. **KUMA fails as an exact fixed Koopman operator.** Inadmissible strawman.
   Its released README explicitly calls the mechanism input dependent and
   Koopman inspired. Full released training is healthy, MSE{kuma['mse']:.9f}.
6. **Continuous partial observability necessarily needs a new counterfactual
   solver even for the small shared-Gaussian-alignment family.** Rejected in
   the frozen18 cases. Existing Gaussian inference and generic SDP satisfy the
   precision gate; maximum relative gap{maximum_gap:.9g}. Do not repeatedly
   tune this solved toy to create a new-method story.
7. **Smoothing once across causal models or using PSD moment bounds is new.**
   Rejected by Haugh/Singal2023 and established SDP/moment methods. Generic
   WorldKernel discussions overlap with PSD partial identification too.
8. **A speculative triangular coarse/fine transform refutes COFE's theorem.**
   Not established or executed. Initial-law, learned-class and conditional
   density assumptions must be reconciled with the unavailable full proof;
   a thought experiment is not a theorem counterexample.
9. **The runtime crashes demonstrate model weakness or out-of-memory.**
   Rejected. They are native-process execution failures, retained separately.

No accepted research direction results from these rejections. This file records
negative scientific decisions; it does not declare the autonomous goal complete.
''')
 write('literature_review.md','''# Literature and novelty boundary: cycle5

Primary paper bytes and SHA256 are in `outputs/literature/registry.json`.
This extends previous cycles; it is not a replacement for their evidence.

| Primary source | Relevant boundary |
|---|---|
| [COFE, IJCAI2026](https://www.ijcai.org/proceedings/2026/0602.pdf) | Coarse/fine decomposition and innovation sparsity already studied. Its theory treats coarse evolution as deterministic; smooth positive-density wording explicitly concerns the fine process. The archived nine-page PDF refers to an unavailable full Appendix B. Do not assert an unverified theorem counterexample. Native objective/scale findings here come from independent code execution. |
| [KUMA, ICML2026](https://proceedings.mlr.press/v306/xiong26j.html) | Input-dependent Koopman decomposition plus multilevel attention already proposed. This cycle reproduces the released full PEMS03 configuration; not an exact fixed-operator identification model. Linked paper PDF retrieval did not succeed; do not hide that boundary. |
| [Haugh and Singal, ICML2023](https://proceedings.mlr.press/v202/haugh23a/haugh23a.pdf) | Dynamic hidden-state counterfactual bounds already optimize compatible causal mechanisms and reuse factual posterior samples across candidates. Their implemented setting is finite-state; posterior reuse alone is not a continuous-state contribution. |
| [Wasserstein DR Kalman filtering, NeurIPS2018](https://proceedings.neurips.cc/paper/2018/file/15212f24321aa2c3dc8e9acf820f3c15-Paper.pdf) | Robust Gaussian estimation and convex reformulations are established. Its ambiguity set differs from action-indexed cross-world coupling; do not equate the estimands or claim Gaussian robust inference is new. |
| [Orthogonal trace-sum SDP, 2021preprint/2022SIAM](https://arxiv.org/abs/2110.05701) | Orthogonal quadratic relaxations, tightness and local algorithms already studied. The exact model differs from our arbitrary single-matrix quadratic, so its theorem is not blindly transferred; generic SDP is tested directly. |
| [Improved approximation algorithms,2025](https://arxiv.org/abs/2501.02942) | Vectorized orthogonal quadratic SDP and rounding are mature controls, not a new world-model mechanism. Primary text screened; PDF not part of the archived11. |
| [WorldKernel,2026June](https://arxiv.org/abs/2606.10934) | PSD moment constraints for binary potential outcomes overlap a generic coupling-bound narrative. Not a continuous-state shared-rotation solver; no numerical or theorem equivalence claimed. |
| [Robust PNSCM counterfactual policy,2026Aug](https://arxiv.org/abs/2608.02893) | Time-homogeneous causal sensitivity and retrospective policy bounds already studied in discrete MDPs; actualized stochastic semantics differ from deterministic shared-noise SCMs. |
| [iVDFM,2026March/July](https://arxiv.org/abs/2603.22886) | Context-conditioned innovation identification with diagonal dynamics already proposed. Conditional-density identification is not automatically realized-state or arbitrary intervention identification. No native reproduction here. |
| [Physical parameter identifiability,2026July](https://arxiv.org/abs/2607.27017) | Recoverability certificates, objective-versus-information attribution, modality/horizon controls already studied. Generic latent-probe failure is not novel. The primary paper says code forthcoming. |
| [Counterfactual flow matching,2026Oct](https://arxiv.org/abs/2610.01193) | Coupling-sensitive generation error is existing work. Generation under a chosen source-target coupling differs from sharp dynamic partial identification; not interchangeable. |
| [OSSM,2026September](https://arxiv.org/abs/2609.33566) | Observer-based state estimation and shared transition dynamics already proposed. Do not rename observer/transition separation as new. |
| [Pythia,2026October](https://arxiv.org/abs/2610.05240) | Decoupled latent forecasting and probabilistic readout is existing work; code/checkpoint release is not verified available. |
| [Meta-iLaD repository](https://github.com/GoForIt-YYB/Meta-iLaD) | Joint state/environment/dynamics identification through meta-learning already claimed. Pinned repository has zero Python files, explicitly code forthcoming; not executable evidence. |
| [TIME repository](https://github.com/zqiao11/TIME) | Task-centric fresh-data zero-shot evaluation, not a counterfactual truth benchmark. Source archived; no arbitrary model migration mislabeled original reproduction. |

Further primary scope checks, not executed or accepted directions:
[CHiLD,NeurIPS2025](https://arxiv.org/abs/2510.18310) identifies hierarchical
latent dynamics using temporal context and sparsity, with code linked in paper.
[Stochastic multiscale normalizing flows,May2026](https://arxiv.org/abs/2605.09718)
studies effective slow dynamics under latent fast processes.
[Bayesian Langevin comparison,Aug2026](https://arxiv.org/abs/2608.00846) studies
sampling-dependent model discrimination. Therefore a generic claim that
slow/fast stochastic dynamics or sampling identifiability are unstudied is false.
Their full assumptions/code require further review before choosing a continuation.

No private experimental details were posted publicly. Public retrieval queries
used public paper titles/topics, not unpublished result numbers or private paths.
''')
 write('hypothesis_space.md','''# Competing hypotheses and falsification

| Hypothesis | Decisive existing evidence/test | Decision |
|---|---|---|
| H5.1: COFE's native auxiliary objective uniquely teaches coarse/fine mechanisms | Zero the free auxiliary head; measure unchanged distribution and unchanged core gradients. Repeat at full trained checkpoints. | Attribution rejected; auxiliary optimum is independent of forecasting parameters. No claim that every trained representation is bad. |
| H5.2: stochastic hidden-state forecasting necessarily lacks calibrated variance | Repair only mathematically incorrect dispersion mapping, same model/data/budget. | Synthetic shift defect repaired by an ordinary implementation change; no general new-method case. |
| H5.3: apparently healthy KUMA scores hide incomplete training | Execute full released engine, no batch caps, validation selection, native test. | Healthy completed reproduction. Cannot infer physical mechanism fidelity from its MSE. |
| H5.4: a new continuous partially observed counterfactual solver is needed in a small Gaussian alignment class | Ordinary augmented Kalman sufficient statistics + existing generic SDP and feasible witnesses;18 frozen cases. | Rejected in tested regime; all satisfy1% gate. |
| H5.5: a residual nonlinear/hierarchical mechanism gap survives existing methods | First audit CHiLD/Meta-iLaD/observer and multiscale literature, define a recoverability certificate and strongest ordinary generative/filtering controls. | UNTESTED. No chosen architecture or new-method claim. Do not escalate Gaussian toy complexity merely to manufacture failure. |

Future hypotheses must specify an actual task, target query, admissible input
information, independently recoverable signal, and an established repair baseline.
Do not infer a gap from a source bug, missing code, lower MSE or one probe failure.
''')
 write('research_ideas.md','''# Candidate research questions after cycle5

**Accepted directions: none.** No top-conference potential is currently established.

Two candidate routes were explicitly stopped: (i) mathematical variance-scale
repair or auxiliary-head collapse as a new architecture; (ii) custom numerical
certification for the small partially observed Gaussian coupling family.
They have concrete evidence but insufficient novelty/residual need.

One screening priority remains, not a recommendation: determine whether a native
hierarchical stochastic world model has an experimentally recoverable,
decision-relevant failure after ordinary filtering, likelihood learning,
temporal-context and model-budget controls. Inspect CHiLD and observer models
alongside recent multiscale work first. Identifiability of a distribution,
recovery of a realized hidden state, and online belief inference are different
tasks. A training smoother using future context is not automatically a leak or
an invalid forecasting model. Establish the actual native deployment contract.

Reject this screening priority if existing filtering/generative models resolve
the controlled task at comparable information, capacity and computation, if the
signal is unavailable from admissible history, or if closest prior work already
solves the exact claim. No method name, architecture or positive result is invented.
''')
 write('final_research_direction.md','''# Current research decision

**No final direction has met the required evidence and novelty standard.**
The autonomous objective remains ACTIVE. Cycle5 is a completed evidence checkpoint.

**Research Question.** Which concrete deficiency of a time-series world model
survives healthy native execution, correct probabilistic computation, and strong
existing inference/optimization controls? No surviving precise claim is yet fixed.

**Evidence.** Full released KUMA training is healthy. Full COFE-model training
exposes objective and dispersion implementation explanations. The18-case
continuous partially observed Gaussian pilot is handled by existing generic tools.
These findings eliminate candidate explanations; they do not prove an unsolved gap.

**Existing Methods.** Ordinary probability-scale transformations, generative
likelihood/filtering, reused-posterior counterfactual inference, Gaussian
sufficient statistics and orthogonal quadratic SDP already cover the tested repairs.

**Proposed Idea.** None endorsed. Treat a native hierarchical stochastic-model
deployment audit as the next screening target, conditional on a precise novelty
boundary and recoverability controls. Do not rename known techniques.

**Falsification.** Stop any candidate if ordinary existing controls solve its
declared task at matched input/capacity/compute, the information is unavailable,
the native model does not actually make the tested claim, or closest prior work
already solves it. Gaussian custom solver and variance-fix novelty have stopped.

**Feasibility.** The single RTX5090 completed full source-model runs in minutes;
small numerical experiments run in an isolated NumPy/SciPy/CVXPY environment.
Current data/checkpoints/source pins/logs are reproducible and preserved.

**Next Experiment.** Audit the actual CHiLD hierarchical generator and its
inference/deployment interfaces against recent observer/meta-learning/multiscale
work. Only after a distinct residual is identified, freeze an independent
controlled cohort and compare the native full model with an ordinary probabilistic
filter/generative control and a signal-recoverability certificate. No long run or
complex architecture is justified before that gate.
''')
 write('CONTINUATION.md','''# Continue autonomous discovery

Cycles1-4 remain preserved. This cycle has four completed trained configurations
(one KUMA, three COFE), component/trained-checkpoint attribution,18 new Gaussian
cases, and retained execution failures. No remaining live scientific job is
required for these completed results. All accepted-directions lists are empty.

Do not repeat these solved routes: free auxiliary-noise-head collapse as novelty,
location/scale correction as novelty, reused-posterior Gaussian counterfactual
optimization, or tiny shared-rotation solver tweaks. Their evidence is negative.

Before any next training, inspect source availability and native deployment
contracts of hierarchical/observer models, with CHiLD as a new code-audit target.
Meta-iLaD has no released Python code at its pinned revision. TIME is zero-shot
forecasting, not a causal intervention truth source. Return to broad discovery
if no precise residual survives existing methods; do not force an idea.

Server profile that passed completed retries: process-local affinity0xffff0000,
one CPU thread, MKL sequential, GPU5090. This does not prove hardware stability.
Preserve failed first attempts; never restart an existing live process from stale
status alone. Use a new cycle directory after sealing this one.
''')
 task={'objective_status':'active','objective_achieved':False,'accepted_research_directions':[],'cycle5':{'scientific_jobs_complete':True,'completed_native_training_configurations':4,'controlled_counterfactual_cases':18,'failed_first_attempts_retained':True,'all_new_results_private':True},'next':'Inspect actual hierarchical/observer model contracts and closest prior work; no further Gaussian toy solver tuning'}
 (ROOT/'outputs/task_status.json').write_text(json.dumps(task,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
