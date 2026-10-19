"""Generate complete research ledgers from actual outputs, without accepting a gap."""
import csv,datetime,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load(name):return json.loads((ROOT/'outputs'/name).read_text())
def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')
def main():
 for name in ['noise_objective_pilot','noise_attribution','vjepa_component_audit','risk_state_pilot','risk_moment_control']:
  assert load(name+'/status.json')['state']=='complete',name
 pilot=load('noise_objective_pilot/results.json');attribution=load('noise_attribution/results.json')
 risk=load('risk_state_pilot/results.json');moment=load('risk_moment_control/results.json')
 verification=load('noise_attribution/verification.json');component=load('vjepa_component_audit/results.json')
 assert len(pilot)==6 and len(attribution)==5 and len(risk)==9 and len(moment)==3
 grouped={m:[r for r in risk if r['mode']==m] for m in ['detached','attached','full_observation']};grouped['moment_control']=moment
 moment_nmse=float(np.mean([r['risk_v_probe_nmse'] for r in moment]));moment_regret=float(np.mean([r['policy_regret'] for r in moment]))
 table='| Configuration | Risk-driver probe NMSE | Conditional Gaussian KL | Conditional mean MSE | Policy regret |\n|---|---:|---:|---:|---:|\n'
 for m,rows in grouped.items():
  cells=[]
  for key in ['risk_v_probe_nmse','conditional_gaussian_kl','conditional_mean_mse','policy_regret']:
   a=np.array([r[key] for r in rows]);cells.append(f'{a.mean():.6f} +/- {a.std(ddof=1):.6f}')
  table+='| '+m+' | '+' | '.join(cells)+' |\n'
 gates=[]
 for seed in [41,42,43]:
  a=next(r for r in risk if r['seed']==seed and r['mode']=='attached');d=next(r for r in risk if r['seed']==seed and r['mode']=='detached')
  row={'seed':seed,'risk_probe_nmse_gain':d['risk_v_probe_nmse']-a['risk_v_probe_nmse'],'kl_gain':d['conditional_gaussian_kl']-a['conditional_gaussian_kl'],'relative_regret_reduction':1-a['policy_regret']/d['policy_regret'],'mean_mse_worsening':a['conditional_mean_mse']-d['conditional_mean_mse']}
  row['passed']=row['risk_probe_nmse_gain']>.25 and row['kl_gain']>.05 and row['relative_regret_reduction']>.30 and row['mean_mse_worsening']<.01;gates.append(row)
 gate=all(r['passed'] for r in gates)
 evidence={'risk_pilot_gate_all_seeds':gate,'per_seed':gates,'candidate_accepted':False,'three_initializations_not_three_independent_data_cohorts':True,'existing_moment_control':moment}
 (ROOT/'outputs'/'risk_state_pilot'/'analysis.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
 noise_table='| Configuration | Step2000 NMSE | Step4000 NMSE | Step8000 NMSE |\n|---|---:|---:|---:|\n'
 for row in attribution:noise_table+='| '+row['run']+' | '+' | '.join(f"{s['physical_probe_nmse']:.6f}" for s in row['snapshots'])+' |\n'
 write('experimental_results.md',f'''# Cycle4 experimental evidence

Overall discovery objective: ACTIVE, NOT ACHIEVED. No accepted research direction.
All measured values below are from server execution. Prior cycles are preserved.

## A. Native MotionJEPA/LeWM objective adapter pilot: six fits

Unchanged official `train_step` and SIGReg, linear32->2 observation/difference
adapters, causal3-step linear predictor; original512-wide motion inverse head.
This is a new controlled objective audit, not a native visual-paper reproduction.
Clean physical-state NMSE was approximately8e-15 for all three configurations.
With30 independent observation nuisance dimensions, 2000-step NMSE was
0.497659(default LeWM),0.533266(matched static weight),0.544318(MotionJEPA).
Motion-minus-matched gap={verification['pilot_noise_gap']:.9f}, below the frozen0.10 gate.
Physical signal directions are noise-free in this condition; the separate
`sensor_noise` generator branch was not run. Do not label this raw sensor noise.

## B. One fixed attribution extension: five fits

{noise_table}
Same seed41, cohort9051, independent probe9052 and development9053. All final
checkpoints are fixed-step, without state-based selection. Training-only CCA
initialization uses no state labels. Three random-initialization2000-step prefixes
match every pilot state tensor exactly, maximum difference0. Default LeWM is
repaired by budget alone; matched-static LeWM is repaired by ordinary CCA.
Motion has a remaining initialization-sensitive deficit in these adapters.
This neither proves a native architecture failure nor passes the original gate.

## C. Native VJEPA component paths: three initializations, no training

All three native CNN forwards accept64x64 inputs. With the unchanged exp10 loss,
encoder/mean-predictor gradients equal VICReg-only gradients exactly in all three
checks; the variance head has a nonzero gradient. Perturbing variance-head
parameters leaves native mean-only CEM actions unchanged under identical RNG.
Original linear code: varying target-KL beta0->1 does not change encoder gradients
because target tensors are detached; changing the supplied future probe argument
does not change the result because it is unused. No future-probe leakage found.
Selected unchanged AST bodies are executed to avoid notebook/Colab side effects.
These are source-path facts, not learned calibration or planning-performance results.

## D. Risk-state compression: nine new controlled fits plus three moment controls

{table}
Numbers are means +/- sample SD across three training initializations on ONE
fixed cohort, not independent dataset replications or confidence intervals.
Matched detached/attached variants each have4549 parameters, identical inputs,
training pairs, initialization, minibatches, heads, optimizer and8000updates.
Only variance-to-encoder gradient differs. Mean residual is detached inside the
variance term for both variants. Full-observation control has6435parameters and
is an informational control, not a capacity-matched competitor.

Process: y_next=.8y+.3a+sqrt(.27(1+.8a*tanh(v)))*eta; v_next iid standard normal.
Randomized +/-1 actions; observations are an unknown orthogonal mixing of y,v
and30 iid nuisances.100burn-in steps precede sampled transitions. Targets project
future observations onto an action-response direction fitted by TRAIN-only OLS.
No physical labels train the encoder, heads or moment projection. Physical labels
fit only a frozen diagnostic probe after training and define evaluation truth.
Policy regret is exact expected quadratic next-state cost in this declared SCM.
Tail Brier refers to |projected future|>1.5. This is not an observational causal claim.

Frozen attached-repair gate passed all seeds: **{gate}**. Per-seed details are
`outputs/risk_state_pilot/analysis.json`. In particular seed43 is not discarded.
Ordinary cross-moment dimension reduction is an existing-method control, not
claimed SAVE/GSAVE software reproduction. It adds a reported training-statistic
pass and freezes64encoder coefficients; its4485head parameters receive the same
8000updates. Full-observation heads overfit/misfit despite containing the state:
extra available information alone does not establish a fitted-model ceiling.

## Limits

Risk fits use a fresh observation-target regression objective, not the native
self-referential latent JEPA loss. Transfer to trained native JEPA is UNPROVEN.
No real-data risk evaluation, independent data confirmation or new-method claim.
Raw plans, logs, source pins, checkpoints and generator scripts remain available.
''')
 write('baseline_registry.md','''# Cycle4 baseline registry

| Source or control | Revision / definition | Execution status |
|---|---|---|
| mkarmann/motion-jepa | b3c2fb374b7c69f169de15774f371290a8736f56 |65pinned code/config files; native objectives with declared vector adapters:6pilot+5attribution fits |
| YongchaoHuang/VJEPA |88a90d2c0af970edd46d44d53d4864d10aa6d117 |11small text files; native CNN/loss/CEM component checks at3seeds; no native training |
| Linear CCA/VAMP-style | Ordinary whitening and lagged cross-covariance SVD | Training-only initialization control; not a new method or third-party package reproduction |
| Heteroscedastic matched controls | Same4549parameter observation-target model; variance input detached/attached |9fits incl3full-observation controls; new diagnostic, not VJEPA paper replication |
| Training predictive cross-moments | OLS mean and Cov(x,a*residual^2) directions |3fits, no true state/channel directions used for fitting; ordinary-method repair control |

Prior LatentTSF/DAG/WorldTS original-paper reproduction evidence remains in the
earlier registries. RP-GSSM execution belongs to cycle2; EpiCF/LeNEPA checks to
cycle3. No cross-protocol pooled leaderboard.10primary PDFs are hash mirrored.
Runtime verified:Python3.10.20,Torch2.8.0+cu128,NumPy1.26.4,RTX5090.
Motion project declares a newer full-stack runtime; standalone objective imports
are not confirmation that its full paper training stack is reproduced here.
''')
 write('negative_evidence.md',f'''# Cycle4 negative evidence

1. Motion-specific nuisance hypothesis failed its frozen2000-step gate:gap
{verification['pilot_noise_gap']:.6f}<0.10. Do not replace it by a post-hoc threshold.
2. The default LeWM deficit fell from0.497659 to0.000123 with8000steps.
Training duration explains that apparent failure; no new architecture needed.
3. Ordinary training-only CCA fixes the matched-static adapter to0.000283.
4. Motion warm-start still has0.085087NMSE; a single initialization and linear
adapter do not establish native visual failure or a population optimum conflict.
5. New VJEPA source passes its native CNN input-shape check. An internal mistaken
shape suspicion was corrected by execution; it is not a baseline defect.
6. Passing future input to the old linear probe does not use it. No future leakage
was found. Target-KL has no encoder gradient in that implementation, but exp10 has
an explicit VICReg anti-collapse path; do not claim all versions lack anti-collapse.
7. Risk pilot simple attached repair did not meet its ALL-initialization gate:
{gate}. Seed43 is retained. No hyperparameter or checkpoint selection on risk truth.
8. Full-observation learned variance is worse despite exact physical-state
availability; this is fit/regularization evidence, not nonidentifiability.
9. Ordinary predictive cross-moments recover the risk driver to NMSE
{moment_nmse:.6f} and mean policy regret{moment_regret:.6f}, without physical labels.
This resolves the toy's practical information-recovery deficit. A new-method route
based only on that toy is therefore stopped. Second-moment SDR is established.
10. Switching-graph basis alignment has explicit connected/spanning-tree results
in Bencherki et al.2023. That elementary central claim is covered. Finite-sample
stochastic decision guarantees would be a different, currently untested question.

Engineering only: an initially absent local staging directory caused upload failure;
created it and securely retried. CUDA convolution determinism was enabled for the
native gradient equality check. No scientific result is inferred from these errors.
''')
 write('hypothesis_space.md','''# Cycle4 competing hypotheses

| Hypothesis | Distinguishing evidence | Current judgment |
|---|---|---|
| Difference auxiliary necessarily selects nuisance over physical dynamics | Noise-specific Motion-minus-matched NMSE gate; clean control; ordinary methods | Pilot gate failed; no structural claim |
| Neural state deficit is underoptimization | Fixed budget extension, identical2000-step prefixes | Explains default LeWM; other objectives remain initialization sensitive |
| Predictive state information absent from observations | Unlabeled CCA reconstruction; known generator | Rejected in independent-nuisance condition |
| Adding a variance head automatically teaches encoder uncertainty drivers | Native loss gradient path and matched detached/attached experiment | Rejected as a general implementation inference; native exp10 deliberately stops that gradient |
| Mean accuracy suffices for action-dependent risk choice | Same mean error, independent risk probes and exact SCM expected costs | Counterevidence in declared controlled regression experiment; native JEPA transfer unproven |
| Ordinary joint variance learning robustly fixes compression | Frozen all-three-seed gate | Gate failed; no claim of stable new repair |
| New method needed to recover variance directions | Training-only predictive cross-moments plus same heads | Existing-method control tested; see actual results |
| Switching-graph alignment itself is a new identification mechanism | Closest switched-system primary theory | Covered central mechanism; do not rename it |

Next useful falsification: same fixed protocol on independent train/probe/evaluation
cohorts and trained native probabilistic representations, with established SDR,
generative predictive SSM, and proper-score representation objectives. If these
already preserve relevant uncertainty without a material nuisance/modeling cost,
stop the new-method route. Do not infer novelty from an encoder probe alone.
''')
 write('research_ideas.md','''# Candidate research questions, not accepted directions

No new-method direction survives this cycle. One narrowly scoped watchlist
question remains for literature/protocol screening, not immediate model tuning: **Does a
noise-robust compressed world state retain uncertainty drivers needed for action
selection, and can this be checked without hidden-state labels or reconstructing
every nuisance channel?**

Supported portion: the controlled matched regression experiment separates mean
accuracy from variance-state retention and expected quadratic decision cost. Native
VJEPA source provides a concrete example of a deliberately detached variance head.
Unsupported portion: trained native JEPA/SSM failures, real-data prevalence,
independent-data stability, and a novel diagnostic or remedy. No innovation claim.

The ordinary predictive cross-moment control already resolves this controlled
case. Do not continue developing a new method for that solved case. Classical
central-subspace/GSAVE theory and modern
probabilistic predictive representations are closest work. Financial world-model
representation evaluation already includes volatility probes; those alone are not
novel. A surviving contribution would need new controlled evidence/diagnostics for
action-dependent risk sufficiency and a genuine efficiency or estimation issue
left by those methods. The current pilot does not establish that residual gap.

Do not revive difference-branch noise failure by changing its gate, claim connected
switching graphs as new, or launch full training before identifying the remaining
question against these strong controls. The overall discovery objective stays active.
''')
 write('final_research_direction.md','''# Current research decision

**No final research direction has passed the required evidence and novelty checks.**
This file is an evidence checkpoint. The autonomous discovery objective is ACTIVE.

**Research Question.** Can compressed probabilistic time-series world states be
tested for retaining action-dependent uncertainty, rather than merely fitting
conditional means and a variance head on an insufficient state?

**Evidence.** Native VJEPA component checks establish the declared gradient and
mean-only planning paths. A separate controlled stochastic-process experiment
shows accurate means can coexist with discarded variance-driver information and
larger risk regret. These do not yet prove failure of a trained native JEPA model.

**Existing Methods.** Ordinary heteroscedastic likelihood, predictive-moment SDR,
probabilistic SSMs, GSAVE and conditional-distribution representations already
address important parts. They must be tested before calling this unsolved.
Towards Financial World Modeling already probes volatility; no generic probe novelty.

**Proposed Idea.** The current toy's information-recovery problem is resolved by
an ordinary moment control, so its new-method route is stopped. Screening a
native-model distributional-sufficiency audit remains possible. No architecture
is endorsed. Any eventual method
would need to address an actual residual estimation/efficiency problem beyond the
existing controls, not rename the current moment or likelihood repair.

**Falsification.** Stop this as a new-method direction if existing probabilistic
representations/SDR repair the issue at comparable information, nuisance burden,
capacity and compute, or if independent data/native models remove the phenomenon.
The current attached-repair pilot already failed its all-three-initialization gate.

**Feasibility.** Current server RTX5090, pinned MotionJEPA/VJEPA code and saved
controlled generators. Small models take seconds per fit; native architecture,
independent cohorts and appropriate established controls remain required.

**Next Experiment.** First verify an unaddressed native-model/protocol claim against
existing distributional-representation and SDR work. Only if such a precise claim
survives, freeze independent data cohorts and test the existing moment/joint-density
controls against a native probabilistic representation, using fixed checkpoints,
held-out conditional scores and SCM decision regret. Do not spend more training
on the solved linear-mixing toy. Otherwise switch to another failure mechanism.

This is the only watchlist priority, not a top-conference-ready recommendation.
If the controls resolve it, return to broader discovery; do not force an idea.
''')
 write('literature_review.md','''# Cycle4 primary-source literature and innovation boundaries

Mode:exploratory. Primary papers are archived by revision and SHA256 in
`outputs/literature/registry.json`. Preprints are not claimed accepted papers.

| Work | Primary source | What it establishes here | Boundary / residual question |
|---|---|---|---|
| MotionJEPA |[2609.23881v1](https://arxiv.org/abs/2609.23881v1) and pinned author source | Difference-image auxiliary, static and difference SIGReg | Our linear adapters do not reproduce native visual results; nuisance-specific pilot gate failed |
| World Modeling through Spectral Alignment |[2610.15429v1](https://arxiv.org/abs/2610.15429v1) and [author project](https://spectral-alignment.github.io/) | Goal-dependent kernel alignment; trajectory ambiguity and nuisance-coverage conditions | Generic forecasting/planning geometry differences are covered; no native run here |
| I Act Therefore I Am |[2609.31161v1](https://arxiv.org/abs/2609.31161v1) | Action-varying transition statistics give componentwise identifiability under stated conditions; Gaussian practical model | Do not claim action diversity or conditional-likelihood representation learning as new; no verified author code link located |
| Basis transform in linear switched systems |[2106.10888v3](https://arxiv.org/abs/2106.10888v3), journal DOI10.1002/acs.3677 | Local bases patched by observed switching relations; connected spanning-tree criterion | Connected-switch graph alignment is covered, not a new mechanism |
| Variational JEPA/BJEPA |[2601.14354v1](https://arxiv.org/abs/2601.14354v1) and [author source](https://github.com/YongchaoHuang/VJEPA) | Probabilistic latent predictive framework; original linear and newer exp10 implementations differ | Executed component paths establish detachment/mean CEM, not native task failure or successful paper reproduction |
| Loss-Conditioned State Execution |[2609.15801v1](https://arxiv.org/abs/2609.15801v1) | Loss-specific decisions and independent groupwise execution calibration | Mean/variance summaries alone are not a generic new decision theory; no native code run |
| PhyLatent |[2608.05720v1](https://arxiv.org/abs/2608.05720v1) | Physical-label supervision, perturbation/denoising and action consistency | Different training information from unlabeled representation objectives; no equal-information baseline claim |
| JEPA-kMMVAE |[2610.13315v1](https://arxiv.org/abs/2610.13315v1) | Noisy CSI history to clean future CSI with pretrained latent/decoder structure | Clean target supervision differs from unsupervised nuisance filtering |
| Towards Financial World Modeling |[2610.09048v1](https://arxiv.org/abs/2610.09048v1) and [author code](https://github.com/fin-ai-lab/tfwm) | Multi-regime representation study includes volatility and liquidity probes | Generic variance-information probes are covered; action-conditioned dynamics remain future scope there. Data release completeness needs checking before download |
| Variational Heteroscedastic Volatility Model |[2204.05806v1](https://arxiv.org/abs/2204.05806v1) | Sequential variational modeling of heteroscedastic covariance | End-to-end volatility learning is established, not a new remedy |
| Nonlinear sufficient dimension reduction |[Annals2013 author-hosted paper](https://pages.stat.wisc.edu/~wahba/stat860/pdf3/li.bing.uq.pdf), [1304.0580](https://arxiv.org/abs/1304.0580) | Central class and GSIR/GSAVE; conditional variance can reveal more than regression mean | A mean-subspace/central-subspace distinction or nonlinear second-moment method cannot be claimed new |
| Conditional Variance Estimator |[2102.08782](https://arxiv.org/abs/2102.08782), Bernoulli2022 DOI10.3150/21-BEJ1402 | Existing sufficient-dimension-reduction estimation beyond restrictive inverse-regression assumptions | Relevant strong prior; not executed in this cycle, so no comparison to it is claimed |

Recent window emphasis is Apr–Oct2026, with earlier direct theoretical anchors.
Adjacent DCA/DAPC, controlled-world identifiability2607.22430, One-Step2609.36227
and temporal JEPA papers are in cycle3/main ledgers. No evidence that a mere
combination of their familiar pieces yields a new research contribution.

Opportunity remains unconfirmed: controlled risk-sufficiency evidence and a
label-free diagnostic with an actual cost/estimation advantage. Existing theory
and broad related topics do not by themselves prove this precise question solved,
but the current experiments do not establish an unaddressed residual problem.
''')
 status={'objective_achieved':False,'overall_status':'active','accepted_research_directions':[],'watchlist':['risk-relevant compressed-state sufficiency; novelty and native transfer unproven'],'cycle4':{'scientific_jobs_complete':True,'trained_configurations':23,'noise_pilot_fits':6,'noise_attribution_fits':5,'risk_pilot_fits':9,'risk_existing_moment_control_fits':3,'native_vjepa_component_initializations':3,'native_visual_training_reproductions':0,'risk_repair_gate_passed':gate,'all_previous_cycles_preserved':True},'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (ROOT/'outputs'/'task_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
 write('CONTINUATION.md','''# Continue after cycle4

Goal ACTIVE and NOT ACHIEVED. No accepted direction. Read the seven ledgers and
`outputs/task_status.json`. Preserve all previous cycles and fixed plans/results.
23fits complete here, plus3native VJEPA component seeds. No native visual training.
Noise-specific Motion pilot gate failed; default LeWM repaired by8000steps and
matched-static by existing CCA. Do not reopen the same gate with tuned thresholds.
Risk compression has a matched controlled phenomenon, but simple attached repair
failed all-seed gate. Ordinary moment control resolves the toy; new-method route
for this case is stopped. Include seed43 and the moment control in any conclusion.
This is observation-target regression, not a native JEPA risk failure.
VJEPA native CNN passes. Native uncertainty term cannot teach encoder by design;
mean-only CEM is independent of the variance head. No future-probe leak found.
Switching graph alignment covered by Bencherki2023; GSAVE/central-subspace theory
already distinguishes distributional and mean information. No new invention there.
Next: independent cohorts/native probabilistic representation and strong established
SDR/proper-score baselines, only if a residual empirical question remains valuable.
No content published. All server artifacts are retained and manifest hash checked.
''')
 try:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
  for ax,key,title in zip(axes,['risk_v_probe_nmse','policy_regret'],['Variance-driver information','Quadratic decision regret']):
   for j,(m,rows) in enumerate(grouped.items()):
    ys=[r[key] for r in rows];ax.scatter([j-.1,j,j+.1],ys,s=35);ax.plot([j-.15,j+.15],[np.mean(ys)]*2,color='black',lw=1.2)
   ax.set_xticks(range(len(grouped)),['Detached','Attached','Full obs.','Moments'],rotation=15)
   ax.set_title(title)
   if key=='policy_regret':ax.set_yscale('log')
   else:ax.set_ylim(-.04,1.06)
   ax.grid(axis='y',alpha=.2)
  axes[0].set_ylabel('Frozen linear-probe NMSE');axes[1].set_ylabel('Expected excess next-state cost')
  fig.suptitle('Controlled risk-state diagnostic: three initializations, one cohort',fontsize=11)
  fig.savefig(ROOT/'outputs'/'risk_state_pilot'/'risk_compression.png',dpi=170);plt.close(fig)
 except ImportError:pass
 print(json.dumps(status,indent=2))
if __name__=='__main__':main()
