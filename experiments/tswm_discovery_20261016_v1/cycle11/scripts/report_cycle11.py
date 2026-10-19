"""Generate raw research ledgers from verified actual results."""
import datetime,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def ms(values):return f'{np.mean(values):.8g} +/- {np.std(values,ddof=1):.3g}'
def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
 verification=read(OUT/'evidence_verification.json');assert verification['all_checks_passed']
 oracle=read(OUT/'score_folding/results.json');learn=read(OUT/'score_learning/results.json')
 grid=[]
 for p in oracle['config']['flip_probabilities']:
  a=[r for r in oracle['score_rows'] if r['p']==p and r['horizon']==1 and r['normalization']=='Gaussian radial']
  b=[r for r in oracle['score_rows'] if r['p']==p and r['horizon']==8 and r['normalization']=='Gaussian radial']
  den=[r for r in oracle['density_rows'] if r['p']==p]
  grid.append(f"| {p:g} | {ms([r['fold_minus_identity_ES']['mean'] for r in a])} | {ms([r['fold_minus_identity_ES']['mean'] for r in b])} | {ms([r['predictive_MI_lost_by_fold']['mean'] for r in den])} | {den[0]['analytic_raw_MSE_increase']:.6g} |")
 table=[]
 for arm in learn['config']['arms']:
  vals=[r for r in learn['rows'] if r['arm']==arm]
  table.append(f"| {arm} | {ms([r['probes']['ridge']['raw_state_MSE'] for r in vals])} | {ms([r['probes']['extra_trees']['raw_state_MSE'] for r in vals])} | {ms([r['sign_fiber_separation_ratio'] for r in vals])} | {vals[0]['active_trainable_parameters']} |")
 energy=[r for r in learn['rows'] if r['arm']=='energy'];minm=min(r['probes']['ridge']['raw_state_MSE'] for r in energy);maxm=max(r['probes']['ridge']['raw_state_MSE'] for r in energy)
 docs={
 'experimental_results.md':f'''# Cycle11 verified experimental results

## What actually ran

The exact-kernel diagnostic uses3independent context seeds x5flip probabilities x2horizons x2normalizations=60score rows, with32768iid stationary contexts per seed. Full configs, common random design arrays, logs and every grid result are retained. No parameter fitting is involved. Separately,12neural fits used4arms x3seeds,6000updates each on the privateRTX5090server. NativeSIGReg and nativeInfo-LDM logdet-entropy functions were reused unchanged with explicit controlled settings. These are new controlled experiments and source-component transfers, not original-paper reproductions.

## Oracle score discriminator

Negative score difference means a lower oracle latentEnergy risk for the fold. Information loss is in nats. Entries show mean+/-sampleSD across3MonteCarlo seeds, not a confidence interval. The p-grid shares designs within seed and is not a set of independent replications.

| flip p | folded-minus-identity ES,H1 | folded-minus-identity ES,H8 | one-step predictive MI lost | exact extra raw BayesMSE |
|---|---:|---:|---:|---:|
{chr(10).join(grid)}

Both frame marginals are exactlyN(0,I), dimension2. Unit-circle normalization also retains the counterexample atp=.25; all results are in theJSON. Identical frame marginals do not mean identical future-block joint distributions. The correct fixed-target conditional law is used on each side; no propriety theorem is violated. The analytic positive-noise existence argument and exact rawMSE formula are inSCORE_WITNESS.md.

## Actual learned-state test

Final fixed-budget checkpoints, independent state readouts, mean+/-sampleSD over3training seeds. StateMSE is summed over the2raw coordinates. Energy/information active parameter count differs from decoded arms; compute and active capacity are not claimed matched. No test/readout metric selected a checkpoint.

| arm | linear readout stateMSE | ExtraTrees stateMSE | sign-fiber separation ratio | active parameters |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Identity readout controlMSE is{learn['probe_controls']['identity']['ridge']['raw_state_MSE']:.6g}; exact-fold control is{learn['probe_controls']['exact_fold']['ridge']['raw_state_MSE']:.8g}. A ratio near2preserves sign fibers in a near-affine embedding; exactfold gives0. The simple energy-only arm has stateMSE range[{minm:.8g},{maxm:.8g}], not the order2error of folding. Its three covariance spectra remain well conditioned. All12fits retain the observed state; no actual alias failure is demonstrated. Small absolute readout differences among healthy arms do not justify a new method.

Training-plus-probe elapsed{learn['elapsed_seconds']:.3f}s. PyTorch tracked peak tensor allocation{learn['max_cuda_memory_allocated']}bytes, excluding driver/context/reserved memory. The verifier replayed all60score rows from stored draws, checked analyticGaussian Energy risks, checked the weightedMC estimator against independent knownGaussian optima, safely loaded all12checkpoints and replayed all linear readouts. No scientific fit was rerun by verification.
''',
 'baseline_registry.md':'''# Cycle11 baseline and execution registry

| baseline/component | provenance | executed scope |
|---|---|---|
| exact stationary mixture kernel +identity/fold maps | full equations/config inSCORE_AUDIT_PLAN.md |60oracle score rows,15density rows,3retained independent design arrays |
| mixture likelihood/information principle | standard existingMDN/NLL with equal Gaussian frame entropy | exact oracle density control; not nativeInfo-LDM reproduction |
| nativeSIGReg | AC-MTM43c96f66bb12c95208be68b0858ee658c7e1d363, unchanged class | reused in12new neural fits;17knots/64projections,weight1; full original model not reproduced |
| native logdet_entropy | Info-LDMba705dd626c1d669acd478079e85e1901a0bc0d5 | unchanged component in information-control arm; new mixture predictor rather than released fixed-Gaussian protocol |
| simple Energy; Energy+information; Energy+observed; observed-only | same frozen controlled protocol,3seeds each |12complete6000-update fits, last checkpoints and curves retained |
| Ridge and ExtraTrees | fixed independent states/estimator settings | state-sufficiency probes; identity/exactfold controls |
| LENS867780b5ed196c9f017f8defec3967835026d03f | full52-filetree,14selected files hashed | source review only, no training/inference |
| Branch-JEPAv3 | exact primaryPDF/method/gradient-scope review | existing decoded/EMA/reconstruction boundary; no full native execution |

PINfiles identify exact commits and retained-file scope. NoAGENTS occurred in fullLENS/Info-LDMtrees. AC-MTMAGENTS was retained/read and its append-only research ledger is inprogress/experiment-log.md. NoModal/W&Bjob or public upload. Scripts,configs,rawdraws,held-outstates,12checkpoints,12latentdatasets,curves and rawlogs are retained. PrimaryPDFversions/SHAs are inoutputs/literature/registry.json. Main/cycles2-10 remain sealed and unchanged.
''',
 'negative_evidence.md':'''# Cycle11 negative evidence and rejected explanations

The oracle preference is not a finite-predictor-capacity failure: both sides use their exact conditional laws, and an analytic positive-noise existence argument supports the construction. Scale/dimension/marginal differences do not explain it: the frame marginals and dimensions are equal. Frame-unit normalization does not remove it. Information loss matters for the declared raw prediction task, as the exact.405BayesMSE increase atp=.25and conditionalNLL/MI control establish. Atp=.5the sign is a null for prediction; do not call its removal harmful.

However, the key practical hypothesis failed in the executed random-initialization learning study: all3energy-only fits and all9existing-control fits retained state to small error, with near-affine sign-fiber separation. A bad global criterion-level construction does not show that ordinary training finds it. Stop this toy route; do not change the data or initialization merely to force a positive result.

No claim that a new mixture-information or decoded-space method is needed is supported. Both principles already exist, and the simple baseline is already healthy. Relative percentage gains on tiny readout errors would be misleading.

Fixed-target propriety remains intact. Branch-JEPAv3usesEMA targets, decodedEnergy and reconstruction; our full-gradient radial experiment is not its full method. Info-LDM's recovery theorem assumes information saturation and exact exponential-family distribution matching; the two-Gaussian transition and our new mixture predictor are outside its released fixed-Gaussian reproduction. Earlier self-predictiveRLwork already analyzes online/detached/EMA optimization. No broad theorem refutation or novelinformationprinciple is claimed.

Source-pin utility initially exceeded its selected-file guard(45candidates); selection was narrowed to18relevant files, retaining the fullGit tree. This was a utility issue, not a failed scientific fit. All scientific jobs exited0and independent verification passed. Some next-sourceZenodo requests fail localDNS, while the server's officialAPI metadata fetch succeeds; this is transport state, not scientific evidence.
''',
 'hypothesis_space.md':'''# Cycle11 hypothesis and falsification ledger

| hypothesis | discriminator | evidence and current judgment |
|---|---|---|
| fixed-target propriety alone compares endogenous representations fairly | exact oracle risks with identical frame marginals | false in constructed family; no fixed-target theorem refuted |
| observed preference is only a small-particle approximation | exact conditional kernels and boundary-continuity argument | rejected |
| lower latent score means no useful information was lost | exact NLL/MI difference and raw BayesMSE | rejected forp<.5in the displayed counterexample regime |
| a stochastic predictor removes all representational shortcuts | exact oracle score | insufficient as a universal guarantee |
| ordinary random neural training finds this harmful fold |12frozen fits, independent linear/nonlinear probes and fiber diagnostics | unsupported in tested task; simple energy baseline retains state in all3seeds |
| adding a new objective repairs a necessary observed defect | existing information/raw-space controls versus healthy simple arm | no demonstrated defect to repair; no newmethod claim |
| published fullBranch-JEPA/Info-LDM guarantees fail here | compare gradient scopes and assumptions | not established; protocols and assumptions differ |

The remaining distinction is between criterion-level possible optima and the actual bias of an optimizer/regularizer. The present experiment does not establish a practical unresolved gap on that distinction. New work should move to realistic native dynamics rather than escalate this construction.
''',
 'research_ideas.md':'''# Cycle11 research-idea triage

No direction is accepted; no quota of three is filled.

Candidate question: which probabilistic self-prediction criteria preserve predictive states when targets are also learned? The exact example gives a specific negative boundary, and log-loss/information or raw observation scores provide existing controls. It does not yet establish new theoretical scope beyond prior information/self-predictive literature, and random-initialized neural training is healthy. The missing evidence is a repeatable defect in an actual strong method beyond its stated assumptions and standardEMA/decoded/information controls. This is currently a low-priority theoretical note, not the recommended paper direction.

Possible rescue would require a new precise characterization or real-model failure with an improvement that cannot be explained by a standard existing repair. No further toy tuning is authorized by the evidence. The user clarified that the target is a general time-series world model, with WorldTS as a scope reference. The next action is to identify a reusable state/conditional-transition capability and a cross-system falsification protocol. QG source/weights preparation is retained only as an optional validation tool; optimizing this physical dataset is not the research direction.
''',
 'final_research_direction.md':'''# Cycle11 decision: final research direction remains unconfirmed

**Research Question:** Can correct probabilistic latent prediction still reward removing predictive observable-state information when the representation defines both context and target?

**Evidence:** An exact stationary mixture-kernel construction and60oracle score rows show a folded target can lowerEnergy risk while losing predictiveMI and increasing raw BayesMSE. However,12real neural fits under a frozen4-arm/3-seed protocol preserve state; the simple baseline alone is healthy. This fails the practical-failure requirement for a newmethod motivation in this task.

**Existing Methods:** Information/density-ratio principles, mixtureNLL, decoded-space proper scoring, reconstruction andEMA/stop-gradient targets are established. Their scopes differ. The current evidence neither shows them insufficient on a common realistic task nor refutes their conditional guarantees.

**Proposed Idea:** None accepted. No new architecture, mixture objective or decoded loss is proposed as an innovation. Keep the exact score boundary and the negative neural evidence together.

**Falsification:** Healthy ordinary training or an existing control solving the observed defect rejects a new-repair claim. The healthy baseline result has already triggered that rejection for this controlled task. Merely constructing another bad map would not reverse it.

**Feasibility:** The12fits ran on the private singleRTX5090 with nativeSIGReg/entropy source components and modest memory. Code,fullconfigs,seeds,checkpoints,held-outstates,draws,rawlogs and independent verification are preserved. No new user resource or permission is needed.

**Next Experiment:** Re-anchor discovery in a general time-series world model, following the user's scope correction. Compare reusable predictive-state and conditional-transition capabilities across systems rather than optimizing a single dataset. Audit WorldTS and the closest shared-pretraining/observer models, then freeze a capability-specific discriminator with ordinary recurrent/state-estimation controls before model development. QG is only an optional test environment; no QG training or physical-transfer result is claimed.

The full autonomous discovery goal remains active and unachieved. A final top-conference research direction is not established by this completed audit.
'''
 }
 for name,text in docs.items():(ROOT/name).write_text(text,encoding='utf-8')
 status={'cycle11':{'scientific_jobs_complete':True,'fits':12,'updates':72000,'evidence_verification_passed':True,'accepted_directions':0},'objective_achieved':False,'decision':'retain exact criterion-level example and negative neural evidence; stop toy method direction; prioritize general time-series world-model capability and cross-system evidence','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'task_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
 # Standard plotting, from saved actual numbers; raw evidence figure, no generation model.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for H,color in [(1,'#1976a3'),(8,'#b65c2e')]:
  ps=oracle['config']['flip_probabilities'];vals=[[r['fold_minus_identity_ES']['mean'] for r in oracle['score_rows'] if r['p']==p and r['horizon']==H and r['normalization']=='Gaussian radial'] for p in ps];means=np.mean(vals,axis=1);std=np.std(vals,axis=1,ddof=1);axes[0].errorbar(ps,means,yerr=std,marker='o',color=color,label=f'H={H}, MC seed SD')
 axes[0].axhline(0,color='0.5',lw=.8);axes[0].set(xlabel='Sign-flip probability',ylabel='Fold minus identity: oracle latent ES',title='Constructed score preference');axes[0].legend(frameon=False,fontsize=8)
 labels=['Latent ES','ES + information','ES + observed','Observed only']
 for i,arm in enumerate(learn['config']['arms']):
  v=[r['probes']['ridge']['raw_state_MSE'] for r in learn['rows'] if r['arm']==arm];axes[1].scatter(i+np.array([-.1,0,.1]),v,color='#1976a3',s=24);axes[1].plot([i-.2,i+.2],[np.mean(v)]*2,color='black',lw=1)
 axes[1].axhline(learn['probe_controls']['exact_fold']['ridge']['raw_state_MSE'],color='#b65c2e',ls='--',label='Exact fold control');axes[1].set(yscale='log',ylabel='Raw state MSE: frozen linear readout',title='Actual learned encoders: p=0.25',xticks=range(4),xticklabels=labels);axes[1].tick_params(axis='x',labelrotation=18);axes[1].legend(frameon=False,fontsize=8)
 for ax in axes:ax.spines[['top','right']].set_visible(False)
 folder=OUT/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'score_vs_learning.png',dpi=160);fig.savefig(folder/'score_vs_learning.pdf');plt.close(fig)
 print(json.dumps(status))
if __name__=='__main__':main()
