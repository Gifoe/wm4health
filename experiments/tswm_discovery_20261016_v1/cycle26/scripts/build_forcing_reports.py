"""Create audit reports only after independent verification passes."""
import runtime_profile
import csv,collections,hashlib,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/forcing_audit';DEST=ROOT/'outputs/analysis'

def main():
 result=json.loads((OUT/'results.json').read_text());verified=json.loads((OUT/'verification.json').read_text())
 probe_path=ROOT/'outputs/native_control_import_probe.json'
 native_probe=json.loads(probe_path.read_text()) if probe_path.exists() else {'status':'not_yet_verified'}
 assert verified['status']=='passed'
 assert verified['main_results_sha256']==hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()
 with (OUT/'rows.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 groups=collections.defaultdict(list)
 for r in rows:
  key=(int(r['nonlinear']),int(r['k']),r['method'],float(r['force']),r['coordinates'])
  groups[key].append(r)
 aggregate=[]
 for (nonlinear,k,method,force,coordinates),rs in sorted(groups.items()):
  assert len(rs)==3
  row=dict(nonlinear=nonlinear,k=k,h=.01*k,method=method,force=force,coordinates=coordinates,n_mc=3)
  for metric in ['response_path_rel_rmse','response_mean_rel_rmse','response_mean_endpoint_true','response_mean_endpoint_pred','response_mean_endpoint_rel_error','autonomous_path_max_abs_error','variance_response_endpoint_true','variance_response_endpoint_pred']:
   v=np.array([float(r[metric]) for r in rs]);row[metric+'_mean']=float(v.mean());row[metric+'_min']=float(v.min());row[metric+'_max']=float(v.max())
  aggregate.append(row)
 DEST.mkdir(parents=True,exist_ok=True)
 with (DEST/'aggregate.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
 def summary(nonlin,k,method):
  selected=[r for r in rows if int(r['nonlinear'])==nonlin and int(r['k'])==k and r['method']==method and r['coordinates']=='observed_x1']
  v=np.array([float(r['response_path_rel_rmse']) for r in selected]);return dict(mean=float(v.mean()),min=float(v.min()),max=float(v.max()),n_rows=len(selected))
 headline={str(i):{m:summary(i,64,m) for m in ['endpoint','symmetric','split4']+(['exact_discrete_input'] if i==0 else [])} for i in [0,1]}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
 colors={'endpoint':'#B64939','symmetric':'#315C8C','split4':'#27826C','exact_discrete_input':'#6B5594'}
 for i,ax in enumerate(axes):
  for method in ['endpoint','symmetric','split4']+(['exact_discrete_input'] if i==0 else []):
   vals=[summary(i,k,method) for k in [1,4,16,64]]
   ax.plot([.01,.04,.16,.64],[100*v['mean'] for v in vals],label=method.replace('_',' '),color=colors[method],marker='o',ms=4,lw=1.8)
   ax.fill_between([.01,.04,.16,.64],[100*v['min'] for v in vals],[100*v['max'] for v in vals],color=colors[method],alpha=.12,lw=0)
  ax.set_xscale('log');ax.set_xticks([.01,.04,.16,.64]);ax.set_xticklabels(['0.01','0.04','0.16','0.64'])
  ax.set_xlabel('Coarse observation interval');ax.set_title('Linear control' if i==0 else 'Nonlinear source triad')
  ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Paired response path error (%)');axes[0].legend(frameon=False,fontsize=8,loc='upper left')
 fig.suptitle('Perfect passive dynamics do not fix a mismatched forcing interface',fontsize=12)
 fig.text(.5,.015,'Means over 3 MC cohorts and 4 signed amplitudes; shading is full range, not a confidence interval.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.05,1,.93))
 for ext in ['png','svg','pdf']:fig.savefig(DEST/f'forcing_interface.{ext}',dpi=180)
 plt.close(fig)
 compact=dict(status='verified',headline=headline,main=result,verification=verified,n_aggregate=len(aggregate),
     scientific_decision='REJECT_ARCHITECTURE_GAP_FROM_FORCING_INTERFACE_ARTIFACT',accepted_final_directions=0,
     goal_status='ACTIVE_UNACHIEVED')
 (DEST/'results.json').write_text(json.dumps(compact,indent=2),encoding='utf-8')
 table=['| Case at h=.64 | Endpoint | Symmetric | Split4 | Exact discrete input |','|---|---:|---:|---:|---:|']
 for i,label in [(0,'Linear control'),(1,'Nonlinear source')]:
  table.append('| '+label+' | '+' | '.join(f"{100*headline[str(i)][m]['mean']:.4f}%" if m in headline[str(i)] else 'not applicable' for m in ['endpoint','symmetric','split4','exact_discrete_input'])+' |')
 table='\n'.join(table)
 scope='This is a source-derived full-state oracle audit, not a learned-model comparison, partial-observation solution, or native-paper reproduction. The native one-Euler-step forcing definition is correct for its discrete source. We test a deliberately coarsened extension. No new training, test-selected setting or architecture claim.'
 source='[Falasca2025 exact v4](https://arxiv.org/html/2506.22552v4), [Falasca–Zanna2026 exact v5](https://arxiv.org/html/2602.13847v5), [pinned source](https://github.com/FabriFalasca/Physics-and-causally-constrained-neural-models/tree/398082d6dcee2e9d42832dfcd39febfa48c383dd).'
 reports={
 'experimental_results.md':f'''# Cycle26: forcing-interface falsifier

Decision: response error from this protocol does not establish a new world-model architecture gap. General reusable time-series world modeling remains the goal; no final direction accepted.

{scope}

Three fresh Monte Carlo cohorts,8192 independent initial-state/noise realizations each,linear/nonlinear cases of the same triad family,four cadences,four nonzero signed forcing amplitudes,common Brownian increments. These are not8192 different parameterized systems. {len(rows)} score rows;{len(aggregate)} three-cohort aggregates. Five methods for the linear case,four for nonlinear. All retained.

{table}

Values are mean normalized paired path RMSE of x1 over all common observation times,averaged across signed amplitudes and MC cohorts. Individual signs/amplitudes,full-state scores,mean responses and variance responses remain in rows.csv/aggregate.csv. Full-state path/mean RMSE uses all3 coordinates;endpoint means,endpoint relative errors,endpoint SE and variance responses always refer to x1,also in full_state rows. The raw legacy field n_systems_per_mc means stochastic path realizations,not different parameterized plants. They are not cross-dataset generalization or training-seed statistics. Zero autonomous discrepancy is an exact per-path gate,not merely comparable forecast MSE.

The analytic discrete input operator matches the linear forced reference within numerical precision. Symmetric insertion and ordinary subcycling are old numerical controls. This is direct counterevidence to attributing the endpoint artifact to latent-state learning; it does not prove all nonlinear partial-observation response problems solved.

Main runtime {result['runtime_s']:.2f}s; peak CUDA allocated {result['gpu_peak_allocated_bytes']/2**20:.2f}MiB. Runtime includes generation,compression and scoring,not just GPU computation. {result['environment']}. Raw arrays server-only;compact outputs mirrored. Source repository36files2,688,628bytes,exact Git blobs verified locally; source inspection only.

Independent verification: {verified['counts']}; maxima {verified['maxima']}. Main resultsSHA256={verified['main_results_sha256']}. NumPy replay,all score recomputation and independent matrix-power response checks passed. Discrete reference only:no continuous-SDE refinement claim. Protocol fixes the native dt=.01 process and exposes all3 coordinates.

{source}
''',
 'negative_evidence.md':f'''# Negative evidence

1. Candidate claim: a model that accurately simulates unforced trajectories yet responds poorly requires a new dynamics/representation architecture. This observation alone is insufficient: exact passive maps show the failure under endpoint forcing. Linear input discretization and ordinary finer input insertion are necessary controls. Table in experimental_results.md contains all predeclared cases;no method selected after scoring.
2. Generic forecast-versus-simulation,non-Markovian residuals,multiplicative noise,response validation and physics-map-plus-memory claims are occupied by {source} The 2026v5 Sec5.4 memory extension must not be omitted.
3. Source-native h=.01 is not erroneous. The failure is in an explicitly coarsened extension. This audit does not reproduce original neural-model scores or refute the paper.
4. Full-state exact-source control cannot establish a remaining learned partial-observation gap. No final direction accepted. Earlier cycle25 universal harmful-update claim remains downgraded;do not recycle it as motivation.
''',
 'baseline_registry.md':f'''# Baseline registry

| Item | Status | Role and information |
|---|---|---|
| Native WorldTS | Previously executed cycle22 | General latent model reference;no new run here |
| Falasca2025 triad | Equation-faithful native discrete dynamics;modified full observation | I=1 source parameters;I=0 explicitly modified control |
| Perfect autonomous K-step map | Executed | Has full3D state,physical equations and exact shared fine increments;not a data-driven competitor |
| Endpoint insertion | Executed all settings | Deliberately coarsened forcing extension;native K1 exact |
| Symmetric insertion | Executed all settings | Old splitting control;K1 uses native endpoint convention |
| Four input subcycles | Executed all settings | Same fine dynamics calls;more input insertions;not a cheaper learned-map inference claim |
| Exact discrete ZOH operator | Executed linear case | Oracle A known;not inferred from passive data or transferred to nonlinear case |
| Falasca–Zanna2026 native code | Archived/source-audited only | 36files pinned398082d;no native training/checkpoints/data released in inspected tree;climate/MSM source absent |
| Otto/Peitz/Rowley bilinear generator EM | Five exact-source files archived;import probe status:{native_probe['status']};not trained | Pin ef830a6d4a6cff699492d54230fc2ad1729cab20;next mature hidden-state action-modulation control |

Do not rank these as a common-information neural leaderboard. Task-local pinned NumBa/llvmlite target leaves the main environment unchanged. First missing-numba import and transfer/TLS recovery are retained in EXECUTION_NOTES.md;readiness is not a paper reproduction. {source}
''',
 'literature_review.md':f'''# Literature and novelty boundary

Three independent source-search perspectives are retained in research/modern_frontier.md,research/strong_prior_art.md,research/adjacent_mechanisms.md. Their citations and publication/preprint status are explicit. Root rechecked the exact two Falasca manuscripts and released response function. Source evidence is not numerical reproduction.

{source}

Falasca2025 covers stationary-versus-response mismatch,full versus partial observation,multiplicative noise and temporal averaging. Falasca–Zanna2026v5 goes further with structured discrete maps and two stochastic residual-memory layers (Sec5.4);broad structured-state-plus-memory proposals therefore lack differentiation. Its public CdV source implements paired control/forced rollouts with endpoint dt_phys forcing;at native Euler step that is appropriate. The source tree and hash manifest are outputs/source_manifest.json and upstream/falasca_zanna.

DySLIM [ICML2024](https://proceedings.mlr.press/v235/schiff24b.html) already trains invariant measures along with prediction;neural SDEs [ICML2021](https://proceedings.mlr.press/v139/kidger21b.html) already model stochastic paths;CoDA [ICML2022](https://proceedings.mlr.press/v162/kirchmeyer22a.html) already provides shared structure plus environment contexts. These defeat generic primitive claims but do not prove every interaction or practical finite-data regime solved.

Next-nearest control: [Otto/Peitz/Rowley actuated Koopman generators](https://arxiv.org/abs/2209.09977),[official implementation](https://github.com/samotto1/KoopmanGeneratorEM). Partial observations,hidden-state inference and bilinear input-modulated generators have existing direct solutions. A future transfer claim must compare this class,GRU/NARX and learned stochastic state-space models using the same histories/action data. Newer frontier notes identify action-memory and multistep latent constraints already present in BeliefGraph-JEPA/SG-JEPA;do not rename those ideas.

Current scientific synthesis: failure attribution is a prerequisite to selecting a new framework. This cycle rejects an interface confound as architecture evidence. The still open candidate is an empirical finite-data transfer interaction,not a claimed theorem or accepted Research Gap.
''',
 'hypothesis_space.md':'''# Competing hypotheses after cycle26

| Hypothesis | Distinguishing evidence | Current decision |
|---|---|---|
| Forced-response mismatch necessarily reflects wrong learned state | Exact full-state autonomous oracle plus mismatched endpoint forcing | Rejected as a general inference |
| Input discretization alone can create mismatch | Correct discrete linear input operator and same-noise path checks | Supported for the controlled extension;old mechanism |
| Ordinary splitting/subcycling is an adequate repair here | All amplitudes,cadences and three cohorts | Quantified in full result table;not a new method |
| Nonlinear partially observed pretrained state fails under actuator-induced observability changes beyond competent controlled identification | Native bilinear EM,GRU/NARX,stochastic SSM versus frozen/adapted pretrained state;held-out system parameters and action sequences | Untested;next candidate only |
| Missing intervention excitation or an arbitrary latent-action map explains a future failure | Give all methods the same known physical actuator and calibration data;oracle-filter information control | Required before any architecture claim |

No hypothesis accepted merely because earlier primitives are old. Require an interaction that strong existing combinations fail to solve at matched information/cost,then an independently confirmed light method benefit.
''',
 'research_ideas.md':'''# Candidate research questions,not accepted directions

One provisional lead remains: **Can a reusable pretrained time-series state support rapid system identification when actuation changes both dynamics and which hidden variables become observable?** The target is a common state/update/conditional rollout mechanism across systems,not motor- or triad-specific scoring.

Nearest obstacle to novelty: controlled Koopman-generator EM and recurrent state-space identification already address partial observation and bilinear actuation. Existing context adaptation,history-aware neural SDEs and action-conditioned representation objectives are strong composites. Merely adding action,loops,memory or a response loss is insufficient.

First falsifier: execute the native published slow-manifold control example,then a separately source-verified DC-motor benchmark with hidden current/observed speed. The motor would be a cross-benchmark transfer of bilinear EM,not an author-native example:the actual released repository has slow-manifold and Duffing notebooks,and the paper's third example is fluidic pinball. Compare bilinear EM at several latent sizes,ordinary GRU/NARX and pretrained-state adaptation. Calibration budgets and held-out physical initial conditions/action dwell times fixed before test scoring. Distinguish fully observed,oracle-filtered and history-only information. If an ordinary learned bilinear/readout/recurrent repair matches the proposal,the proposed new mechanism is unnecessary.

The response-interface lead is rejected as new architecture evidence by cycle26. Hidden-memory intervention initialization remains a conceptual possibility but is not prioritized without a same-information,known-actuator nonlinear example that defeats standard controlled filters. Do not build another toy solely to exclude competent baselines.
''',
 'final_research_direction.md':f'''# General time-series world-model direction: not yet established

**Research Question:** Can one reusable learned state adapt efficiently across physical systems whose inputs change both dynamics and observability,while retaining predictive and simulation fidelity?

**Evidence:** This particular question is **not established by current experiments**. Cycle26 supplies counterevidence to a tempting alternative explanation:exact autonomous maps can have wrong forced responses solely due to endpoint insertion at coarse time.

{table}

**Existing Methods:** Bilinear controlled-generator EM,GRU/NARX,stochastic state-space models,CoDA and physics-map+MSM are direct controls. Falasca2025/2026 already cover broad response and memory motivation. Their sufficiency in a properly matched transfer experiment remains to be tested.

**Proposed Idea:** No new architecture selected yet. A potential transferable state/filter/generator interface must first show a residual finite-data interaction that these existing controls do not solve. Generic action-conditioning,posterior correction,memory,semigroup consistency and input integration are not innovations by themselves.

**Falsification:** Abandon the novelty claim if ordinary controlled identification,regularized latent/readout adaptation or sufficient excitation removes the failure;if physical timing/interface corrections explain the gain;or if apparent benefits depend on privileged observations,test tuning or one plant.

**Feasibility:** Native small controlled generators and motor/slow-manifold equations fit one5090. This cycle measured {result['runtime_s']:.2f}s and {result['gpu_peak_allocated_bytes']/2**20:.2f}MiB peak allocated for its different oracle audit;these timings do not predict neural-training cost. No global runtime/dependency change.

**Next Experiment:** Execute and independently verify the exact-source bilinear-EM positive control before designing a modern transfer model. Freeze cross-system calibration/held-out action protocol only after source fidelity checks. Do not infer novelty from failure of an untrained/frozen forecasting head alone.

Final accepted directions: **0**. Goal **ACTIVE,UNACHIEVED**. This is an evidence-driven candidate ledger,not a completed top-conference research claim. Dataset optimization is not the final target.
'''}
 for name,value in reports.items():(ROOT/name).write_text(value,encoding='utf-8')
 print(json.dumps(dict(headline=headline,rows=len(rows),aggregate=len(aggregate),reports=list(reports)),indent=2),flush=True)

if __name__=='__main__':main()
