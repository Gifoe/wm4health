"""Generate all ledgers and a diagnostic plot from verified saved values."""
import datetime,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def ms(v):return f'{np.mean(v):.6f} +/- {np.std(v,ddof=1):.6f}'

def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
 verify=read(OUT/'evidence_verification.json');assert verify['all_checks_passed'] and verify['fits_replayed']==24
 families={name:read(OUT/name/'results.json') for name in ['acquisition_controls','residual_controls']}
 table=[];effects=[]
 for family,result in families.items():
  for dataset in ['ETTh1','BikeSharing']:
   for group in ['q0_constant','q0_single_switch','q0_fast_switch','q1_constant','q1_single_switch','q1_fast_switch','fully_observed']:
    a=[r for r in result['rows'] if r['dataset']==dataset and r['arm']=='source'];b=[r for r in result['rows'] if r['dataset']==dataset and r['arm']=='augmented']
    av=[r['tests'][group]['MAE'] for r in a];bv=[r['tests'][group]['MAE'] for r in b]
    table.append(f'| {dataset} | {family} | {group} | {ms(av)} | {ms(bv)} | {ms(np.array(bv)-av)} |')
    effects.append({'dataset':dataset,'family':family,'query':group,'source_MAE_mean':float(np.mean(av)),'augmented_MAE_mean':float(np.mean(bv)),
      'paired_aug_minus_source_MAE':list(np.array(bv)-av),'percent_change_of_mean':float(100*(np.mean(bv)/np.mean(av)-1))})
 elapsed=sum(r['elapsed_seconds'] for r in families.values());parameters=sorted({r['active_parameters'] for f in families.values() for r in f['rows']})
 (OUT/'comparison_summary.json').write_text(json.dumps({'effects':effects,'training_seconds_sum':elapsed,'fits':24,'updates':48000,'active_parameters':parameters},indent=2),encoding='utf-8')
 docs={
 'experimental_results.md':f'''# Verified cross-system ordinary controls

New controlled acquisition-policy experiment,not original-paper reproduction. Two real observed targets:ETTh1oil temperature and BikeSharing rental counts. L96,H24;chronological60/20/20;training-only standardization. BikeSharing has17379original rows on a17544hour grid:165naturally absent hours. Windows crossing any original gap were excluded. No future value was imputed into a context. The two datasets have145and123test roots respectively,with nonoverlapping24step target blocks.

There are24fits:2ordinary GRU families x2acquisition recipes x2datasets x3seeds,2000updates each. Same input information and active parameter counts{parameters};seasonal skip adds no parameters. Both recipes know causal-MAR/sensor-MNAR type and the configured time-varying policy coefficient. They do not know hidden values or Bernoulli draws. Both families select checkpoints only on the same fixed source validation queries. Seed variation includes initialization/training sampling; physical data are shared.

Values are normalized MAE mean+/-sampleSD over3seeds,not confidence intervals. Negative paired difference favors ordinary augmentation. All conditions are retained; comparing error across different mask laws alone cannot establish regret because the observed information differs.

| Dataset | Family | Query | Source training | Augmented training | Paired difference |
|---|---|---|---:|---:|---:|
{chr(10).join(table)}

Full MSE,observation rates,reference persistence/past-day scores,predictions and curves are in the rawJSON/NPZ. The verifier also computes a descriptive training-only hour-of-week median reference; it is not a test-selected model. Training elapsed sum{elapsed:.3f}s. This excludes setup and verification.

Independent verification checked40saved query groups,raw timestamps/normalization/labels,causal sampling laws,zeroed unobserved values,forward-fill/gaps,common queries,checkpoint selection and all24saved predictors/metrics. Original GPU replay error is exactly0. CPU versus original cuDNN predictions differs by at most{verify['maximum_CPU_vs_GPU_prediction_error']:.8g};with cuDNN TF32 disabled,representative group checks across24models agree with CPU within{verify['maximum_no_cudnn_TF32_vs_CPU_control_error']:.8g}. This backend precision difference is retained,not hidden by claiming exact CPU equality. No training was rerun by verification.
''',
 'baseline_registry.md':'''# Baseline/data execution registry

Direct GRU and conventional last-day residual GRU:24executed ordinary fits,not new methods. Source and augmented recipes have matched parameters,inputs,updates and source-validation selection. Persistence and past-day are fixed observed-input references. A training-only calendar median is a descriptive extra reference. These are controlled transfers using real series,not public benchmark leaderboard scores.

ETTh1is read-only reused from sealed cycle5with its SHA256. UCI BikeSharing is publisher-downloaded CC-BY4.0,with archive/source hashes and attribution in outputs/data_registry.json. Simulated context masks do not alter the underlying series. No physical latent-state or causal ground-truth claim is made.

Timeflies source snapshot:40b67ed216f4d1457080d3def2cb49fb6db479fd,complete four-file tree,two text files retained and hashed,noAGENTS or credential pattern flagged. No training code is present. Native WorldTS,OSSM,LatentODE,Under-Cali,NeuMISE,MIRRAMS are literature/source-scope controls,not executed models in this cycle. Six exact primary PDFs are archived. Earlier completed cycles are preserved.
''',
 'negative_evidence.md':'''# Negative evidence and limits

The direct GRU does not exhibit a marked new collapse on the composed acquisition queries. Augmentation yields modest improvements also on constant and fully observed queries,so its benefit is not specific evidence of a new dynamics mechanism. The residual family is worse on several masked conditions and better on fully observed BikeSharing;do not hide those results or select a family using test scores.

No Bayes oracle exists on these real observed series. Differences between sampling laws may reflect different information content. No model is proved converged,optimal or physically faithful. Ordinary controls alone do not prove all latent world models adequate;they also provide no positive motivation for a new policy-modular architecture here. Do not enlarge this benchmark solely to force a failure.

Timeflies' abstract advertises code,but its current complete public tree has none. The older pinned PRO-DYNtree has no advertised ILI/NYSEfiles. Those availability facts changed the experimental route;they are not scientific defects. CPU replay initially exceeded a strict2e-5threshold. Exact original-GPU replay and TF32-disabled controls localized this to backend numerical precision;the observed max discrepancy and controls are saved. No scientific result was replaced.
''',
 'hypothesis_space.md':'''# Competing hypotheses and evidence

| Hypothesis | Discriminator | Current status |
|---|---|---|
| New sampling-policy composition causes an acute general forecasting failure | same physical roots,target labels,known metadata;source versus composed queries | not demonstrated by the direct GRU in these two processes;cross-law raw error is not Bayes regret |
| Ordinary conditioning/augmentation already explains the tested variation | matched source and augmented GRUs,three seeds | modest broad benefit;no need for a new architecture established |
| A prediction-head/optimization choice explains apparent weakness | ordinary seasonal residual family with same parameters/information | material family dependence;residual is not uniformly stronger;do not attribute it to latent physics |
| Hidden context values leaked through mask construction | independent raw-value/causal-MAR/sensor-MNAR replay | checks pass;simulator-only MNAR trigger is not provided to the predictor |
| The protocol tests unknown-policy robustness | inspect supplied metadata | false:known policy parameters are an explicit part of the information set |
| World models universally cannot handle acquisition changes | strong native latent models and state-estimation comparison | not tested;no such claim is justified |

Known nonignorable-shift theory already allows the optimal predictor to change. The present experiment does not reinterpret that fact as novelty. No final method is selected by test performance.
''',
 'research_ideas.md':'''# Candidate triage

Candidate:reusable time-series dynamics with acquisition-policy-dependent state updates. It is a general capability question rather than a dataset optimization. Generic observation-process factorization,observers,informative missingness and missingness-shift robustness already have substantial prior art.

The current real-series discriminator has not isolated a residual beyond ordinary history modeling,policy metadata and augmentation. Treat this candidate as low priority,not a validated gap. A rescue would require a repeatable native latent-state/transition reuse deficiency beyond its declared task,without confounding information differences,capacity or optimization. A source-specific implementation issue or an elementary MNAR example would not suffice.

No second or third direction is filled merely to satisfy a quota. QG remains optional infrastructure. The research target remains a general world-model contribution at the WorldTS level,without mandatory large shared pretraining.
''',
 'final_research_direction.md':'''# Research direction remains unconfirmed

**Research Question:** Can a time-series world model reuse dynamics when a known observation policy is composed differently,while correctly updating its predictive state?

**Evidence:**24ordinary neural fits on two real processes,matched source/composed queries,three seeds and independent replay. The direct GRU does not show the sought acute composition-specific failure. Augmentation benefits are modest and broad. Seasonal residual results are mixed. This is insufficient practical evidence for a new mechanism.

**Existing Methods:**Observers,joint observation-time likelihoods,missingness-shift theory,robust prediction and online calibration already cover the broad principles. Their existence alone does not resolve every finite-data residual,but no concrete residual has survived this test.

**Proposed Idea:**None accepted. Do not rename ordinary filtering,policy conditioning or data augmentation as a new world model.

**Falsification:**A healthy ordinary control or an existing repair explaining the putative failure defeats the new-method motivation for that test. Current evidence fails to establish the motivation;it does not prove universal optimality or physical correctness.

**Feasibility:**The private singleRTX5090completed24fits/48000updates. Raw data hashes,plans,seeds,checkpoints,query arrays,curves,metrics,source metadata and precision diagnostics are retained. There is no resource blocker.

**Next Experiment:**Return to native-model capability discovery that directly tests predictive-state and conditional-transition reuse. Do not expand the present missingness benchmark or create a new architecture from its scores. Any next candidate must specify a real native task/output contract and a discriminator beyond the known repairs before substantial training.

The overall autonomous research goal remains active and unachieved. Zero final directions currently meet all evidence/novelty requirements.
'''}
 for name,content in docs.items():(ROOT/name).write_text(content,encoding='utf-8')
 status={'cycle13_scientific_complete':True,'verified_fits':24,'updates':48000,'objective_achieved':False,'accepted_directions':0,
   'decision':'No new acquisition-policy mechanism motivated by this test;retain evidence and stop expanding this benchmark',
   'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'task_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
 for i,dataset in enumerate(['ETTh1','BikeSharing']):
  for j,family in enumerate(families):
   ax=axes[i,j]
   for q,color in [(0,'#1976a3'),(1,'#b65c2e')]:
    vectors=[next(e['paired_aug_minus_source_MAE'] for e in effects if e['dataset']==dataset and e['family']==family and e['query']==f'q{q}_{s}') for s in ['constant','single_switch','fast_switch']]
    ax.errorbar(range(3),np.mean(vectors,axis=1),yerr=np.std(vectors,axis=1,ddof=1),marker='o',color=color,label='MAR' if q==0 else 'MNAR')
   ax.axhline(0,color='0.5',lw=.8);ax.set(title=f'{dataset}: '+('direct GRU' if j==0 else 'seasonal residual GRU'),xticks=range(3),xticklabels=['Constant','One switch','Fast switches'],ylabel='Augmented minus source MAE')
   ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('Ordinary augmentation effect; paired training-seed SD, not CI')
 folder=OUT/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'ordinary_controls.png',dpi=150);fig.savefig(folder/'ordinary_controls.pdf');plt.close(fig)
 print(json.dumps(status))

if __name__=='__main__':main()
