"""Recompute scored arrays, apply the pre-heldout rule, retain every seed."""
from common57 import *
import csv,io,numpy as np
METHODS_FINAL=['native','independent','running09c','running01c','full','proper']
NAMES={'native':'Native same-batch','independent':'Independent bank','running09c':'Running .9 centered','running01c':'Running .1 centered','full':'Full training bank','proper':'Proper likelihood'}

def group(rows,sensor,method):return sorted([r for r in rows if r['sensor']==sensor and r['method']==method],key=lambda r:r['seed'])
def values(rows,key):return np.array([r[key] if key!='primary' else r['primary_h20_plan_effect_mse']['mean'] for r in rows],dtype=float)
def aggregate(rows):
    x=values(rows,'primary')
    return {'n':len(rows),'stochastic_fits':rows[0]['stochastic_fit'],'mean_primary':float(x.mean()),'seed_sd':float(x.std(ddof=1)) if len(x)>1 else None,'seeds':[r['seed'] for r in rows],'per_fit_primary':x.tolist(),'mean_absolute_rho_error':float(values(rows,'absolute_rho_error').mean()),'mean_filtered_r2':float(values(rows,'filtered_state_r2').mean()),'mean_train_map_r2':float(values(rows,'train_map_r2').mean()),'mean_training_seconds':float(values(rows,'training_seconds').mean()),'max_peak_cuda_bytes':int(max(r['peak_training_cuda_bytes'] for r in rows))}

def main():
    source_check();evaluation=read(OUT/'EVALUATION57_V1.json');models=read(OUT/'MODEL_SEAL57_V1.json');maps=read(OUT/'MAP_SEAL57_V1.json');opener=read(OUT/'HELDOUT_OPEN_RECEIPT57_V1.json')
    assert evaluation['status']=='ALL_32_MODELS_SCORED_FIXED_HELDOUT'
    assert evaluation['open_receipt_sha256']==sha(OUT/'HELDOUT_OPEN_RECEIPT57_V1.json')
    assert maps['models_sha256']==sha(OUT/'MODEL_SEAL57_V1.json')
    assert opener['map_seal_sha256']==sha(OUT/'MAP_SEAL57_V1.json')
    assert opener['decision_rule_sha256']==sha(ROOT/'DECISION_RULE_V1.md')
    def dt(x):return datetime.datetime.fromisoformat(x)
    assert dt(models['utc'])<dt(maps['utc'])<dt(opener['utc'])<dt(evaluation['utc'])
    rows=evaluation['records'];assert len(rows)==32
    arrays={};run_records=[];max_recompute=0.
    for row in rows:
        assert sha(row['arrays'])==row['arrays_sha256']
        with np.load(row['arrays']) as a:
            P=a['predictions'];Q=a['oracle']
            delta=(P[1:]-P[:1])-(Q[1:]-Q[:1])
            primary=np.square(delta[:,:,-1]).mean(axis=(0,2))
            error=max(float(abs(primary.mean()-row['primary_h20_plan_effect_mse']['mean'])),float(np.max(abs(primary-a['primary']))))
            assert error<1e-12,(row['tag'],error)
            assert np.isfinite(P).all() and np.isfinite(primary).all()
            arrays[row['tag']]=primary.copy();max_recompute=max(max_recompute,error)
        record=read(OUT/'runs'/row['tag']/'result.json')
        assert record['checkpoint_sha256']==row['checkpoint_sha256'] and sha(record['checkpoint'])==row['checkpoint_sha256']
        assert record['heldout_training_reads']==record['hidden_state_training_reads']==record['validation_training_reads']==0
        assert record['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
        assert np.isfinite(record['common_train_objective'])
        run_records.append(record)
    for sensor in SENSORS:
        for seed in SEEDS:
            rr=[r for r in run_records if r['sensor']==sensor and r['seed']==seed and r['method']!='proper']
            assert len(rr)==5 and len({r['initial_tensor_sha256'] for r in rr})==1
    aggregates={sensor:{method:aggregate(group(rows,sensor,method)) for method in METHODS_FINAL} for sensor in SENSORS}
    comparisons=[];paired=[]
    from scipy.stats import t as student_t
    for sensor in SENSORS:
        N=aggregates[sensor]['native']['mean_primary'];P=aggregates[sensor]['proper']['mean_primary'];gap=N-P
        for method in METHODS_FINAL[1:-1]:
            C=aggregates[sensor][method]['mean_primary'];base=group(rows,sensor,'native');control=group(rows,sensor,method)
            assert [x['seed'] for x in base]==[x['seed'] for x in control]==SEEDS
            deltas=[]
            for a,b in zip(base,control):
                d=arrays[a['tag']]-arrays[b['tag']];mean=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(len(d)));q=student_t.ppf(.975,len(d)-1)
                pair={'sensor':sensor,'method':method,'seed':a['seed'],'native_minus_control':mean,'sequence_se':se,'sequence_interval95':[float(mean-q*se),float(mean+q*se)],'scope':'Within this pair of fitted models; not an across-seed confidence interval.'}
                paired.append(pair);deltas.append(mean)
            comparisons.append({'sensor':sensor,'method':method,'native_primary':N,'proper_primary':P,'control_primary':C,'native_minus_proper_gap':gap,'actionable_native_gap':bool(gap>.005),'closure_fraction':float((N-C)/gap) if gap>.005 else None,'paired_seed_improvements':deltas,'improves_all_three_seeds':bool(all(x>0 for x in deltas)),'reference_cost_role':'attribution upper control' if method=='full' else 'small-reference adaptation/control'})
    gate=[]
    for method in ['independent','running09c','running01c','full']:
        a=next(r for r in comparisons if r['sensor']=='asinh' and r['method']==method)
        lin=next(r for r in comparisons if r['sensor']=='linear' and r['method']==method)
        degradation=lin['control_primary']-lin['native_primary']
        passed=bool(a['actionable_native_gap'] and a['closure_fraction']>=.8 and a['improves_all_three_seeds'] and degradation<=.005)
        gate.append({'method':method,'passes_fixed_feasibility_rule':passed,'nonlinear_gap_actionable':a['actionable_native_gap'],'nonlinear_closure_fraction':a['closure_fraction'],'all_nonlinear_seeds_improve':a['improves_all_three_seeds'],'linear_degradation':degradation,'is_small_reference_control':method!='full'})
    repair=[r['method'] for r in gate if r['passes_fixed_feasibility_rule'] and r['is_small_reference_control']]
    if not next(r for r in comparisons if r['sensor']=='asinh')['actionable_native_gap']:decision='NO_ACTIONABLE_NATIVE_NONLINEAR_GAP_BY_FIXED_RULE'
    elif repair:decision='EXISTING_SMALL_REFERENCE_CONTROL_REPAIRS_THIS_SYNTHETIC_DEFICIT'
    elif next(r for r in gate if r['method']=='full')['passes_fixed_feasibility_rule']:decision='COMPLETE_REFERENCE_REPAIRS_DEFICIT_SMALL_REFERENCE_REPAIR_NOT_ESTABLISHED'
    else:decision='REPAIR_NOT_ESTABLISHED_MECHANISM_AND_OPTIMIZATION_UNRESOLVED'
    failure=read(OUT/'FAILED_ATTEMPT57_V1.json')
    for p in failure['retained_files']:assert sha(p['path'])==p['sha256']
    assert failure['attempted_updates']==10000
    neural=[r for r in run_records if r['method']!='proper']
    assert len(neural)==30 and all(r['updates']==10000 for r in neural)
    summary={'utc':utc(),'status':'C57_ALL_FIXED_MODELS_INTERPRETED','scientific_decision':decision,'passing_small_reference_controls':repair,'groups':aggregates,'comparisons':comparisons,'feasibility_gate':gate,'paired_sequence_comparisons':paired,'max_primary_array_recomputation_error':max_recompute,'valid_terminal_models':32,'neural_updates_valid':sum(r['updates'] for r in neural),'retained_failed_updates':10000,'total_neural_update_attempts':sum(r['updates'] for r in neural)+10000,'deterministic_proper_fits':2,'proper_iterations':[r['updates'] for r in run_records if r['method']=='proper'],'evaluation_sha256':sha(OUT/'EVALUATION57_V1.json'),'model_seal_sha256':sha(OUT/'MODEL_SEAL57_V1.json'),'map_seal_sha256':sha(OUT/'MAP_SEAL57_V1.json'),'open_receipt_sha256':sha(OUT/'HELDOUT_OPEN_RECEIPT57_V1.json'),'decision_rule_sha256':sha(ROOT/'DECISION_RULE_V1.md'),'script_sha256':sha(__file__),'accepted_final_directions':0,'overall_goal':'ACTIVE_UNACHIEVED'}
    dump(OUT/'SUMMARY57_V1.json',summary)
    csvout=io.StringIO(newline='');writer=csv.writer(csvout)
    writer.writerow(['tag','sensor','method','seed','stochastic_fit','primary_h20_effect_mse','sequence_se','sequence_low95','sequence_high95','rho','rho_error','filtered_r2','train_map_r2','training_seconds','parameters','peak_training_cuda_bytes'])
    for r in rows:
        p=r['primary_h20_plan_effect_mse'];writer.writerow([r['tag'],r['sensor'],r['method'],r['seed'],r['stochastic_fit'],p['mean'],p['sequence_mc_se'],*p['sequence_interval_95'],r['rho'],r['absolute_rho_error'],r['filtered_state_r2'],r['train_map_r2'],r['training_seconds'],r['parameters'],r['peak_training_cuda_bytes']])
    with (OUT/'results57_v1.csv').open('x',encoding='utf-8',newline='') as f:f.write(csvout.getvalue())
    lines=['| Sensor | Control | Fits | h20 effect MSE | Seed SD | rho absolute error | Filtered R2 |','|---|---|---:|---:|---:|---:|---:|']
    for sensor in SENSORS:
        for method in METHODS_FINAL:
            a=aggregates[sensor][method];sd='N/A: deterministic' if a['seed_sd'] is None else f"{a['seed_sd']:.6g}"
            lines.append(f"|{sensor}|{NAMES[method]}|{a['n']}|{a['mean_primary']:.6g}|{sd}|{a['mean_absolute_rho_error']:.6g}|{a['mean_filtered_r2']:.6g}|")
    main_table='\n'.join(lines)
    glines=['| Control | Nonlinear closure | All 3 seeds improve | Linear degradation | Passes fixed rule |','|---|---:|:---:|---:|:---:|']
    for r in gate:
        fraction='N/A: no actionable gap' if r['nonlinear_closure_fraction'] is None else f"{r['nonlinear_closure_fraction']:.4f}"
        glines.append(f"|{NAMES[r['method']]}|{fraction}|{r['all_nonlinear_seeds_improve']}|{r['linear_degradation']:.6g}|{r['passes_fixed_feasibility_rule']}|")
    gate_table='\n'.join(glines)
    claim=('Existing small-reference controls pass the fixed repair criterion in these systems: '+', '.join(NAMES[m] for m in repair)+'.' if repair else 'No small-reference repair is established by the fixed feasibility rule in these systems.')
    if decision=='NO_ACTIONABLE_NATIVE_NONLINEAR_GAP_BY_FIXED_RULE':claim='The native-to-proper nonlinear error gap is not actionable under the fixed .005 threshold; no gap-closure ratio is used.'
    scope='The evidence covers one controlled linear latent system with two paired sensors, a restricted in-model rotation prior and a privileged train-state mean diagnostic. It does not demonstrate arbitrary nonlinear latent dynamics, covariance calibration, causal validity on real data or a novel general framework. The two sensors are not two independent physical systems. No final research direction is accepted.'
    reports={
        'README.md':f'''# C57 complete controlled-sensor audit

{claim}

{main_table}

{gate_table}

The protocol, numerical amendments and 80% gap-closure rule were fixed before the first heldout opening. This is not an untouched before-training preregistration: the unshifted adaptation failed in training and was explicitly replaced by separately qualified centered controls before heldout access. Every failed log/checkpoint is retained. Valid neural fits total 300,000 updates, with another 10,000 failed adaptation updates. Proper controls are two deterministic fits, not three seed replications.

Primary values are h20 changed-plan conditional-mean effect MSE, averaging zero/one/alternating plans. Neural seed spread and within-model sequence uncertainty are separate. See outputs/results57_v1.csv, outputs/EVALUATION57_V1.json and outputs/SUMMARY57_V1.json for all seeds, horizons and controls. INDEPENDENT_REFERENCE_CAVEAT_V1.md explains why the unsafe independent-bank control cannot establish failure of competent existing methods.

{scope}
''',
        'experimental_results.md':f'''# C57 experimental results

{main_table}

{gate_table}

{claim}

All 32 model/mean-map hashes were sealed before heldout opening; terminal model -> train-only map -> heldout receipt -> scoring timestamp ordering is verified. Primary values were independently recomputed from every prediction array (maximum discrepancy {max_recompute:.3g}). The heldout set has 256 independent simulated sequences; the physical process/data draw is fixed. Paired sequence intervals condition on each fitted model, while the table's SD is across the three neural initializations. There is no validation or test checkpoint selection and no best-seed reporting.

The proper control receives the same prior structure and a small parametric invertible-sensor family, learns its parameters without state labels, and uses full-data float64 L-BFGS. The five neural controls share matched initial tensors, architecture, Adam step budget, data and training precision. Independent/running variants encode 64 sequences per update, native 32 and full reference 544; total compute is not silently equated. Their runtime and peak allocation are in the CSV. The proper and auxiliary losses have different units/definitions and are not ranked as a common objective.

{scope}
''',
        'baseline_registry.md':'''# C57 baseline registry

RP-GSSM source a75225d5edceb972bb872dc9a335608733567dd4 remains unchanged across 21 pinned files. This is an adapted invariant-rotation prior and qualified PyTorch loss port, not the authors' unrestricted action network/Trainer reproduction. Native moment/loss/gradient, CUDA float32, analytic positive-relative kernels, gamma-one gradients and proper Gaussian/flow likelihood have separate qualification receipts.

Native, independent, centered running .9, centered running .1 and complete-reference neural controls each execute three seeds for both sensors. Architecture: 2-64-64-2 ReLU recognition mean, learned constant diagonal site precision, three prior parameters, 4,487 total parameters. Proper controls have seven parameters and two deterministic full-train fits. Their normalizing-Kalman principle is prior art, not a new method. All capacities and budgets remain explicit.

The first unshifted running .9 fit completed 10,000 attempted updates but failed its terminal finite-reference diagnostic. It is excluded from valid fitted-model counts, with malformed partial result, checkpoint, curves and stderr retained. Its three preceding valid fits were reused by exact hashes. No package, driver, upstream file or old C54-C56 result was modified. Large artifacts remain on the authorized server disk, code and compact evidence on wm4health.
''',
        'literature_review.md':'''# C57 literature and originality boundary

See LITERATURE_BOUNDARY_V1.md for pinned source and primary links. RP-GSSM supplies the tested probabilistic recognition model. SogCLR and prior contrastive gradient-bias work already supply global/running-normalizer optimization ideas. Normalizing Kalman filtering already supplies invertible-sensor Gaussian likelihood learning. Analytic score centering is an equivalent loss representation, not a renamed architecture contribution.

Pythia's two-stage representation/readout recipe and its explicit observational-forecasting scope are prior art. EpiCF-Bench, DoTime and TIME already supply complementary evaluation/data machinery; their existence does not prove arbitrary controlled-world simulation is solved, but prevents a claim that distributional/intervention evaluation is entirely missing. No fit of those benchmarks or Pythia was executed in C57.

The observed source-objective failure and its possible repair must be evaluated against these existing mechanisms. This audit alone cannot establish a new field-wide problem or novelty, and no final direction is accepted.
''',
        'negative_evidence.md':f'''# C57 negative evidence

{claim}

The unshifted running-denominator adaptation failed numerically despite passing first-step gradient checks. Its unbounded Gaussian scores do not inherit the original bounded-similarity image pipeline's behavior; this does not refute SogCLR. The independent-bank adaptation is also unsafe: the separately executed support counterexample has loss .5*log(4/3)-b^2/12, unbounded below on that batch. Its poor training cannot establish that existing competent normalizer repairs fail.

C55 already rejected unavoidable ideal scalar population bias. C56 demonstrated finite-reference gradient deviations; those facts do not imply that a new architecture is necessary. The current fixed-rule result is {decision}. All baseline outcomes, unstable trajectories, seed variability and resource differences remain visible.

{scope}
''',
        'hypothesis_space.md':f'''# C57 competing hypotheses

1. Ideal scalar population objective necessarily biased: rejected by C55 in the realizable setting.
2. Finite references affect learned persistence/action effects: C56 true-point gradients and C57 trained matched controls test this without substituting representation R2 for simulation quality.
3. A naive independent or unshifted running normalizer is automatically a competent repair: contradicted by the retained training pathology and independent-support counterexample. These are adapted controls, not full original-method reproductions.
4. An existing stable global/running estimate suffices in this small controlled setting: current fixed-rule decision {decision}. {claim}
5. Neural representation capacity, optimizer dynamics, nonstationary tracking or data limits explain remaining errors: not globally ruled out by this protocol. A failed repair alone cannot justify a novel framework.

Use the complete per-seed outcomes and tracking statistics, not only a selected headline row. No new method or final direction is accepted here.
''',
        'research_ideas.md':f'''# C57 direction disposition

{claim}

Generic finite-reference correction cannot be promoted as a new time-series world-model contribution. The experiment tests an existing repair family within a controlled model and does not establish a broad unsolved problem. Avoid extending this scalar/rotation route into an open-ended optimizer/gamma sweep to manufacture a positive method.

Any further candidate needs independent system evidence, relevant nonlinear/real observations, competent existing probabilistic and predictive-state baselines, and a specific capability beyond this repaired or unresolved objective. Components being known is not alone fatal to a useful framework, but a coherent framework must show an independently useful mechanism and heldout benefit rather than rename SogCLR or Kalman filtering. Accepted final directions: zero.
''',
        'final_research_direction.md':f'''# C57 research disposition

**Research Question:** Does correcting finite-reference normalization with existing mechanisms repair plan-conditioned world-model conditional means in a controlled nonlinear-sensor setting?

**Evidence:** Qualified native-loss/CUDA equations, 30 matched neural fits, two deterministic proper controls, sealed train-only mean maps and one fixed heldout evaluation. {claim}

**Existing Methods:** Running/global normalizer optimization and proper invertible-sensor Gaussian likelihoods are established approaches. The unsafe independent-bank control and failed unshifted adaptation cannot be used as evidence against those competent approaches.

**Proposed Idea:** None accepted from this audit. An architectural novelty claim would exceed the current evidence.

**Falsification:** The fixed rule evaluates every specified control and all three seeds. Its outcome is {decision}. Passing eliminates a necessity for a new correction here; failing does not establish a new gap without resolving optimization/tracking and broader applicability.

**Feasibility:** Single RTX5090, unchanged isolated runtimes, G: data/checkpoint/artifact subtree, 300,000 valid neural updates plus 10,000 retained failed updates. Code/configuration/logs and metric evidence are preserved for reproduction.

**Next Experiment:** Do not repeat the same normalization seed/gamma sweep. A further research candidate must distinguish itself from the existing repair and show a new useful failure mechanism on independent controlled/nonlinear systems or real observations. First audit the nearest existing framework and decisive falsification test; no final direction is selected yet.

{scope}
'''}
    assert all(not (ROOT/name).exists() for name in reports)
    for name,text in reports.items():
        with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text)
    closure={'utc':utc(),'status':'C57_COMPLETE_SCIENTIFIC_INTERPRETATION','summary_sha256':sha(OUT/'SUMMARY57_V1.json'),'evaluation_sha256':sha(OUT/'EVALUATION57_V1.json'),'reports_sha256':{p:sha(ROOT/p) for p in reports},'results_csv_sha256':sha(OUT/'results57_v1.csv'),'analysis_script_sha256':sha(__file__),'all_32_model_and_array_hashes_verified':True,'initialization_match_verified':True,'seal_before_heldout_order_verified':True,'failed_attempt_retained_hashes_verified':True,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'CLOSURE57_V1.json',closure)
    print(json.dumps({'status':closure['status'],'decision':decision,'passing_small_reference_controls':repair,'accepted_final_directions':0}),flush=True)

if __name__=='__main__':main()
