"""Independent array/receipt checks and bounded conclusions for both complete campaigns."""
from common60 import *
import numpy as np,csv,io
def main():
    e=read(OUT/'EVALUATION60_V2.json');seal=read(OUT/'COMBINED_MODEL_SEAL60_V1.json');opened=read(OUT/'TEST_OPEN60_V2.json');data=read(OUT/'DATA60_V2.json')
    assert e['status']=='ALL_SELECTED_NSSM_GRU_AND_ORDINARY_CONTROLS_SCORED'
    assert e['model_seal_sha256']==opened['model_seal_sha256']==sha(OUT/'COMBINED_MODEL_SEAL60_V1.json')
    dt=datetime.datetime.fromisoformat;assert dt(seal['utc'])<dt(opened['utc'])<dt(e['utc'])
    assert len(seal['models'])+len(seal['failed'])==12
    source_check();maxerr=0.;training=[]
    for row in e['records']:
        assert sha(row['arrays'])==row['arrays_sha256']
        with np.load(row['arrays']) as a:
            sy=data['systems'][row['system']]['normalization']['y_std'];error=np.square((a['prediction']-a['truth'])/sy)
            finite=bool(np.isfinite(error).all());assert finite==row['finite_rollout']
            if finite:
                value=float(error.mean());maxerr=max(maxerr,abs(value-row['primary']['mean']));assert abs(value-row['primary']['mean'])<max(1e-10,abs(value)*1e-12)
        if row['method'] in ['native_component_nssm','ordinary_gru']:
            recpath=OUT/'runs'/row['tag']/'result.json';rec=read(recpath);saved=next(x for x in seal['models'] if x['tag']==row['tag']);assert sha(recpath)==saved['result_sha256']
            assert sha(rec['selected']['checkpoint'])==row['checkpoint_sha256']==rec['selected']['checkpoint_sha256']
            assert sha(rec['terminal_checkpoint'])==rec['terminal_checkpoint_sha256']
            assert sha(OUT/'runs'/row['tag']/'curve.jsonl')==rec['curve_sha256']
            assert rec['updates']==5000 and rec['training_reads']==['train','validation'] and not rec['test_score_selection']
            training.append(rec)
    groups={};methods=['native_component_nssm','ordinary_gru','static','play','ema','hybrid','arx']
    for system in ['N87','pick_place']:
        groups[system]={}
        for method in methods:
            rows=[r for r in e['records'] if r['system']==system and r['method']==method];neural=method in methods[:2]
            failures=[r for r in seal['failed'] if (r['tag'].startswith('GRU_') if method=='ordinary_gru' else not r['tag'].startswith('GRU_')) and system in r['tag']] if neural else []
            vals=[r['primary']['mean'] for r in rows if r['finite_rollout']]
            complete=(len(rows)==(3 if neural else 1) and len(vals)==len(rows) and not failures)
            groups[system][method]={'expected_fits':3 if neural else 1,'completed_fits':len(rows),'training_failures':len(failures),'finite_rollouts':len(vals),'all_expected_outcomes_finite':complete,'mean_primary_all_expected':float(np.mean(vals)) if complete else None,'seed_sd':float(np.std(vals,ddof=1)) if complete and neural else None,'per_fit_primary':[{'tag':r['tag'],'seed':r['seed'],'value':r['primary']['mean']} for r in rows]}
    buf=io.StringIO(newline='');w=csv.writer(buf);w.writerow(['system','tag','method','seed','selected_step','parameters','updates','training_seconds','finite_rollout','test_mse_fit_variance'])
    for r in e['records']:w.writerow([r['system'],r['tag'],r['method'],r['seed'],r.get('selected_step'),r['parameters'],r.get('train_updates'),r.get('training_seconds'),r['finite_rollout'],r['primary']['mean']])
    with (OUT/'results60_v1.csv').open('x',encoding='utf-8',newline='') as f:f.write(buf.getvalue())
    table=['| System | Control | Completed / expected | Finite rollouts | Mean test MSE / fit variance | Neural seed SD |','|---|---|---:|---:|---:|---:|']
    for system in groups:
        for method,a in groups[system].items():
            mean='N/A: incomplete/nonfinite' if a['mean_primary_all_expected'] is None else f"{a['mean_primary_all_expected']:.6g}";sd='N/A' if a['seed_sd'] is None else f"{a['seed_sd']:.6g}"
            table.append(f"|{system}|{method}|{a['completed_fits']}/{a['expected_fits']}|{a['finite_rollouts']}|{mean}|{sd}|")
    table='\n'.join(table)
    summary={'utc':utc(),'status':'C60_SEALED_TWO_SYSTEM_CONTROLS_INTERPRETED','groups':groups,'valid_neural_fits':len(training),'failed_training_attempts':seal['failed'],'neural_updates_completed':sum(r['updates'] for r in training),'synthetic_qualification_updates_separate':45,'max_primary_recomputation_absolute_error':maxerr,'evaluation_sha256':sha(OUT/'EVALUATION60_V2.json'),'training_records':[{'tag':r['tag'],'selected_step':r['selected']['step'],'selected_validation_mse':r['selected']['validation_mse'],'final_short_training_loss':r['final_training_loss'],'seconds':r['training_seconds']} for r in training],'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'SUMMARY60_V1.json',summary)
    scope='Two separately fitted measured systems; the N87 split has historical C59 test exposure and remains exploratory. Pick-place contributes one chronological test record, not an independent-device population. All neural seeds are shown; record-level intervals are not seed intervals. No shared checkpoint, new causal identification, new architecture or field-wide unresolved gap is established.'
    reports={
      'README.md':f'''# C60: ordinary latent-state and GRU controls on two measured systems

{table}

This completes the previously missing generic recurrent controls. Native-component NSSM and ordinary GRU use identical per-seed window-sampler code, 5000 Adam updates, observed history100, free future128 and full validation selection. They have 6312 and 6241 parameters respectively; state dimension, update law, device and runtime differ. This is not a pure stability-only ablation or equal-compute claim. The GRU addition was triggered by validation-only NSSM divergence before the first C60 neural test score; the amendment and superseded unopened waiter are preserved.

All selected model hashes were sealed before new scoring. Independent re-scoring of stored arrays gives maximum absolute discrepancy {maxerr:.6g}. See all seed rows in outputs/results60_v1.csv and outputs/EVALUATION60_V2.json.

{scope}
''',
      'literature_review.md':'''# C60 prior-art and protocol boundary

The source neural-state components come from Forgione, Mejari and Piga, Learning neural state-space models: do we need a state estimator? https://arxiv.org/abs/2206.12928 and https://github.com/forgi86/sysid-neural-estimator . Pinned commit 7473be56be61b53f4e035db6c41cf4efbdc8cff3 and the MIT license remain intact. The current post-update output timing, derived input increment and cross-data recipe are explicit adaptations; the paper's complete training or benchmark table is not reproduced.

MetaLIC is a direct meta-state-space comparator, https://github.com/merlresearch/MetaLIC , but its released benchmark evaluation receives true output history in every test window and uses encoder/readout prediction. Its released loss does not train the computed x_plus term. That route must not be advertised as a free latent simulator. An adapted autoregressive rollout can be separately defined, but poor one-step-to-rollout migration would not itself refute the paper's original task.

MagNetX and PI-HNO already target history-conditioned material response; GRU/SSM state estimation and long-horizon training are established modeling mechanisms. C58 records direct stochastic, trajectory-coherence and memory prior art. None of those components can be renamed into a contribution here.
''',
      'baseline_registry.md':'''# C60 executed model identities

The 12 original source/data/license objects are byte/Git-blob/SHA256 verified. Native residual state update, linear output and LSTM state estimator are unchanged; the adapter emits after each known input update. Float64 output/gradient and scripted qualification passed; synthetic profiling chose CPU for the tiny scripted NSSM. Standard GRU history encoder/future decoder hidden30 has 6241 parameters and independently qualified CPU/CUDA wiring/gradients; it runs on CUDA concurrently. These are ordinary existing methods.

Both systems use separate fitted weights with one fixed modeling recipe. Native data loader full-record normalization is replaced by fit-only normalization. All C59 N87 short-prefix deterministic controls stay immutable; mechanical static/play/EMA/hybrid/ARX counterparts are validation-selected from 40 candidates before test scoring. The initial prepared-cache path collision and both waiting-scorer versions are retained. No source, package, driver or older result is overwritten.
''',
      'experimental_results.md':f'''# C60 executed results

{table}

Neural updates completed in valid fits: {summary['neural_updates_completed']}. The 45 synthetic qualification optimizer steps are separate. Model-seal -> first new test-scoring receipt -> evaluation ordering, original source hashes, selected/terminal checkpoints and every stored prediction hash were verified. Recomputed maximum primary discrepancy: {maxerr:.6g} (relative checks also protect very large finite errors).

Native-component NSSM validation showed instances of short-sequence loss improving while full free rollout diverged. This motivated a parameter-comparable ordinary GRU control before new test scoring; the original NSSM fits were neither stopped nor retuned. Test outcomes do not select a new recipe. Full individual seeds, validation-selected steps and training runtime are in the CSV/JSON. The table reports a complete group mean only when every expected fit has finite outcomes; incomplete/nonfinite results are not silently averaged away.

{scope}
''',
      'negative_evidence.md':'''# C60 negative evidence

A low finite-budget neural score cannot establish a research gap. The source unconstrained residual state update can be unstable over horizons much longer than its training window. Full validation selects a checkpoint but does not turn insufficient long-horizon optimization into a structural lower bound. The added ordinary GRU is a direct conventional repair control, not a new architecture.

The two neural mechanisms differ in state dimension/update law despite similar parameter counts; they also use different devices and training runtime. Differences cannot be attributed exclusively to boundedness. The N87 test split was already viewed in C59, and mechanical data comes from one continuous device record. Unknown earlier magnetic state, finite context, coverage, basis order and output-family adequacy remain possible explanations for residual risk. No test hyperparameter sweep or best-seed selection was performed.
''',
      'hypothesis_space.md':'''# C60 hypothesis disposition

1. Ordinary persistent/decaying bases are always sufficient: not asserted. C59 provided one-material evidence; the mechanical controls and generic neural fits broaden that check without proving universality.
2. Generic latent neural dynamics necessarily cannot simulate these systems: not established by a finite unconstrained NSSM run. Optimization, horizon mismatch and stability remain explanations.
3. A conventional gated recurrent update repairs the observed long-rollout deficit: examined by all three GRU seeds on both systems, with test values retained regardless of outcome. Passing would reduce necessity for a new mechanism here; failing alone would not establish it.
4. Every released state-space evaluation is already free simulation: contradicted by the explicit MetaLIC one-step-history source route. Task migration must be labeled separately.
5. A reusable new world-model framework has been found: unsupported. No candidate new method is tested or accepted in this cycle.
''',
      'research_ideas.md':'''# C60 research disposition

Do not turn native-component NSSM instability into a new-memory story before the ordinary gated control, horizon-matched learning and adequate state initialization are considered. Conversely, a two-system ordinary repair does not prove that all world-model capabilities are solved. Components being known does not by itself reject an efficient useful framework; this cycle has not supplied a distinct additional mechanism or a verified unsolved capability.

The next candidate must earn its scope through a concrete cross-system residual or measurable learning/compute advantage beyond competent existing recurrent/probabilistic models. It must avoid a fresh test-tuned architecture, a privileged physical-state probe mistaken for deployment, and a claim of shared-weight transfer when models were separately fitted. Accepted research directions remain zero.
''',
      'final_research_direction.md':f'''# C60 final direction status

**Research Question:** Does a generic latent-state model or a conventional gated control leave an unexplained conditional-simulation deficit on independent measured mechanisms?

**Evidence:** Locked raw provenance, unchanged native components, qualified output/gradient wiring, two complete fit/validation/test protocols, up to twelve fixed-budget neural fits and immutable/validation-selected ordinary banks. All outcomes are in the table and full machine-readable receipts.

{table}

**Existing Methods:** State estimation, residual neural state-space, ordinary encoder-decoder GRU, play/relaxation and stable ARX are mature controls. Short-horizon instability and mismatched one-step evaluation cannot be repackaged as novelty.

**Proposed Idea:** None accepted. The evidence is control completion and failure attribution, not a new framework.

**Falsification:** A claimed new-memory necessity should be dropped wherever adequate ordinary controls already repair it. A surviving poor model needs initialization/horizon/optimization attribution and independent data before it counts as a gap.

**Feasibility:** Actual CPU/CUDA profiling guided execution; raw data/checkpoints/predictions remain on G:, compact evidence/code on GitHub. No environment changes.

**Next Experiment:** Use any surviving residual to define a matched, independently heldout mechanism test. Do not repeat a viewed split to tune a positive method or require capabilities the original user did not request.

{scope}
'''}
    for name,text in reports.items():
        with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text)
    closure={'utc':utc(),'status':'C60_TWO_SYSTEM_CONTROL_COMPARISON_INTERPRETED','summary_sha256':sha(OUT/'SUMMARY60_V1.json'),'evaluation_sha256':sha(OUT/'EVALUATION60_V2.json'),'reports_sha256':{k:sha(ROOT/k) for k in reports},'max_primary_recomputation_error':maxerr,'selected_and_terminal_model_hashes_verified':True,'source_objects_verified':12,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'CLOSURE60_V1.json',closure);print(json.dumps(closure),flush=True)
if __name__=='__main__':main()
