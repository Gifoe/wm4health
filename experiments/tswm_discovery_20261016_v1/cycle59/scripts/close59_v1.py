"""Independently check stored predictions and preserve the bounded scientific conclusion."""
from common59 import *
import csv,io,numpy as np
def main():
    result=read(OUT/'BASELINE_RESULTS59_V1.json');seal=read(OUT/'MODEL_SEAL59_V1.json');split=read(OUT/'SPLIT59_V1.json');opening=read(OUT/'TEST_SCORE_OPEN59_V1.json');audit=read(OUT/'DATA_AUDIT59_V2.json');core=read(OUT/'CORE_QUALIFICATION59_V1.json');download=read(OUT/'DOWNLOAD59_V2.json')
    assert result['status']=='TEN_ORDINARY_CONTROLS_SCORED_ON_FIXED_HELDOUT_ROWS'
    assert core['status']=='PASS_EXTRACTED_OFFICIAL_EQUATION_CORES_AND_LINEAR_ANALYTIC_SOLVER'
    assert result['model_seal_sha256']==opening['model_seal_sha256']==sha(OUT/'MODEL_SEAL59_V1.json')
    assert seal['source_split_sha256']==sha(OUT/'SPLIT59_V1.json') and split['protocol_sha256']==sha(ROOT/'BASELINE_SCREEN59_V1.md')
    dt=datetime.datetime.fromisoformat
    assert dt(split['utc'])<dt(seal['utc'])<dt(opening['utc'])<dt(result['utc'])
    for f in download['files']:assert sha(f['server_path'])==f['sha256']
    rawH=np.loadtxt(data_root()/'N87_5_H.csv.gz',delimiter=',');rawB=np.loadtxt(data_root()/'N87_5_B.csv.gz',delimiter=',')
    tr=split['rows']['train'];test=split['rows']['test'];hs=float(rawH[tr].std())
    assert hs==seal['hscale_train_only'] and float(rawB[tr].std())==seal['bscale_train_only']
    maxerror=0.;points=[]
    for row in result['models']:
        assert sha(row['model_file'])==row['model_sha256'] and sha(row['prediction_file'])==row['prediction_sha256']
        with np.load(row['prediction_file']) as a:
            assert a['rows'].tolist()==test and np.array_equal(a['truth'],rawH[test,row['prefix']:])
            assert np.isfinite(a['prediction']).all()
            errors=np.square((a['prediction']-a['truth'])/hs).mean(axis=1)
            discrepancy=max(float(np.max(abs(errors-a['row_mse_train_scale']))),abs(float(errors.mean())-row['test_mse_train_H_variance']))
            maxerror=max(maxerror,discrepancy);assert discrepancy<1e-12
        points.append([row['tag'],row['prefix'],row['coefficient_count'],row['ridge'],row['a'],row['validation_mse'],row['test_mse_train_H_variance'],row['test_mean_row_nrmse'],row['test_median_row_nrmse']])
    assert len(points)==10 and len(seal['validation_candidates'])==80
    out=io.StringIO(newline='');writer=csv.writer(out);writer.writerow(['tag','prefix','coefficients','ridge','AR_a','validation_mse','test_mse_train_H_variance','test_mean_row_nrmse','test_median_row_nrmse']);writer.writerows(points)
    with (OUT/'results59_v1.csv').open('x',encoding='utf-8',newline='') as f:f.write(out.getvalue())
    table=['| History samples | Control | Coefficients | Test MSE / train H variance | Mean row NRMSE |','|---:|---|---:|---:|---:|']
    for row in result['models']:table.append(f"|{row['prefix']}|{row['method']}|{row['coefficient_count']}|{row['test_mse_train_H_variance']:.6g}|{row['test_mean_row_nrmse']:.6g}|")
    table='\n'.join(table)
    conclusion='Standard play plus exponential memory is useful on this heldout N87 split; adding those known primitives is not a new method. This result neither establishes an unexplained residual nor proves that all neural/state-space frameworks are unnecessary.'
    scope='One tutorial material, one temperature, 128 raw rows, one group-disjoint 76/25/27 split. No cross-material transfer, neural-seed stability, identified measurement noise floor, reset-history validation or full official trainer reproduction. All raw arrays were inspected for metadata before this screen; predictive test scores were first computed after the ten selected models were sealed. Overall goal remains active / unachieved; accepted final research directions: zero.'
    reports={
      'README.md':f'''# C59 real-measurement qualification and ordinary baseline screen

{conclusion}

{table}

38 official source/data objects are pinned to commit ab4913ca63ebda1f8eab429c64d3e93f96204713 and verified by byte count, Git blob identity and SHA256. Raw B/H/T remain on G:. Ten selected deterministic models come from 80 train/validation candidate fits; all candidates and coefficients are retained. Primary risk is dimensionless test MSE divided by train-only H variance; NRMSE uses each test row's true future RMS for evaluation only. Confidence intervals condition on this fixed split.

{scope}
''',
      'literature_review.md':'''# C59 literature and task boundary

Official MagNet Challenge 2 already asks for future H given past B/H, future B and temperature, including time-step and initialization handling: https://github.com/minjiechen/magnetchallenge-2 . The COMPEL 2025 foundation-model paper explicitly discusses frequency/time-step independence and finite-history assumptions: https://collaborate.princeton.edu/en/publications/unified-time-domain-foundation-models-for-hysteretic-passive-comp/ . These are direct prior capabilities, not newly discovered method ideas.

PI-HNO already combines recurrent dynamic response and global hysteresis context, with material-specific models on a 14-material dataset: https://arxiv.org/html/2608.02965v1 . Its global attention is Preisach-inspired, not an implementation of classical physical hysterons. The current tutorial sample has only N87 at 25 C and cannot be silently identified with that 14-material release. C58/research/adjacent58.md records mature Preisach/PNLSS/GLE controls and their assumptions. This screen does not reproduce any paper's reported table.
''',
      'baseline_registry.md':f'''# C59 baseline registry

Original tutorial sources and MPL2 license are unchanged. The existing JAX runtime lacks Equinox, Diffrax and Optimistix; no packages were installed. Exact decorator-free equation AST bodies and the original smooth functions were compared to independent NumPy math in 32 random cases (max scaled discrepancy {core['max_scaled_derivative_discrepancy']:.6g}); the linear equation was integrated by SciPy DOP853 against an analytic solution (max absolute discrepancy {core['linear_analytic_max_absolute_error']:.6g}). This qualifies those mathematical cores, not the full native model/Trainer.

Actual fitted controls are static nonlinear ridge, a 16-play bank, eight exponential states, their ordinary combination, and scalar stable ARX. Each has two independently fitted prefix conditions. Regularization and AR coefficient are chosen exclusively by validation; all 80 candidate scores are in MODEL_SEAL59_V1.json. The selected ten coefficient files and complete heldout predictions are retained with hashes on G:. These are adapted ordinary baselines, not new methods or an official benchmark reproduction.
''',
      'experimental_results.md':f'''# C59 executed results

{table}

All ten stored prediction arrays were independently checked against original raw target rows and re-scored. Maximum discrepancy: {maxerror:.6g}. Training-only scales, file hashes and split -> selected-model seal -> score receipt -> final result timestamp order passed. Original validation decisions and all candidate scores remain visible. Elapsed baseline script time: {result['seconds']:.6g} seconds on the existing CPU runtime, with no neural updates.

{conclusion}

{scope}
''',
      'negative_evidence.md':f'''# C59 negative evidence and limitations

Generic mixed-memory architecture necessity is not established: ordinary fixed play/relaxation banks already explain much of the static model's deficit. Their remaining error has no verified noise floor, unknown prehistory and finite feature/order effects. It must not be called a field-wide research gap.

The original periodic tutorial parser subtracts each full H row's mean, picks an H-dependent phase and circularly rolls the full record, then uses periodic interpolation. Those operations can be appropriate for its periodic demonstration but cannot be silently reused when future H is unavailable in a boundary-conditioned deployment protocol. The actual fitted screen uses raw chronological rows and prefix-only H calibration instead. This is task separation, not an accusation about the tutorial.

Only one material and one temperature are available in this sample. Clock-stretching the input while retaining its original H would invent a false response ground truth for rate-dependent physics; no such OOD test was performed. No measurement reset or original device IDs were inferred from filenames. {scope}
''',
      'hypothesis_space.md':'''# C59 competing hypotheses

1. Static input mapping is enough on these heldout rows: all static, ARX and memory outcomes are retained for comparison; static risk is substantially larger in this split.
2. Known persistent and decaying memory features explain a useful part of the error: supported descriptively by the ordinary hybrid bank; this is established modeling structure, not a new contribution.
3. A new learned memory framework is required: unproven. The screen lacks neural and fully fitted nonlinear-state-space comparators, independent systems and a noise/initialization attribution.
4. Preprocessing or split leakage explains apparent learning: whole-record H transformations and circular history were excluded; identical B+T rows form disjoint groups. Original acquisition/device independence remains external metadata, not proven by array hashes.
5. Remaining risk comes from unknown initial state, finite sample, basis choice or target noise: still live explanations; no architecture claim can bypass them.
''',
      'research_ideas.md':'''# C59 research disposition

No method is accepted. A generic context + play + exponential filters proposal has a competent ordinary implementation and close prior art. The result provides a real-data control rather than a new framework. Known components do not automatically make a broader efficient framework worthless, but any such framework must demonstrate a distinct useful capability or learning/compute advantage under fair context, state initialization and output-family budgets.

Next screen should obtain independent complete-run systems/materials with preserved chronology and physical time, verify context/initial-state contracts, and compare competent learned state-space / recurrent / proper generative controls at stated budgets. Do not claim cross-system benefit by relabeling this one-material row split or by increasing difficulty after seeing the test results.
''',
      'final_research_direction.md':f'''# C59 final direction status

**Research Question:** Can a defensible real-measurement conditional-response audit expose an unexplained world-model memory failure after ordinary controls?

**Evidence:** Pinned 38-object provenance, raw 128x5008 sample audit, official-equation core qualification, 80 deterministic candidate fits, ten sealed selected controls and independent heldout re-scoring. {conclusion}

**Existing Methods:** Classical play/relaxation, stable ARX, nonlinear state-space and the official ODE/neural tutorial are direct controls. MagNetX and PI-HNO already cover relevant conditional-response formulations.

**Proposed Idea:** None accepted. Combining known memory banks alone does not establish novelty; the residual has not been attributed.

**Falsification:** Any candidate whose gains disappear after fair initialization, ordinary mixed-memory controls, equal context or adequate nonlinear/state-space capacity should be rejected. This screen cannot by itself decide an unseen-system framework claim.

**Feasibility:** The actual screen required {result['seconds']:.6g} seconds and no neural updates; single-5090 compute remains available. Native full-trainer dependency compatibility and broad independent-system data are not yet qualified.

**Next Experiment:** Resolve independent complete-run data and initialization semantics before a shared neural/latent model comparison. Do not repeat the current fixed heldout split to tune a new method.

{scope}
'''}
    for name,text in reports.items():
        with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text)
    obj={'utc':utc(),'status':'C59_DATA_CORE_QUALIFICATION_AND_TEN_BASELINE_SCREEN_CLOSED','raw_rows':128,'split_counts':result['split_counts'],'candidate_fits':80,'selected_fits':10,'max_prediction_recomputation_error':maxerror,'source_objects_verified':len(download['files']),'reports_sha256':{k:sha(ROOT/k) for k in reports},'baseline_results_sha256':sha(OUT/'BASELINE_RESULTS59_V1.json'),'model_seal_sha256':sha(OUT/'MODEL_SEAL59_V1.json'),'core_qualification_sha256':sha(OUT/'CORE_QUALIFICATION59_V1.json'),'close_script_sha256':sha(__file__),'no_local_result_files_written':True,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'CLOSURE59_V1.json',obj);print(json.dumps(obj),flush=True)
if __name__=='__main__':main()
