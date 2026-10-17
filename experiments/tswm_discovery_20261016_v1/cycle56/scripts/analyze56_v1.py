"""Close all fixed Monte Carlo cells and record narrow scientific conclusions."""
from common56 import *
import csv,io
import numpy as np

def table(rows):
    lines=['| r | T | fixed q | B | a-gradient deviation | MC SE | interval low | interval high |',
           '|---:|---:|:---:|---:|---:|---:|---:|---:|']
    for row in rows:
        s=row.get('primary_bonferroni_deviation',row['finite_reference_deviation'])
        lines.append(f"|{row['r']}|{row['T']}|{row['fixed_q']}|{row['B']}|{s['mean'][0]:.8g}|{s['mc_se'][0]:.3g}|{s['interval_low'][0]:.8g}|{s['interval_high'][0]:.8g}|")
    return '\n'.join(lines)

def main():
    pins=source_check();result=read(OUT/'GRADIENT56_V1.json')
    assert result['status']=='FINITE_REFERENCE_GRADIENT_AUDIT_COMPLETE_PENDING_INTERPRETATION'
    assert result['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
    rows=result['records'];assert len(rows)==144
    primary=[r for r in rows if r['primary_family_member']];assert len(primary)==8
    nonzero=[r for r in primary if r['primary_bonferroni_deviation']['interval_low'][0]>0 or r['primary_bonferroni_deviation']['interval_high'][0]<0]
    fixed=[r for r in rows if r['fixed_q']]
    max_cancel=max(abs(r['finite_reference_deviation']['mean'][0]) for r in fixed)
    assert max_cancel<1e-11,('Fixed-q cancellation failed',max_cancel)
    trace_receipts=[]
    for p in sorted(OUT.glob('case*_B*_j*.json')):
        rec=read(p);path=pathlib.Path(rec['path']);assert sha(path)==rec['sha256']
        with np.load(path) as arr:
            assert all(np.isfinite(arr[k]).all() for k in arr.files)
            for row in rec['rows']:
                k=f"fixed{int(row['fixed_q'])}_"
                dev=arr[k+row['reference']+'_gradient']-arr[k+'integrated_gradient']
                assert np.max(abs(dev.mean(0)-row['finite_reference_deviation']['mean']))<1e-12
        trace_receipts.append({k:rec[k] for k in ['case','B','jitter','path','sha256']})
    assert len(trace_receipts)==24
    for law in result['laws']:assert sha(law['path'])==law['sha256']
    qualify=read(OUT/'QUALIFICATION56_V1.json')
    status='FINITE_REFERENCE_DEVIATION_PRESENT_IN_PRIMARY_FAMILY' if nonzero else 'NO_PRIMARY_FINITE_REFERENCE_DEVIATION_DETECTED'
    summary={'utc':utc(),'status':status,'primary_tests':8,'primary_nonzero_a_deviations':len(nonzero),'primary_nonzero_cases':[{k:r[k] for k in ['case','r','T','fixed_q','B']} for r in nonzero],'fixed_q_max_absolute_paired_a_deviation':max_cancel,'all_diagnostic_rows':144,'independent_banks_per_cell':256,'native_gradient_max_error':max(r['errors']['gradient'] for r in qualify['native']),'sigma_point_max_error':max(r['max_error'] for r in qualify['sigma_point_expectation']),'central_difference_max_error':max(r['max_gradient_error'] for r in qualify['central_difference']),'native_training_updates':0,'accepted_final_directions':0,'overall_goal':'ACTIVE_UNACHIEVED','gradient_results_sha256':sha(OUT/'GRADIENT56_V1.json'),'qualification_sha256':sha(OUT/'QUALIFICATION56_V1.json'),'raw_gradient_receipts':trace_receipts,'raw_bank_receipts':result['laws']}
    dump(OUT/'SUMMARY56_V1.json',summary)
    buf=io.StringIO(newline='');writer=csv.writer(buf)
    writer.writerow(['case','r','T','B','actual_reference_size','targets','jitter','fixed_q','reference','parameter','population_gradient','raw_mean','cv_mean','deviation_mean','mc_se','ci95_low','ci95_high','primary_a_family','primary_ci_low','primary_ci_high'])
    for row in rows:
        s=row['finite_reference_deviation'];primary_s=row.get('primary_bonferroni_deviation')
        for k,param in enumerate(row['parameter_order']):
            writer.writerow([row['case'],row['r'],row['T'],row['B'],row['actual_reference_size'],row['targets'],row['jitter'],row['fixed_q'],row['reference'],param,row['population_gradient'][k],row['raw_gradient']['mean'][k],row['gradient_control_variate']['mean'][k],s['mean'][k],s['mc_se'][k],s['interval_low'][k],s['interval_high'][k],bool(primary_s and k==0),primary_s['interval_low'][k] if primary_s and k==0 else '',primary_s['interval_high'][k] if primary_s and k==0 else ''])
    with (OUT/'gradients56_v1.csv').open('x',encoding='utf-8',newline='') as f:f.write(buf.getvalue())
    primary_table=table(primary)
    trend=table([r for r in rows if r['reference']=='self' and not r['fixed_q'] and r['jitter']==1e-5])
    finding=(f"{len(nonzero)}/8 primary a-gradient deviation intervals exclude zero after the prespecified Bonferroni correction. "
             "The intervals describe Monte Carlo uncertainty at fixed known parameters, not learned-model generalization.")
    scope=('This is a scalar realizable Gaussian objective test. It does not establish the cause or effect size of C54 multidimensional learning failure, nonlinear applicability, a practical repair, or originality. The existing EM option changes recognition as well as prior learning; cancellation of its direct a-gradient deviation is not evidence that its full training succeeds. No final research direction is accepted.')
    reports={
        'README.md':f'''# C56 true-point finite-reference audit

{finding}

{primary_table}

The primary intervals are two-sided family-wise 0.05 intervals for eight a tests. Full gradients use the paired exact-reference control variate; raw gradients, recognition blocks, all reference sizes, external and leave-one-out controls remain in the JSON and CSV. Fixed-q a deviations cancel structurally (maximum absolute mean {max_cancel:.3g}); they do not validate the full EM learning trajectory.

{scope}

See PROTOCOL.md, DERIVATION.md and outputs/QUALIFICATION56_V1.json. Run qualify56_v1.py, run56_v1.py and analyze56_v1.py in a fresh output directory with the pinned source/runtime. start56_v1.ps1 durably runs qualification and Monte Carlo stages. Source pins, exact bank hashes and all raw replicate arrays are retained. Large arrays stay on configured server storage; no local result files were synchronized.
''',
        'experimental_results.md':f'''# C56 results

{finding}

Primary family:

{primary_table}

Descriptive native full-gradient reference-size sweep (95% intervals outside the B=32 primary family):

{trend}

There are 144 diagnostic rows (24 data-law/reference-size/jitter cells, two gradient conventions and three reference laws), 432 parameter summaries and 256 independent Monte Carlo banks per cell. Reference prefixes and target sequences are paired across controls. These are not neural training runs or real datasets. Full numeric records include population gradients under literal jitter, raw and integrated-target gradients, recognition blocks and all uncertainty estimates. No optimizer updates were performed.

{scope}
''',
        'negative_evidence.md':f'''# C56 negative evidence and limits

The fixed-posterior finite-reference a deviation cancels for algebraic reasons; stopping q does not supply an unbiased recognition update. C54 already showed that the released EM option does not consistently repair learned conditional-risk deficits. Those trained negative results remain unchanged.

{finding}

C55 rules out unavoidable ideal scalar population bias in the realizable case. C56 measures a different, finite-reference effect and does not reverse that conclusion. Large-bank trends are descriptive, and target-prefix averaging increases variance relative to using every target. No nonlinear or real-data validation, model improvement, novelty proof or accepted direction follows.
''',
        'hypothesis_space.md':f'''# C56 competing explanations

1. An unavoidable ideal scalar population optimum bias: rejected in the C55 realizable setting by the properness/tightness argument.
2. Finite-reference gradients differ at a known truth: tested here without optimizer trajectories. {finding}
3. Coupled recognition/prior updates move q away from correctness: remains plausible. The fixed-q direct a cancellation does not constrain recognition-block deviations or subsequent iterates.
4. Insufficient optimization, finite training data and multidimensional conditioning: remain possible contributors to C54. This diagnostic cannot assign their relative contributions.

The controls can separate stationary-point effects from optimization but cannot establish a field-wide research gap. A new architecture is not warranted by this test alone.
''',
        'baseline_registry.md':'''# C56 execution registry

Native derivative qualification: RP-GSSM a75225d5edceb972bb872dc9a335608733567dd4, 21 source files hash-checked unchanged; full and fixed-posterior loss derivatives qualified against unchanged source. Independent scalar equations follow C55, including native covariance jitter and recomputed adjacent cross covariance. Existing isolated JAX 0.4.35 CPU / NumPy 1.26.4 runtime, x64 enabled, four pinned logical CPU cores. No new installation or GPU training.

Four analytic data laws, no train/test selection. All bank seeds, configurations, protocol and code hashes, raw gradients, bank hashes and failed-run logs (if any) are retained. External and leave-one-out references are mathematical controls rather than original-paper reproductions.
''',
        'literature_review.md':'''# C56 novelty boundary

[RP-GSSM](https://proceedings.mlr.press/v306/hromadka26a.html) supplies the audited objective and already discusses approximation error. A finite-reference diagnostic is not by itself a new world-model method. [SogCLR](https://proceedings.mlr.press/v162/yuan22b.html) and [contrastive gradient-bias work](https://proceedings.neurips.cc/paper_files/paper/2022/hash/db174d373133dcc6bf83bc98e4b681f8-Abstract-Conference.html) preclude claiming generic global-normalizer correction as novel.

A fresh [biosignal world-model perspective](https://arxiv.org/html/2610.17510v1) explicitly discusses delayed effects, unobserved drivers, endogenous state and intervention validation. Those broad motivations are existing prior art, not new directions. It does not establish a correction to the scalar RP objective. [DoTime](https://github.com/thummd/dotime) and [EpiCF-Bench](https://github.com/complex-ai-lab/epi-cf-benchmark) are potential controlled-effect benchmarks, not executed C56 evidence. This is a boundary check, not a comprehensive new survey or evidence of either benchmark's current protocol validity.
''',
        'research_ideas.md':'''# C56 candidate disposition

No direction is accepted. Generic finite-batch normalizer correction has substantial prior art. Directly rebranding an exact Gaussian denominator or fixed-posterior update would not constitute an adequate contribution. A broader reusable world-model framework would need fresh nonlinear/controlled-system failures, competent existing repairs and held-out validation before a specific mechanism is proposed.
''',
        'final_research_direction.md':f'''# C56 research disposition

**Research Question:** Does the finite-reference native loss have a stationary-point deviation at a known correct population model?

**Evidence:** Unchanged-source AD qualification, independent analytic-reference checks, four fixed Gaussian laws and 256-bank gradient estimates. {finding}

**Existing Methods:** Proper Gaussian likelihood learning already solves the realizable linear setting; generic reference/gradient-bias corrections already exist. Their nonlinear applicability and practical cost must be established rather than renamed.

**Proposed Idea:** None accepted. This scalar diagnosis is not a reusable time-series world-model contribution.

**Falsification:** Zero population gradients, native derivative agreement, exact fixed-q a cancellation and all negative/noisy cells are retained. A claimed universal learned-dynamics bias is already contradicted by C55 and ordinary EM controls.

**Feasibility:** Existing CPU runtime, no new training, compact records on D:, larger raw banks on the configured G: artifact subtree, source/results published only to wm4health.

**Next Experiment:** Do not extend scalar controls into an open-ended hyperparameter sweep. Assess whether the candidate failure persists in a nonlinear controlled-system setting after competent existing normalizer/inference controls. If existing repairs close it, drop finite-normalizer correction as the research direction. A final broad framework and its minimal validation remain outstanding.

{scope}
'''}
    assert all(not (ROOT/p).exists() for p in reports)
    for p,text in reports.items():
        with (ROOT/p).open('x',encoding='utf-8') as f:f.write(text)
    assert source_check()==pins
    closure={'utc':utc(),'status':'C56_NUMERICS_AND_INTERPRETATION_CLOSED','summary_sha256':sha(OUT/'SUMMARY56_V1.json'),'report_sha256':{p:sha(ROOT/p) for p in reports},'csv_sha256':sha(OUT/'gradients56_v1.csv'),'analysis_script_sha256':sha(__file__),'all_raw_replicate_hashes_verified':True,'all_21_native_source_hashes_verified':True,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'CLOSURE56_V1.json',closure)
    print(json.dumps({'status':closure['status'],'primary_nonzero_a_deviations':len(nonzero),'fixed_q_a_cancel_max':max_cancel,'accepted_final_directions':0}),flush=True)

if __name__=='__main__':main()
