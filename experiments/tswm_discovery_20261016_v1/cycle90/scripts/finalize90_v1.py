from common90_v1 import *
import shutil,subprocess

def md(name,s):
    with (C/name).open('x',encoding='utf-8') as f:f.write(s.strip()+'\n')

def main():
    assert not (O/'EVIDENCE_CLOSURE90_V1.json').exists()
    fit=read(O/'FIT90_2026901_V1.json');screen=read(O/'CAL_SCREEN90_2026901_V1.json');ordinary=read(O/'ORDINARY_CONTROL90_V1.json');library=read(O/'LIBRARY90_V1.json');classifier=read(O/'CLASSIFIER90_V2.json');confirm=read(O/'ORDINARY_CONFIRMATION90_V2.json');verify=read(O/'CONFIRMATION_VERIFICATION90_V1.json');nativeverify=read(O/'VERIFICATION90_V1.json');campaign=read(O/'campaign_v1/EXIT.json')
    assert campaign['returncode']==0 and campaign['completed_seeds']==[2026901]
    assert screen['status']=='CAL_COMPETENCE_FAIL_DO_NOT_OPEN_EVAL' and library['status']=='CAL_LIBRARY_PASS'
    assert not (O/'FIT90_2026902_V1.json').exists() and not (O/'FIT90_2026903_V1.json').exists()
    assert not (O/'COUPLING_CAL90_V1.json').exists()
    assert all(r['competence_screen_passed'] for r in confirm['summary'])
    livecmd="$p=@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {$_.CommandLine -match 'cycle90.*(capture90_v1|train90_v1|screen90_v1|classifier90_v[12]|confirm_ordinary90_v[12])'}); ConvertTo-Json -InputObject $p -Compress -Depth 2"
    result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',livecmd],capture_output=True,text=True,timeout=25);assert result.returncode==0 and json.loads(result.stdout or '[]')==[]
    table='| Ordinary control | System | Energy | Exact reference | Excess vs reference | Physical residual ratio | Event probability MAE (two events) |\n|---|---|---:|---:|---:|---:|---|\n'
    for r in confirm['summary']:
        excess=100*(r['energy_score']/r['reference_population_energy']-1)
        table+=f"| {r['arm']} | {'Duffing' if r['family']==0 else 'Reservoir'} | {r['energy_score']:.9f} | {r['reference_population_energy']:.9f} | {excess:.4f}% | {r['off_manifold_ratio']:.7f} | {r['event_probability_MAE']} |\n"
    md('literature_review.md','''# Literature and novelty boundaries

The detailed current source boundary is in LITERATURE_BOUNDARIES90_V1.md. Generic query/marginalization coherence, shared-mechanism branching, marginal-preserving noise couplings, Bayesian mixtures and ensemble reordering are occupied. This cycle does not introduce a new primitive or claim that changing the query from time/channel to action plan creates novelty.

Known methods succeeding in a privileged toy do not prove all learned-model settings solved. Here the evidence is stronger than the C89 oracle-ID stress: ordinary models actually learn nonlinear transition laws from TRAIN data and are checked on independent action queries. Nevertheless, their known fully observed Markov/polynomial assumptions delimit the result. No conclusion about arbitrary hidden-state observation maps or high-dimensional pretrained world models follows.

No final research direction is accepted. A future candidate must identify a distinct, useful feasible quality/physics/probability/cost tradeoff beyond these controls; mathematical consistency alone or one unsuitable neural model failing is insufficient.
''')
    md('baseline_registry.md',f'''# Model and evidence registry

| Model/control | Executed | Status |
|---|---|---|
| Native source-pinned CSDI | One seed,20000 updates,{fit['parameters']} parameters | Duffing CAL support passes; reservoir fails; later seeds and EVAL not opened |
| Empty-component-pruned unsupervised mixture | Same C89 fitted coefficients, TRAIN-only support repair | Both CAL and independent confirmation pass |
| Group-supervised polynomial library | Two closed-form TRAIN fits | Both CAL identification accuracies100%; independent confirmation passes |
| Flat MLP interpreter | One seed,10000 updates | CAL selection by CE gives accuracy{classifier['cal']['accuracy']:.6f}, macro{classifier['cal']['macro_accuracy']:.6f}; disqualified |
| Native semantic coupling comparison | Code/protocol authored | NOT executed because the generator did not qualify across both families |

Native source is unchanged `ermongroup/CSDI@7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`. It is a controlled cross-task transfer, not original-paper table reproduction. TRAIN8192/CAL256 arrays and original normalization/masks remain unchanged. The full-budget fit takes{fit['elapsed_seconds']:.3f}s; the selected step is{fit['selected_step']}. No new environment installation occurred.

The unsupervised safeguard removes Duffing components[0,4,8], total mass{ordinary['families'][0]['removed_probability']:.12g}, solely because expected TRAIN membership<1. All retained coefficients are byte-exact to the failed original checkpoint; no trajectory is removed or clipped. Reservoir retains all components. The original failure remains in C89.

The library uses TRAIN group identities, known component count, a degree5 fully observed Markov basis and family-specific parameters. The unsupervised mixture also has strong Markov/polynomial bias. These are positive controls, not a matched-information/capacity/compute leaderboard against CSDI. Deployment identification receives observed/generated states and actions, not physical parameter values or hidden class IDs.
''')
    md('experimental_results.md',f'''# Actual results and independent confirmation

## Training sufficiency check

Native CSDI completed exactly20000 updates. CAL denoising MSE={fit['cal_denoising_mse']:.9f}, selected step{fit['selected_step']}. Physical residual ratios are Duffing{screen['families'][0]['off_manifold_ratio']:.6f} and reservoir{screen['families'][1]['off_manifold_ratio']:.6f}. Both beat the deterministic posterior mean on trajectory energy; only Duffing meets the<=.5 support threshold. The overall configuration is disqualified. Seeds2026902/2026903 were not launched; sealed comparison EVAL seed2026905 was not generated. No qualified native coupling failure claim exists.

## Independent baseline-only confirmation

This separately registered confirmation uses fresh seed20269080,16 independent four-plan groups per family, no selection or tuning on these queries. Both ordinary models were already frozen and CAL-qualified. It is not a favorable subset of a native-model comparison.

{table}

Energy is evaluated for the exact learned finite-mixture law against the exact nine-world reference, without forecast Monte Carlo error. Physical ratio is mean nearest-reference trajectory RMS divided by the reference mixture spread. The two actual events are max abs x>.75 and terminal x>.25. They were fixed before opening the fresh queries. Grouped bootstrap intervals for energy improvements over the deterministic mean are all below zero; see the machine-readable receipt. One fit per arm is not optimizer-seed robustness.

## Common persistent component versus independent branch sampling

For K4, independent nearest-reference-bin agreement is about2.6–2.7%. Shared-component sampling yields89.60%/92.22% for the unsupervised mixture and100%/100% for the supervised library. Every learned single-query marginal is unchanged by construction. Bin agreement is descriptive, not a certificate of exact common continuous physics.

The independent dense161-point k-grid evaluator gives K4 common-parameter RMS residuals of0.003873/0.000919 for the unsupervised control and0.00000965/0.000857 for the library (TRAIN-standardized state units). The common-versus-separate parameter-fit excess is small but not identically zero for all learned models. Exact shared reference worlds have gap<=1e-12. These are approximate discrete-simulator checks, not continuous-time or parameter-identification theorems.

## Replay and failures

Native CAL metrics replay{nativeverify['metric_values_replayed']} values, maximum error{nativeverify['independent_gram_distance_max_abs']:.12g}. The independent confirmation replays{verify['values_recomputed']} values, maximum error{verify['max_abs_error']:.12g}. Data/payload/checkpoint hashes and source immutability are checked.

The first classifier invocation stopped at step0 because int32 labels were invalid for CUDA CE; V2 only makes labels torch.long. The first confirmation invocation stopped during import before generating queries; V2 only adds the frozen simulator module path. The source attempts remain. Neither repair changes the scientific protocol, hyperparameters, sample selection or reported metrics.
''')
    md('negative_evidence.md','''# Negative evidence and stopped routes

1. The3000-update CSDI pilot was not enough to exclude training insufficiency. A fixed20000-update fit substantially improves physical support, but reservoir still fails the registered screening threshold. Do not relabel it a qualified generator or extend training again on these outcomes.
2. A flat neural mechanism classifier is not automatically a reliable physical evaluator. CAL-selected accuracy is71.48%, macro53.35%; no coupling interpretation uses that head. A structured supervised transition library reaches100% on the same CAL recordings, so data non-identifiability is not an explanation for this specific classification failure.
3. The original polynomial model's numeric failure comes from an unsupported component. Removing all components with less than one expected TRAIN member changes only about3e-6 probability and fixes the ordinary pipeline. This is engineering, not innovation.
4. The repaired unsupervised model and the supervised library learn useful stochastic world laws on independent queries. Both preserve single-plan marginals under shared-component sampling, with strong physical/energy/risk results. The current controlled task does not establish a need for a new coupling architecture.
5. These positive controls have strong assumptions and do not solve all high-dimensional, partially observed or representation-shift settings. Their success is a local falsification of the current justification, not a theorem that the research field has no gap.
6. No native multi-action coupling result or independently confirmed harmful probability drift was obtained. Do not turn authored code or ineligible experiments into evidence. The sealed native EVAL remains unopened.

Decision: stop the current new-architecture route for this controlled setting. Do not spend more runs tuning CSDI, the failed MLP or a renamed Bayesian/IPF/ECC primitive to manufacture novelty.
''')
    md('hypothesis_space.md','''# Hypothesis update

| Hypothesis | Evidence and decision |
|---|---|
| CSDI pilot failure is entirely intrinsic to value-space stochastic generators | Unsupported; longer training improves support substantially, one family now passes |
| The fixed native configuration is qualified across the two systems after20000 updates | Rejected by reservoir CAL support ratio.6627 |
| Mechanism identities in the CAL data are irrecoverable | Rejected within the strong structured-library assumptions:100% identification |
| Polynomial mixture divergence demonstrates a scientific limitation | Rejected: an unsupported near-empty component explains the failure |
| New coupling architecture is necessary for the present controlled problem | Not supported; two ordinary learned controls succeed independently |
| Common latent index alone certifies exact shared true physics | False inference; independent physical trajectory fitting remains necessary |
| Strictly preserving an inaccurate base forecast is always preferable | Not assumed; actual energy/event calibration must be assessed |
| A distinct practical learned-model tradeoff remains beyond current controls | Unverified; needs concrete new evidence rather than a generic consistency claim |

The broader discovery objective remains active. No candidate is accepted merely because a narrow baseline or proof does not cover all possible settings.
''')
    md('research_ideas.md','''# Research candidate decision

The C89 candidate concerns probabilistically coherent multi-action time-series simulation. Its broad concepts overlap prior work, and its controlled new-method justification has now failed an ordinary learned-model confirmation. Retire the new-architecture route for this setting.

A future *evaluation* contribution remains conceivable only if qualified representative models expose a different, independently confirmed practical defect that existing shared-latent/coupling controls do not already resolve with comparable information/cost. No such residual is established here. Do not promote a partly observed or high-dimensional variant solely by changing task labels.

The reusable outcome is a pair of independently checked positive controls, native stochastic training/provenance, calibrated physical/outcome evaluation tools and a clear stopped route. These are evidence infrastructure, not the final research idea. Accepted final directions=0. No extra candidates are invented to fill a list.
''')
    md('final_research_direction.md','''# Decision: current method route stopped; full research goal unachieved

**Research Question:** Do practical time-series world models need a new mechanism to jointly simulate action alternatives with shared dynamics and trustworthy probabilities?

**Evidence:** In the present fully observed two-system setting, ordinary learned persistent mixtures achieve strong independent energy, physical residual and risk calibration while preserving per-action marginals. A native CSDI full-budget configuration remains unqualified across both systems. Thus the present evidence cannot justify a new coupling architecture or claim a native qualified-model coupling defect.

**Existing Methods:** See LITERATURE_BOUNDARIES90_V1.md. Shared-latent generation and marginal-preserving coupling are established. Current ordinary models provide executable positive controls, not just oracle mathematical existence arguments. Their known Markov/polynomial/group-supervision assumptions must remain explicit.

**Proposed Idea:** None accepted. Stop developing a new architecture for this controlled setting. A useful new composition/evaluation framework remains possible only after a distinct residual is empirically established.

**Falsification:** This cycle supplies the intended falsification: ordinary constructions learn useful, physically coherent action alternatives without query-induced single-plan probability changes. Do not keep the original architecture claim alive by excluding these controls or modifying success criteria after EVAL.

**Feasibility:** Selected generator/checkpoints, raw arrays, data, source and reports remain on the server; CPU positive-control fits and native RTX5090 training actually ran. Qualified ordinary controls can be reused in later appropriate settings. No global runtime/hardware changes occurred.

**Next Experiment:** Audit already completed native-model evidence for a concrete residual capability and select one decisive comparison with an independently frozen protocol. Do not launch another CSDI/classifier hyperparameter rescue or repeat generic C23–25 Bayesian updating/C39 forecast revisions/C82 counterfactual-coupling principles. Preserve the independent full-budget C71 campaign and inspect its final assessment when actually terminal. A new hypothesis requires a specific observed failure and closest feasible controls, not another text-only name.

**Recommendation:** No final direction can responsibly be recommended from C90. The complete research objective remains ACTIVE and UNACHIEVED; accepted final directions=0.
''')
    previous=ROOT/'CURRENT_RESEARCH_STATUS.md';archive=ROOT/'CURRENT_RESEARCH_STATUS_BEFORE90_V1.md';assert not archive.exists();shutil.copyfile(previous,archive)
    status='''# Current time-series world-model research status

C90 completes fixed20000-update native stochastic qualification and independent ordinary-model confirmation. **Full research goal ACTIVE/UNACHIEVED; accepted final directions0.**

Native CSDI seed2026901 completes all updates; Duffing CAL support passes, reservoir ratio.662679 fails. Seeds2/3 and sealed comparison EVAL are not opened. No native learned-generator coupling defect is claimed.

The old ordinary mixture failure is repaired by a TRAIN-only empty-component safeguard with unchanged coefficients. The repaired unsupervised mixture and a group-supervised polynomial library pass independently frozen confirmations in both systems, near exact reference energy and with small real-event probability errors. Shared persistent components preserve single-plan laws; independent physical fitting checks show small residuals. These strong low-dimensional Markov/polynomial controls do not prove every general world-model setting solved.

Decision: stop the current new-coupling-architecture route for this controlled task. Generic query/marginal consistency and shared physics are already occupied; no final novel framework is accepted. See cycle90/final_research_direction.md and all negative evidence. No further CSDI/MLP tuning is authorized by a positive interpretation of these results.

All C90 scientific workers are terminal. Independent C71 original-budget work remains preserved and was live at the last direct process inspection. All raw data, checkpoints and prediction arrays remain server-only. GitHub publication includes authored code/protocols, bounded metrics and seven reports.
'''
    with (ROOT/'CURRENT_RESEARCH_STATUS_V90.md').open('x',encoding='utf-8') as f:f.write(status)
    previous.write_text(status,encoding='utf-8')
    receipt=dict(utc=utc(),status='C90_CONTROLLED_NEW_ARCHITECTURE_ROUTE_STOPPED',previous_goal_turn='PROGRESS',current_goal_turn='PROGRESS',native_fits_completed=1,native_updates=20000,native_all_family_qualification=False,seeds2_3_launched=False,native_coupling_diagnostic_executed=False,sealed_native_eval_opened=False,ordinary_independent_arm_family_screens_passed=4,failed_MLP_retained=True,all_current_scientific_workers_terminal=True,full_goal_achieved=False,accepted_final_directions=0,method_route_stopped_for_current_setting=True,do_not_generalize_control_success_to_all_world_models=True,previous_canonical_sha256=sha(archive),current_canonical_sha256=sha(previous),evidence_receipts={n:sha(O/n) for n in ['FIT90_2026901_V1.json','CAL_SCREEN90_2026901_V1.json','ORDINARY_CONTROL90_V1.json','CLASSIFIER90_V2.json','LIBRARY90_V1.json','ORDINARY_CONFIRMATION90_V2.json','VERIFICATION90_V1.json','CONFIRMATION_VERIFICATION90_V1.json']})
    write(O/'EVIDENCE_CLOSURE90_V1.json',receipt);log(dict(stage='c90_evidence_closed',status=receipt['status'],accepted_final_directions=0))

if __name__=='__main__':main()
