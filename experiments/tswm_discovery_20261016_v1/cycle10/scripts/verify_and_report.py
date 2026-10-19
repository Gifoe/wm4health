"""Verify saved evidence and close the local audit; never reruns scientific fits."""
import runtime_profile
from fractions import Fraction as R
import datetime, hashlib, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def mat(rows): return [[R(str(x)) for x in row] for row in rows]
def eye(n): return [[R(i == j) for j in range(n)] for i in range(n)]
def transpose(a): return list(map(list, zip(*a)))
def mul(a, b): return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def sub(a, b): return [[x-y for x,y in zip(r,s)] for r,s in zip(a,b)]
def inv(a):
    n=len(a); b=[row[:] + ident for row,ident in zip(a,eye(n))]
    for i in range(n):
        k=next(k for k in range(i,n) if b[k][i]); b[i],b[k]=b[k],b[i]
        p=b[i][i]; b[i]=[v/p for v in b[i]]
        for j in range(n):
            if i != j:
                p=b[j][i]; b[j]=[x-p*y for x,y in zip(b[j],b[i])]
    return [row[n:] for row in b]
def strings(a): return [[str(x) for x in row] for row in a]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (ROOT/'ARTIFACT_MANIFEST.json').exists(), 'sealed cycle is immutable'
    witness=json.loads((OUT/'lag_mask/results.json').read_text())
    control=json.loads((OUT/'profile_probe/results.json').read_text())
    cfg=json.loads((OUT/'profile_probe/config.json').read_text())
    B=mat([[0,0,0],[1,0,0],[-1,'3/5',0]])
    B2=mat([[0,0,0],[1,0,0],[1,'-2/5',0]])
    S=mat([[1,'-1/15','2/3'],[0,1,0],[0,0,1]])
    D=mat([['1/5',0,0],[0,'1/2',0],[0,0,'7/10']])
    I=eye(3); L=inv(sub(I,B)); L2=inv(sub(I,B2)); O=mul(inv(L2),mul(S,L))
    F=mul(L,D); F2=mul(S,mul(F,inv(S))); D2=mul(inv(L2),F2)
    C=mat([[0,1,0],[0,0,1]])
    assert mul(O,transpose(O)) == I
    assert mul(C,inv(S)) == C
    # Equal structural noise covariance after the exact hidden-state change.
    SL=mul(S,L)
    assert mul(SL,transpose(SL)) == mul(L2,transpose(L2))
    assert mul(L2,D2) == F2
    proof={'arithmetic':'Python fractions.Fraction, no floating approximation',
           'B':strings(B),'B_alternative':strings(B2),'S':strings(S),
           'L':strings(L),'L_alternative':strings(L2),'O':strings(O),
           'D_at_scale1':strings(D),'D_alternative_at_scale1':strings(D2),
           'identities_verified':['O O^T = I','C S^-1 = C','S L L^T S^T = L2 L2^T','L2 D2 = S L D S^-1'],
           'effect_original':'3/5','effect_alternative':'-2/5',
           'scope':'Alternative cross-lag class; no positive-scale identification theorem for the original diagonal-lag class.'}
    write_json(OUT/'rational_proof.json',proof)
    rows=witness['native_checks']; checks=witness['matrix_checks']
    assert len(rows)==15 and len(checks)==5
    for row in rows:
        data=np.load(OUT/'lag_mask'/f"paths_scale{row['scale']:g}_seed{row['seed']}.npz")
        assert all(np.isfinite(data[k]).all() for k in data.files)
        first,second=data['observations'],data['observations_alternative']
        err=float(np.max(np.abs(first[:,1:]-second[:,1:])))
        assert np.isclose(err,row['native_observed_path_max_error'],atol=1e-15,rtol=1e-12)
        assert first.shape==second.shape==(512,3)
        for k,obs,truth,field in [('intervention',first,.6,'instant_effect_original'),('intervention_alternative',second,-.4,'instant_effect_alternative')]:
            effect=float(data[k][256,2]-obs[256,2])
            assert abs(effect-truth)<5e-6 and abs(effect-row[field])<1e-14
        assert np.max(np.abs(data['noise_alternative']-data['noise']@np.asarray([[float(x) for x in r] for r in O]).T))<1e-14
    assert len(control['profiles'])==10
    assert sum(len(r['all_starts']) for r in control['profiles'])==120
    for row in control['profiles']:
        assert row['best']['residual_l2']==min(x['residual_l2'] for x in row['all_starts'])
    probes=control['probe_runs']; assert len(probes)==1200
    assert len({(r['scale'],r['world'],r['n'],r['seed']) for r in probes})==1200
    for r in probes:
        assert r['sign_correct']==(r['slope']*r['truth']>0)
        assert r['covers_truth']==(r['low95']<=r['truth']<=r['high95'])
        assert r['interval_excludes_zero']==(r['low95']*r['high95']>0)
    for summary in control['probe_summary']:
        subset=[r for r in probes if r['world']==summary['world'] and r['n']==summary['n']]
        assert len(subset)==summary['runs']==200
        for raw,key in [('sign_correct','sign_correct_fraction'),('covers_truth','coverage95'),('interval_excludes_zero','excludes_zero_fraction')]:
            assert abs(np.mean([r[raw] for r in subset])-summary[key])<1e-14
        assert abs(np.sqrt(np.mean([(r['slope']-r['truth'])**2 for r in subset]))-summary['slope_rmse'])<1e-14
    source=json.loads((ROOT/'sources/dotime/SOURCE_PIN.json').read_text())
    for row in source['files']:
        assert digest(ROOT/'sources/dotime'/row['path'])==row['sha256']
        # Check the files actually imported by the native run, not just retained copies.
        assert digest(ROOT.parent/'upstream/dotime'/row['path'])==row['sha256']
    verification={'native_checks':15,'covariance_fits':120,'randomized_controls':1200,
        'unique_probe_seed_blocks_per_world_and_N':40,'pooled_scales_per_seed':5,
        'rational_identities_verified':True,'source_revision':source['revision'],
        'source_files_verified_against_execution_dependency':len(source['files']),
        'all_saved_paths_and_summaries_verified':True,
        'max_native_observed_error':max(r['native_observed_path_max_error'] for r in rows),
        'max_double_observed_error':max(r['exact_linear_observed_path_max_error'] for r in rows),
        'max_spectral_error':max(r['heldout_spectral_frobenius_max_error'] for r in checks),
        'objective_achieved':False}
    write_json(OUT/'evidence_verification.json',verification)
    profile_table='\n'.join(f"| {r['scale']:g} | {r['best']['residual_l2']:.9g} | {r['independent_spectral_max_error']:.9g} |" for r in control['profiles'] if r['fixed_b']==-.4)
    probe_table='\n'.join(f"| {r['world']} | {r['n']} | {r['sign_correct_fraction']:.3f} | {r['coverage95']:.3f} | {r['excludes_zero_fraction']:.3f} | {r['slope_rmse']:.6f} |" for r in control['probe_summary'])
    docs={
    'experimental_results.md':f'''# Cycle10 experimental evidence

All scientific jobs ran on the user's private Windows server. These are controlled source transfers, not a reproduction of official benchmark scores. No neural model or new method is trained here. Raw configs/logs/arrays/results are retained.

The unmodified pinned DoTime TemporalSCM, TemporalDAG and mechanism classes were instantiated with explicitly supplied linear parameters. The alternative permits cross-lags, outside the original diagonal-lag class except at zero scale. Five lag scales x three independent noise seeds =15 native paired runs. Full observed-process equality follows from the rational identities in `outputs/rational_proof.json`, including independent equal-variance Gaussian structural noise. Native float32 parameters in float64 buffers give maximum observed path difference {verification['max_native_observed_error']:.9g}; direct float64 recurrence maximum {verification['max_double_observed_error']:.9g}. Maximum independent-frequency spectral error {verification['max_spectral_error']:.9g}. Both systems are stable and no clipping occurs. Onset response is +0.6 vs -0.4 in every run.

## Existing diagonal-lag covariance-fit control

12 starts per fixed slope and scale;120 total fits. The true-slope warm start fits exactly. Opposite-slope local fits yield:

| lag scale | normalized covariance residual L2 | holdout spectral max error |
|---|---:|---:|
{profile_table}

Nonzero local residual is not a global identification certificate. Zero-scale numerical equality is consistent with the exact iid witness. No claim is made that positive scales identify the effect in the original class.

## Additional-information randomized control

OLS on independent one-step hard-do A=+/-1 experiments, with stationary initial states. These trials contain extra intervention labels; they are not an input-matched predictive-model improvement. The table pools five lag scales per each of40 seeds.200 rows per cell are correlated within seed block; no200-independent-seed significance claim is valid.

| world | N per trial | correct sign | 95% interval coverage | interval excludes0 | slope RMSE |
|---|---:|---:|---:|---:|---:|
{probe_table}

1200 control trials in total. The N=32 sign result is100% on this fixed finite design. It is not a general sample-complexity guarantee. Native witness elapsed {witness['elapsed_seconds']:.3f}s; follow-up controls {control['elapsed_seconds']:.3f}s, excluding transport/verification. No GPU-hour performance comparison is claimed.
''',
    'baseline_registry.md':'''# Cycle10 baseline and provenance registry

| component | role | protocol/status |
|---|---|---|
| thummd/dotime at8f227406a5b17c2040cfdcc680b493729ef47d00 | unmodified simulator classes |15 controlled linear paired runs; not released-suite replication |
| source-free linear recurrence/spectral covariance | independent contract |15 paths plus5 stationary/spectral checks; exact rational identities retained |
| SciPy least_squares | existing covariance matching |120 local fits, diagonal-lag model only; no global certificate |
| ordinary OLS + Student-t intervals | existing randomized experiment estimator |1200 trials; extra intervention data;40 unique seed blocks per world/N |

All code, fixed grids, seeds and raw numerical outputs are retained. `sources/dotime/SOURCE_PIN.json` hashes every relevant retained native source and verifies the actual imported dependency against it. The whole upstream native package was already retained and verified in sealed cycle9. No modification to prior experiments, native dependency or shared environment. Main runtime uses Python3.10.20 and Torch2.8+cu128 with the recorded runtime profile. No unexecuted paper method is represented as reproduced.
''',
    'negative_evidence.md':'''# Cycle10 negative evidence

Reject a proposed new predictive architecture justified solely by passive causal nonidentification: the exact pair is indistinguishable from all observed A,Y data. More capacity cannot distinguish it. Generic latent causal ambiguity, sensitivity analysis and experimental design already exist.

Reject a claim that a tiny relative lag change overturns any strongly persistent model. Scaling the off-diagonal norm toward zero here also scales all persistence toward zero. The relative perturbation is not uniformly small. This is compatible with weak identification near an iid boundary.

Reject an implementation-bug explanation of this specific witness: orthogonal structural-noise rotation, unchanged observed coordinates, stable dynamics, rational matrix equalities, source-free recurrences and independent spectra agree. Native float32 parameter roundoff remains explicitly measured.

Reject a claim that the original published diagonal-lag class has been disproved at positive lag scales: the alternative is outside that class. Failure of120 local fits to find another diagonal solution is not a theorem.

Reject a need for a new estimator on the randomized control: ordinary OLS obtains the correct sign for every N=32 trial. The extra data restore information, not neural architecture.

The onset+1 definition uses the factual onset A retrospectively to define a paired counterfactual slope. It is not a prospective deployment policy supplied with unavailable future observations. No test-set HPO or model selection was performed.
''',
    'hypothesis_space.md':'''# Cycle10 competing hypotheses

| hypothesis | discriminator | outcome |
|---|---|---|
| full observed equality is finite-sample coincidence | exact rational similarity/noise/projection identities; independent spectra | rejected for this constructed pair |
| equality requires correlated structural innovations | exact orthogonal O; iid spherical Gaussian innovations | rejected |
| observed units or channels were changed | C S^-1=C exactly | rejected |
| instability or clipping causes false counterfactual difference | spectral radii<1; amplitudes below native thresholds; direct recurrences | rejected |
| native diagonal-lag prior and all nearby broader models share the same effect | construct a cross-lag alternative | rejected as a universal claim; distance/persistence caveat essential |
| unchanged positive-lag diagonal class is itself nonidentified | local covariance optimization | unresolved; no global theorem |
| ordinary randomized data resolve the tested ambiguity | independent randomized hard-do OLS | supported at the tested sample sizes |
| this witness alone yields a new top-conference method | closest causal-ID/sensitivity literature and ordinary repair | unsupported |

No failed local optimization is promoted into proof. No new neural model is justified by a target whose value is not identified by its available inputs.
''',
    'research_ideas.md':'''# Cycle10 research candidate assessment

No direction is accepted. Do not fill a quota of three ideas.

Candidate: assumption-aware intervention prediction for time-series foundation models. Evidence: exact matching observed laws can support opposite effects after relaxing a lag restriction. Existing methods already cover latent graphical identification, corrected-past instrumental variables, causal sensitivity and posterior averaging. The witness establishes an evaluation boundary but does not show that these methods fail on an identifiable, informative task. A residual potentially worth testing is whether deployed amortized models can report the right *identified set* under explicitly supplied assumption uncertainty. This is a low-confidence question, not an innovation claim; static CSA-PFN and CAFE are mandatory prior-art controls, with their differing assumptions preserved.

Rescue condition: identify a realistic task with independently justified assumption uncertainty, show a repeatable defect in current strong methods, and demonstrate that an existing sensitivity/profile/posterior baseline does not solve it at comparable information and computation. Current linear toy does not meet that condition. Stop optimizing this example.

The older inverse-action lead is also not rescued by simply adding endpoint state-change grounding: PSG-JEPA already explicitly proposes multi-horizon state-change grounding and discusses inverse-action ambiguity. ACPC, CoCo and Dueling World Models cover related robustness/action-response diagnostics and ordinary contrastive/readout fixes. Their scopes differ; none is evidence that every possible residual is solved.
''',
    'final_research_direction.md':'''# Cycle10 decision: no final direction accepted

**Research Question:** How much does an inferred intervention response depend on an untested temporal structural restriction, even when all observed dynamics are matched?

**Evidence:**15 native paired checks and exact rational identities provide two stable Gaussian hidden-state systems with identical observed processes and opposite instantaneous slopes(+0.6,-0.4).120 covariance fits inspect the narrower diagonal class without claiming global identification.1200 randomized OLS controls show an existing information-restoring repair.

**Existing Methods:** Time-series causal identification, nuisance/corrected-past IV, latent VAR identification, sensitivity analysis and posterior counterfactual inference already address parts of this problem. Their assumptions must be checked separately. The present evidence does not establish their insufficiency on a common well-posed task.

**Proposed Idea:** None accepted. An assumption-aware identified-set output remains an unvalidated residual question, not a renamed novel method. No long training is justified on the current toy.

**Falsification:** An existing profile/sensitivity/posterior method solving the eventual matched task, or a target not identifiable from the declared information, rejects an architecture claim. This cycle already rejects the broad version.

**Feasibility:** Reproducible source-based construction and controls ran on the private RTX5090 Windows server using the project runtime; no new compute resource is required. Raw outputs and exact source pins are retained.

**Next Experiment:** Move away from this solved Gaussian construction. Audit a distinct real open-source mechanism, freeze a task with identifiable observable targets, and compare the current native implementation against a proper existing repair before designing a new method. Near-term source triage considers explicit temporal path laws and partially observed stochastic dynamics; no claim or job is reported until an executable discriminator is frozen.

The autonomous research objective remains active and unachieved. A completed audit is not a completed discovery goal.
'''
    }
    for name,text in docs.items(): (ROOT/name).write_text(text,encoding='utf-8')
    write_json(OUT/'task_status.json',{'cycle10':{'scientific_jobs_complete':True,'evidence_verification_passed':True,'accepted_directions':0},'objective_achieved':False,'decision':'retain model-class boundary evidence; reject broad novelty; change mechanism before further training','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps(verification,allow_nan=False))

if __name__=='__main__': main()
