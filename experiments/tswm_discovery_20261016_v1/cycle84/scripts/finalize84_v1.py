"""Build a server-only immutable publication snapshot after numerical/source checks."""
import json,hashlib,datetime,shutil,sys,platform
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'cycle84';O=C/'outputs';P=O/'publication_v1'
def digest(b):return hashlib.sha256(b).hexdigest()
def raw(path):return path.read_bytes()
def load(path):return json.loads(raw(path))
def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    b=data.encode() if isinstance(data,str) else data
    with path.open('xb') as f:f.write(b)
def obj(path,data):put(path,json.dumps(data,indent=2,allow_nan=False))

def main():
    r=load(O/'QUALIFICATION84_V1.json');v=load(O/'VERIFICATION84_V1.json')
    assert v['status']=='PASS' and v['result_sha256']==digest(raw(O/'QUALIFICATION84_V1.json'))
    assert r['protocol_sha256']==digest(raw(C/'QUALIFICATION_PROTOCOL84_V1.md'))
    originals=[O/'SOURCE_CONTRACT84_V1.json',O/'SOURCE_CSR84_V1.json']
    repos=[];verified=0
    for manifest in originals:
        m=load(manifest)
        for repo in m['repositories']:
            public={k:z for k,z in repo.items() if k not in ['source_root','selected']}
            public['selected']=[]
            for z in repo['selected']:
                b=raw(Path(z['server_cache']))
                assert len(b)==z['bytes'] and digest(b)==z['sha256']
                assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==z['git_blob_sha1']
                public['selected'].append({k:w for k,w in z.items() if k!='server_cache'});verified+=1
            repos.append(public)
    assert verified==328 and len(repos)==5
    P.mkdir(exist_ok=False)
    for p in sorted(C.rglob('*')):
        if not p.is_file() or 'outputs' in p.relative_to(C).parts or '__pycache__' in p.parts:continue
        if p.suffix not in ['.md','.py','.json','.ps1']:continue
        put(P/p.relative_to(C),raw(p))
    public_r=dict(r);public_r.pop('raw_array_path')
    public_r['raw_array_location']='server-artifact:cycle84/qualification_v1/arrays.npz'
    public_r['original_server_result_sha256']=digest(raw(O/'QUALIFICATION84_V1.json'))
    obj(P/'evidence'/'QUALIFICATION84_PUBLIC_V1.json',public_r)
    obj(P/'evidence'/'VERIFICATION84_V1.json',v)
    obj(P/'evidence'/'SOURCE_CONTRACT84_PUBLIC_V1.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='328_PINNED_SOURCE_FILES_REHASHED_NOT_EXECUTED',repositories=repos,original_manifest_sha256=[digest(raw(p)) for p in originals],cached_text_files=verified,local_result_files_written=0))
    report=f'''# C84 results: the broad attribution requirement is insufficient

The source audit identifies substantial prior art for observation adaptation, selective updates and residual compensation. The numerical qualification additionally rejects a universal requirement to name the changed internal component before producing correct observable action predictions. No new learned-model research gap is established.

| Check | Actual result |
|---|---:|
| Common-control trajectory pairs | {r['counts']['common_action_equivalence']} |
| Maximum observable discrepancy | {r['max_common_action_equivalence_error']:.12g} |
| Independent continuous-ODE pairs | {r['counts']['independent_continuous_fields']} |
| Maximum independent ODE discrepancy | {r['max_independent_ode_error']:.12g} |
| Matched action-response pairs checked | {v['common_action_response_pairs']} |
| Maximum paired response discrepancy | {v['max_paired_action_response_error']:.12g} |
| Ordinary affine fits / independent test rollouts | 8 / 64 |
| Maximum ordinary-fit rollout discrepancy | {v['max_ordinary_fit_rollout_error']:.12g} |
| Saved trajectory scalar values independently checked | {v['saved_trajectory_scalar_values_checked']} |
| Pinned public text sources rehashed | {verified} |

These are exact-construction numerical witnesses, not 192 independent scientific tasks or neural training seeds. The eight ordinary fits have the correct linear family, noiseless full observations and nested 64/256-transition budgets. Their success cannot establish nonlinear or partially observed adaptation performance. Independent exact linear propagation differs from the main RK4 reference by at most {v['max_exact_linear_reference_discrepancy']:.12g}; independent QR and SVD coefficient fits differ by at most {v['max_independent_qr_coefficient_difference']:.12g}.

## Why component labels do not define the objective

Let x'=f(x,u), y=g(x), with invertible g. An alternative model uses state z=g(x), identity observation, and z'=J_g(g^(-1)(z)) f(g^(-1)(z),u). For the same initial observable state and every allowed action plan, both models produce the same y. This is established state-space conjugacy. One tested nonlinear transformation also preserves the known input vector (0,1) and the zero reset; rank-deficient known actuation alone does not name every latent coordinate.

The 96 separate named-coordinate intervention pairs differ in the physical operator applied. They have positive recipient discrepancies, but those cannot be reported as action-response errors: do(x_0=0.8) and do(z_0=0.8) are different interventions after changing coordinates. Physical anchors or known kinematic structure can exclude the ambiguity; we make no claim that it survives those assumptions.

## Decision and limits

The generic attribution-first formulation is not an accepted direction. The narrower question of physically anchored response calibration remains only a development question. None of ILA, CSR, MoVie, Cross-Embodiment Transfer or AdaJEPA was executed here. This audit does not refute CSR's restricted theorem, reproduce an adaptation paper, demonstrate a failure after valid physical anchoring, or validate a new method.

Original research objective: ACTIVE / UNACHIEVED. Qualified final directions: 0. All raw arrays stay on the server; the GitHub snapshot contains code, compact results, source hashes and reports.
'''
    put(P/'experimental_results.md',report)
    readme='''# C84: qualify the shift-adaptation research question

Read [the results](experimental_results.md), [prior-art boundary](literature_review.md), [negative evidence](negative_evidence.md) and [current research judgment](final_research_direction.md).

This cycle completed an original numerical qualification and primary-source audit. It did not reproduce a neural adaptation baseline or establish a new research direction. The established conjugacy construction prevents an unanchored component label from being treated as a universal ground truth.

For a fresh numerical run, use Python 3.10 with NumPy and SciPy, place a STORAGE_LOCATION_V1.json containing an artifact_root entry in the project parent of cycle84, and point it to a fresh external artifact directory. Run scripts/qualification84_v1.py, then scripts/verify84_v1.py. Both scripts derive the project root from their saved location. Existing evidence and artifact directories are protected against overwriting; use a fresh checkout/output location for reproduction. The source-contract audit caches are required only for building the publication snapshot, not for reproducing the numerical qualification.

The frozen protocol is QUALIFICATION_PROTOCOL84_V1.md. Public evidence uses logical server artifact locations; raw signal, checkpoint and prediction payloads are not included. Four control seeds are specified; no neural training seeds are represented by the witness counts.
'''
    put(P/'README.md',readme)
    status=f'''# C84 current research judgment

Unanchored observation-versus-dynamics component attribution is not a sound universal endpoint: distinct internal repairs can agree on the allowed action interface. Numerical qualification and independent checks completed. Eight ordinary linear fits also predict that interface without a component label. Generic alignment, selective adaptation and residual repair have direct prior art.

The only retained development question concerns observable action-response calibration under explicitly stated physical anchors and coupled changes. No new neural-model failure or improving mechanism has been established. Qualified final directions: 0. Overall objective: ACTIVE / UNACHIEVED.

Completed evidence: 192 common-control witness pairs, 168 paired action responses, 12 independent ODE pairs, 8 ordinary fits and 64 fitted-model rollouts. Independent verification PASS. Five public repositories, 328 selected text files hash-verified. This is not a model-performance leaderboard.

C71's unchanged original-budget campaign is separate; C84 does not imply its completion. Raw arrays and cached source remain on the server. Only code, compact evidence and reports are uploaded to GitHub; no local result-payload synchronization.
'''
    put(ROOT/'CURRENT_RESEARCH_STATUS_V84.md',status)
    rows=[]
    for p in sorted(P.rglob('*')):
        if not p.is_file():continue
        b=raw(p);rows.append(dict(path=p.relative_to(P).as_posix(),server_snapshot_file=str(p),bytes=len(b),sha256=digest(b),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
    manifest=dict(schema='C84_PUBLICATION_V1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows,original_result_sha256=digest(raw(O/'QUALIFICATION84_V1.json')),local_result_files_written=0,qualified_directions=0)
    obj(P/'manifest.json',manifest)
    closure=dict(status='READY_FOR_GITHUB_QUALIFICATION_PUBLICATION_NOT_RESEARCH_COMPLETION',manifest_sha256=digest(raw(P/'manifest.json')),files=len(rows),source_files_checked=verified,numerical_verification='PASS',local_result_files_written=0,python=sys.version,platform=platform.platform())
    obj(O/'CLOSURE84_V1.json',closure);print(json.dumps(closure,indent=2),flush=True)

if __name__=='__main__':main()
