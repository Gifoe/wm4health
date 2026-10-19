"""Independently enumerate Bayes risks, identity and global margin constants."""
import pathlib,json,hashlib,time
import numpy as np
from fractions import Fraction as F
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/compatibility_witness_v1'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
start=time.perf_counter();r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
assert r['status']=='passed_fixed_target_compatibility_witness'
assert sha(OUT/'arrays.npz')==r['arrays_sha256']
assert sha(ROOT/'scripts/run_compatibility_witness.py')==r['script_sha256']
assert sha(ROOT/'SURROGATE_COMPATIBILITY_PLAN.md')==r['plan_sha256']
manifest=json.loads((ROOT/'upstream/worldts_loss/manifest.json').read_text())
assert sha(ROOT/'upstream/worldts_loss/manifest.json')==r['source_manifest_sha256']
for row in manifest['files']:assert sha(ROOT/'upstream/worldts_loss'/row['path'])==row['sha256']
a=np.load(OUT/'arrays.npz');assert len(a['y'])==40
P=[[F(9,20),F(8,20),F(3,20)],[F(12,20),F(2,20),F(6,20)]]
Y=[-1,0,1]
raw_risks=[[sum(p*abs(value-act) for p,value in zip(prob,Y)) for act in Y] for prob in P]
raw_medians=[Y[min(range(3),key=lambda j:raw_risks[c][j])] for c in range(2)]
assert raw_medians==[0,-1]
bayes=sum(min(rows) for rows in raw_risks)/2
shared=min(sum(raw_risks[c][j] for c in range(2))/2 for j in range(3))
assert bayes==F(13,20) and shared==F(7,10) and shared-bayes==F(1,20)
assert abs(float(shared-bayes)-r['unavoidable_excess_MAE'])<1e-14
table=a['phi_table'];assert table.shape==(3,2)
assert np.linalg.matrix_rank(table[1:]-table[0])==2
# Direct affine identity independently solved from the three codebook rows.
coeff=np.linalg.solve(np.c_[table,np.ones(3)],np.array(Y,dtype=float))
reconstructed=np.c_[a['phi_samples'],np.ones(40)]@coeff
assert np.max(np.abs(reconstructed-a['y']))<1e-12
assert all(np.array_equal(np.bincount(((a['y'][a['context']==c])+1).astype(int),minlength=3),
                          [int(p*20) for p in P[c]]) for c in range(2))
coordinate_rows=[]
for c in range(2):
    conditional=a['phi_samples'][a['context']==c]
    med=np.median(conditional,axis=0);assert np.array_equal(med,[100,100])
    # Native mean-over-dimension scaling + balanced context contribution.
    highprob=[P[c][1],P[c][2]]
    margin=[F(5,1)*(1-2*p)/2 for p in highprob]
    coordinate_rows.append({'context':c,'latent_median':med.tolist(),
                            'joint_coordinate_L1_margins':[float(x) for x in margin]})
assert min(x for row in coordinate_rows for x in row['joint_coordinate_L1_margins'])==.5
for cert in r['certificates']:
    weight=cert['reg'];R=60+weight
    norm=np.linalg.norm([100,100])-R
    slope=.5-7.5/norm-weight/8
    assert slope>0 and abs(slope-cert['strict_joint_margin'])<1e-14
    assert abs(R-cert['outside_ball_radius'])<1e-14
# Independent grid of the native decoded MAE, including interval interiors.
grid=np.linspace(-2,2,4001)
gridrisks=np.array([np.abs(a['y']-pred).mean() for pred in grid])
assert abs(gridrisks.min()-.7)<1e-14 and grid[gridrisks.argmin()]==-1
mean=a['conditional_means'];code=np.c_[np.ones(2),(mean-100)/3]
probs=np.c_[1-code[:,1:].sum(1),code[:,1:]]
readouts=[]
for prob in probs:
    risk=np.array([np.sum(prob*np.abs(np.array(Y)-act)) for act in Y])
    readouts.append(Y[int(risk.argmin())])
assert readouts==raw_medians
for row in r['native_checks']:
    assert row['abs_error']<2e-10+2e-12*abs(row['numpy'])
result={'status':'passed','result_sha256':sha(OUT/'result.json'),'verifier_sha256':sha(__file__),
        'rational_raw_Bayes_MAE':str(bayes),'rational_shared_MAE':str(shared),'rational_excess':str(shared-bayes),
        'affine_identity_reconstruction_max_abs':float(np.max(np.abs(reconstructed-a['y']))),
        'global_margin_checks':len(r['certificates']),'coordinate_margin_checks':coordinate_rows,
        'native_formula_checks':len(r['native_checks']),'decoded_grid_points':len(grid),
        'ordinary_squared_target_and_refitted_MAE_readout':readouts,'seconds':time.perf_counter()-start,
        'scope':'fixed-target component witness only; no native trained-model failure or new-method claim'}
assert not (OUT/'verification.json').exists()
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
