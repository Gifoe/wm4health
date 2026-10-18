"""Privileged post-fit support diagnostic; never used to select/fit a model."""
from common85_v1 import *
from scipy.spatial import Delaunay,cKDTree

FOLDER=O/'attribution_v1';rows=[]
for p in sorted(FOLDER.glob('CASE85_ATTRIBUTION_*_V1.json')):
    case=json.loads(p.read_text());tag=f"brusselator_{case['seed']}_{case['case']}";seal=json.loads((FOLDER/f'SELECTION_SEALED85_ATTRIBUTION_{tag}_V1.json').read_text())
    assert sha(case['array_path'])==case['array_sha256'];d=np.load(case['array_path'],allow_pickle=False);cal=np.load(seal['calibration_path'],allow_pickle=False)
    physical=(d['physical_reference']-d['mu'])/d['scale']
    for allocation in ['short','long']:
        anchors=cal[allocation+'_anchor_source'];hull=Delaunay(anchors)
        outside=(hull.find_simplex(physical[:,1:].reshape(-1,2))<0).reshape(16,128)
        lo=anchors.min(axis=0);hi=anchors.max(axis=0);outbox=np.any((physical[:,1:]<lo)|(physical[:,1:]>hi),axis=-1)
        y,u=cal[allocation+'_y'],cal[allocation+'_u'];yv,uv=cal[allocation+'_val_y'],cal[allocation+'_val_u']
        train=np.column_stack([y[:,:-1].reshape(-1,2),u.reshape(-1,2)]);query=np.column_stack([d['reference'][:,:-1].reshape(-1,2),d['actions'].reshape(-1,2)]);validation=np.column_stack([yv[:,:-1].reshape(-1,2),uv.reshape(-1,2)])
        scale=np.maximum(train.std(axis=0),1e-8);tree=cKDTree(train/scale);vd=tree.query(validation/scale,workers=1)[0];qd=tree.query(query/scale,workers=1)[0]
        statehull=Delaunay(y[:,:-1].reshape(-1,2));outside_observed=statehull.find_simplex(d['reference'][:,:-1].reshape(-1,2))<0
        rows.append(dict(seed=case['seed'],case=case['case'],allocation=allocation,eval_frames=2048,physical_eval_outside_anchor_convex_hull_fraction=float(outside.mean()),physical_after_action_change_outside_anchor_fraction=float(outside[:,32:].mean()),physical_eval_outside_anchor_box_fraction=float(outbox.mean()),observable_eval_outside_train_state_hull_fraction=float(outside_observed.mean()),eval_nearest_training_feature_distance_median=float(np.median(qd)),eval_nearest_training_feature_distance_p95=float(np.quantile(qd,.95)),val_nearest_training_feature_distance_p95=float(np.quantile(vd,.95)),eval_fraction_beyond_val_distance_p95=float(np.mean(qd>np.quantile(vd,.95)))))
assert len(rows)==24
record=dict(status='POST_FIT_DIAGNOSTIC_COMPLETE',utc=utc(),rows=rows,physical_states_are_privileged_diagnostic=True,used_for_model_fitting_or_selection=False,distances_are_not_a_certificate=True,interpretation='Changed dynamics can move trajectories beyond the static physical anchor region. This is a support/identification confound; the audit does not establish an unavoidable failure or a new mechanism.',local_result_payload_files_written=0)
write(O/'COVERAGE85_V1.json',record)
print(json.dumps(record,indent=2),flush=True)
