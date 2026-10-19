"""Derived descriptive statistics and independent stronger-probe arithmetic/traversal."""
import runtime_profile
import hashlib,json,pathlib,time
import joblib
import numpy as np
import torch
import innovation_pilot as p
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/innovation_pilot';STRONG=ROOT/'outputs/strong_probes';OUT=ROOT/'outputs/analysis'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def manual_forest(forest,x,classifier):
    predictions=[]
    for estimator in forest.estimators_:
        tree=estimator.tree_;leaf=np.zeros(len(x),dtype='int64')
        for i in range(len(x)):
            node=0
            while tree.children_left[node]!=-1:
                node=tree.children_left[node] if x[i,tree.feature[node]]<=tree.threshold[node] else tree.children_right[node]
            leaf[i]=node
        value=tree.value[leaf]
        if classifier:
            value=value[:,0,:];predictions.append(value[:,1]/value.sum(1))
        else:predictions.append(value[:,:,0])
    return np.mean(predictions,axis=0)
def main():
    tick=time.perf_counter();torch.set_num_threads(1);assert json.loads((BASE/'independent_verification.json').read_text())['pass']
    result=json.loads((BASE/'results.json').read_text());fits={r['key']:r for r in json.loads((BASE/'fit_records.json').read_text())}
    stronger=json.loads((STRONG/'results.json').read_text());original={r['key']:r for r in result['rows']};OUT.mkdir(exist_ok=True)
    data={f:np.load(BASE/'data'/(f+'_test.npz'),allow_pickle=False) for f in p.FAMILIES};derived=[];floors={}
    for family,saved in data.items():
        raw=saved['current'];u=saved['action'];mean=saved['mean'];scale=saved['scale'];n=len(raw)
        lo=p.physical_step(raw[:,:2],np.full(n,.4),u,family);hi=p.physical_step(raw[:,:2],np.full(n,1.6),u,family)
        blind=(lo+hi)/2
        floors[family]=dict(empirical_gain_blind_oracle_MSE=float(np.mean(((blind-saved['future'][:,:2])/scale[:2])**2)),
            theoretical_gain_blind_MSE=(.36*.35**2/(2*scale[0]**2) if family=='coupled_linear' else
                (.36*.3**2/(2*scale[1]**2) if family=='rotating' else .36*(.4*np.tanh(1))**2/(2*scale[0]**2))))
    for row in result['rows']:
        file=BASE/'predictions'/row['prediction_file'];assert sha(file)==row['prediction_sha256'];pred=np.load(file,allow_pickle=False);saved=data[row['family']]
        labels=(saved['current'][:,:2]-saved['mean'][:2])/saved['scale'][:2]
        r=dict(row);r['current_physical_decoding_MSE']=float(np.mean((pred['current_probe'][:,:2]-labels)**2))
        r['gain_blind_oracle_MSE']=floors[row['family']]['theoretical_gain_blind_MSE'];derived.append(r)
    traversal_errors=[];strong_arithmetic=[]
    for row in stronger['rows']:
        model_file=STRONG/'checkpoints'/row['checkpoint'];predfile=STRONG/'predictions'/row['prediction_file']
        assert sha(model_file)==row['checkpoint_sha256'] and sha(predfile)==row['prediction_sha256']
        pred=np.load(predfile,allow_pickle=False);saved=data[row['family']];truth=(saved['future'][:,:2]-saved['mean'][:2])/saved['scale'][:2]
        accuracy=float(np.mean((pred['gain_probability']>.5)==(saved['current'][:,2]>1)))
        mse=float(np.mean((pred['successor_prediction']-truth)**2))
        strong_arithmetic.extend([abs(accuracy-row['tree_test_gain_accuracy']),abs(mse-row['tree_test_successor_MSE'])])
        state=torch.load(BASE/'checkpoints'/fits[row['key']]['checkpoint'],map_location='cpu',weights_only=True);world=p.Model(row['method']);world.load_state_dict(state['world_model']);world.eval()
        ready=p.prep({k:saved[k] for k in ['current','future','action','trajectory_id']},saved['mean'],saved['scale'],saved['rotation'])
        with torch.no_grad():encoded=world.encoder(torch.from_numpy(ready['x'][:128])).numpy()
        trees=joblib.load(model_file)  # Own diagnostic checkpoints, not external pickle files.
        gp=manual_forest(trees['classifier'],encoded,True)
        forecast=manual_forest(trees['regressor'],np.column_stack([encoded,ready['u'][:128]]),False)
        traversal_errors.extend([float(np.max(np.abs(gp-pred['gain_probability'][:128]))),float(np.max(np.abs(forecast-pred['successor_prediction'][:128])))])
    assert max(strong_arithmetic)<1e-12 and max(traversal_errors)<1e-12
    groups=[]
    metric_keys=['gain_accuracy','native_physical_MSE','frozen_encoder_successor_probe_MSE','native_modulation_relative_RMSE',
        'frozen_encoder_modulation_relative_RMSE','current_physical_decoding_MSE','oracle_successor_decoding_MSE','encoder_effective_rank','inverse_joint_action_accuracy']
    for family in p.FAMILIES:
        for method in p.METHODS:
            rows=[r for r in derived if r['family']==family and r['method']==method];g=dict(family=family,method=method,n_initialization_seeds=len(rows),metrics={})
            for key in metric_keys:
                values=[r[key] for r in rows if r[key] is not None]
                if values:g['metrics'][key]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),min=float(min(values)),max=float(max(values)))
            groups.append(g)
    strong_groups=[]
    for family in p.FAMILIES:
        for method in ['joint_idm','fullobs','joint_idm_full']:
            rows=[r for r in stronger['rows'] if r['family']==family and r['method']==method]
            strong_groups.append(dict(family=family,method=method,gain_accuracy_mean=float(np.mean([r['tree_test_gain_accuracy'] for r in rows])),
                gain_accuracy_min=min(r['tree_test_gain_accuracy'] for r in rows),gain_accuracy_max=max(r['tree_test_gain_accuracy'] for r in rows),
                selected_successor_MSE_mean=float(np.mean([r['selected_successor_test_MSE'] for r in rows]))))
    analysis=dict(groups=groups,derived_rows=derived,gain_blind_oracles=floors,stronger_probe_groups=strong_groups,
        independent_stronger_probe_check=dict(checkpoint_hashes_verified=27,prediction_hashes_verified=27,metric_values_recomputed=54,
            direct_tree_traversal_cases=27*128,max_metric_arithmetic_error=max(strong_arithmetic),max_tree_traversal_error=max(traversal_errors)),
        main_result_sha256=sha(BASE/'results.json'),strong_result_sha256=sha(STRONG/'results.json'),script_sha256=sha(pathlib.Path(__file__)),
        seconds=time.perf_counter()-tick,accepted_novel_directions=0,objective_achieved=False,
        scope='Descriptive pilot and post-pilot stronger probe. Gain-blind oracle assumes gain omitted but q,p,action available; near-chance probes do not prove exact zero mutual information.')
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8');print(json.dumps(dict(groups=len(groups),strong_probe_checks=analysis['independent_stronger_probe_check'],seconds=analysis['seconds'])),flush=True)
if __name__=='__main__':main()
