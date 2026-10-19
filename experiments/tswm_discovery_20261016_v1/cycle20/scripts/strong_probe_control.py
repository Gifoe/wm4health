"""Exploratory stronger readout check after the initial pilot; frozen encoders, CPU."""
import runtime_profile
import hashlib,json,pathlib,sys,time
import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
import innovation_pilot as p
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/innovation_pilot';OUT=ROOT/'outputs/strong_probes'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    tick=time.perf_counter();torch.set_num_threads(1);assert json.loads((BASE/'independent_verification.json').read_text())['pass']
    assert not OUT.exists();OUT.mkdir(parents=True);(OUT/'checkpoints').mkdir();(OUT/'predictions').mkdir()
    fits=json.loads((BASE/'fit_records.json').read_text());original={r['key']:r for r in json.loads((BASE/'results.json').read_text())['rows']};rows=[]
    settings=dict(n_estimators=64,max_depth=16,min_samples_leaf=3,max_features=1.,n_jobs=1)
    for family in p.FAMILIES:
        data={name:np.load(BASE/'data'/(family+'_'+name+'.npz'),allow_pickle=False) for name in ['train','val','test']}
        ready={name:p.prep({k:s[k] for k in ['current','future','action','trajectory_id']},s['mean'],s['scale'],s['rotation']) for name,s in data.items()}
        for fit in [r for r in fits if r['family']==family and r['method'] in ['joint_idm','fullobs','joint_idm_full']]:
            checkpoint=BASE/'checkpoints'/fit['checkpoint'];assert sha(checkpoint)==fit['checkpoint_sha256']
            state=torch.load(checkpoint,map_location='cpu',weights_only=True);model=p.Model(fit['method']);model.load_state_dict(state['world_model']);model.eval()
            encoded={}
            with torch.no_grad():
                for name in ready:encoded[name]=model.encoder(torch.from_numpy(ready[name]['x'])).numpy()
            classifier=ExtraTreesClassifier(random_state=20262071+fit['seed'],**settings)
            classifier.fit(encoded['train'][:8192],data['train']['current'][:8192,2]>1)
            regressor=ExtraTreesRegressor(random_state=20262072+fit['seed'],**settings)
            regressor.fit(np.column_stack([encoded['train'][:8192],ready['train']['u'][:8192]]),ready['train']['future_labels'][:8192])
            val_accuracy=float(classifier.score(encoded['val'],data['val']['current'][:,2]>1));test_accuracy=float(classifier.score(encoded['test'],data['test']['current'][:,2]>1))
            val_pred=regressor.predict(np.column_stack([encoded['val'],ready['val']['u']]))
            test_pred=regressor.predict(np.column_stack([encoded['test'],ready['test']['u']]))
            val_MSE=float(np.mean((val_pred-ready['val']['future_labels'])**2));test_MSE=float(np.mean((test_pred-ready['test']['future_labels'])**2))
            old_val=min(np.asarray(fit['successor_probe_validation_MSE']).mean(1));selected_tree=bool(val_MSE<old_val)
            out=OUT/'checkpoints'/(fit['key']+'.joblib');joblib.dump(dict(classifier=classifier,regressor=regressor,settings=settings),out,compress=3)
            predfile=OUT/'predictions'/(fit['key']+'.npz');np.savez_compressed(predfile,gain_probability=classifier.predict_proba(encoded['test'])[:,1],successor_prediction=test_pred)
            row=dict(key=fit['key'],family=family,method=fit['method'],seed=fit['seed'],tree_validation_gain_accuracy=val_accuracy,tree_test_gain_accuracy=test_accuracy,
                original_test_gain_accuracy=original[fit['key']]['gain_accuracy'],tree_validation_successor_MSE=val_MSE,tree_test_successor_MSE=test_MSE,
                tree_selected_for_successor_by_validation=selected_tree,
                selected_successor_test_MSE=test_MSE if selected_tree else original[fit['key']]['frozen_encoder_successor_probe_MSE'],
                checkpoint=out.name,checkpoint_sha256=sha(out),prediction_file=predfile.name,prediction_sha256=sha(predfile))
            rows.append(row);print(json.dumps(dict(completed=len(rows),key=row['key'],gain_accuracy=test_accuracy)),flush=True)
    result=dict(rows=rows,settings=settings,seconds=time.perf_counter()-tick,script_sha256=sha(pathlib.Path(__file__)),
        scope='Exploratory post-pilot probe check on unchanged encoders/data; tree settings fixed before tree test outcomes. Not an independent confirmation, method update or proof of zero mutual information.',
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),accepted_novel_directions=0,objective_achieved=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(completed=27,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
