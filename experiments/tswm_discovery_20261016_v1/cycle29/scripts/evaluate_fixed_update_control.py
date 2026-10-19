"""Original frozen test and scoring definitions for the registered fixed-update pair."""
import runtime_profile
import json,pathlib,time
import numpy as np
import torch
from world_data import ROOT,sha,dump
from world_models_v2 import build
from train_world_comparison_v2 import setup,SEEDS,hashes
from train_fixed_update_control import OUT,KINDS,control_hashes
from evaluate_world_comparison_v2 import load_test,neural_predict,metric_rows,csvwrite

def main():
    setup();summary=json.loads((OUT/'training_summary.json').read_text());assert summary['status']=='all_training_complete' and summary['runs']==6 and summary['control_hashes']==control_hashes()
    directory=OUT/'evaluation';assert not directory.exists();directory.mkdir();start=time.monotonic();data=load_test();rows=[];files=[]
    for seed in SEEDS:
        for kind in KINDS:
            folder=OUT/'training'/f'{kind}_{seed}';r=json.loads((folder/'result.json').read_text());assert r['steps']==32768 and r['best_weight_sha256']==sha(folder/'best.pt')
            state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);assert state['hashes']==hashes();model=build(kind).cuda().eval();model.load_state_dict(state['model'],strict=True)
            for family,test in data.items():
                pred,elapsed=neural_predict(model,test);assert np.isfinite(pred).all();path=directory/f'{family}_{kind}_{seed}.npz';np.savez_compressed(path,prediction=pred)
                files.append(dict(path=path.name,sha256=sha(path),weight_sha256=sha(folder/'best.pt'),seconds=elapsed));rows+=metric_rows(family,kind,seed,pred,test)
            del model;torch.cuda.empty_cache()
    assert len(rows)==18432;csvwrite(directory/'per_system_metrics.csv',rows)
    result=dict(status='completed',training_summary_sha256=sha(OUT/'training_summary.json'),plan_sha256=sha(ROOT/'FIXED_UPDATE_CONTROL_PLAN.md'),source_sha256=sha(pathlib.Path(__file__)),predictions=files,n_metric_rows=len(rows),seconds=time.monotonic()-start,scope='registered before test inspection; same original test, fixed additional control, not independent replication',accepted_final_directions=0)
    dump(directory/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='predictions'}),flush=True)

if __name__=='__main__':main()
