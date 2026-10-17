"""Record outcome-dependent negative evidence without editing locked reports."""
from common60 import *

def main():
    s=read(OUT/'SUMMARY60_V1.json');g=s['groups'];rows=[]
    assert s['neural_updates_completed']==60000 and not s['failed_training_attempts']
    for system,groups in g.items():
        assert all(groups[m]['all_expected_outcomes_finite'] for m in ['native_component_nssm','ordinary_gru'])
        best=min(['static','play','ema','hybrid','arx'],key=lambda m:groups[m]['mean_primary_all_expected'])
        n=groups['native_component_nssm']['mean_primary_all_expected'];b=groups[best]['mean_primary_all_expected']
        rows.append({'system':system,'native_nssm_three_seed_mean':n,'best_ordinary_bank_name':best,'best_bank_mse':b,'relative_reduction_vs_best_bank':1-n/b,'ordinary_gru_three_seed_mean':groups['ordinary_gru']['mean_primary_all_expected']})
    assert all(r['native_nssm_three_seed_mean']<r['best_bank_mse'] for r in rows)
    text='''# C60 outcome-dependent interpretation

All twelve fixed-budget neural fits completed and all twelve validation-selected complete test rollouts were finite. The ordinary residual neural state-space model, with its ordinary prefix state estimator, achieved substantially lower observed error than all five fixed memory-bank controls on both measured mechanisms. It also had lower three-seed mean error than the parameter-comparable ordinary GRU under this fixed-update recipe. Devices, runtime and state dimension differ; this is not a matched-compute neural ranking.

'''
    for r in rows:
        text+=f"- {r['system']}: NSSM mean {r['native_nssm_three_seed_mean']:.9g}; strongest fixed bank {r['best_ordinary_bank_name']} {r['best_bank_mse']:.9g}; descriptive relative reduction {100*r['relative_reduction_vs_best_bank']:.3f}%; GRU mean {r['ordinary_gru_three_seed_mean']:.9g}.\n"
    text+='''
These outcomes contradict the proposed necessity, on this evidence, for a new persistent/decaying-memory mechanism. They also show that selected-model test instability is not the surviving failure: transient full-validation divergence occurred during optimization, but the final sealed selected models were finite. Claiming a structural simulation failure from those transient checkpoints would be misleading.

The stronger conclusion is limited to these two separately fitted measured systems and this recipe. N87 is an exploratory reuse of the previously viewed C59 split. Pick-place is a chronological segment of one physical record. No independent-device statistical generalization, shared pretrained framework, stochastic uncertainty calibration or causal identifiability is established. A single fitted architecture reused across these systems is a conventional baseline, not a new world-model contribution.

Decision: retire mixed persistent/relaxing-memory necessity as a currently supported novel-method route. Preserve it as negative evidence. Search for a materially different, demonstrable capability deficit in an adequate existing world model; do not spend another cycle tuning these viewed test splits to produce a positive result. Accepted final directions remain zero, and the overall research goal remains active and unachieved.
'''
    with (ROOT/'INTERPRETATION60_V1.md').open('x',encoding='utf-8') as f:f.write(text)
    dump(OUT/'INTERPRETATION60_V1.json',{'utc':utc(),'status':'ORDINARY_NSSM_REPAIRS_BOTH_MEASURED_CONTROLS_MEMORY_NECESSITY_NOT_SUPPORTED','systems':rows,'summary_sha256':sha(OUT/'SUMMARY60_V1.json'),'report_sha256':sha(ROOT/'INTERPRETATION60_V1.md'),'accepted_final_directions':0,'overall_goal':'ACTIVE_UNACHIEVED'})
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
