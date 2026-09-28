"""Post-run alignment and completeness gates before reporting or publication."""
import json
import numpy as np
import pandas as pd
from common import *

REQUIRED=['data_summary.json','ensemble_model_summary.csv','support_reference_summary.json',
          'support_scores.csv','error_support_correlations.csv','failure_detection_metrics.csv',
          'risk_coverage_metrics.csv','horizon_reliability_metrics.csv','selective_rollout_metrics.csv',
          'subgroup_metrics.csv','integrity_checks.json']

def main():
    checks={}
    checks['required_metrics_exist']=all((OUT/x).exists() for x in REQUIRED)
    checks['figures_exist']=all((ROOT/'plots'/f'figure{i}_{name}.png').exists() for i,name in
      [(1,'support_vs_error'),(2,'failure_detection'),(3,'risk_coverage'),
       (4,'horizon_reliability'),(5,'representative_patients'),(6,'support_pca')])
    models=pd.read_csv(OUT/'ensemble_model_summary.csv')
    checks['five_seeds_exact']=models.seed.tolist()==CFG['seeds'] and (models.parameters==58946).all()
    scores=pd.read_csv(OUT/'support_scores.csv')
    checks['test_window_count_exact']=len(scores)==83198
    checks['test_case_anchor_unique']=not scores.duplicated(['caseid','anchor_t']).any()
    checks['core_scores_finite']=np.isfinite(scores[['U_ensemble','D_state_knn','D_action',
      'D_joint_knn','D_cond','D_perp','U_plus_best_geometry']]).all().all()
    source=np.load(SOURCE/'outputs/prediction_true.npz')['prediction']
    round3=np.load(ARRAYS/'test_pred_seed0.npy',mmap_mode='r')
    delta=float(np.max(np.abs(source-round3)))
    checks['seed0_predictions_identical_to_round2']=delta<1e-4
    checks['seed0_max_absolute_prediction_difference']=delta
    check=json.loads((OUT/'integrity_checks.json').read_text())
    checks['subject_splits_disjoint']=check['subject_splits_disjoint']
    checks['no_outcome_or_ce_in_support_features']=check['CE_absent_from_geometry'] and check['future_physiology_absent_from_geometry']
    main=pd.read_csv(OUT/'main_trust_table.csv')
    checks['main_table_complete']=all(x in main['Trust signal'].tolist() for x in
      ['Random','U_ensemble','D_raw_physiology','D_state_knn','D_state_mahalanobis',
       'D_action','D_joint_knn','D_joint_mahalanobis','D_cond','D_perp','U_plus_best_geometry'])
    checks['main_table_finite']=np.isfinite(main.select_dtypes(include='number')).all().all()
    jwrite('completion_checks.json',checks)
    print('COMPLETION',checks,flush=True)
    assert all(v for k,v in checks.items() if isinstance(v,(bool,np.bool_)))

if __name__=='__main__':main()
