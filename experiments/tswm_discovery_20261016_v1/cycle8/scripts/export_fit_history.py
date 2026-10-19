"""Export original score arrays retained by fitted sklearn estimators."""
import runtime_profile
import json,pathlib
import joblib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 base=joblib.load(ROOT/'outputs/clock_rpm_controls/fitted_models.joblib');strong=joblib.load(ROOT/'outputs/strong_transition_controls/fitted_models.joblib');models={'clock':base['clock'],**{k+'_'+str(j):m for k,ms in strong.items() for j,m in enumerate(ms)}};rows=[]
 for name,m in models.items():rows.append({'model':name,'iterations':int(m.n_iter_),'training_score':m.train_score_.tolist(),'internal_validation_score':m.validation_score_.tolist(),'score_definition':'Native sklearn retained negative half squared-error scores; internal training-only split, not held-out recording or test validation.'})
 out=ROOT/'outputs/fitted_training_history.json';out.write_text(json.dumps({'scope':'Original retained estimator score arrays exported after terminal fits; not reconstructed or fabricated stdout. Per-recording raw metrics and ridge validation-alpha grids are stored separately.','rows':rows},indent=2),encoding='utf-8');print('Exported',len(rows),'original training histories')
if __name__=='__main__':main()
