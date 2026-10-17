"""First new neural heldout scoring after all predeclared attempts and bank seals."""
from common60 import *
from model60_v1 import *
from gru60_v1 import GRUControl
import numpy as np,sys
from scipy.io import loadmat
from scipy.stats import t as student_t
sys.path.insert(0,str(PROJECT/'cycle59/scripts'))
from baselines59_v1 import features,predict_arx

def summarize(error):
    if not np.isfinite(error).all():return {'mean':None,'record_values':None,'record_interval95':None,'finite':False}
    v=error.mean(1);interval=None
    if len(v)>1:
        se=v.std(ddof=1)/np.sqrt(len(v));q=student_t.ppf(.975,len(v)-1);interval=[float(v.mean()-q*se),float(v.mean()+q*se)]
    return {'mean':float(v.mean()),'record_values':v.tolist(),'record_interval95':interval,'finite':True}

def main():
    torch.set_num_threads(4);source=source_check();models=read(OUT/'COMBINED_MODEL_SEAL60_V1.json');banks=read(OUT/'BANK_SEAL60_V1.json');data=read(OUT/'DATA60_V2.json')
    assert models['status']=='ALL_TWELVE_PREDECLARED_ATTEMPTS_TERMINAL_SELECTED_MODELS_SEALED'
    for row in models['models']:
        assert sha(OUT/'runs'/row['tag']/'result.json')==row['result_sha256']
        assert sha(row['selected']['checkpoint'])==row['selected']['checkpoint_sha256']
    for row in banks['models']:assert sha(row['checkpoint'])==row['checkpoint_sha256']
    dump(OUT/'TEST_OPEN60_V2.json',{'utc':utc(),'status':'FIRST_C60_NEURAL_TEST_SCORING_AFTER_FIXED_SELECTION','model_seal_sha256':sha(OUT/'COMBINED_MODEL_SEAL60_V1.json'),'bank_seal_sha256':sha(OUT/'BANK_SEAL60_V1.json'),'data_manifest_sha256':sha(OUT/'DATA60_V2.json'),'prior_metadata_audits_read_raw_arrays':True,'prior_C59_N87_scores_already_known':True,'test_based_selection':False})
    result_records=[];cache={};p=100;folder=paths()['artifact_root']/'test_scores_v2';folder.mkdir(exist_ok=False)
    old=PROJECT/'cycle59/outputs';rawsource=read(old/'DOWNLOAD59_V2.json');oldsplit=read(old/'SPLIT59_V1.json')['rows']['test']
    n87h=np.loadtxt(next(x['server_path'] for x in rawsource['files'] if x['path'].endswith('N87_5_H.csv.gz')),delimiter=',')[oldsplit]
    mat=loadmat(next(x['server_path'] for x in source['files'] if x['kind']=='data'));pph=np.asarray(mat['y']).reshape(-1)[50000::10]
    for system in ['N87','pick_place']:
        ds=data['systems'][system];row=next(x for x in ds['splits'] if x['split']=='test');assert sha(row['file'])==row['sha256']
        with np.load(row['file']) as a:u=a['u'].copy();y=a['y'].copy()
        truth=n87h[:,p:] if system=='N87' else pph[None,p:];sy=ds['normalization']['y_std'];my=ds['normalization']['y_mean'];assert truth.shape==y[:,p:,0].shape
        cache[system]={'u':u,'y':y,'truth':truth,'sy':sy,'my':my}
    for row in models['models']:
        rec=read(OUT/'runs'/row['tag']/'result.json');system=rec['system'];d=cache[system]
        model=GRUControl() if row['tag'].startswith('GRU_') else torch.jit.script(PostUpdateSSM());ck=torch.load(rec['selected']['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(ck['model'],strict=True);model.eval()
        u=torch.from_numpy(d['u']).transpose(0,1);y=torch.from_numpy(d['y']).transpose(0,1)
        with torch.no_grad():pred=model(u[:p],y[:p],u[p:]).transpose(0,1)[:,:,0].numpy().astype('float64')
        rawpred=pred*d['sy']+d['my'];error=np.square((rawpred-d['truth'])/d['sy']);metric=summarize(error)
        dest=folder/(row['tag']+'.npz')
        with dest.open('xb') as f:np.savez(f,prediction=rawpred,truth=d['truth'],squared_error_train_units=error)
        horizons=[1,16,128,min(512,error.shape[1]),error.shape[1]]
        result_records.append({'system':system,'method':'ordinary_gru' if row['tag'].startswith('GRU_') else 'native_component_nssm','seed':rec['seed'],'tag':row['tag'],'selected_step':rec['selected']['step'],'validation_mse':rec['selected']['validation_mse'],'parameters':rec['parameters'],'train_updates':rec['updates'],'training_seconds':rec['training_seconds'],'primary':metric,'mean_record_nrmse':float(np.mean(np.sqrt(np.mean(np.square(rawpred-d['truth']),1)/np.mean(np.square(d['truth']),1)))) if metric['finite'] else None,'horizon_prefix_mse':{str(h):float(error[:,:h].mean()) if np.isfinite(error[:,:h]).all() else None for h in horizons},'checkpoint_sha256':rec['selected']['checkpoint_sha256'],'arrays':str(dest),'arrays_sha256':sha(dest),'finite_rollout':metric['finite']})
        print(json.dumps({'status':'NEURAL_TEST_SCORED','tag':row['tag'],'primary_mse':metric['mean']}),flush=True)
    d=cache['pick_place'];F=features(d['u'][:,:,0].astype('float64'));h=d['y'][:,:,0].astype('float64')
    for row in banks['models']:
        with np.load(row['checkpoint']) as a:beta=a['beta']
        method=row['method']
        if method=='arx':
            fs=np.concatenate([F['static'],np.ones((*F['static'].shape[:2],1))],-1);pred=predict_arx(fs,beta,row['a'],h,p)
        else:pred=h[:,:p].mean(1,keepdims=True)+(F[method][:,p:]-F[method][:,:p].mean(1,keepdims=True))@beta
        rawpred=pred*d['sy']+d['my'];error=np.square((rawpred-d['truth'])/d['sy']);metric=summarize(error);dest=folder/f'pick_place_{method}.npz'
        with dest.open('xb') as f:np.savez(f,prediction=rawpred,truth=d['truth'],squared_error_train_units=error)
        result_records.append({'system':'pick_place','method':method,'seed':None,'tag':f'pick_place_{method}','validation_mse':row['validation_mse'],'parameters':row['coefficients'],'train_updates':0,'primary':metric,'checkpoint_sha256':row['checkpoint_sha256'],'arrays':str(dest),'arrays_sha256':sha(dest),'finite_rollout':metric['finite']})
    oldresults=read(old/'BASELINE_RESULTS59_V1.json');assert len(oldresults['models'])==10
    for row in oldresults['models']:
        if row['prefix']!=100:continue
        assert sha(row['prediction_file'])==row['prediction_sha256']
        result_records.append({'system':'N87','method':row['method'],'seed':None,'tag':'C59_'+row['tag'],'validation_mse':row['validation_mse'],'parameters':row['coefficient_count'],'train_updates':0,'primary':{'mean':row['test_mse_train_H_variance'],'record_values':row['test_row_mse'],'record_interval95':row['test_mse_row_interval95'],'finite':True},'arrays':row['prediction_file'],'arrays_sha256':row['prediction_sha256'],'finite_rollout':True,'historical_C59_control':True})
    obj={'utc':utc(),'status':'ALL_SELECTED_NSSM_GRU_AND_ORDINARY_CONTROLS_SCORED','records':result_records,'failed_training_attempts':models['failed'],'model_seal_sha256':sha(OUT/'COMBINED_MODEL_SEAL60_V1.json'),'bank_seal_sha256':sha(OUT/'BANK_SEAL60_V1.json'),'test_open_sha256':sha(OUT/'TEST_OPEN60_V2.json'),'script_sha256':sha(__file__),'scope':'Two separately fitted measured systems. N87 reuses a historically viewed exploratory split; pick-place has one complete chronological test record. No shared-checkpoint transfer or causal intervention claim.','overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'EVALUATION60_V2.json',obj)
if __name__=='__main__':main()
