import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from engine67_v2 import *
cm=read(OUT/'CASE_MANIFEST67_V1.json');rec=cm['subsets']['TRAIN'];assert sha(rec['path'])==rec['sha256'];z=np.load(rec['path']);engine=Engine();engine.set_case(z['history'][0],z['known'][0]);out,log=engine.forecast('bayes_ignored',2021)
torch.manual_seed(2021);np.random.seed(2021)
with torch.inference_mode():direct=engine.model.latent_bayesian_transition(engine.normalized,n_channels=4,physio_channels=[0],context_length=L,pred_length=H,num_states=10,action_channels=[1,2,3],condition_bayes=False,num_traj_sampled=4,latent_encoder=engine.model)[0]*engine.sd[0,0,0]+engine.mean[0,0,0]
error=float(np.max(np.abs(out-direct.cpu().numpy())));assert error==0
indexed,lg=engine.forecast('bayes_indexed',2021);original=engine.normalized.clone();engine.normalized[:,0,L:]=111;altered,_=engine.forecast('bayes_indexed',2021);assert np.array_equal(indexed,altered);engine.normalized=original
shapes={}
for mode in MODES[2:]:pred,_=engine.forecast(mode,2021);assert pred.shape==(H,) and np.isfinite(pred).all();shapes[mode]=list(pred.shape)
dump(OUT/'NATIVE_QUALIFICATION67_V1.json',dict(utc=utc(),status='PASS',fixture='first TRAIN case; no target scores',case=rec['cases'][0]['id'],exact_native_ignored_max_abs_difference=error,hidden_future_response_perturbation_max_abs_difference=float(np.max(np.abs(indexed-altered))),action_feature_control_shapes=shapes,model_parameters=sum(p.numel() for p in engine.model.parameters()),state_elements=sum(p.numel() for p in engine.model.state_dict().values()),sampler_AST_sha256=engine.sampler_AST_sha256,source_model_files_sha256=sha(PROJECT/'cycle66'/'outputs'/'MODEL_FILES66_V1.json'),train_case_receipt_sha256=sha(OUT/'CASE_MANIFEST67_V1.json'),scientific_benchmark_scores_computed=False));print(json.dumps(dict(status='PASS',native_max_abs_difference=error,hidden_future_max_abs_difference=0,control_shapes=shapes)),flush=True)
