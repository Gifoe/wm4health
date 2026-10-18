import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native66_v1 import *
q=read(OUT/'INTERFACE66_V2.json');b=read(OUT/'BAYES_INTERFACE66_V1.json');assert sha(q['array_artifact'])==q['array_sha256'] and sha(b['arrays'])==b['arrays_sha256']
z=np.load(q['array_artifact'],allow_pickle=False);w=np.load(b['arrays'],allow_pickle=False);errors={};counts=dict(array_artifact_hashes=2,loaded_npz_arrays=len(z.files)+len(w.files),interface_output_shapes=0,interface_scalar_metrics=0,transition_laws=0,CPU_GPU_native_outputs=0)
def check(name,actual,expected,tolerance):
 error=float(np.max(np.abs(np.asarray(actual,dtype='float64')-np.asarray(expected,dtype='float64'))));errors[name]=error;assert error<=tolerance,(name,error,tolerance)
for row in q['simulate_runs']:
 x=z[row['name']];assert list(x.shape)==row['shape'] and np.isfinite(x).all();counts['interface_output_shapes']+=1
 if row['name']!='native_late_covariate_perturbed':
  check(row['name']+'_known_output',np.max(np.abs(x[:,:160,1]-z['fixture'][:,:,1])),row['known_future_output_max_abs_change'],1e-7)
  u=z['fixture'][0,:,1].astype('float64');nu=(u-u.mean())/(u.std(ddof=1)+1e-8)
  trace=z[row['name']+'_encoder_trace1'][0,1,128:144];check(row['name']+'_encoder_known',np.max(np.abs(trace-nu[128:144])),row['second_encoder_known_first_future_block_max_abs_change'],1e-6);counts['interface_scalar_metrics']+=2
check('hidden_future_invariance',z['native_simulate'],z['native_hidden_perturbed'],0)
check('late_plan_early_response',np.max(np.abs(z['native_simulate'][:,128:144,0]-z['native_late_covariate_perturbed'][:,128:144,0])),q['late_covariate_perturbation_earlier_response_max_abs_change'],0)
assert w['observed_encoder'].shape==(2,8,768);enc=w['observed_encoder'];flat=enc.transpose(1,2,0).reshape(8,1536);assert np.array_equal(flat,w['flattened_encoder']);assert np.array_equal(flat[:,w['action_indices']],enc[1]);assert not np.array_equal(flat[:,-768:],enc[1])
for row in b['outputs']:
 name=row['name'];P=w[name+'_transition'];assert np.isfinite(w[name+'_output']).all() and w[name+'_output'].shape==(1,16);check(name+'_row_sum',np.max(np.abs(P.sum(1)-1)),row['max_transition_row_sum_error'],0);assert np.array_equal(P,w['native_unconditional_transition']);assert np.array_equal(w[name+'_centroids'],w['native_unconditional_centroids']);assert np.array_equal(w[name+'_std'],w['native_unconditional_std']);counts['transition_laws']+=1
check('indexing_only_changes_output',np.max(np.abs(w['indexed_conditional_output']-w['native_conditional_output'])),b['indexed_vs_native_conditional_output_max_abs_change'],0)
model,mod,manifest=load(device='cpu')
with torch.inference_mode():
 pred=model.predict(torch.from_numpy(z['fixture'][:,:128].copy()),prediction_length=32,max_pred_length=16)
 fresh_enc,_,_=model.forward_encoder(torch.from_numpy(w['input_normalized'][:,:,:128].copy()),masking=False)
check('CPU_GPU_predict',pred.numpy(),z['native_predict'],2e-4);check('CPU_GPU_history_encoder',fresh_enc.numpy(),enc,2e-4);counts['CPU_GPU_native_outputs']=2
obj=dict(utc=utc(),status='PASS',counts=counts,max_absolute_errors=errors,CPU_GPU_absolute_tolerance=2e-4,metric_float64_replay_tolerance=1e-6,native_fixture_receipt_sha256=sha(OUT/'INTERFACE66_V2.json'),bayes_fixture_receipt_sha256=sha(OUT/'BAYES_INTERFACE66_V1.json'),source_model_files_sha256=sha(OUT/'MODEL_FILES66_V1.json'),scope='Independent saved-array scalar/layout replay, normalized transition checks, exact shared fitted transition artifacts, fresh CPU native predict/encoder vs original GPU outputs. No task performance or causal truth.',accepted_final_directions=0)
dump(OUT/'INDEPENDENT_REPLAY66_V1.json',obj);print(json.dumps(obj),flush=True)
