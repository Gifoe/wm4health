"""Frozen native-order qualification and paired simulator comparison; zero fitting."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import pathlib,json,hashlib,sys,datetime
import numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OLD=PROJECT/'cycle88';sys.path.insert(0,str(OLD/'scripts'))
from plan_models88_v1 import make_model,qualify_sources,tensor_state_sha
from plan_data88_v1 import advance,observe
from evaluate_plan88_v1 import predict
from gru_plan88_v1 import OrdinaryGRU
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,j):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
qualify_sources();STORAGE=read(PROJECT/'STORAGE_LOCATION_V1.json');OUT=ROOT/'outputs';OUT.mkdir(exist_ok=False);ART=pathlib.Path(STORAGE['artifact_root'])/ROOT.name;ART.mkdir(exist_ok=False)
row=next(x for x in read(OLD/'outputs/PLAN_DATA88_V1.json')['files'] if x['split']=='val');assert sha(row['path'])==row['sha256']
with np.load(row['path'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
START=48;T=96;Q=65;B=64;assert len(data['actions'])==B
obs=data['observations'][:,START:START+T+1];actions=data['actions'][:,START:START+T].copy();truth=obs[:,Q:];family=data['family'];theta=data['parameters'];center=data['center'];scale=data['scale']
permutations={'identity':[0,1,2,3],'swap_outer':[3,1,2,0],'swap_middle':[0,2,1,3],'reverse_blocks':[3,2,1,0]}
dump('MATRIX_V1.json',dict(protocol_sha256=sha(ROOT/'PROTOCOL.md'),validation_data_sha256=row['sha256'],models=['stock','lagged','acrkn','gru'],permutations=permutations,trajectories=B,optimizer_updates=0,checkpoint_reselection=False,independent_eval_accessed=False))
arrays=dict(family=family,system_id=data['system_id'],factual_truth=truth,actions=actions);branches={};replay=[]
for name,order in permutations.items():
 a=actions.copy();index=np.concatenate([np.arange(64+8*k,72+8*k) for k in order]);a[:,64:]=actions[:,index]
 assert np.array_equal(np.sort(a[:,64:],axis=1),np.sort(actions[:,64:],axis=1))
 target=truth.copy()
 for f in [0,1]:
  ids=np.flatnonzero(family==f);state=data['states'][ids,START+64].copy()
  for t in range(64,T):
   state=advance(state,a[ids,t].astype(np.float64),theta[ids],f);sensed=(observe(state)+data['observation_noise'][ids,START+t+1]).astype(np.float32);target[ids,t-64]=((sensed-center)/scale).astype(np.float32)
 if name=='identity':
  error=float(abs(target-truth).max());assert error<=1e-5;replay.append(dict(name='simulator_identity',max_abs_error=error));target=truth.copy()
 else:
  first=next(k for k in range(4) if order[k]!=k)*8
  assert np.array_equal(target[:,:first],truth[:,:first])
 branches[name]=(a,target,index);arrays[name+'_truth']=target
old_result=read(OLD/'outputs/PAIRED_PLAN_RESULTS88_V1.json');old_gru=read(OLD/'outputs/GRU_RESULTS88_V1.json')
assert sha(old_result['payload_path'])==old_result['payload_sha256'] and sha(old_gru['payload_path'])==old_gru['payload_sha256']
records=[];abstraction=[];fits=[]
for arm in ['stock','lagged','acrkn','gru']:
 fit=read(OLD/'outputs'/('GRU_FIT88_V1.json' if arm=='gru' else 'PLAN_FIT88_V1_'+arm+'.json'));assert sha(fit['selected_path'])==fit['selected_sha256']
 model=OrdinaryGRU().cuda() if arm=='gru' else make_model(arm);model.load_state_dict(torch.load(fit['selected_path'],map_location='cpu',weights_only=False)['model']);model.eval();initial=tensor_state_sha(model)
 base,var=predict(model,obs,actions);refpath=old_gru['payload_path'] if arm=='gru' else old_result['payload_path']
 with np.load(refpath,allow_pickle=False) as z:original=z[arm+'_base']
 error=float(abs(base-original).max());assert error<=1e-6;replay.append(dict(name=arm+'_original_forecast',max_abs_error=error));arrays[arm+'_base']=base
 if arm=='stock':
  def agg(a,t):
   x=torch.cat([a,t],dim=-1);m,v=model._absActEnc(x);prior,pc=model._intialize_mean_covar(len(a),diagonal=True);p,pv=model._action_Infer(prior,pc,m,v,None);return torch.cat([p,pv],dim=-1)
  with torch.no_grad():
   a=torch.as_tensor(actions[:,64:],device='cuda');time=model._create_time_embedding(B,32);original=agg(a,time)
   for name,order in permutations.items():
    pi=np.concatenate([np.arange(8*k,8*k+8) for k in order]);index=torch.as_tensor(pi,device='cuda');fixed=agg(a[:,index],time);joint=agg(a[:,index],time[:,index]);jn=float(abs(joint-original).max());bound=1e-6*max(1.,float(original.abs().max()));assert jn<=bound
    abstraction.append(dict(permutation=name,fixed_timestamp_max_abs_change=float(abs(fixed-original).max()),joint_action_timestamp_permutation_max_abs_change=jn,negative_control_bound=bound))
 for name,(a,target,index) in branches.items():
  pred,_=predict(model,obs,a);arrays[arm+'_'+name]=pred
  if name=='identity':assert np.array_equal(pred,base)
  for f in [0,1]:
   choose=family==f;effect=target[choose].astype(np.float64)-truth[choose].astype(np.float64);response=pred[choose].astype(np.float64)-base[choose].astype(np.float64);err=response-effect;den=float(np.mean(effect**2));num=float(np.mean(err**2));dot=float(np.sum(effect*response));norm=float(np.linalg.norm(effect)*np.linalg.norm(response));first=32 if name=='identity' else next(k for k in range(4) if permutations[name][k]!=k)*8
   records.append(dict(arm=arm,family=f,permutation=name,trajectories=int(choose.sum()),factual_mse=float(np.mean((base[choose]-truth[choose]).astype(np.float64)**2)),edited_mse=float(np.mean((pred[choose]-target[choose]).astype(np.float64)**2)),true_response_rms=float(np.sqrt(den)),model_response_rms=float(np.sqrt(np.mean(response**2))),response_error_rms=float(np.sqrt(num)),zero_response_reference_rms=float(np.sqrt(den)),response_error_over_zero_reference=float(np.sqrt(num/den)) if den>0 else None,response_cosine=dot/norm if norm>0 else None,endpoint_response_error_rms=float(np.sqrt(np.mean(err[:,-1]**2))),prefix_before_first_edit_max=float(abs(response[:,:first]).max()) if first else None))
 fits.append(dict(arm=arm,parameters=fit['parameters'],checkpoint_sha256=fit['selected_sha256'],selected_step=fit.get('selected_step',fit.get('selected',{}).get('step')),same_selected_weights=True));assert initial==tensor_state_sha(model) and sha(fit['selected_path'])==fit['selected_sha256'];del model;torch.cuda.empty_cache()
payload=ART/'order_predictions_v1.npz'
with payload.open('xb') as f:np.savez_compressed(f,**arrays)
qualify_sources();dump('RESULTS_V1.json',dict(status='FROZEN_ACTION_ORDER_DIAGNOSTIC_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=records,abstraction=abstraction,qualification=replay,fits=fits,payload_path=str(payload),payload_sha256=sha(payload),protocol_sha256=sha(ROOT/'PROTOCOL.md'),old_source_manifest_sha256=sha(OLD/'outputs/SOURCE_MANIFEST88_V1.json'),validation_data_sha256=row['sha256'],optimizer_updates=0,independent_confirmation=False,accepted_final_directions=0))
print(json.dumps(dict(status='FROZEN_ACTION_ORDER_DIAGNOSTIC_COMPLETE',abstraction=abstraction,rows=[r for r in records if r['permutation']!='identity'],optimizer_updates=0)),flush=True)
