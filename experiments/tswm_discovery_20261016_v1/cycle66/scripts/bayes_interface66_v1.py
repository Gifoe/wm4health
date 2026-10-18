import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native66_v1 import *
import ast,textwrap
q=read(OUT/'INTERFACE66_V2.json');assert q['status']=='NATIVE_PRETRAINED_INTERFACE_QUALIFIED_DIAGNOSTICS_ONLY';assert sha(q['array_artifact'])==q['array_sha256']
model,mod,manifest=load();lb=importlib.import_module('normwear66_native.latent_bayesian');z=np.load(q['array_artifact']);raw=torch.from_numpy(z['fixture']).to('cuda');mean=raw[:,:128].mean(1,keepdim=True);sd=raw[:,:128].std(1,keepdim=True);inp=((raw-mean)/(sd+1e-8)).permute(0,2,1).contiguous();inp[:,0,128:]=0
with torch.inference_mode():
 enc,_,_=model.forward_encoder(inp[:,:,:128],masking=False);action,_,_=model.forward_encoder(inp[:,[1],:],masking=False)
D=enc.shape[-1];assert enc.shape==(2,8,768);flat=enc.permute(1,2,0).flatten(1);indices=np.arange(1,2*D,2)
assert torch.equal(flat[:,indices],enc[1]);assert not torch.equal(flat[:,-D:],enc[1])
sentinel=torch.arange(2*3*4).reshape(2,3,4);sf=sentinel.permute(1,2,0).flatten(1);assert torch.equal(sf[:,1::2],sentinel[1])
orig=lb.get_traj_of_state;tree=ast.parse(textwrap.dedent(inspect.getsource(orig)));changed=0
for n in ast.walk(tree):
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='centroids_action_emb':
  assert isinstance(n.value,ast.Subscript);n.value.slice.elts[1]=ast.Name(id='_audit_action_indices66',ctx=ast.Load());changed+=1
assert changed==1;ast.fix_missing_locations(tree);scope=dict(lb.__dict__);scope['_audit_action_indices66']=indices;exec(compile(tree,'<authored_action_coordinate_index66_v1>','exec'),scope);repaired=scope['get_traj_of_state']
arrays={'input_normalized':inp.cpu().numpy(),'observed_encoder':enc.cpu().numpy(),'flattened_encoder':flat.cpu().numpy(),'future_action_encoder':action.cpu().numpy(),'action_indices':indices,'layout_sentinel':sentinel.numpy()};rows=[]
for name,conditional,fn in [('native_unconditional',False,orig),('native_conditional',True,orig),('indexed_conditional',True,repaired)]:
 lb.get_traj_of_state=fn;torch.manual_seed(2021);np.random.seed(2021)
 try:
  with torch.inference_mode():result=model.latent_bayesian_transition(inp,n_channels=2,physio_channels=[0],context_length=128,pred_length=16,num_states=3,action_channels=[1],condition_bayes=conditional,num_traj_sampled=4,latent_encoder=model,return_transit_matrix=True)
 finally:lb.get_traj_of_state=orig
 out,P,C,S,E=result;assert out.shape==(1,16) and torch.isfinite(out).all();assert np.isfinite(P).all() and np.max(np.abs(P.sum(1)-1))<1e-12;assert torch.equal(E,flat)
 arrays[name+'_output']=out.cpu().numpy();arrays[name+'_transition']=P;arrays[name+'_centroids']=C;arrays[name+'_std']=np.stack([S[i] for i in range(3)])
 rows.append(dict(name=name,shape=list(out.shape),finite=True,transition_shape=list(P.shape),max_transition_row_sum_error=float(np.max(np.abs(P.sum(1)-1))),state_clusters=len(C)))
ap=artifacts()/'BAYES_INTERFACE_ARRAYS66_V1.npz';assert not ap.exists();np.savez_compressed(ap,**arrays)
obj=dict(utc=utc(),status='NATIVE_INSIGHT_INTERFACE_EXECUTED_DIAGNOSTICS_ONLY',native_module_sha256=sha(pathlib.Path(lb.__file__)),model_files_receipt_sha256=sha(OUT/'MODEL_FILES66_V1.json'),source_revision=manifest['revision'],actual_history_encoder_shape=list(enc.shape),flattened_shape=list(flat.shape),native_suffix_contains_nonaction_coordinates=D//2,required_action_coordinate_indices=indices.tolist(),sentinel_correct_index_exact=True,real_encoder_correct_index_exact=True,original_and_indexed_sampler_executed=True,repair_scope='One centroid action-feature subscript only. Does not prove separate-channel and joint-channel latent geometries are aligned.',repair_AST_sha256=hashlib.sha256(ast.unparse(tree).encode()).hexdigest(),outputs=rows,indexed_vs_native_conditional_output_max_abs_change=float(np.max(np.abs(arrays['indexed_conditional_output']-arrays['native_conditional_output']))),arrays=str(ap),arrays_sha256=sha(ap),performance_benchmark=False,causal_identification_test=False,accepted_final_directions=0)
dump(OUT/'BAYES_INTERFACE66_V1.json',obj);print(json.dumps({k:v for k,v in obj.items() if k!='required_action_coordinate_indices'}),flush=True)
