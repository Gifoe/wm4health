"""Frozen selected-weight Validation measurement and predeclared branch; no selection on Test."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,sys,json,hashlib,csv,types,copy,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=pathlib.Path('D:/baseline_reproduction_seed2021_v1');SRC=BASE/'upstream/LatentTSF'
sys.path.insert(0,str(BASE/'scripts'));import common as original
sys.path.insert(0,str(SRC));import torch,numpy as np
from my_utils import model_dict,get_data
from my_AE import get_autoencoder
from my_train import encoder_encode,encoder_decode,build_dec_inp_latent,build_dec_inp_original
OUT=ROOT/'outputs';cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert read(OUT/'decoded_controls_v1_exit.json')['exit_code']==0 and read(OUT/'decoded_control_v1/RESULTS_V1.json')['status']=='SIX_OFFICIAL_DECODED_CONTROLS_CLOSED'
dest=OUT/'decoded_validation_v1';dest.mkdir(exist_ok=False)
arrays=pathlib.Path(cfg['artifact_root'])/ROOT.name/'decoded_validation_v1';arrays.mkdir(exist_ok=False)
fixed=read(OUT/'matched_cap_v1/RESULTS_V1.json');new=read(OUT/'decoded_control_v1/RESULTS_V1.json');rows=[]
cells=[dict(x,mode=x['mode']) for x in fixed['rows']]+[dict(x,mode='decoded') for x in new['rows']]
for cell in cells:
 mode=cell['mode'];run=pathlib.Path(cell['metrics_path']).parent;m=read(run/'metrics.json');args=types.SimpleNamespace(**read(run/'resolved_config.json'))
 assert sha(run/'metrics.json')==cell['metrics_sha256'] and sha(m['checkpoint_path'])==m['checkpoint_sha256'] and sha(run/'predictions.npz')==m['predictions_sha256']
 assert args.step==60 and args.label_len==0 and args.seq_len==720 and args.batch_size==32 and m['epochs']==100
 # Independent Test arithmetic is a verification only; it never enters the branch.
 if mode=='decoded':
  with np.load(run/'predictions.npz',allow_pickle=False) as z:
   ss=sa=0.;n=0
   for i in range(0,len(z['pred']),17):
    e=z['pred'][i:i+17].astype(np.float64)-z['true'][i:i+17];ss+=np.einsum('ijk,ijk->',e,e);sa+=abs(e).sum(dtype=np.float64);n+=e.size
  assert abs(ss/n-m['reproduced_mse'])<2e-11 and abs(sa/n-m['reproduced_mae'])<2e-11
  assert read(run/'config.json')['arguments']['perceptual_weight']==1 and read(run/'config.json')['arguments']['mse_weight']==0 and read(run/'config.json')['arguments']['cosine_weight']==0
 aearg=read(run/'config.json');enc=None;ae_before=None
 if mode!='raw':
  assert sha(args.autoencoder_path)==aearg['ae_checkpoint_sha256'];enc=get_autoencoder(args).float().cuda();enc.load_state_dict(torch.load(args.autoencoder_path,map_location='cpu',weights_only=False),strict=True);enc.eval();enc.requires_grad_(False);ae_before=original.tensor_sha(enc)
 modelargs=copy.copy(args)
 if enc is not None:modelargs.enc_in=modelargs.dec_in=modelargs.c_out=args.d_model
 model=model_dict[args.model].Model(modelargs).float().cuda();model.load_state_dict(torch.load(m['checkpoint_path'],map_location='cpu',weights_only=False),strict=True);model.eval();model_before=original.tensor_sha(model)
 data,_=get_data(args,flag='val');loader=torch.utils.data.DataLoader(data,batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=0)
 assert sha(pathlib.Path(args.root_path)/args.data_path)==m['data_sha256']
 om=[];oa=[];H=cell['horizon']
 with torch.no_grad():
  for bx,by,bxm,bym in loader:
   bx=bx.float().cuda();by=by[:,-H:,:].float();bxm=bxm.float().cuda();bym=bym.float().cuda()
   if enc is None:pred=model(bx,bxm,build_dec_inp_original(by,0,H,'cuda'),bym)[:,-H:,:]
   else:
    zx=encoder_encode(enc,bx,'AE',None);zp=model(zx,bxm,build_dec_inp_latent(zx,0,H,args.d_model,'cuda'),bym)[:,-H:,:];pred=encoder_decode(enc,zp,'AE')
   e=pred.cpu().numpy().astype(np.float64)-by.numpy();assert np.isfinite(e).all() and e.shape[1:]==(H,321)
   om.extend(np.mean(e*e,axis=(1,2)).tolist());oa.extend(np.mean(abs(e),axis=(1,2)).tolist())
 assert len(om)==len(data) and sha(m['checkpoint_path'])==m['checkpoint_sha256'] and original.tensor_sha(model)==model_before
 if enc is not None:assert original.tensor_sha(enc)==ae_before and sha(args.autoencoder_path)==aearg['ae_checkpoint_sha256']
 hist=list(csv.DictReader((run/'history.csv').open(encoding='utf-8-sig')));assert len(hist)==100
 selected=min(hist,key=lambda x:float(x['valid_loss']));assert int(selected['epoch'])==m['best_epoch']
 p=arrays/f"{mode}_H{H}_seed{cell['seed']}.npz";np.savez_compressed(p,origin_mse=np.array(om),origin_mae=np.array(oa))
 row=dict(mode=mode,horizon=H,seed=cell['seed'],validation_origins=len(om),validation_mse=float(np.mean(om)),validation_mae=float(np.mean(oa)),native_selection_statistic=float(selected['valid_loss']),selection_epoch=m['best_epoch'],test_mse=m['reproduced_mse'],test_mae=m['reproduced_mae'],checkpoint_sha256=m['checkpoint_sha256'],metrics_sha256=cell['metrics_sha256'],arrays=dict(path=str(p),sha256=sha(p)),epoch_reselected=False)
 with (dest/f"{mode}_H{H}_seed{cell['seed']}.json").open('x') as f:json.dump(row,f,indent=2)
 rows.append(row);print(json.dumps({k:row[k] for k in ['mode','horizon','seed','validation_mse','native_selection_statistic','test_mse']}),flush=True)
 del model,enc,data,loader;torch.cuda.empty_cache()
summary=[]
for h in (336,720):
 q=dict(horizon=h)
 for mode in ('raw','latent','decoded'):
  selected=[x for x in rows if x['mode']==mode and x['horizon']==h];q[mode+'_validation_mean']=float(np.mean([x['validation_mse'] for x in selected]));q[mode+'_test_mean']=float(np.mean([x['test_mse'] for x in selected]))
 q['decoded_to_raw_validation_ratio']=q['decoded_validation_mean']/q['raw_validation_mean'];q['within_registered_validation_band']=q['decoded_to_raw_validation_ratio']<=1.05;summary.append(q)
branch='STOP_CURRENT_ARCHITECTURE_MOTIVATION_OBJECTIVE_CONTROL_SUFFICIENT' if all(x['within_registered_validation_band'] for x in summary) else 'RUN_REGISTERED_ITRANSFORMER_COMPATIBILITY_CONTROLS'
obj=dict(status='EIGHTEEN_FROZEN_SELECTED_MODELS_VALIDATION_MEASURED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),table=rows,summary=summary,decision=branch,decision_uses='deterministic full Validation MSE only',threshold=1.05,selected_epochs_unchanged=True,protocol_sha256=sha(ROOT/'OFFICIAL_DECODED_CONTROL_PROTOCOL_V1.md'),test_already_previously_viewed=True,new_method=False,accepted_final_directions=0)
(dest/'RESULTS_V1.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
print(json.dumps(dict(status=obj['status'],summary=summary,decision=branch)),flush=True)
