"""No-learning six-path composition audit, with qualified baselines and error identities."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,json,sys,hashlib,copy,types,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=pathlib.Path('D:/baseline_reproduction_seed2021_v1');SRC=BASE/'upstream/LatentTSF'
sys.path.insert(0,str(BASE/'scripts'));import common as original
sys.path.insert(0,str(SRC));import torch,numpy as np
from my_utils import model_dict,get_data
from my_AE import get_autoencoder
from my_train import encoder_encode,encoder_decode,build_dec_inp_latent
OUT=ROOT/'outputs';reg=json.loads((OUT/'matched_cap_v1/RESULTS_V1.json').read_text());dec=json.loads((OUT/'decoded_control_v1/RESULTS_V1.json').read_text());val=json.loads((OUT/'decoded_validation_v1/RESULTS_V1.json').read_text())
assert val['decision']=='STOP_CURRENT_ARCHITECTURE_MOTIVATION_OBJECTIVE_CONTROL_SUFFICIENT'
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
dest=OUT/'transport_audit_v1';dest.mkdir(exist_ok=False);arrays=pathlib.Path(cfg['artifact_root'])/ROOT.name/'transport_audit_v1';arrays.mkdir(exist_ok=False)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def dump(name,j):
 with (dest/name).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
arms=['raw','raw_after_input_recon','output_recon_after_raw','both_recon_transport','latent_native','latent_decoded']
matrix=[dict(horizon=h,seed=s,split=split) for h in (336,720) for s in (42,43,44) for split in ('val','test')]
source_files=[SRC/x for x in ['models/DLinear.py','my_AE.py','my_train.py','my_utils.py','data_provider/data_loader.py']]
before={p.as_posix():sha(p) for p in source_files}
dump('MATRIX_V1.json',dict(matrix=matrix,arms=arms,protocol_sha256=sha(ROOT/'FROZEN_TRANSPORT_PROTOCOL_V1.md'),source_sha256=before,optimizer_updates=0,test_selection=False))
records=[]
for H in (336,720):
 for seed in (42,43,44):
  cells={mode:next(x for x in reg['rows'] if x['mode']==mode and x['horizon']==H and x['seed']==seed) for mode in ('raw','latent')}
  cells['decoded']=next(x for x in dec['rows'] if x['horizon']==H and x['seed']==seed)
  models={};metrics={};args=None;predrefs={};truthref=None;checkpoint_receipts={}
  for mode,cell in cells.items():
   folder=pathlib.Path(cell['metrics_path']).parent;m=read(folder/'metrics.json');a=types.SimpleNamespace(**read(folder/'resolved_config.json'))
   assert sha(folder/'metrics.json')==cell['metrics_sha256'] and sha(m['checkpoint_path'])==m['checkpoint_sha256'] and sha(folder/'predictions.npz')==m['predictions_sha256']
   assert a.model=='DLinear' and a.seq_len==720 and a.pred_len==H and a.step==60 and a.label_len==0 and not a.use_latent_norm and m['epochs']==100
   if args is None:args=copy.copy(a)
   metrics[mode]=m;ma=copy.copy(a)
   if mode!='raw':ma.enc_in=ma.dec_in=ma.c_out=a.d_model
   model=model_dict['DLinear'].Model(ma).float().cuda();model.load_state_dict(torch.load(m['checkpoint_path'],map_location='cpu',weights_only=False),strict=True);model.eval();model.requires_grad_(False);models[mode]=model
   with np.load(folder/'predictions.npz',allow_pickle=False) as z:predrefs[mode]=z['pred'];t=z['true']
   if truthref is None:truthref=t
   else:assert np.array_equal(truthref,t)
   checkpoint_receipts[mode]=dict(path=m['checkpoint_path'],sha256=m['checkpoint_sha256'],tensor_sha256=original.tensor_sha(model))
  aa=types.SimpleNamespace(**read(pathlib.Path(cells['latent']['metrics_path']).parent/'resolved_config.json'))
  aecfg=read(pathlib.Path(cells['latent']['metrics_path']).parent/'config.json');aesha=aecfg['ae_checkpoint_sha256'];assert sha(aa.autoencoder_path)==aesha
  enc=get_autoencoder(aa).float().cuda();enc.load_state_dict(torch.load(aa.autoencoder_path,map_location='cpu',weights_only=False),strict=True);enc.eval();enc.requires_grad_(False);encbefore=original.tensor_sha(enc)
  def forecasts(bx,bxm,bym):
   # No future target is accepted by this function. Both decoder inputs are known zeros.
   zx=encoder_encode(enc,bx,'AE',None);rx=encoder_decode(enc,zx,'AE')
   zero=torch.zeros(bx.size(0),H,321,device=bx.device,dtype=bx.dtype)
   raw=models['raw'](bx,bxm,zero,bym)[:,-H:,:];rin=models['raw'](rx,bxm,zero,bym)[:,-H:,:]
   rout=enc.decode(enc.encode(raw));rwhole=enc.decode(enc.encode(rin))
   zin=build_dec_inp_latent(zx,0,H,aa.d_model,'cuda')
   native=encoder_decode(enc,models['latent'](zx,bxm,zin,bym)[:,-H:,:],'AE')
   decoded=encoder_decode(enc,models['decoded'](zx,bxm,zin,bym)[:,-H:,:],'AE')
   return [raw,rin,rout,rwhole,native,decoded],rx
  for split in ('val','test'):
   data,_=get_data(args,flag=split);loader=torch.utils.data.DataLoader(data,batch_size=32,shuffle=False,drop_last=False,num_workers=0)
   assert sha(pathlib.Path(args.root_path)/args.data_path)==metrics['raw']['data_sha256']
   n=len(data);scores=np.empty((n,6));maes=np.empty((n,6));shifts=np.empty((n,6));crosses=np.empty((n,6));input_recon=np.empty(n);truth_recon=np.empty(n)
   offset=0;baseline_errors={k:0. for k in ('raw','latent','decoded')};truth_error=0.;identity_error=0.
   with torch.no_grad():
    for bx,by,bxm,bym in loader:
     bx=bx.float().cuda();bxm=bxm.float().cuda();bym=bym.float().cuda()
     ps,rx=forecasts(bx,bxm,bym)
     # Future labels become available only after all forecast paths have been computed.
     y=by[:,-H:,:].float().numpy().astype(np.float64);pred=[p.cpu().numpy().astype(np.float64) for p in ps]
     assert all(p.shape==y.shape and np.isfinite(p).all() for p in pred)
     count=len(bx);sl=slice(offset,offset+count);base=pred[0]-y
     for col,p in enumerate(pred):
      e=p-y;delta=p-pred[0];scores[sl,col]=np.mean(e*e,axis=(1,2));maes[sl,col]=np.mean(abs(e),axis=(1,2));shifts[sl,col]=np.mean(delta*delta,axis=(1,2));crosses[sl,col]=2*np.mean(base*delta,axis=(1,2))
      identity_error=max(identity_error,float(np.max(abs(scores[sl,col]-scores[sl,0]-shifts[sl,col]-crosses[sl,col]))))
     input_recon[sl]=np.mean((rx.cpu().numpy().astype(np.float64)-bx.cpu().numpy())**2,axis=(1,2))
     yrec=enc.decode(enc.encode(by[:,-H:,:].float().cuda())).cpu().numpy().astype(np.float64)
     truth_recon[sl]=np.mean((yrec-y)**2,axis=(1,2))
     if split=='test':
      truth_error=max(truth_error,float(np.max(abs(y-truthref[sl]))))
      for mode,col in [('raw',0),('latent',4),('decoded',5)]:baseline_errors[mode]=max(baseline_errors[mode],float(np.max(abs(pred[col]-predrefs[mode][sl]))))
     offset+=count
   assert offset==n and identity_error<2e-10
   qualification={}
   for mode,col in [('raw',0),('latent',4),('decoded',5)]:
    expected=metrics[mode]['reproduced_mse'] if split=='test' else next(x['validation_mse'] for x in val['table'] if x['mode']==mode and x['horizon']==H and x['seed']==seed)
    assert abs(float(scores[:,col].mean())-expected)<2e-6
    if split=='test':
     bound=2e-5*max(1.,float(abs(predrefs[mode]).max()));assert baseline_errors[mode]<=bound and truth_error==0.
     qualification[mode]=dict(prediction_max_abs_error=baseline_errors[mode],bound=bound,truth_max_abs_error=truth_error)
   path=arrays/f'H{H}_seed{seed}_{split}.npz';np.savez_compressed(path,origin_mse=scores,origin_mae=maes,shift_mse=shifts,error_cross_term=crosses,history_reconstruction_mse=input_recon,true_future_reconstruction_mse=truth_recon,relative_origin=np.arange(n)*60)
   rec=dict(horizon=H,seed=seed,split=split,origins=n,arms=arms,forecasts={name:dict(mse=float(scores[:,i].mean()),mae=float(maes[:,i].mean()),shift_mse=float(shifts[:,i].mean()),error_cross_term=float(crosses[:,i].mean())) for i,name in enumerate(arms)},diagnostics=dict(history_reconstruction_mse=float(input_recon.mean()),true_future_reconstruction_mse=float(truth_recon.mean()),true_future_reconstruction_is_not_forecast=True),identity_max_abs_error=identity_error,baseline_qualification=qualification,checkpoint_receipts=checkpoint_receipts,ae_sha256=aesha,data_sha256=metrics['raw']['data_sha256'],arrays=dict(path=str(path),sha256=sha(path)),optimizer_updates=0,epoch_reselected=False,future_targets_only_for_scoring=True)
   dump(f'H{H}_seed{seed}_{split}.json',rec);records.append(rec)
   print(json.dumps(dict(horizon=H,seed=seed,split=split,mse={k:v['mse'] for k,v in rec['forecasts'].items()},diagnostics=rec['diagnostics'],identity_max_abs_error=identity_error)),flush=True)
  for mode,m in models.items():assert original.tensor_sha(m)==checkpoint_receipts[mode]['tensor_sha256'] and sha(metrics[mode]['checkpoint_path'])==checkpoint_receipts[mode]['sha256']
  assert original.tensor_sha(enc)==encbefore and sha(aa.autoencoder_path)==aesha
  del models,enc,predrefs,truthref,data,loader;torch.cuda.empty_cache()
assert {p.as_posix():sha(p) for p in source_files}==before
dump('RESULTS_V1.json',dict(status='TWELVE_FROZEN_COMPOSITION_CELLS_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=records,source_sha256=before,protocol_sha256=sha(ROOT/'FROZEN_TRANSPORT_PROTOCOL_V1.md'),new_training=False,new_method=False,independent_confirmation=False,accepted_final_directions=0))
