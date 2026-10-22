"""Formula-based TTC probe on actual frozen native checkpoints; zero optimizer steps."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,json,sys,hashlib,types,copy,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];CAMPAIGN=ROOT.parent;PROJECT=CAMPAIGN.parent;BASE=pathlib.Path('D:/baseline_reproduction_seed2021_v1');SRC=BASE/'upstream/LatentTSF'
sys.path.insert(0,str(BASE/'scripts'));import common as original
sys.path.insert(0,str(SRC));import numpy as np,torch
from my_utils import get_data,model_dict
from my_AE import get_autoencoder
from my_train import encoder_encode,encoder_decode,build_dec_inp_latent
cfg=json.loads((PROJECT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));ART=pathlib.Path(cfg['artifact_root'])/CAMPAIGN.name/'metric_validity_v1';ART.mkdir(exist_ok=False)
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,j):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
def cosine_pairs(a):
 a=np.asarray(a,dtype=np.float64);x=a[:,:-1];y=a[:,1:];nx=np.linalg.norm(x,axis=-1);ny=np.linalg.norm(y,axis=-1)
 value=np.sum(x*y,axis=-1)/(np.maximum(nx,1e-8)*np.maximum(ny,1e-8))
 return float(value.mean()),int(((nx==0)|(ny==0)).sum())
def descriptor(bank):
 a=np.asarray(bank,dtype=np.float64);mean=a.mean(axis=(0,1),keepdims=True);ttc,zero=cosine_pairs(a);cttc,czero=cosine_pairs(a-mean)
 return dict(ttc=ttc,centered_ttc=cttc,zero_norm_pairs=zero,centered_zero_norm_pairs=czero,pairs=a.shape[0]*(a.shape[1]-1))
source={p.as_posix():sha(p) for p in [SRC/'my_train.py',SRC/'my_AE.py',SRC/'models/DLinear.py',SRC/'my_utils.py',SRC/'data_provider/data_loader.py']}
dump('MATRIX_V1.json',dict(datasets=['ETTh1','Electricity'],native_backbone='DLinear',horizon=96,seed=42,offset_multipliers=[0,1,4,16],null_permutations=32,null_seed=20261023,protocol_sha256=sha(ROOT/'PROTOCOL.md'),source_sha256=source,optimizer_updates=0,metric_formula_reimplementation=True,original_paper_table_reproduction=False))
rows=[]
for dataset in ['ETTh1','Electricity']:
 run=BASE/f'results/latenttsf/{dataset}_DLinear_H96_seed42';m=read(run/'metrics.json');config=read(run/'config.json');args=types.SimpleNamespace(**read(run/'resolved_config.json'))
 assert m['source_commit']=='7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2' and m['run_type']=='fresh_run'
 assert sha(m['checkpoint_path'])==m['checkpoint_sha256'] and sha(args.autoencoder_path)==config['ae_checkpoint_sha256'] and sha(pathlib.Path(args.root_path)/args.data_path)==m['data_sha256']
 assert args.model=='DLinear' and args.pred_len==96 and args.label_len==0 and not args.use_latent_norm
 enc=get_autoencoder(args).float().cuda();enc.load_state_dict(torch.load(args.autoencoder_path,map_location='cpu',weights_only=False),strict=True);enc.eval();enc.requires_grad_(False)
 ma=copy.copy(args);ma.enc_in=ma.dec_in=ma.c_out=args.d_model
 model=model_dict['DLinear'].Model(ma).float().cuda();model.load_state_dict(torch.load(m['checkpoint_path'],map_location='cpu',weights_only=False),strict=True);model.eval();model.requires_grad_(False)
 before_model=original.tensor_sha(model);before_ae=original.tensor_sha(enc)
 train,_=get_data(args,flag='train');points=train.data_x[np.linspace(0,len(train.data_x)-1,512,dtype=np.int64)]
 with torch.no_grad():tz=enc.encode(torch.from_numpy(points).float().unsqueeze(0).cuda()).cpu().numpy().astype(np.float64)
 rms=float(np.sqrt(np.mean(tz*tz)));assert rms>0;scale=float(2.**np.round(np.log2(rms)));offsets=[k*scale for k in [0,1,4,16]]
 val,_=get_data(args,flag='val');raw=val.data_x[args.seq_len:].astype(np.float32);encoded=[]
 with torch.no_grad():
  for i in range(0,len(raw),512):encoded.append(enc.encode(torch.from_numpy(raw[i:i+512]).unsqueeze(0).cuda()).cpu().numpy()[0])
 oz=np.concatenate(encoded)[None];rawbank=raw[None];indices=np.linspace(0,len(val)-1,32,dtype=np.int64);assert len(np.unique(indices))==32
 loader=torch.utils.data.DataLoader(torch.utils.data.Subset(val,indices.tolist()),batch_size=8,shuffle=False,num_workers=0)
 predicted=[];max_recovered=0.;max_forecast=0.;prediction_max=0.
 with torch.no_grad():
  for bx,_,bxm,bym in loader:
   bx=bx.float().cuda();bxm=bxm.float().cuda();bym=bym.float().cuda();zx=encoder_encode(enc,bx,'AE',None);decin=build_dec_inp_latent(zx,0,96,args.d_model,'cuda');zp=model(zx,bxm,decin,bym)[:,-96:,:];baseline=encoder_decode(enc,zp,'AE');prediction_max=max(prediction_max,float(baseline.abs().max()));predicted.append(zp.cpu().numpy())
   for c in offsets:
    recovered=((zx.double()+c)-c).float();max_recovered=max(max_recovered,float(abs(recovered-zx).max()))
    shifted=model(recovered,bxm,build_dec_inp_latent(recovered,0,96,args.d_model,'cuda'),bym)[:,-96:,:].double()+c
    physical=encoder_decode(enc,(shifted-c).float(),'AE');max_forecast=max(max_forecast,float(abs(physical-baseline).max()))
 assert max_recovered==0.;assert max_forecast<=2e-5*max(1.,prediction_max)
 forecast=np.concatenate(predicted);assert forecast.shape==(32,96,args.d_model)
 # Existing Test arrays are qualification fixtures only, never metric/model-selection data.
 test,_=get_data(args,flag='test');tl=torch.utils.data.DataLoader(torch.utils.data.Subset(test,list(range(32))),batch_size=8,shuffle=False,num_workers=0)
 with np.load(run/'predictions.npz',allow_pickle=False) as z:ref=z['pred'][:32];truth=z['true'][:32]
 offset=0;pe=te=0.
 with torch.no_grad():
  for bx,by,bxm,bym in tl:
   bx=bx.float().cuda();bxm=bxm.float().cuda();bym=bym.float().cuda();zx=encoder_encode(enc,bx,'AE',None);yp=encoder_decode(enc,model(zx,bxm,build_dec_inp_latent(zx,0,96,args.d_model,'cuda'),bym)[:,-96:,:],'AE').cpu().numpy();target=by[:,-96:,:].float().numpy();sl=slice(offset,offset+len(bx));pe=max(pe,float(abs(yp-ref[sl]).max()));te=max(te,float(abs(target-truth[sl]).max()));offset+=len(bx)
 assert offset==32 and te==0. and pe<=2e-5*max(1.,float(abs(ref).max()))
 banks={'observed_encoding':oz,'predicted_latent':forecast};result=[];saved=dict(raw_observations=rawbank,observed_encoding=oz,predicted_latent=forecast,validation_origin_indices=indices)
 for name,bank in banks.items():
  rng=np.random.default_rng(20261023);perms=np.asarray([[rng.permutation(bank.shape[1]) for _ in range(bank.shape[0])] for _ in range(32)],dtype=np.int32);saved[name+'_permutations']=perms
  for c in offsets:
   a=bank.astype(np.float64)+c;d=descriptor(a);null=[];cnull=[]
   for perm in perms:
    shuffled=np.take_along_axis(a,perm[:,:,None],axis=1);n=descriptor(shuffled);null.append(n['ttc']);cnull.append(n['centered_ttc'])
   reversed_metric=descriptor(a[:,:,::-1]);assert abs(reversed_metric['ttc']-d['ttc'])<2e-12
   result.append(dict(bank=name,offset=float(c),offset_train_rms_scale=scale,**d,shuffle_ttc_mean=float(np.mean(null)),shuffle_ttc_sd=float(np.std(null,ddof=1)),shuffle_centered_ttc_mean=float(np.mean(cnull)),shuffle_scores=null,shuffle_centered_scores=cnull,coordinate_reversal_error=abs(reversed_metric['ttc']-d['ttc'])))
  centered=[x['centered_ttc'] for x in result if x['bank']==name];assert max(centered)-min(centered)<2e-12
 p=ART/(dataset+'_state_banks.npz');np.savez_compressed(p,**saved)
 rec=dict(dataset=dataset,checkpoint_sha256=m['checkpoint_sha256'],ae_sha256=config['ae_checkpoint_sha256'],data_sha256=m['data_sha256'],original_native_run_type=m['run_type'],train_rms=rms,power2_scale=scale,raw_observation_descriptor=descriptor(rawbank),rows=result,qualification=dict(test_fixture_origins=32,test_prediction_max_abs_error=pe,test_truth_max_abs_error=te,translated_encoder_input_recovery_max_abs_error=max_recovered,translated_forecast_max_abs_error=max_forecast,bound=2e-5*max(1.,prediction_max)),state_banks=dict(path=str(p),sha256=sha(p)),optimizer_updates=0,forecast_hyperparameter_tuning=False)
 dump(dataset+'_RESULTS_V1.json',rec);rows.append(rec)
 print(json.dumps(dict(dataset=dataset,qualification=rec['qualification'],rows=[{k:x[k] for k in ['bank','offset','ttc','centered_ttc','shuffle_ttc_mean']} for x in result])),flush=True)
 assert original.tensor_sha(model)==before_model and original.tensor_sha(enc)==before_ae and sha(m['checkpoint_path'])==m['checkpoint_sha256'] and sha(args.autoencoder_path)==config['ae_checkpoint_sha256']
 del model,enc,train,val,test,loader,tl;torch.cuda.empty_cache()
assert {p:sha(p) for p in source}==source
dump('RESULTS_V1.json',dict(status='TWO_NATIVE_CHECKPOINTS_METRIC_VALIDITY_PROBE_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=rows,protocol_sha256=sha(ROOT/'PROTOCOL.md'),source_sha256=source,optimizer_updates=0,new_method=False,original_paper_geometry_table_reproduced=False,independent_method_confirmation=False,accepted_final_directions=0))
