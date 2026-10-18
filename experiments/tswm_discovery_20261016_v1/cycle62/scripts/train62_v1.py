from native62 import *
import argparse,time,platform
p=argparse.ArgumentParser();p.add_argument('arm');p.add_argument('seed',type=int);args=p.parse_args()
assert read(OUT/'QUALIFICATION62_V1.json')['status']=='PASS'
verify_sources();ad,models,compat=load_native();df,dp=load_data();seed(args.seed)
key=args.arm+'_'+str(args.seed);pp=paths();receipt=OUT/(key+'_fit62_v1.json');assert not receipt.exists()
model=adapter(ad,models,args.arm);history=[];counter={'training_forward_batches':0,'training_examples':0}
Original=ad.EarlyStopping
class RecordingEarlyStopping(Original):
 def __call__(self,val_loss,net):
  assert np.isfinite(val_loss),'Non-finite native validation score'
  super().__call__(val_loss,net)
  history.append({'epoch':len(history)+1,'validation_mse_native_unweighted_batch_mean':float(val_loss),'selected':self.counter==0,**counter})
  state('fit_progress62_v1.json',{'key':key,'history':history,'status':'NATIVE_TRAINING'})
ad.EarlyStopping=RecordingEarlyStopping
Base=models[args.arm]
class Instrumented(Base):
 def forward(self,*x,**kwargs):
  if self.training:
   counter['training_forward_batches']+=1;counter['training_examples']+=len(x[0])
  return super().forward(*x,**kwargs)
model.model_class=Instrumented
t0=time.perf_counter();model.forecast_fit(df.iloc[:int(len(df)*.8)].copy(),train_ratio_in_tv=.75);elapsed=time.perf_counter()-t0
assert model.early_stopping.check_point is not None
tr=int(int(len(df)*.8)*.75);assert np.allclose(model.scaler.mean_,df.iloc[:tr].values.mean(0),atol=1e-10,rtol=1e-12)
ck=pp['checkpoint_root']/key;ck.mkdir(exist_ok=False)
terminal=ck/'terminal.pt';selected=ck/'selected.pt'
def cpu(state):return {k:v.detach().cpu() for k,v in state.items()}
torch.save(cpu(model.model.state_dict()),terminal)
torch.save(cpu(model.early_stopping.check_point),selected)
selected_epoch=[r['epoch'] for r in history if r['selected']][-1]
dump(receipt,{'utc':utc(),'key':key,'arm':args.arm,'seed':args.seed,'status':'SEALED_FOR_TEST','source_revision':'733d986d349e9e71ded5d9cdeeeed8d3323d83b4','protocol_sha256':sha(ROOT/'PROTOCOL62_V1.md'),'data_sha256':sha(dp),'parameters':sum(p.numel() for p in model.model.parameters()),'config':vars(model.config),'history':history,'selected_epoch':selected_epoch,'training_seconds':elapsed,'formal_optimizer_updates':counter['training_forward_batches'],'scaler_mean':model.scaler.mean_.tolist(),'scaler_scale':model.scaler.scale_.tolist(),'scaler_training_rows':tr,'selected_checkpoint':str(selected),'selected_sha256':sha(selected),'terminal_checkpoint':str(terminal),'terminal_sha256':sha(terminal),'compatibility':compat,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0)}})
print(json.dumps({'key':key,'status':'SEALED_FOR_TEST','epochs':len(history),'selected_epoch':selected_epoch,'updates':counter['training_forward_batches'],'seconds':elapsed}),flush=True)
