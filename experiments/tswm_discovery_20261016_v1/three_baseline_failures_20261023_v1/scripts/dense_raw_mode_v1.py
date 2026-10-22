"""Frozen-model hourly-origin audit, streamed; native helper and raw-target checks."""
import sys,pathlib,json,hashlib,types,copy,time,datetime,csv
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=pathlib.Path('D:/baseline_reproduction_seed2021_v1');SRC=BASE/'upstream/LatentTSF'
sys.path.insert(0,str(BASE/'scripts'))
import common as original
sys.path.insert(0,str(SRC))
import numpy as np,pandas as pd,torch
from my_utils import model_dict,get_data
from my_AE import get_autoencoder
from my_train import build_dec_inp_original
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
ART=pathlib.Path(cfg['artifact_root'])/ROOT.name
OUT=ROOT/'outputs/raw_dense_v1';OUT.mkdir(exist_ok=False,parents=True)
ARRAYS=ART/'raw_dense_v1';ARRAYS.mkdir(exist_ok=False)
resultpath=ROOT/'outputs/raw_control_v1/RESULTS_V1.json'
registry=json.loads(resultpath.read_text(encoding='utf-8-sig'))
assert registry['status']=='ALL_SIX_RAW_MODE_CONTROLS_CLOSED'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
records=[];rawframe=None
for row in registry['rows']:
    begin=time.perf_counter();run=pathlib.Path(row['metrics_path']).parent
    assert sha(row['metrics_path'])==row['metrics_sha256']
    m=json.loads((run/'metrics.json').read_text(encoding='utf-8-sig'))
    assert sha(m['checkpoint_path'])==m['checkpoint_sha256']
    args=types.SimpleNamespace(**json.loads((run/'resolved_config.json').read_text(encoding='utf-8-sig')))
    assert args.encoder_type=='AE' and args.ae_type=='MLP' and args.features=='M' and not args.inverse and not args.use_latent_norm
    assert args.step==60 and args.num_workers==0 and args.batch_size==32
    assert not args.use_latent and args.label_len==0
    ae_sha=None;enc=None
    modelargs=copy.copy(args)
    model=model_dict[args.model].Model(modelargs).float().cuda();model.load_state_dict(torch.load(m['checkpoint_path'],map_location='cpu',weights_only=False),strict=True);model.eval()
    before_model=original.tensor_sha(model);before_enc=None
    args=copy.copy(args);args.step=1
    data,loader=get_data(args,flag='test')
    assert isinstance(loader.sampler,torch.utils.data.SequentialSampler) and not loader.drop_last
    rawpath=pathlib.Path(args.root_path)/args.data_path;assert sha(rawpath)==m['data_sha256']
    if rawframe is None:rawframe=pd.read_csv(rawpath)
    names=[x for x in rawframe.columns if x not in ['date',args.target]]+[args.target]
    values=rawframe[names].to_numpy(dtype=np.float64);train_stop=int(len(values)*.7);test_count=int(len(values)*.2)
    mean=values[:train_stop].mean(axis=0);scale=values[:train_stop].std(axis=0);scale[scale==0]=1
    assert np.allclose(mean,data.scaler.mean_,rtol=2e-11,atol=2e-11)
    assert np.allclose(scale,data.scaler.scale_,rtol=2e-11,atol=2e-11)
    border=len(values)-test_count-args.seq_len
    expect=(values[border:]-mean)/scale
    raw_target_error=float(np.max(abs(expect-data.data_x)));assert raw_target_error<2e-9
    with np.load(run/'predictions.npz',allow_pickle=False) as z:sparse_pred=z['pred'];sparse_true=z['true']
    N=len(data);H=args.pred_len;C=args.enc_in
    assert sparse_pred.shape==sparse_true.shape==(len(range(0,N,60)),H,C)
    origin_mse=np.empty(N);origin_mae=np.empty(N);channel_sse=np.zeros(C);channel_sae=np.zeros(C);lead_sse=np.zeros(H)
    offset=0;max_pred=0.;max_true=0.;checked_sparse=0
    with torch.no_grad():
        for bx,by,bxm,bym in loader:
            bx=bx.float().cuda();by=by[:,-H:,:].float();bxm=bxm.float().cuda();bym=bym.float().cuda()
            inp=build_dec_inp_original(by,args.label_len,H,'cuda')
            pred=model(bx,bxm,inp,bym)[:,-H:,:].cpu().numpy();truth=by.numpy()
            assert pred.shape==truth.shape==(len(bx),H,C) and np.isfinite(pred).all() and np.isfinite(truth).all()
            indices=np.arange(offset,offset+len(bx));pick=np.flatnonzero(indices%60==0)
            if len(pick):
                ref=indices[pick]//60;max_pred=max(max_pred,float(np.max(abs(pred[pick]-sparse_pred[ref]))));max_true=max(max_true,float(np.max(abs(truth[pick]-sparse_true[ref]))));checked_sparse+=len(pick)
            error=pred.astype(np.float64)-truth;ss=error**2;aa=abs(error)
            origin_mse[indices]=ss.mean(axis=(1,2));origin_mae[indices]=aa.mean(axis=(1,2));channel_sse+=ss.sum(axis=(0,1));channel_sae+=aa.sum(axis=(0,1));lead_sse+=ss.sum(axis=(0,2));offset+=len(bx)
            if offset%2048<32:print(json.dumps(dict(seed=row['seed'],horizon=H,scored=offset,total=N)),flush=True)
    assert offset==N and checked_sparse==len(sparse_pred) and max_true==0.
    bound=2e-5*max(1.,float(abs(sparse_pred).max()));assert max_pred<=bound,(max_pred,bound)
    coarse_mse=float(origin_mse[::60].mean());coarse_mae=float(origin_mae[::60].mean())
    assert abs(coarse_mse-m['reproduced_mse'])<2e-6 and abs(coarse_mae-m['reproduced_mae'])<2e-6
    timestamps=pd.to_datetime(rawframe['date']).iloc[len(values)-test_count:len(values)-test_count+N]
    hours=timestamps.dt.hour.to_numpy();assert len(hours)==N
    phases=[dict(phase=p,n=int(np.sum(np.arange(N)%60==p)),mse=float(origin_mse[np.arange(N)%60==p].mean()),mae=float(origin_mae[np.arange(N)%60==p].mean())) for p in range(60)]
    hour_scores=[dict(hour=h,n=int(np.sum(hours==h)),mse=float(origin_mse[hours==h].mean()),mae=float(origin_mae[hours==h].mean())) for h in range(24) if np.any(hours==h)]
    array=ARRAYS/f"H{H}_seed{row['seed']}_origin_metrics.npz"
    np.savez_compressed(array,origin_mse=origin_mse,origin_mae=origin_mae,raw_target_origin=np.arange(len(values)-test_count,len(values)-test_count+N),forecast_hour=hours,channel_mse=channel_sse/(N*H),channel_mae=channel_sae/(N*H),lead_mse=lead_sse/(N*C))
    assert sha(m['checkpoint_path'])==m['checkpoint_sha256']
    assert original.tensor_sha(model)==before_model
    rec=dict(seed=row['seed'],horizon=H,native_sparse_origins=len(sparse_pred),dense_origins=N,native_sparse_mse=coarse_mse,native_sparse_mae=coarse_mae,dense_mse=float(origin_mse.mean()),dense_mae=float(origin_mae.mean()),native_sparse_hours=sorted(set(map(int,hours[::60]))),dense_hours=sorted(set(map(int,hours))),phase_scores=phases,hour_scores=hour_scores,raw_target_max_abs_error=raw_target_error,native_prediction_max_abs_error=max_pred,native_prediction_qualification_bound=bound,source_checkpoint_sha256=m['checkpoint_sha256'],data_sha256=m['data_sha256'],ae_checkpoint_sha256=ae_sha,arrays=dict(path=str(array),sha256=sha(array)),wall_seconds=time.perf_counter()-begin,weights_changed=False,test_selection=False)
    dump(f"H{H}_seed{row['seed']}_result.json",rec);records.append(rec)
    print(json.dumps({k:rec[k] for k in ['seed','horizon','native_sparse_mse','dense_mse','native_sparse_hours','dense_hours','native_prediction_max_abs_error','wall_seconds']}),flush=True)
    del model,enc,sparse_pred,sparse_true,loader,data;torch.cuda.empty_cache()
dump('RESULTS_V1.json',dict(utc=now(),status='SIX_FROZEN_MODELS_DENSE_HOURLY_AUDIT_COMPLETE',records=records,source_registry_sha256=sha(resultpath),protocol_sha256=sha(ROOT/'PROTOCOL.md'),scope='Changed inference workload on previously inspected benchmark; not fresh method generalization or original-paper score',new_method=False,accepted_final_directions=0))
