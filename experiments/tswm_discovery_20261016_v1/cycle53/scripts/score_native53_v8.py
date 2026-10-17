"""Locked native VAL plus new-crop audit after all source selections finish."""
from native53 import *
import gc
import importlib.util
import time

import numpy as np
import torch

torch.set_num_threads(8)
supervisor=json.loads((R/'outputs/supervisor_v7.json').read_text())
assert supervisor['status']=='ALL_40_NATIVE_FITS_COMPLETE_TRANSFER_UNOPENED'
seal_path=R/'outputs/SELECTION_SEAL_V6.json'
seal=json.loads(seal_path.read_text());assert not seal['second_batch_evaluation_opened']
assert len(seal['selected_checkpoints'])==40
for selection in seal['selected_checkpoints']:assert sha(selection['checkpoint'])==selection['sha256']
OUT=ART/'evaluation_v1';assert not OUT.exists();OUT.mkdir()
check_source()
dump(OUT/'OPENING_RECEIPT.json',{'utc':now(),'pid':os.getpid(),'selection_seal_sha256':sha(seal_path),'protocol_sha256':sha(R/'PROTOCOL.md'),'evaluation_supplement_sha256':sha(R/'EVALUATION_SUPPLEMENT.md'),'all_40_fits_terminal':True,'transfer_used_for_selection':False})

from common.gpu_dataset import prepare_gpu,_materialize
from common.windowing import discover_schema
from dataset_utils.greenhouse.dataset import GreenhouseDataset,ACTION_CHANNELS,TARGET_CHANNELS
from observational_space.engine import forward_loss as obs_forward
from latent_space.engine import forward_loss as latent_forward
from fusion_train import fusion_forward_loss
from encode_train import encoding_forward_loss

probe_folder=UP/'RQ6_intervention/scripts';sys.path.insert(0,str(probe_folder))
import probe_intervention as native_probe
from action_space import ActionSpace
from RQ6_intervention.priors import channels,priors_for
native_probe.AE_DIR=CODEC
native_probe.DEVICE='cuda'
from replay53 import load_compatible_cell
assert json.loads((R/'outputs/concat_qualification_v1/qualification.json').read_text())['status']=='PASS_NATIVE_CONCAT_TRAINING_AND_CHECKPOINT_FORMAT'

data=prepare_gpu('greenhouse',context_length=256,horizon=16,stride=80,val_ratio=.2,seed=0,root=DATA,device='cpu',cache_dir=CACHE)
source_bank={'n':len(data.val),'tensors':data.val.tensors,'schema':vars(data.schema),'combo_to_id':None,'unit_ids':['reference']*len(data.val)}
root2=pathlib.Path(STORAGE['data_root'])/'greenhouse3_original_v1/experiment_2/TimeSeries'
raw2=GreenhouseDataset(root2,seq_len=272,stride=80)
schema2=discover_schema(raw2[0]);assert vars(schema2)==vars(data.schema)
transfer_tensors=_materialize(raw2,schema2,256,16,None)
transfer_bank={'n':len(raw2),'tensors':transfer_tensors,'schema':vars(schema2),'combo_to_id':None,'unit_ids':[team for team,start in raw2._windows],'starts':[int(start) for team,start in raw2._windows]}
assert source_bank['n']==163 and transfer_bank['n']==803
val_raw=GreenhouseDataset(DATA,teams=['reference'],seq_len=272,stride=80)
assert len(val_raw)==163
source_bank['starts']=[int(start) for team,start in val_raw._windows]
banks={'source_val':source_bank,'new_crop_all':transfer_bank}
bank_receipts={}
for name,bank in banks.items():
    path=OUT/(name+'__inputs.pt');torch.save(bank,path)
    bank_receipts[name]={'path':path.name,'sha256':sha(path),'n':bank['n'],'teams':sorted(set(bank['unit_ids']))}
    dump(OUT/(name+'__identities.json'),{'teams':bank['unit_ids'],'starts':bank['starts']})
dump(OUT/'input_manifest.json',bank_receipts)

def metrics(pred,target):
    error=pred.astype(np.float64)-target.astype(np.float64)
    return {'MAE':float(np.mean(np.abs(error))),'MSE':float(np.mean(error**2)),'RMSE':float(np.sqrt(np.mean(error**2))),
            'channel_MAE':np.mean(np.abs(error),(0,1)).tolist(),'channel_MSE':np.mean(error**2,(0,1)).tolist(),
            'horizon_MAE':np.mean(np.abs(error),(0,2)).tolist(),'horizon_MSE':np.mean(error**2,(0,2)).tolist()}

def xmatrix(tensors):
    # Target future is deliberately excluded from the feature construction.
    names=['target_history','continuous_history','exog_history','mark_history','continuous_future','exog_future','mark_future']
    return np.concatenate([tensors[k].numpy().reshape(len(tensors[k]),-1) for k in names],axis=1).astype(np.float64)

train_x=xmatrix(data.train.tensors);train_y=data.train.tensors['target_future'].numpy().reshape(len(data.train),-1).astype(np.float64)
xmean=train_x.mean(0);xscale=train_x.std(0);xscale[xscale<1e-6]=1
z=(train_x-xmean)/xscale;ym=train_y.mean(0);yc=train_y-ym
kernel=z@z.T
val_z=(xmatrix(source_bank['tensors'])-xmean)/xscale
truth_val=source_bank['tensors']['target_future'].numpy()
ridge_candidates=[];ridge_solutions=[]
for alpha in [.1,1.,10.,100.,1000.]:
    w=np.linalg.solve(kernel+alpha*np.eye(len(z)),yc)
    prediction=(val_z@z.T@w+ym).reshape(truth_val.shape)
    per_batch=[float(np.mean((prediction[s:s+128]-truth_val[s:s+128])**2)) for s in range(0,len(prediction),128)]
    ridge_candidates.append({'alpha':alpha,'native_style_val_mse':float(np.mean(per_batch)),**metrics(prediction,truth_val)})
    ridge_solutions.append(w)
chosen=min(range(len(ridge_candidates)),key=lambda i:(ridge_candidates[i]['native_style_val_mse'],i))
coeff=z.T@ridge_solutions[chosen]
np.savez_compressed(OUT/'ridge_selected.npz',xmean=xmean,xscale=xscale,coeff=coeff,ymean=ym,alpha=ridge_candidates[chosen]['alpha'])
dump(OUT/'ridge_selection.json',{'candidates':ridge_candidates,'selected_alpha':ridge_candidates[chosen]['alpha'],'selection_data':'first-batch VAL only; native-style equal-batch MSE','transfer_used':False})
summaries=[];raw_manifest=[]
for bank_name,bank in banks.items():
    truth=bank['tensors']['target_future'].numpy()
    controls={'persistence':np.repeat(bank['tensors']['target_history'][:,-1:,:].numpy(),16,axis=1),'ridge':(((xmatrix(bank['tensors'])-xmean)/xscale)@coeff+ym).reshape(truth.shape)}
    for tag,pred in controls.items():
        path=OUT/f'{tag}__{bank_name}.npz';np.savez_compressed(path,prediction=pred,target=truth)
        row={'tag':tag,'arm':tag,'model':'ordinary_control','seed':None,'bank':bank_name,'n':len(pred),**metrics(pred,truth)}
        summaries.append(row);raw_manifest.append({'tag':tag,'bank':bank_name,'path':path.name,'sha256':sha(path),'n':len(pred)})

forward_checks=[]
for selection in seal['selected_checkpoints']:
    tag=selection['tag'];model,arm_name,seed_text=tag.split('__');seed=int(seed_text[4:])
    arm={'obs':'obs','ae_concat':'concat','ae_gate':'gate','ae_gate_soft':'gate_soft'}[arm_name]
    native_probe.ckpt_path=lambda *unused,checkpoint=selection['checkpoint'],**kwargs:pathlib.Path(checkpoint)
    cell,restore_receipt=load_compatible_cell(native_probe,model,arm,seed,selection['checkpoint'],source_bank,data.schema)
    assert not isinstance(cell,str),cell
    batch={k:v[:128].to('cuda') for k,v in source_bank['tensors'].items()}
    with torch.no_grad():
        fw=native_probe.Forward(cell,batch,arm)
        probed=fw.predict(batch['continuous_future'],batch['categorical_future'],batch['exog_future'])
        if arm=='obs':
            direct=obs_forward(cell.adapter,cell.embedder,batch,4,'cuda',torch.nn.MSELoss())[1]
        elif arm=='concat':
            direct=latent_forward(cell.adapter,cell.embedder,cell.target_ae,batch,cell.n_latent,4,'cuda',torch.nn.MSELoss())[1]
        elif arm=='gate_soft':
            direct=encoding_forward_loss(cell.adapter,cell.embedder,cell.target_ae,cell.encoder,cell.fusion,batch,cell.n_latent,'cuda',torch.nn.MSELoss())[1]
        else:
            direct=fusion_forward_loss(cell.adapter,cell.embedder,cell.target_ae,cell.fusion,batch,cell.n_latent,'cuda',torch.nn.MSELoss())[1]
        difference=float((direct-probed).abs().max())
        assert torch.allclose(direct,probed,atol=2e-5,rtol=2e-5),(tag,difference)
        corrupt={k:v.clone() for k,v in batch.items()};corrupt['target_future']=torch.flip(corrupt['target_future'],dims=[0])+7.
        invariant_fw=native_probe.Forward(cell,corrupt,arm)
        invariant=invariant_fw.predict(corrupt['continuous_future'],corrupt['categorical_future'],corrupt['exog_future'])
        future_error=float((invariant-probed).abs().max());assert future_error==0.,(tag,future_error)
    forward_checks.append({'tag':tag,'native_probe_vs_train_forward_max_error':difference,'future_target_corruption_max_error':future_error,'restore_receipt':restore_receipt})
    del fw,direct,probed,batch,invariant_fw,invariant,corrupt
    for bank_name,bank in banks.items():
        parts=[]
        with torch.no_grad():
            for start in range(0,bank['n'],128):
                batch={k:v[start:start+128].to('cuda') for k,v in bank['tensors'].items()}
                fw=native_probe.Forward(cell,batch,arm)
                pred=fw.predict(batch['continuous_future'],batch['categorical_future'],batch['exog_future'])
                assert torch.isfinite(pred).all();parts.append(pred.cpu().numpy())
        pred=np.concatenate(parts);truth=bank['tensors']['target_future'].numpy()
        path=OUT/(tag+'__'+bank_name+'.npz');np.savez_compressed(path,prediction=pred,target=truth)
        row={'tag':tag,'arm':arm_name,'model':model,'seed':seed,'bank':bank_name,'n':bank['n'],**metrics(pred,truth)}
        if bank_name=='source_val':assert abs(row['MAE']-float(cell.recorded))<2e-5,(tag,row['MAE'],cell.recorded)
        n=bank['n'];lo=torch.tensor([x[1] for x in ACTION_CHANNELS],dtype=torch.float64,device='cuda').expand(n,-1).contiguous();hi=torch.tensor([x[2] for x in ACTION_CHANNELS],dtype=torch.float64,device='cuda').expand(n,-1).contiguous()
        aspace=ActionSpace('greenhouse','minmax01',tuple(x[0] for x in ACTION_CHANNELS),lo,hi,torch.zeros((n,len(ACTION_CHANNELS)),device='cuda'),{})
        original_acc=native_probe.Acc
        captured_acc=[]
        class RecordedAcc(original_acc):
            def __init__(self):
                super().__init__();captured_acc.append(self)
        native_probe.Acc=RecordedAcc
        try:
            mechanism=native_probe.run_cell(cell,bank,aspace,priors_for('greenhouse',tier='mechanistic'),channels('greenhouse'),arm,batch_size=128,ladders='none',max_windows=None)
        finally:
            native_probe.Acc=original_acc
        assert len(captured_acc)==1
        mechanism['_native_score_counts']=captured_acc[0].n
        mechanism['_native_score_totals']=captured_acc[0].s
        dump(OUT/(tag+'__'+bank_name+'__native_mechanisms.json'),mechanism)
        summaries.append(row);raw_manifest.append({'tag':tag,'bank':bank_name,'path':path.name,'sha256':sha(path),'n':bank['n'],'checkpoint_sha256':selection['sha256']})
        print(json.dumps({'tag':tag,'bank':bank_name,'MAE':row['MAE'],'MSE':row['MSE']}),flush=True)
    dump(OUT/'summary.json',summaries);dump(OUT/'raw_manifest.json',raw_manifest);dump(OUT/'forward_checks.json',forward_checks)
    del cell;gc.collect();torch.cuda.empty_cache()
assert len(summaries)==84 and len(raw_manifest)==84 and len(forward_checks)==40
dump(OUT/'COMPLETE.json',{'utc':now(),'status':'NATIVE_AND_NEW_CROP_PREDICTIONS_COMPLETE_UNVERIFIED','groups':len(summaries),'checkpoint_count':40,'source_files_rechecked':check_source(),'raw_manifest_sha256':sha(OUT/'raw_manifest.json'),'summary_sha256':sha(OUT/'summary.json'),'opening_receipt_sha256':sha(OUT/'OPENING_RECEIPT.json')})


