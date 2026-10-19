"""Locked held-out quality; raw joint draws retained, then separate verifier."""
from common46 import *
from control46 import AlignedSSM
import datetime,os
torch.set_num_threads(1)

def metrics(draws,truth):
 tensor=torch.from_numpy(draws);flat=tensor.flatten(2);target=torch.from_numpy(truth).flatten(1);distance=torch.cdist(flat,flat,compute_mode='donot_use_mm_for_euclid_dist');distance.diagonal(dim1=-2,dim2=-1).zero_();es=((flat-target[:,None]).norm(dim=-1).mean(1)-distance.sum((1,2))/(2*64*63)).numpy()/math.sqrt(FUTURE*Y);mse=((draws.mean(1)-truth)**2).mean((1,2));lo,hi=np.quantile(draws,[.05,.95],axis=1);coverage=((truth>=lo)&(truth<=hi)).mean((1,2));return es,mse,coverage
def main():
 sup=read(R/'outputs/supervisor_v1.json');assert sup['status']=='FITS_COMPLETE_EVAL_UNOPENED';fits=[]
 for s in SEEDS:
  for o,d in [(0,0),(0,1),(1,0),(1,1)]:
   path=R/f'outputs/fits/o{o}d{d}_seed{s}';summary=read(path/'fit_summary.json');assert summary['status']=='FIT_COMPLETE_EVAL_UNOPENED' and sha(path/'selected.pt')==summary['selected_sha256'];fits.append((path,summary))
 selection=read(R/'outputs/ordinary_control_v1/selection.json');assert selection['status']=='VAL_SELECTION_COMPLETE_EVAL_UNOPENED'
 out=R/'outputs/eval_v1';assert not out.exists();out.mkdir(parents=True);dump(out/'OPENING_RECEIPT.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'protocol_sha256':sha(R/'PROTOCOL.md'),'control_supplement_sha256':sha(R/'CONTROL_SUPPLEMENT.md'),'selected_weights':{p.name:s['selected_sha256'] for p,s in fits},'control_selection_sha256':sha(R/'outputs/ordinary_control_v1/selection.json'),'prior_EVAL_access_by_training_or_selection':False})
 manifest=read(R/'data/MANIFEST.json');assert sha(R/'data/sealed_eval.npz')==manifest['files']['sealed_eval.npz'];evaluation=dict(np.load(R/'data/sealed_eval.npz'));rows=[];records=[]
 def retain(tag,k,draws):
  truth=evaluation[k][:,HIST:];assert draws.shape==(len(truth),64,FUTURE,Y) and np.isfinite(draws).all();es,mse,cov=metrics(draws,truth);file=out/f'{tag}__{k}.npz';np.savez_compressed(file,draws=draws,truth=truth,energy=es,mse=mse,coverage=cov);records.append({'path':file.name,'sha256':sha(file),'model':tag,'cell':k,'origins':len(truth)})
  for j in range(len(truth)):rows.append({'model':tag,'cell':k,'heldout':int(k[1])==int(k[-1]),'origin':j,'energy':float(es[j]),'mse':float(mse[j]),'coverage90':float(cov[j])})
  print(json.dumps({'model':tag,'cell':k,'origins':len(truth),'energy':float(es.mean()),'mse':float(mse.mean()),'coverage90':float(cov.mean())}),flush=True)
 for path,summary in fits:
  ckpt=torch.load(path/'selected.pt',weights_only=True,map_location='cpu');cfg=ckpt['config'];seed(cfg['seed']);model=World(cfg['obs'],cfg['dyn']);model.load_state_dict(ckpt['model']);before=checksum(model)
  for j,k in enumerate(sorted(evaluation)):
   values=evaluation[k];pieces=[]
   for b in range(0,len(values),8):pieces.append(model.sample(values[b:b+8,:HIST],int(k[1]),int(k[-1]),64,202646800+cfg['seed']*100+j*10000+b).cpu().numpy())
   retain(path.name,k,np.concatenate(pieces))
  assert before==checksum(model) and sha(path/'selected.pt')==summary['selected_sha256']
 train=dict(np.load(R/'data/train.npz'));anchors=np.load(R/'data/anchors.npz')['X'];control=AlignedSSM(train,anchors,selection['selected_ridge']);control.save(out/'control_replay.npz');original=np.load(R/'outputs/ordinary_control_v1/selected_parameters.npz');replay=np.load(out/'control_replay.npz');assert set(original.files)==set(replay.files) and all(np.array_equal(original[k],replay[k]) for k in original.files)
 for j,k in enumerate(sorted(evaluation)):
  values=evaluation[k];v=int(k[1]);s=int(k[-1]);retain('ordinary_aligned_PCA_LGSSM',k,control.sample(values[:,:HIST],v,s,64,202646900+j));pool=np.concatenate([x[:,HIST:] for key,x in train.items() if key.startswith(f'v{v}_')]);rng=np.random.default_rng(202646950+j);retain('ordinary_unconditional_view_paths',k,pool[rng.integers(len(pool),size=(len(values),64))]);retain('zero_point_control',k,np.zeros((len(values),64,FUTURE,Y),dtype='float32'));retain('persistence_point_control',k,np.broadcast_to(values[:,None,HIST-1:HIST],(len(values),64,FUTURE,Y)).copy())
 dump(out/'raw_manifest.json',records);dump(out/'all_rows.json',rows);groups={}
 for r in rows:groups.setdefault((r['model'],r['heldout']),[]).append(r)
 summary=[]
 for (tag,heldout),rs in groups.items():summary.append({'model':tag,'heldout':heldout,'n_origins':len(rs),'energy':float(np.mean([r['energy'] for r in rs])),'mse':float(np.mean([r['mse'] for r in rs])),'coverage90':float(np.mean([r['coverage90'] for r in rs])),'point_control_intervals_not_probabilistic':tag.endswith('point_control')})
 dump(out/'summary.json',{'status':'SCORED_PENDING_INDEPENDENT_VERIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'groups':summary,'raw_groups':len(records),'origins':len(rows),'raw_manifest_sha256':sha(out/'raw_manifest.json'),'rows_sha256':sha(out/'all_rows.json'),'opening_sha256':sha(out/'OPENING_RECEIPT.json'),'source_sha256':sha(__file__)});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
