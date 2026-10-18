from common86_v1 import *
from scipy.linalg import expm

STAGE=O/'paired_v1'
def network(x,state,kind):
    h=x.reshape(len(x),-1).astype(np.float64)
    for layer in [0,2,4]:
        h=h@state[f'net.{layer}.weight'].numpy().astype(np.float64).T+state[f'net.{layer}.bias'].numpy().astype(np.float64)
        if layer<4:h=np.tanh(h)
    if kind=='direct':return h.reshape(-1,16,2)
    p=.8*np.tanh(h);a=np.zeros((len(x),2,2));a[:,0,0]=p[:,0];a[:,0,1]=a[:,1,0]=p[:,1];a[:,1,1]=p[:,2];t=np.repeat(np.eye(2)[None],len(a),axis=0);power=np.repeat(np.eye(2)[None],len(a),axis=0)
    for k in range(1,5):power=power@a*.05/k;t+=power
    v=x[:,-1].astype(np.float64);result=[]
    for _ in range(16):v=np.einsum('nij,nj->ni',t,v);result.append(v.copy())
    return np.stack(result,axis=1)
def ordinary(x,lam):
    out=[]
    for row in x.astype(np.float64):
        z=np.concatenate([row[:-1],np.sqrt(lam)*np.eye(2)]);b=np.concatenate([row[1:]-row[:-1],np.zeros((2,2))]);delta=np.linalg.pinv(z,rcond=1e-15)@b;t=np.eye(2)+delta;v=row[-1];pred=[]
        for _ in range(16):v=t.T@v;pred.append(v.copy())
        out.append(pred)
    return np.asarray(out)
def main():
    result=json.loads((STAGE/'RESULT86_PAIRED_V1.json').read_text());assert json.loads((STAGE/'WORKER_EXIT86_V1.json').read_text())['exit_code']==0;sealpath=STAGE/'SELECTION_SEALED86_V1.json';assert sha(sealpath)==result['selection_seal_sha256'];seal=json.loads(sealpath.read_text());assert seal['eval_generated'] is False and seal['fits']==result['fits'];assert sha(result['eval_array_path'])==result['eval_array_sha256'] and datetime.datetime.fromisoformat(seal['utc']).timestamp()<=Path(result['eval_array_path']).stat().st_mtime
    eval=np.load(result['eval_array_path'],allow_pickle=False);truth=eval['reference'];assert truth.shape==(1024,16,2);physical_error=0.
    for j in range(1024):
        for t in range(16):physical_error=max(physical_error,float(np.max(np.abs(expm((t+8)*.05*eval['a'][j])@eval['initial'][j,0]-truth[j,t]))))
    assert physical_error<1e-10
    group_masks=dict(all=np.ones(1024,dtype=bool),selected=eval['selected_population'],rejected=~eval['selected_population']);lookup={r['label']:r for r in result['results']};data={};source_max=0.
    for entry in result['sources']:
        assert sha(entry['path'])==entry['sha256'];d=np.load(entry['path'],allow_pickle=False);data[entry['seed']]=d;pop=json.loads((O/f"POPULATION86_{entry['seed']}_V1.json").read_text());original=np.load(pop['array_path'],allow_pickle=False)
        assert np.array_equal(d['whole_filter_train_ids'],d['resampled_ic_train_ids']) and np.array_equal(d['whole_filter_val_ids'],d['resampled_ic_val_ids'])
        for arm in ['whole_filter','resampled_ic','prefix_all']:
            train=d[arm+'_train_ids'];val=d[arm+'_val_ids'];assert len(train)==len(np.unique(train))==1024 and not set(train)&set(val) and np.all(original['split'][train]==0) and np.all(original['split'][val]==1)
            if arm!='prefix_all':assert np.all(original['p1_3_mask'][train]) and np.all(original['p1_3_mask'][val])
            for part,ids in [('train',train),('val',val)]:
                clean=d[arm+'_'+part+'_clean'];assert np.isfinite(clean).all()
                for j in range(0,len(ids),32):
                    for t in [0,7,23]:source_max=max(source_max,float(np.max(np.abs(expm(t*.05*original['a'][ids[j]])@clean[j,0]-clean[j,t]))))
                if arm!='resampled_ic':assert np.allclose(clean[:,0],original['initial'][ids,0],atol=1e-12,rtol=1e-12)
            if arm=='resampled_ic':assert not np.array_equal(d['whole_filter_train_clean'][:,0],d['resampled_ic_train_clean'][:,0])
        for part in ['train','val']:
            left={int(k):j for j,k in enumerate(d['whole_filter_'+part+'_ids'])};right={int(k):j for j,k in enumerate(d['prefix_all_'+part+'_ids'])}
            for k in set(left)&set(right):assert np.array_equal(d['whole_filter_'+part+'_y'][left[k]],d['prefix_all_'+part+'_y'][right[k]])
    assert source_max<1e-10;checks=[];total=0;max_trace_difference=0.;maximum_replay=0.
    for fit in result['fits']:
        seed,kind,arm=fit['seed'],fit['architecture'],fit['arm'];d=data[seed];trace=fit['trace'];assert fit['updates']==3000 and [r['step'] for r in trace]==list(range(0,3001,300));assert fit['selected']==min(trace,key=lambda z:z['val_mse']);x=d[arm+'_val_y'][:,:8];y=d[arm+'_val_y'][:,8:];initials=[]
        for row in trace:
            assert sha(row['checkpoint_path'])==row['checkpoint_sha256'];cp=torch.load(row['checkpoint_path'],map_location='cpu',weights_only=False);assert cp['step']==row['step'];prediction=network(x,cp['model'],kind);error=abs(float(np.mean((prediction-y)**2))-row['val_mse']);max_trace_difference=max(max_trace_difference,error);assert np.isclose(np.mean((prediction-y)**2),row['val_mse'],atol=1e-5,rtol=1e-5)
            if row['step']==0:initials=[cp['model'][k].numpy().tobytes() for k in sorted(cp['model'])]
            if row==fit['selected']:selected=cp['model']
        assert hashlib.sha256(b''.join(initials)).hexdigest()==fit['initial_state_sha256'];rng=np.random.default_rng(np.random.SeedSequence([seed,6000]));schedules=np.asarray([rng.integers(0,1024,64) for _ in range(3000)],dtype=np.int64);assert hashlib.sha256(schedules.tobytes()).hexdigest()==fit['schedule_sha256'];persistence=float(np.mean((x[:,-1,None]-y)**2));assert np.isclose(persistence,fit['persistence_val_mse'],atol=1e-7,rtol=1e-7);assert fit['source_qualified']==(fit['selected']['val_mse']<fit['persistence_val_mse'])
        label=f'{seed}_{kind}_{arm}';p=network(eval['history'],selected,kind);expected=eval[label+'_prediction'];replay=float(np.max(np.abs(p-expected)));maximum_replay=max(maximum_replay,replay);assert np.allclose(p,expected,atol=1e-4,rtol=1e-4);total+=expected.size;checks.append(dict(label=label,source_qualified=fit['source_qualified'],selected_step=fit['selected']['step'],max_numpy_replay_difference=replay))
    for seed in SEEDS:
        for kind in ['generator','direct']:
            group=[r for r in result['fits'] if r['seed']==seed and r['architecture']==kind];assert len(set(r['schedule_sha256'] for r in group))==len(set(r['initial_state_sha256'] for r in group))==1
    for record in result['linear']:
        d=data[record['seed']];x=d['prefix_all_val_y'][:,:8];y=d['prefix_all_val_y'][:,8:]
        for candidate in record['candidates']:assert np.isclose(np.mean((ordinary(x,candidate['ridge'])-y)**2),candidate['val_mse'],atol=1e-9,rtol=1e-9)
        assert record['selected']==min(record['candidates'],key=lambda r:r['val_mse']);p=ordinary(eval['history'],record['selected']['ridge']);assert np.allclose(p,eval[f"linear_{record['seed']}_prediction"],atol=1e-9,rtol=1e-9);total+=p.size
    assert np.array_equal(eval['persistence_prediction'],np.repeat(eval['history'][:,-1,None],16,axis=1));total+=eval['persistence_prediction'].size
    for label,record in lookup.items():
        p=eval[label+'_prediction'];failed=(~np.isfinite(p).all(axis=(1,2)))|(np.max(np.abs(p),axis=(1,2))>1e4);mse=np.sum((p-truth)**2,axis=(1,2))/32;nrmse=np.sqrt(mse)/np.maximum(np.sqrt(np.sum(truth**2,axis=(1,2))/32),.1);mse[failed]=1e12;nrmse[failed]=1e12;assert np.array_equal(failed,eval[label+'_failed']) and np.allclose(mse,eval[label+'_mse'],atol=1e-12,rtol=1e-12) and np.allclose(nrmse,eval[label+'_nrmse'],atol=1e-12,rtol=1e-12)
        for name,mask in group_masks.items():
            actual=record['summary'][name];assert actual['n']==int(mask.sum()) and actual['failed']==int(failed[mask].sum());assert np.isclose(actual['mean_mse'],mse[mask].mean(),atol=1e-12,rtol=1e-12) and np.isclose(actual['median_nrmse'],np.median(nrmse[mask]),atol=1e-12,rtol=1e-12) and np.isclose(actual['mean_nrmse'],nrmse[mask].mean(),atol=1e-12,rtol=1e-12)
    assert len(checks)==18 and total==720896
    out=dict(status='PASS',utc=utc(),result_sha256=sha(STAGE/'RESULT86_PAIRED_V1.json'),prediction_scalars=total,neural_fits=18,checkpoint_traces_checked=198,checks=checks,max_numpy_prediction_difference=maximum_replay,max_numpy_val_loss_difference=max_trace_difference,max_eval_scipy_reference_difference=physical_error,max_source_scipy_reference_difference=source_max,paired_initializations_and_schedules_verified=True,all_selectors_source_validation_only=True,protocol_sha256=sha(C/'PAIRED_PRETRAIN86_V1.md'),verification_protocol_sha256=sha(C/'VERIFICATION86_V1.md'),released_checkpoint_tested=False,new_method_validated=False,local_result_payload_files_written=0);write(STAGE/'VERIFICATION86_PAIRED_V1.json',out);print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2),flush=True)
if __name__=='__main__':main()
