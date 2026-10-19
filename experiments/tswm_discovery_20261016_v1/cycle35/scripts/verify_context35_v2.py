"""Independent raw-window, checkpoint selection and native forward replay audit."""
import context_common35_v2 as c
import json, numpy as np, pandas as pd, torch, time, pathlib
OUT=c.ROOT/'outputs/pilot_v2';started=time.perf_counter()
summary=json.loads((OUT/'summary.json').read_text());assert summary['status']=='completed'
assert summary['test_scored'] is False and summary['accepted_directions']==0
assert summary['locks']['common_sha256']==c.sha(c.ROOT/'scripts/context_common35_v2.py')
assert summary['locks']['runner_sha256']==c.sha(c.ROOT/'scripts/run_context35_v2.py')
assert summary['locks']['plan_sha256']==c.sha(c.ROOT/'MECHANISM_PILOT_PLAN_V2.md')
c.source_check();Model,_=c.native();checks=[];max_error=0.;score_error=0.;baselines=[]

for di,name in enumerate(c.DATA):
    meta=next(x for x in summary['metadata'] if x['name']==name)
    rel,fields,expected=c.DATA[name];path=c.BASE/'cycle34/assets/fev'/rel
    assert c.sha(path)==expected==meta['data_sha256']
    row=pd.read_parquet(path).iloc[0]
    # Separate raw extraction and finite-value mean/std, retaining every time index.
    raw=np.column_stack([np.array(row[f],dtype=np.float64) for f in fields]);tr=int(.7*len(raw));va=int(.8*len(raw))
    mu=[];sd=[]
    for j in range(raw.shape[1]):
        x=raw[:tr,j];x=x[np.isfinite(x)];mu.append(x.mean());sd.append(x.std(ddof=0))
    mu=np.asarray(mu);sd=np.asarray(sd)
    np.testing.assert_allclose(mu,meta['mu'],rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(sd,meta['sd'],rtol=1e-12,atol=1e-10)
    scaled=((raw[:va]-mu)/sd).astype(np.float32)
    origins=np.array([o for o in range(tr,va-c.H+1,c.H) if np.isfinite(raw[o-c.T:o+c.H]).all()],dtype=np.int64)
    assert len(origins)==meta['valid_origin_counts']['val'] and (np.diff(origins)>=c.H).all()
    all_train=np.arange(c.T,tr-c.H+1,dtype=np.int64)
    bad=np.r_[0,np.cumsum(~np.isfinite(raw[:tr]).all(1))]
    all_train=all_train[(bad[all_train+c.H]-bad[all_train-c.T])==0]
    saved_schedule=np.load(OUT/name/'arm_schedule.npy')
    recreated=np.random.default_rng(c.SEED+10000+di).choice(all_train,size=(2000,c.B),replace=True)
    assert np.array_equal(saved_schedule,recreated)
    history=json.loads((OUT/name/'teacher/history.json').read_text())
    teacher_state=torch.load(OUT/name/'teacher/decoder_best.pt',map_location='cpu',weights_only=True)
    for stage in ['latent','decoder']:
        file=OUT/name/'teacher'/(stage+'_best.pt')
        assert c.sha(file)==history[stage]['best_checkpoint_sha256']
        assert min(x['val_objective'] for x in history[stage]['curve'])<float('inf')
        sched=np.load(OUT/name/'teacher'/(stage+'_schedule.npy'))
        exp=np.random.default_rng(c.SEED+(stage=='decoder')).choice(all_train,size=(1000,c.B),replace=True)
        assert np.array_equal(sched,exp)
    curves={}
    all_y=np.stack([scaled[o:o+c.H,:1] for o in origins])
    last=np.stack([np.repeat(scaled[o-1:o,:1],c.H,axis=0) for o in origins])
    period=96 if name=='solar' else 144
    seasonal=np.stack([scaled[o+np.arange(c.H)-period,:1] for o in origins])
    for tag,pred in [('last_value',last),('daily_seasonal',seasonal)]:
        baselines.append({'dataset':name,'model':tag,'val_mae':float(np.abs(pred-all_y).mean()),
                          'val_mse':float(np.square(pred-all_y).mean()),'posthoc_reference_only':True})
    for record in [r for r in summary['results'] if r['dataset']==name]:
        arm=record['arm'];folder=OUT/name/arm;curves[arm]=json.loads((folder/'curve.json').read_text())
        assert len(curves[arm])==21
        minimum=min(curves[arm],key=lambda a:a['fused_mae'])
        assert minimum['step']==record['best_step'] and minimum['fused_mae']==record['best_val_mae']
        assert c.sha(folder/'best.pt')==record['best_checkpoint_sha256']
        assert c.sha(folder/'best_val_predictions.npz')==record['predictions_sha256']
        assert c.sha(OUT/name/'arm_schedule.npy')==record['schedule_sha256']
        c.seed();teacher=Model(c.SimpleNamespace(**c.CONFIG)).cuda();teacher.load_state_dict(teacher_state)
        m=c.Arm(teacher,arm,len(fields)-1);m.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True));m.eval()
        assert c.state_sha(m.model.predictor.state_dict())==minimum['predictor_sha256']
        for k,t in teacher_state.items():
            if k.startswith(('target_encoder.','decoder.')):assert torch.equal(m.model.state_dict()[k].cpu(),t),k
        initial_model=c.Arm(teacher,arm,len(fields)-1)
        assert record['initialization']['teacher']==c.state_sha(teacher_state)
        assert record['initialization']['predictor']==c.state_sha(initial_model.model.predictor.state_dict())
        # Arm initializer RNG is independently reseeded for exact g equality.
        c.seed();initial_model=c.Arm(teacher,arm,len(fields)-1)
        if initial_model.g is not None:assert c.state_sha(initial_model.g.state_dict())==record['initialization']['g']
        if initial_model.project is not None:assert c.state_sha(initial_model.project.state_dict())==record['initialization']['project']
        saved=np.load(folder/'best_val_predictions.npz');assert np.array_equal(saved['origins'],origins)
        np.testing.assert_allclose(saved['target'],all_y,rtol=1e-6,atol=1e-6)
        pred=[];numerical=[]
        with torch.no_grad():
            for i in range(0,len(origins),c.B):
                os=origins[i:i+c.B]
                raw_hist=torch.tensor(np.stack([scaled[o-c.T:o] for o in os]),device='cuda')
                yh=raw_hist[:,:,:1];cov=raw_hist[:,:,1:]
                # Invoke original numerical model, then reconstruct the adapter without Arm.forward.
                o=m.model(yh);n=o['h_pred'];hh=o['h_hist'];st=o['target_revin_stats']
                numerical_y=m.model.decode(n,st,hh);yp=numerical_y
                if arm!='N':
                    if arm=='PZ_NC':cov=torch.zeros_like(cov)
                    feat=torch.cat((n.flatten(1),hh[:,-2:].flatten(1),cov.flatten(1)),dim=1)
                    hidden=torch.nn.functional.gelu(torch.nn.functional.linear(feat,m.g[0].weight,m.g[0].bias))
                    delta=torch.nn.functional.linear(hidden,m.g[2].weight,m.g[2].bias).reshape(-1,c.H,c.D)
                    yp=(numerical_y+torch.nn.functional.linear(delta,m.project.weight)*st['stdev'] if arm=='PY'
                        else m.model.decode(n+delta,st,hh))
                pred.append(yp.cpu().numpy());numerical.append(numerical_y.cpu().numpy())
        pred=np.concatenate(pred);numerical=np.concatenate(numerical)
        e=max(float(np.max(np.abs(pred-saved['pred']))),float(np.max(np.abs(numerical-saved['num']))))
        max_error=max(max_error,e);assert e<3e-5,(name,arm,e)
        for tag,vv in [('fused',pred),('num',numerical)]:
            for metric,fn in [('mae',np.abs),('mse',np.square)]:
                val=float(np.mean(fn(vv-all_y),dtype=np.float64));key=tag+'_'+metric
                diff=abs(val-minimum[key]);score_error=max(score_error,diff)
                assert diff<3e-6,(name,arm,key,diff)
        checks.append({'dataset':name,'arm':arm,'val_origins':len(origins),'prediction_max_abs_error':e,
                       'checkpoint_selection_passed':True,'data_and_schedule_passed':True,
                       'encoder_decoder_unchanged':True,'test_scored':False})
        del m,initial_model,teacher
    for arm in ['PZ','PY','PZ_NC']:
        assert [x['predictor_sha256'] for x in curves[arm]]==[x['predictor_sha256'] for x in curves['N']]
        assert [x['num_mae'] for x in curves[arm]]==[x['num_mae'] for x in curves['N']]

decision={}
for name in c.DATA:
    rr={r['arm']:r for r in summary['results'] if r['dataset']==name};pz=rr['PZ']['best_val_mae']
    gains={a:1-pz/rr[a]['best_val_mae'] for a in ['PY','JZ_N','PZ_NC']}
    decision[name]={'PZ_relative_improvement':gains,'passes':gains['PY']>=.02 and gains['JZ_N']>=.02 and gains['PZ_NC']>0}
assert decision==summary['decision']
c.dump(OUT/'verification.json',{'status':'passed','summary_sha256':c.sha(OUT/'summary.json'),
       'script_sha256':c.sha(__file__),'source':c.source_check(),'checks':checks,
       'maximum_prediction_error':max_error,'maximum_metric_error':score_error,
       'protected_numerical_trajectories_identical':True,'posthoc_untrained_references':baselines,
       'screen_passes':all(d['passes'] for d in decision.values()),'test_scored':False,
       'seconds':time.perf_counter()-started})
print(json.dumps({'status':'passed','fits':len(checks),'max_replay_error':max_error,'max_metric_error':score_error}))
