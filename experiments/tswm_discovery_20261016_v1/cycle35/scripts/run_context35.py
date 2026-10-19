import context_common35 as c
import torch, numpy as np, time, json, datetime, pathlib, copy
OUT=c.ROOT/'outputs/pilot_v1'
assert not OUT.exists(),'Never silently restart an existing scientific run'
pre=json.loads((c.ROOT/'outputs/preflight.json').read_text());assert pre['status']=='passed'
assert pre['plan_sha256']==c.sha(c.ROOT/'MECHANISM_PILOT_PLAN.md')
assert pre['common_sha256']==c.sha(c.ROOT/'scripts/context_common35.py')
OUT.mkdir(parents=True)
Model,Loss=c.native();source=c.source_check();started=time.perf_counter()
locks={'source':source,'plan_sha256':pre['plan_sha256'],'common_sha256':pre['common_sha256'],
       'runner_sha256':c.sha(__file__),'preflight_sha256':c.sha(c.ROOT/'outputs/preflight.json')}
results=[];metadata=[]

def save_state(path,model):torch.save(model.state_dict(),path)
def read_state(path):return torch.load(path,map_location='cpu',weights_only=True)

@torch.no_grad()
def evaluate_teacher(model,values,origins,criterion,stage):
    model.eval();total=0.;count=0
    for i in range(0,len(origins),c.B):
        yh,_,yf=c.batch(values,origins[i:i+c.B]);o=model(yh,future_target=yf if stage=='latent' else None)
        v=(criterion(o['h_pred'],o['h_target'])['loss'] if stage=='latent' else
           (model.decode(o['h_pred'],o['target_revin_stats'],o['h_hist'])-yf).abs().mean())
        total+=float(v)*len(yh);count+=len(yh)
    return total/count

def fit_teacher(name,values,origins):
    folder=OUT/name/'teacher';folder.mkdir(parents=True)
    c.seed();m=Model(c.SimpleNamespace(**c.CONFIG)).cuda()
    criterion=Loss(latent_mse_weight=10,latent_alignment_loss='mae',latent_cosine_weight=15,
                   latent_sigreg_weight=.1,latent_detach_target=True,latent_regularizer_type='legacy_vc')
    histories={}
    for stage in ['latent','decoder']:
        m.configure_for_stage1() if stage=='latent' else m.configure_for_stage2()
        params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=.001)
        rng=np.random.default_rng(c.SEED+(0 if stage=='latent' else 1))
        schedule=rng.choice(origins['train'],size=(1000,c.B),replace=True)
        np.save(folder/(stage+'_schedule.npy'),schedule)
        history=[];best=float('inf');path=folder/(stage+'_best.pt');st=time.perf_counter()
        for step in range(1001):
            if step%100==0:
                val=evaluate_teacher(m,values,origins['val'],criterion,stage)
                history.append({'step':step,'val_objective':val})
                if val<best:best=val;save_state(path,m)
                print(json.dumps({'dataset':name,'stage':stage,'step':step,'val':val}),flush=True)
            if step==1000:break
            m.train();yh,_,yf=c.batch(values,schedule[step]);opt.zero_grad(set_to_none=True)
            if stage=='latent':
                o=m(yh,future_target=yf);loss=criterion(o['h_pred'],o['h_target'])['loss']
            else:
                with torch.no_grad():o=m(yh)
                loss=(m.decode(o['h_pred'],o['target_revin_stats'],o['h_hist'])-yf).abs().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        m.load_state_dict(read_state(path));histories[stage]={'curve':history,'seconds':time.perf_counter()-st,
             'best_checkpoint_sha256':c.sha(path),'schedule_sha256':c.sha(folder/(stage+'_schedule.npy'))}
        c.dump(folder/'history.json',histories)
    m.eval();return m,histories

for di,name in enumerate(c.DATA):
    values,origins,meta=c.load_data(name);metadata.append(meta)
    c.dump(OUT/'metadata.json',metadata)
    teacher,teacher_history=fit_teacher(name,values,origins)
    teacher_sha=c.state_sha(teacher.state_dict());encoder_sha=c.state_sha(teacher.target_encoder.state_dict())
    decoder_sha=c.state_sha(teacher.decoder.state_dict())
    schedule=np.random.default_rng(c.SEED+10000+di).choice(origins['train'],size=(2000,c.B),replace=True)
    np.save(OUT/name/'arm_schedule.npy',schedule)
    protected_hashes={}
    for arm in c.ARMS:
        folder=OUT/name/arm;folder.mkdir();c.seed();m=c.Arm(teacher,arm,values.shape[-1]-1)
        p,ctx=m.groups();opt=torch.optim.Adam(p+ctx,lr=.0003)
        curves=[];best=float('inf');best_step=None;st=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        init={'teacher':teacher_sha,'predictor':c.state_sha(m.model.predictor.state_dict()),
              'g':None if m.g is None else c.state_sha(m.g.state_dict()),
              'project':None if m.project is None else c.state_sha(m.project.state_dict())}
        for step in range(2001):
            if step%100==0:
                scores,arrays=c.evaluate_arm(m,values,origins['val'])
                row={'step':step,**scores,'predictor_sha256':c.state_sha(m.model.predictor.state_dict())}
                if arm=='JZ_N' and step in [100,500,1000,1500,2000]:
                    yh,ch,yf=c.batch(values,schedule[0]);o=m(yh,ch,yf)
                    fusion=(o['fused']-yf).abs().mean()+(o['z']-o['target']).abs().mean()
                    num=(o['numerical']-yf).abs().mean()
                    fg=torch.autograd.grad(fusion,p,retain_graph=True,allow_unused=True)
                    ng=torch.autograd.grad(num,p,allow_unused=True)
                    dot=sum((a*b).sum() for a,b in zip(fg,ng) if a is not None and b is not None)
                    fn=sum(a.square().sum() for a in fg if a is not None).sqrt()
                    nn=sum(a.square().sum() for a in ng if a is not None).sqrt()
                    row['P_gradient_diagnostic']={'dot':float(dot),'fusion_norm':float(fn),
                          'numerical_norm':float(nn),'cosine':float(dot/(fn*nn).clamp_min(1e-20)),
                          'fixed_train_batch_schedule_index':0}
                curves.append(row)
                if scores['fused_mae']<best:
                    best=scores['fused_mae'];best_step=step;save_state(folder/'best.pt',m)
                    np.savez_compressed(folder/'best_val_predictions.npz',**arrays)
                c.dump(folder/'curve.json',curves)
                c.dump(OUT/'status.json',{'status':'running','dataset':name,'arm':arm,'step':step,'completed_fits':len(results)})
                print(json.dumps({'dataset':name,'arm':arm,'step':step,'val_mae':scores['fused_mae'],'num_mae':scores['num_mae']}),flush=True)
            if step==2000:break
            m.train();yh,ch,yf=c.batch(values,schedule[step]);opt.zero_grad(set_to_none=True)
            o=m(yh,ch,yf);loss,_=m.losses(o,yf);assert bool(torch.isfinite(loss))
            loss.backward()
            # Separate clipping prevents context gradient norms from changing the protected P update.
            torch.nn.utils.clip_grad_norm_(p,1.)
            if ctx:torch.nn.utils.clip_grad_norm_(ctx,1.)
            opt.step()
        assert c.state_sha(m.model.target_encoder.state_dict())==encoder_sha
        assert c.state_sha(m.model.decoder.state_dict())==decoder_sha
        if arm in ['N','PZ','PY','PZ_NC']:
            protected_hashes[arm]=[r['predictor_sha256'] for r in curves]
        rec={'dataset':name,'arm':arm,'seed':c.SEED,'best_step':best_step,'best_val_mae':best,
             'best_metrics':curves[best_step//100], 'seconds':time.perf_counter()-st,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'initialization':init,
             'best_checkpoint_sha256':c.sha(folder/'best.pt'),
             'predictions_sha256':c.sha(folder/'best_val_predictions.npz'),
             'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
             'last_500_relative_val_improvement':(min(r['fused_mae'] for r in curves if r['step']<=1500)-min(r['fused_mae'] for r in curves))/min(r['fused_mae'] for r in curves if r['step']<=1500),
             'encoder_and_decoder_unchanged':True,'updates':2000,'schedule_sha256':c.sha(OUT/name/'arm_schedule.npy')}
        results.append(rec);c.dump(folder/'result.json',rec);c.dump(OUT/'results.json',results)
        del m,opt
    assert all(v==protected_hashes['N'] for v in protected_hashes.values()),'Protected numerical trajectories diverged'
    c.dump(OUT/name/'protected_predictor_identity.json',{'passed':True,'arms':list(protected_hashes),
            'at_steps':list(range(0,2001,100)),'all_state_hashes_equal':True})
    del values,teacher

decision={}
for name in c.DATA:
    rr={r['arm']:r for r in results if r['dataset']==name};pz=rr['PZ']['best_val_mae']
    gains={a:1-pz/rr[a]['best_val_mae'] for a in ['PY','JZ_N','PZ_NC']}
    decision[name]={'PZ_relative_improvement':gains,'passes':gains['PY']>=.02 and gains['JZ_N']>=.02 and gains['PZ_NC']>0}
c.dump(OUT/'summary.json',{'status':'completed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'locks':locks,'metadata':metadata,'results':results,'decision':decision,
       'screen_passes':all(d['passes'] for d in decision.values()),'test_scored':False,'seed_count':1,
       'seconds':time.perf_counter()-started,'accepted_directions':0,'not_a_paper_performance_reproduction':True})
c.dump(OUT/'status.json',{'status':'completed','fits':len(results),'test_scored':False})
print(json.dumps({'status':'completed','fits':len(results),'decision':decision}),flush=True)
