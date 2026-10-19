import context_common35 as c
import torch, json
c.source_check();c.seed();Model,_=c.native();teacher=Model(c.SimpleNamespace(**c.CONFIG)).cuda().eval()
records=[]
for name in c.DATA:
    values,origins,meta=c.load_data(name);yh,ch,yf=c.batch(values,origins['train'][:c.B])
    initial=[];g_hash=[]
    for arm in c.ARMS:
        c.seed();m=c.Arm(teacher,arm,ch.shape[-1]).eval();o=m(yh,ch,yf)
        initial.append(o['fused'].detach());p,ctx=m.groups()
        if m.g is not None:g_hash.append(c.state_sha(m.g.state_dict()))
        fused=(o['fused']-yf).abs().mean()
        context_objective=fused if arm=='PY' else fused+(o['z']-o['target']).abs().mean()
        grad=torch.autograd.grad(context_objective,p,allow_unused=True,retain_graph=True)
        norm=float(torch.sqrt(sum((x.square().sum() for x in grad if x is not None),torch.zeros((),device='cuda'))))
        if arm in ['PZ','PY','PZ_NC']:assert norm==0,(arm,norm)
        else:assert norm>0,(arm,norm)
        loss,_=m.losses(o,yf);loss.backward()
        assert all(p.grad is None for p in m.model.target_encoder.parameters())
        assert all(p.grad is None for p in m.model.decoder.parameters())
        if ctx:assert sum(float(p.grad.abs().sum()) for p in ctx if p.grad is not None)>0
        with torch.no_grad():
            alt=m(yh,ch,yf+12345.)['fused'];assert torch.equal(alt,o['fused'])
        records.append({'dataset':name,'arm':arm,'context_objective_P_gradient_norm':norm,
                        'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
                        'same_initial_forecast':True,'future_target_invariance':True})
    assert all(torch.equal(initial[0],a) for a in initial)
    assert len(set(g_hash))==1
    del values
c.dump(c.ROOT/'outputs/preflight.json',{'status':'passed','records':records,'source':c.source_check(),
    'plan_sha256':c.sha(c.ROOT/'MECHANISM_PILOT_PLAN.md'),'common_sha256':c.sha(c.ROOT/'scripts/context_common35.py'),
    'script_sha256':c.sha(__file__),'test_scored':False})
print(json.dumps({'preflight':'passed','checks':len(records)}))
