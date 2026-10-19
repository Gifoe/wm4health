"""Short implementation diagnostic, not a new scientific comparison."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import context_common35 as c
import torch,json
c.source_check();Model,_=c.native()
values,origins,_=c.load_data('solar');schedule=c.np.load(c.ROOT/'outputs/pilot_v1/solar/arm_schedule.npy')
state=torch.load(c.ROOT/'outputs/pilot_v1/solar/teacher/decoder_best.pt',map_location='cpu',weights_only=True)
for deterministic,foreach in [(False,None),(False,False),(True,False)]:
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic=deterministic
    output=[];reference=[]
    for arm in ['N','PZ','PY','PZ_NC']:
        c.seed();teacher=Model(c.SimpleNamespace(**c.CONFIG)).cuda();teacher.load_state_dict(state)
        c.seed();m=c.Arm(teacher,arm,values.shape[-1]-1);m.train();p,ctx=m.groups()
        opt=torch.optim.Adam(p+ctx,lr=.0003,foreach=foreach)
        rows=[]
        for step in range(12):
            yh,ch,yf=c.batch(values,schedule[step]);opt.zero_grad(set_to_none=True)
            out=m(yh,ch,yf);loss,_=m.losses(out,yf);loss.backward()
            gs=[None if t.grad is None else t.grad.detach().clone() for t in p]
            if arm=='N':reference.append(gs)
            diff=max(float((a-b).abs().max()) for a,b in zip(gs,reference[step]) if a is not None and b is not None)
            torch.nn.utils.clip_grad_norm_(p,1.)
            if ctx:torch.nn.utils.clip_grad_norm_(ctx,1.)
            opt.step()
            rows.append({'step':step+1,'gradient_max_abs_diff_vs_N':diff,'predictor_sha256':c.state_sha(m.model.predictor.state_dict())})
        output.append({'arm':arm,'rows':rows})
        del m,opt,teacher
    identical=all([r['predictor_sha256'] for r in a['rows']]==[r['predictor_sha256'] for r in output[0]['rows']] for a in output)
    record={'deterministic':deterministic,'adam_foreach':foreach,'protected_trajectories_identical':identical,'records':output}
    filename='determinism_'+str(deterministic)+'_'+str(foreach)+'.json'
    c.dump(c.ROOT/'outputs/diagnostics'/filename,record)
    print(json.dumps({'deterministic':deterministic,'adam_foreach':foreach,'identical':identical,
        'first_step_gradient_differences':{a['arm']:a['rows'][0]['gradient_max_abs_diff_vs_N'] for a in output}}),flush=True)
