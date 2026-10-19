import runtime_profile
import hashlib,json,pathlib,time
import torch,numpy as np
from native_context_runtime import ROOT,load,verify_source,restricted_checkpoint
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
manifest=json.loads((ROOT/'outputs/native_context_weights/manifest.json').read_text());rows=[]
for r in manifest['models']:
    f=ROOT/'outputs/native_context_weights'/r['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
    env,model,encoder=load(r['config'],f);cfg=r['config'];device='cuda:0'
    torch.manual_seed(20262399)
    x=torch.randn(5,3,env.state_dim,device=device);u=torch.randn(5,3,env.action_dim,device=device);xn=torch.randn_like(x)
    with torch.inference_mode():
        a=encoder.forward_tensor(x,u,xn)
        b=encoder(x.flatten(0,1),u.flatten(0,1),xn.flatten(0,1),torch.arange(3,device=device).repeat(5),3)
        delta=max(float((a.mean-b.mean).abs().max()),float((a.stddev-b.stddev).abs().max()));assert delta==0
        c=encoder.empty_set_context();assert bool((c.scale>0).all())
        controls=torch.randn(4,3,env.action_dim,device=device)
        yy=model.forward_multi_step(x[0].contiguous(),controls,a.mean)
        assert yy.mean.shape==(5,3,env.state_dim) and torch.isfinite(yy.mean).all()
        noisy=controls.clone();noisy[2:]+=1.0;yy2=model.forward_multi_step(x[0].contiguous(),noisy,a.mean)
        causal=float((yy.mean[:3]-yy2.mean[:3]).abs().max());assert causal==0
    payload=restricted_checkpoint(f)
    assert payload['is_best'] is True and payload['step']<cfg['n_training_steps']
    rows.append(dict(env=cfg['env_name'],seed=cfg['seed'],checkpoint_sha256=r['sha256'],selected_step=payload['step'],validation_loss=float(payload['validation_loss']),ragged_tensor_max_error=delta,causal_prefix_error=causal,parameters=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in encoder.parameters()),allowed_numeric_globals=torch.serialization.get_unsafe_globals_in_checkpoint(f),config=cfg,weight_path=r['path']))
result=dict(status='pass',source_revision=verify_source(),models=rows,compatibility='Process-local native torch.max value adapter for optional torch_scatter; numerical source unchanged; native ragged/tensor outputs exactly equal. Legacy gym0.17.2 isolated in runtime_dependencies. No training or environment mutation.')
(ROOT/'outputs/native_context_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(status='pass',models=len(rows),source_revision=result['source_revision'])))
