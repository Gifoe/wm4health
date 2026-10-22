"""Independent analytic/Torch check of source gradient routing fixture."""
import pathlib,json,hashlib,datetime
import numpy as np,torch
torch.set_num_threads(2);ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'crossmodal_gradient_contract_v1'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
j=read(OUT/'RESULTS_V1.json');assert sha(j['payload_path'])==j['payload_sha256']
with np.load(j['payload_path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
c=torch.from_numpy(a['context']).double();s=torch.from_numpy(a['wm_state']).double().requires_grad_(True);w=torch.from_numpy(a['parameter/wm/ctx_enc/recon_wm/kernel']).double().requires_grad_(True);b=torch.from_numpy(a['parameter/wm/ctx_enc/recon_wm/bias']).double().requires_grad_(True)
r=c@w+b;loss1=((r.detach()-s)**2).mean(-1).sum();g=torch.autograd.grad(loss1,(s,w,b),allow_unused=True);assert g[1] is None and g[2] is None
error=float((g[0]-torch.from_numpy(a['wm_state_gradient']).double()).abs().max());assert error<2e-8,error
assert np.array_equal(a['gradient/wm/ctx_enc/recon_wm/kernel'],np.zeros_like(a['gradient/wm/ctx_enc/recon_wm/kernel'])) and np.array_equal(a['gradient/wm/ctx_enc/recon_wm/bias'],np.zeros_like(a['gradient/wm/ctx_enc/recon_wm/bias']))
assert any(np.max(abs(x))>0 for k,x in a.items() if k.startswith('gradient/wm/ctx_enc/recon_ctxc/'))
result=dict(status='NATIVE_GRADIENT_ROUTE_ANALYTICALLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wm_state_gradient_scalar_components=int(s.numel()),max_target_gradient_error=error,forward_projection_gradient_zero=True,reverse_projection_gradient_nonzero=True,results_sha256=sha(OUT/'RESULTS_V1.json'),physical_model_failure_explained=False,paper_experiment_revision_known=False)
with (OUT/'VERIFICATION_V1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
