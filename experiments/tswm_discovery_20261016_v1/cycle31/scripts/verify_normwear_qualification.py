"""Independent regrouping and one original model-output replay, server only."""
import sys,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'cycle29/scripts'))
import runtime_profile
import hashlib,json,importlib,types,inspect,time
import numpy as np
import torch
from safetensors.torch import load_file
OUT=ROOT/'outputs/normwear_qualification_v3'
UP=ROOT/'upstream/normwear2'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while b:=f.read(4194304):h.update(b)
    return h.hexdigest()
start=time.perf_counter();torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
q=json.loads((OUT/'qualification.json').read_text(encoding='utf-8'))
assert q['status']=='partial_native_execution_and_ordinary_repairs_verified'
assert q['accepted_new_directions']==0 and not q['new_method'] and not q['paper_performance_reproduced']
assert sha(OUT/'raw_arrays.npz')==q['raw_arrays_sha256']
assert sha(ROOT/'scripts/qualify_normwear_v3.py')==q['script_sha256']
assert sha(ROOT/'SCOPE_CORRECTION_20261017.md')==q['scope_correction_sha256']
assert sha(ROOT/'NORMWEAR_QUALIFICATION_PLAN.md')==q['qualification_plan_sha256']
source=json.loads((UP/'source_manifest.json').read_text())
for row in source['source_files']:assert sha(UP/row['path'])==row['sha256']
a=np.load(OUT/'raw_arrays.npz')
assert all(np.isfinite(a[k]).all() for k in a.files)
raw=a['simulate_input'];C=128;L=192
assert raw.shape==(1,L,3)
# Numpy ddof=1 normalization and direct source-action comparisons; no access
# to candidate decoded targets is needed to know the allowed action signal.
u=raw[:,:,2];normalized=(u-u.mean(axis=1,keepdims=True))/(u.std(axis=1,ddof=1,keepdims=True)+1e-8)
trace_rows=[]
for variant in ['native','ordinary_clamp_repair']:
    values=[]
    for step in range(4):
        inp=a[variant+'_encoder_input_'+str(step)]
        end=C+16*(step+1);assert inp.shape==(1,3,end)
        err=float(np.max(np.abs(inp[:,2,:]-normalized[:,:end])))
        values.append(err)
        if variant=='ordinary_clamp_repair':assert err<3e-6
        trace_rows.append({'variant':variant,'step':step,'known_action_max_abs':err})
    assert np.allclose(values,q[variant+'_simulate']['known_action_encoder_errors'],atol=3e-6,rtol=0)
    result=a[variant+'_simulate'];assert result.shape==(1,208,3)
    rawerr=float(np.abs(result[:,:L,2]-u).max())
    assert abs(rawerr-q[variant+'_simulate']['known_action_output_error'])<3e-6
    if variant=='ordinary_clamp_repair':assert rawerr<3e-6
assert max(q['native_simulate']['known_action_encoder_errors'][1:])>.01
# Recover the expected channel using the full [centroid,embedding,channel]
# tensor rather than constructing the candidate's explicit linear indices.
K,E,N=4,768,3
cent=a['bayes_centroids'];assert cent.shape==(K,E*N)
expected=cent.reshape(K,E,N).transpose(0,2,1)[:,2,:]
assert np.array_equal(expected,a['bayes_expected_action_centroids'])
assert np.array_equal(cent[:,-E:],a['bayes_native_action_centroids'])
actualdiff=float(np.max(np.abs(cent[:,-E:]-expected)))
assert actualdiff==q['actual_latent_layout']['max_abs_native_vs_explicit']
# Rebuild observed state transitions using closest saved centroids. This
# independently verifies the matrix used by the conditional sampler.
obs=a['bayes_flat_context'];assert obs.shape==(8,2304)
dist=((obs[:,None,:].astype(np.float64)-cent[None,:,:].astype(np.float64))**2).sum(-1)
label=dist.argmin(-1);counts=np.zeros((K,K),dtype=np.float64)
for prev,nxt in zip(label[:-1],label[1:]):counts[prev,nxt]+=1
trans=(counts+1e-8)/(counts+1e-8).sum(-1,keepdims=True)
trans=(trans+1e-8)/(trans+1e-8).sum(-1,keepdims=True)
transitionerr=float(np.abs(trans-a['bayes_transit']).max());assert transitionerr<1e-12
for row in q['symbolic_latent_layout']['cases']:
    ac=row['action_channels'];expected_tags=np.empty((5,7*len(ac)),dtype=np.int64)
    for p in range(5):
        column=0
        for e in range(7):
            for channel in ac:
                expected_tags[p,column]=(channel*5+p)*7+e;column+=1
    saved=a['symbolic_expected_'+'_'.join(map(str,ac))]
    assert np.array_equal(expected_tags,saved)
assert q['native_bayes_attempt']['status']=='failed' and 'IndexError' in q['native_bayes_attempt']['error']
assert a['ordinary_length_bayes'].shape==(2,64)
assert a['ordinary_length_and_index_bayes'].shape==(2,64)
assert a['native_bayes_no_action'].shape==(3,64)
# Original card replay is unmodified source and direct strict checkpoint,
# independent of the patched runtime functions in the qualification process.
pkg=types.ModuleType('proof_normwear31');pkg.__path__=[str(UP)];sys.modules[pkg.__name__]=pkg
mod=importlib.import_module(pkg.__name__+'.normwear2')
cfg=json.loads((UP/'config.json').read_text())
args={k:v for k,v in cfg.items() if k in inspect.signature(mod.NormWear2.__init__).parameters}
model=mod.NormWear2(**args)
state=load_file(str(UP/'model.safetensors'));assert set(state)==set(model.state_dict())
model.load_state_dict(state,strict=True);del state
model.eval().to('cuda')
with torch.no_grad():
    replay=model.predict(torch.from_numpy(a['card_input'][:,:128,:]).to('cuda'),64).cpu().numpy()
replayerr=float(np.abs(replay-a['card_predict']).max());assert replayerr<1e-6
proof={'status':'passed','qualification_sha256':sha(OUT/'qualification.json'),
       'raw_arrays_sha256':sha(OUT/'raw_arrays.npz'),'verifier_sha256':sha(__file__),
       'native_card_replay_max_abs':replayerr,'numpy_action_trace_checks':trace_rows,
       'centroid_channel_layout_max_abs_difference':actualdiff,'independent_transition_max_abs':transitionerr,
       'finite_arrays':len(a.files),'seconds':time.perf_counter()-start,
       'scope':'execution/interface evidence only; no new mechanism or scientific benchmark claim'}
assert not (OUT/'verification.json').exists()
(OUT/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(proof),flush=True)
