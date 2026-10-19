"""Pinned native model execution and bounded interface arithmetic checks.

This is not a scientific performance benchmark or a novel method.
The unmodified upstream source remains read-only. Ordinary repairs are
isolated runtime functions, identified below and saved for inspection.
"""
import pathlib,sys,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'cycle29/scripts'))
import runtime_profile
import json,hashlib,importlib,types,inspect,random,time,traceback
import numpy as np
import torch
from safetensors.torch import load_file

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
UP=ROOT/'upstream/normwear2'
OUT=ROOT/'outputs/normwear_qualification_v1'
SEED=20263101
arrays={}; report={}; stage='initialize'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()

def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def timed(name,fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    value=fn()
    torch.cuda.synchronize()
    report.setdefault('timings',{})[name]={'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    return value

def store(name,value,shape=None):
    a=value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
    assert np.isfinite(a).all(),name
    if shape is not None:assert tuple(a.shape)==tuple(shape),(name,a.shape,shape)
    arrays[name]=a
    return {'shape':list(a.shape),'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'std':float(a.std())}

def main():
    global stage
    assert not OUT.exists(),'immutable output directory already exists'
    OUT.mkdir(parents=True)
    stage='verify_sources_and_weights'
    manifest=json.loads((UP/'source_manifest.json').read_text())
    for row in manifest['source_files']:assert sha(UP/row['path'])==row['sha256']
    wm=json.loads((UP/'weight_manifest.json').read_text())
    assert sha(UP/wm['path'])==wm['sha256']
    assert sha(UP/wm['demo']['path'])==wm['demo']['sha256']
    seed()
    report.update({'seed':SEED,'task_type':'native_API_qualification_and_autonomous_interface_diagnostic_not_paper_benchmark',
        'repo':wm['repo'],'revision':wm['revision'],'weights_sha256':wm['sha256'],
        'source_manifest_sha256':sha(UP/'source_manifest.json'),'script_sha256':sha(__file__),
        'qualification_plan_sha256':sha(ROOT/'NORMWEAR_QUALIFICATION_PLAN.md'),
        'scope_correction_sha256':sha(ROOT/'SCOPE_CORRECTION_20261017.md'),
        'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'numpy':np.__version__,
        'dependency_choice':'official direct core constructor, also shown in demo cell 2; no transformers installed or global environment changed'})
    stage='strict_load'
    # Bypass only __init__.py, which imports the unavailable HF wrapper.
    # Every loaded core source file above has its pinned manifest SHA verified.
    package=types.ModuleType('audited_normwear31');package.__path__=[str(UP)]
    sys.modules[package.__name__]=package
    mod=importlib.import_module(package.__name__+'.normwear2')
    lb=importlib.import_module(package.__name__+'.latent_bayesian')
    config=json.loads((UP/'config.json').read_text())
    args={k:v for k,v in config.items() if k in inspect.signature(mod.NormWear2.__init__).parameters}
    model=mod.NormWear2(**args)
    state=load_file(str(UP/'model.safetensors'),device='cpu')
    assert all(k.startswith('normwear.') for k in state)
    # HF wrapper owns only the normwear core module; all inference APIs delegate.
    stripped={k[len('normwear.'):]:v for k,v in state.items()}
    loaded=model.load_state_dict(stripped,strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    report['model']={'constructor_args':args,'state_keys':len(state),'strict':True,'missing':loaded.missing_keys,
                     'unexpected':loaded.unexpected_keys,'parameters':sum(p.numel() for p in model.parameters())}
    del state,stripped
    model.eval().to('cuda')
    stage='native_card_predict'
    seed();x=torch.rand(2,256,3,device='cuda')
    arrays['card_input']=x.cpu().numpy()
    with torch.no_grad():
        pred=timed('card_predict',lambda:model.predict(x[:,:128,:].clone(),64))
        report['native_card_predict']=store('card_predict',pred,(2,64,3))
    print(json.dumps({'stage':stage,'result':report['native_card_predict']}),flush=True)

    stage='symbolic_latent_layout'
    E=7;N=3;P=5
    tags=torch.arange(N*P*E).reshape(N,P,E)
    flat=tags.permute(1,2,0).flatten(start_dim=1)
    assert torch.equal(flat.reshape(P,E,N).permute(2,0,1),tags)
    rows=[]
    for ac in ([2],[0],[0,2]):
        native_slice=flat[:,-E*len(ac):]
        expected=tags[ac].permute(1,2,0).flatten(start_dim=1)
        idx=(torch.arange(E)[:,None]*N+torch.tensor(ac)[None,:]).flatten()
        repaired=flat[:,idx]
        assert torch.equal(repaired,expected)
        rows.append({'action_channels':list(ac),'native_matches_expected':bool(torch.equal(native_slice,expected)),
                     'native_differing_entries':int((native_slice!=expected).sum()),'entries':expected.numel(),
                     'explicit_index_matches_expected':True,'explicit_indices':idx.tolist()})
        arrays['symbolic_native_'+'_'.join(map(str,ac))]=native_slice.numpy()
        arrays['symbolic_expected_'+'_'.join(map(str,ac))]=expected.numpy()
    report['symbolic_latent_layout']={'N':N,'P':P,'E':E,'inverse_exact':True,'cases':rows}

    stage='native_and_clamped_simulate'
    C=128;H=64;L=C+H
    t=torch.arange(L,dtype=torch.float32,device='cuda')
    sample=torch.stack((torch.sin(t*.091)+.2*torch.cos(t*.037),
                        torch.cos(t*.057)+.1*torch.sin(t*.13),
                        .01*t+.7*torch.sin(t*.11)),dim=-1)[None]
    arrays['simulate_input']=sample.cpu().numpy()
    original_enc=model.forward_encoder
    native_simulate=model.simulate
    captures=[]
    def wrapped_encoder(self,value,*a,**kw):
        captures.append((value.detach().cpu().numpy().copy(),dict(kw)))
        return original_enc(value,*a,**kw)
    model.forward_encoder=types.MethodType(wrapped_encoder,model)
    source=inspect.getsource(mod.NormWear2.simulate)
    old='pad_context_tensor[:, :, all_visible_length:curr_max_possible_length] = dec_out[:, :, all_visible_length:curr_max_possible_length]'
    new='pad_context_tensor[:, non_visible_channel, all_visible_length:curr_max_possible_length] = dec_out[:, non_visible_channel, all_visible_length:curr_max_possible_length]'
    assert source.count(old)==1
    import textwrap
    fixed_source=textwrap.dedent(source.replace(old,new))
    (OUT/'ordinary_clamp_repair.py').write_text(fixed_source,encoding='utf-8')
    namespace={};exec(compile(fixed_source,'ordinary_clamp_repair.py','exec'),mod.__dict__,namespace)
    repaired_simulate=types.MethodType(namespace['simulate'],model)
    action=sample.permute(0,2,1)[:,2:3,:]
    loc=action.mean(-1,keepdim=True);scale=action.std(-1,keepdim=True)+1e-8
    action_expected=((action-loc)/scale).cpu().numpy()
    traces={}
    for name,fn in [('native',native_simulate),('ordinary_clamp_repair',repaired_simulate)]:
        captures.clear();seed()
        out=timed(name+'_simulate',lambda:fn(sample.clone(),all_visible_length=C,non_visible_channel=[0,1],ar_step=16))
        report[name+'_simulate']=store(name+'_simulate',out,(1,208,3))
        assert len(captures)==4
        errors=[]
        for i,(value,kw) in enumerate(captures):
            arrays[name+'_encoder_input_'+str(i)]=value
            assert value.shape==(1,3,C+16*(i+1))
            assert kw['all_visible_length']==C+16*i and kw['non_visible_channel']==[0,1]
            errors.append(float(np.max(np.abs(value[:,2:3,:]-action_expected[:,:,:value.shape[-1]]))))
        report[name+'_simulate']['known_action_encoder_errors']=errors
        report[name+'_simulate']['known_action_output_error']=float((out[:,:L,2]-sample[:,:,2]).abs().max())
        report[name+'_simulate']['past_target_output_error']=float((out[:,:C,:2]-sample[:,:C,:2]).abs().max())
        traces[name]=[a.copy() for a,k in captures]
    assert report['native_simulate']['known_action_encoder_errors'][0]<1e-6
    assert max(report['ordinary_clamp_repair_simulate']['known_action_encoder_errors'])<1e-6
    assert report['ordinary_clamp_repair_simulate']['known_action_output_error']<1e-6
    # If all channels are forecast, restriction to those channels is identical.
    captures.clear();seed();all_native=native_simulate(sample.clone(),all_visible_length=C,non_visible_channel=[0,1,2],ar_step=16)
    captures.clear();seed();all_fixed=repaired_simulate(sample.clone(),all_visible_length=C,non_visible_channel=[0,1,2],ar_step=16)
    assert torch.equal(all_native,all_fixed)
    report['ordinary_clamp_no_action_equivalence']={'max_abs':float((all_native-all_fixed).abs().max()),'exact':True}
    model.forward_encoder=original_enc

    stage='native_latent_bayesian_and_future_target_invariance'
    # Official demo cell 8's interface, shorter generated input. This is not
    # the unavailable private pretraining dataset demonstration reproduction.
    raw=sample.permute(0,2,1)
    mean=raw[:,:,:C].mean(-1,keepdim=True);std=raw[:,:,:C].std(-1,keepdim=True)+1e-8
    normalized=torch.clamp((raw-mean)/std,-3,3)
    masked=normalized.clone();masked[:,:2,C:]=0
    changed=normalized.clone();changed[:,:2,C:]=torch.arange(H,device='cuda')[None,None,:]*100+50
    arrays['bayes_masked_input']=masked.cpu().numpy()
    bayes_args={'n_channels':3,'physio_channels':[0,1],'context_length':C,'pred_length':H,
                'num_states':4,'action_channels':[2],'condition_bayes':True,'num_traj_sampled':1,
                'latent_encoder':model,'return_transit_matrix':True}
    seed();native_pack=timed('native_bayes',lambda:model.latent_bayesian_transition(masked,**bayes_args))
    seed();changed_pack=timed('changed_future_target_bayes',lambda:model.latent_bayesian_transition(changed,**bayes_args))
    report['native_bayes']=store('native_bayes',native_pack[0],(2,H))
    difference=float((native_pack[0]-changed_pack[0]).abs().max())
    assert difference<1e-6
    report['future_target_invariance']={'max_abs':difference,'passed':True,'same_seed':SEED,'past_and_actions_identical':True}
    arrays['bayes_transit']=native_pack[1];arrays['bayes_centroids']=native_pack[2]
    arrays['bayes_flat_context']=native_pack[4].cpu().numpy()
    # Capture the exact centroid slice and future action embedding used in
    # native get_traj_of_state, then apply the ordinary explicit-index repair.
    original_get=lb.get_traj_of_state
    bcap=[]
    def spy(*a,**kw):
        if kw.get('future_action_enc_out') is not None:
            bcap.append((np.array(a[2],copy=True),kw['future_action_enc_out'].cpu().numpy().copy()))
        return original_get(*a,**kw)
    lb.get_traj_of_state=spy
    seed();spy_pack=model.latent_bayesian_transition(masked,**bayes_args)
    assert torch.equal(spy_pack[0],native_pack[0])
    assert len(bcap)==1
    centroids,future=bcap[0];embed_dim=centroids.shape[-1]//3
    idx=(np.arange(embed_dim)[:,None]*3+np.array([2])[None,:]).reshape(-1)
    arrays['bayes_future_action_embedding']=future
    arrays['bayes_native_action_centroids']=centroids[:,-future.shape[-1]:]
    arrays['bayes_expected_action_centroids']=centroids[:,idx]
    report['actual_latent_layout']={'embedding_dim':embed_dim,'channels':3,'action_channels':[2],
       'slice_shape':list(centroids[:,-future.shape[-1]:].shape),
       'max_abs_native_vs_explicit':float(np.abs(centroids[:,-future.shape[-1]:]-centroids[:,idx]).max())}
    get_source=textwrap.dedent(inspect.getsource(original_get))
    old_slice='centroids[:, -future_action_enc_out.shape[-1]:]'
    new_slice='centroids[:, (np.arange(centroids.shape[-1] // 3)[:, None] * 3 + np.array([2])[None, :]).reshape(-1)]'
    assert get_source.count(old_slice)==1
    get_fixed=get_source.replace(old_slice,new_slice)
    (OUT/'ordinary_action_index_repair_specific_N3_A2.py').write_text(get_fixed,encoding='utf-8')
    ns={};exec(compile(get_fixed,'ordinary_action_index_repair_specific_N3_A2.py','exec'),lb.__dict__,ns)
    lb.get_traj_of_state=ns['get_traj_of_state']
    seed();fixed_pack=timed('ordinary_index_bayes',lambda:model.latent_bayesian_transition(masked,**bayes_args))
    report['ordinary_index_bayes']=store('ordinary_index_bayes',fixed_pack[0],(2,H))
    report['ordinary_index_bayes']['output_difference_vs_native']=float((native_pack[0]-fixed_pack[0]).abs().max())
    lb.get_traj_of_state=original_get
    # Context-only, no-action caller: never supply any future target samples.
    seed();noaction=timed('native_bayes_no_action',lambda:model.latent_bayesian_transition(normalized[:,:,:C],
         n_channels=3,physio_channels=[0,1,2],context_length=C,pred_length=H,num_states=4,action_channels=[],
         condition_bayes=False,num_traj_sampled=1,latent_encoder=model,return_transit_matrix=False))
    report['native_bayes_no_action']=store('native_bayes_no_action',noaction,(3,H))
    stage='write_outputs'
    np.savez_compressed(OUT/'raw_arrays.npz',**arrays)
    report.update({'status':'passed_execution_and_interface_diagnostic_checks','raw_arrays_sha256':sha(OUT/'raw_arrays.npz'),
        'accepted_new_directions':0,'paper_performance_reproduced':False,'new_method':False,
        'limitations':['generated deterministic inputs, no scientific benchmark scores',
        'original simulate returns patch-padded length 208 for input 192; recorded, not silently truncated',
        'ordinary action index repair is intentionally specialized to audited N=3 action=[2]; not released as general API',
        'interface arithmetic defects do not establish a new research gap or invalidate all paper results']})
    (OUT/'qualification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'native_action_encoder_errors':report['native_simulate']['known_action_encoder_errors'],
         'fixed_action_encoder_errors':report['ordinary_clamp_repair_simulate']['known_action_encoder_errors'],
         'actual_layout':report['actual_latent_layout'],'future_target_invariance':report['future_target_invariance'],
         'new_directions':0}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            np.savez_compressed(OUT/'partial_arrays.npz',**arrays)
            (OUT/'failure.json').write_text(json.dumps({'stage':stage,'error':repr(exc),'traceback':traceback.format_exc(),'partial_report':report},ensure_ascii=False,indent=2),encoding='utf-8')
        raise
