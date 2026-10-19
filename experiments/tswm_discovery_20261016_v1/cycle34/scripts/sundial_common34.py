"""Pinned native source, exact single-block noise exposure, no remote import."""
import pathlib,sys,os
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(ROOT/'vendor'));sys.path.insert(1,str(BASE/'cycle29/scripts'))
import runtime_profile
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import json,hashlib,types,importlib,importlib.metadata as metadata,random
import numpy as np
import torch
from safetensors.torch import load_file
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
SOURCE=ROOT/'upstream_source/Sundial';WEIGHT=ROOT/'assets/Sundial/model.safetensors'
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()
def dump(path,data):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)
def state_sha(model):
    h=hashlib.sha256()
    for k,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def validate_sources():
    m=json.loads((SOURCE/'source_manifest.json').read_text());assert m['revision']=='3212e42564493f520593e5414af4367fc4b49226'
    for row in m['files']:assert sha(SOURCE/row['path'])==row['sha256']
    assert sha(WEIGHT)=='414435b508391f92afadd2aaeec418c806776aeccbce12e638d73a139ca5ca78'
    setup=json.loads((ROOT/'outputs/asset_setup_v1/manifest.json').read_text());assert setup['status']=='complete_verified'
    for row in setup['receipts']:assert sha(ROOT/row['path'])==row['sha256']
    return {'source_manifest_sha256':sha(SOURCE/'source_manifest.json'),'asset_manifest_sha256':sha(ROOT/'outputs/asset_setup_v1/manifest.json'),
            'common_sha256':sha(__file__),'plan_sha256':sha(ROOT/'QUALIFICATION_PLAN.md')}
def load_model():
    for name,version in [('transformers','4.40.1'),('tokenizers','0.19.1'),('huggingface-hub','0.23.5'),('regex','2024.5.15')]:
        assert metadata.version(name)==version,(name,metadata.version(name))
        assert pathlib.Path(metadata.distribution(name).locate_file('')).resolve()==(ROOT/'vendor').resolve()
    package=types.ModuleType('sundial_audit34');package.__path__=[str(SOURCE)];sys.modules[package.__name__]=package
    Config=importlib.import_module('sundial_audit34.configuration_sundial').SundialConfig
    Model=importlib.import_module('sundial_audit34.modeling_sundial').SundialForPrediction
    cfg=Config(**json.loads((SOURCE/'config.json').read_text()))
    assert cfg.output_token_lens==[720] and cfg.num_sampling_steps==50
    model=Model(cfg);model.load_state_dict(load_file(str(WEIGHT)),strict=True);model.eval().cuda()
    for p in model.parameters():p.requires_grad_(False)
    return model
def cache_history(model,hist):
    assert hist.ndim==2 and hist.shape[1]==672 and torch.isfinite(hist).all()
    mean=hist.mean(-1,keepdim=True);scale=hist.std(-1,keepdim=True,unbiased=False)+1e-5
    with torch.no_grad():z=model.model(input_ids=(hist-mean)/scale,return_dict=True,use_cache=False).last_hidden_state[:,-1]
    return {'z':z.detach(),'mean':mean.detach(),'scale':scale.detach()}
def from_noise(model,cache,noise,horizon=96):
    # noise order exactly matches native repeat(samples, 1), then reshape.
    assert noise.ndim==3 and noise.shape[1]==len(cache['z']) and noise.shape[2]==720 and 0<horizon<=720
    s,b,_=noise.shape;e=noise.reshape(s*b,720);z=cache['z'].repeat(s,1);x=e
    steps=model.flow_loss.num_sampling_steps;dt=1./steps
    for i in range(steps):
        t=(torch.ones((len(x),))*i/steps).to(x.device)
        pred=model.flow_loss.net(x,t*1000,z);x=x+(pred-e)*dt
    y=x.reshape(s,b,720).transpose(0,1)[:,:,:horizon]
    return y*cache['scale'][:,None,:]+cache['mean'][:,None,:]
def runtime():
    names=['torch','transformers','huggingface-hub','tokenizers','safetensors','numpy','pandas','pyarrow','regex']
    return {'python':sys.executable,'packages':{n:metadata.version(n) for n in names},'gpu':torch.cuda.get_device_name(0),
            'torch_threads':torch.get_num_threads(),'tf32':torch.backends.cuda.matmul.allow_tf32}
