"""Load untouched native numerical source, with a bounded ragged max adapter."""
import runtime_profile
import hashlib,importlib.util,json,pathlib,sys,types
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];SOURCE=ROOT/'upstream/explorethecontext'
sys.path.insert(0,str(ROOT/'runtime_dependencies'));sys.path.insert(0,str(SOURCE))
if importlib.util.find_spec('torch_scatter') is None:
    shim=types.ModuleType('torch_scatter')
    def scatter_max(src,index,dim=0,out=None):
        assert dim==0 and src.ndim==2 and index.ndim==1 and out is not None
        assert index.shape[0]==src.shape[0] and bool((src>=0).all())
        # This native encoder only aggregates nonnegative ReLU features; the
        # caller discards argmax. Torch's max retains its ordinary gradient.
        arg=torch.full_like(out,src.shape[0],dtype=torch.long)
        for group in range(out.shape[0]):
            ids=torch.where(index==group)[0]
            if ids.numel():
                vals,which=src[ids].max(0);out[group]=vals;arg[group]=ids[which]
        return out,arg
    def unsupported(*args,**kwargs):raise RuntimeError('Unsupported optional torch_scatter operation; do not silently approximate it.')
    shim.scatter_max=scatter_max;shim.scatter_add=unsupported;sys.modules['torch_scatter']=shim
from context_exploration.model.context_encoder import get_context_encoder
from context_exploration.model.transition_model import get_transition_model
from context_exploration.data.envs import make_env
def verify_source():
    m=json.loads((ROOT/'outputs/explore_context_all_source_manifest.json').read_text())
    for r in m['files']:assert hashlib.sha256((SOURCE/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    return m['revision']
def restricted_checkpoint(checkpoint):
    # Author checkpoint validation_loss contains a NumPy float64 scalar.
    names=set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))
    assert names<={'numpy.core.multiarray.scalar','numpy.dtype'},names
    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64'))]):
        result=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert set(result)=={'transition_model','context_encoder','validation_loss','step','is_best'}
    assert isinstance(result['validation_loss'],(int,float,np.floating))
    assert all(isinstance(v,torch.Tensor) for k in ['transition_model','context_encoder'] for v in result[k].values())
    return result
def load(config,checkpoint,device='cuda:0'):
    verify_source();payload=restricted_checkpoint(checkpoint)
    env=make_env(config['env_name'])
    model=get_transition_model(config['transition_model_classname'],env.state_dim,env.action_dim,config['context_dim'],config['embedding_dim'],config['transition_model_kwargs'])
    encoder=get_context_encoder(config['context_encoder_classname'],env.state_dim,env.action_dim,config['context_dim'],config['context_encoder_kwargs'])
    model.load_state_dict(payload['transition_model'],strict=True);encoder.load_state_dict(payload['context_encoder'],strict=True)
    return env,model.to(device).eval(),encoder.to(device).eval()
